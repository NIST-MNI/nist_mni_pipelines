#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date   
#

# standard library
import string
import os
import argparse
import pickle

try:
    import cPickle 
except ImportError:
    pass
    
import sys
import json
import csv
# minc
from minc2_simple import minc2_xfm, minc2_file
 

# numpy
import numpy as np

# scikit-learn
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn import tree
#from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import dummy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

# XGB package
try:
    import xgboost as xgb
except ImportError:
    pass

# MINC stuff
from   ipl.minc_tools import mincTools,mincError
import traceback


def prepare_features(images, coords, mask=None, use_coord=True, use_joint=True, patch_size=1, primary_features=1 ):
    features=[]

    # add features dependant on coordinates

    image_no=len(images)
    if primary_features > image_no or primary_features<0 :
        primary_features=image_no
    # use with center at 0 and 1.0 at the edge, could have used preprocessing
    image_idx=0
    if use_coord:
        image_idx=3
        if coords is None:
            c=np.mgrid[ 0:images[0].shape[0] ,
                        0:images[0].shape[1] ,
                        0:images[0].shape[2] ]
                        
            features.append( ( c[2]-images[0].shape[0]/2.0)/ (images[0].shape[0]/2.0) )
            features.append( ( c[1]-images[0].shape[1]/2.0)/ (images[0].shape[1]/2.0) )
            features.append( ( c[0]-images[0].shape[2]/2.0)/ (images[0].shape[2]/2.0) )
            
        else: # assume we have three sets of coords
            features.append( coords[0] )
            features.append( coords[1] )
            features.append( coords[2] )


    # add apparance and context images (patch around each voxel)
    if patch_size>0:
        for i in range(primary_features) :
            for x in range(-patch_size,patch_size+1) :
                for y in range (-patch_size,patch_size+1) :
                    for z in range(-patch_size,patch_size+1) :
                        features.append( np.roll( np.roll( np.roll( images[i], shift=x, axis=0 ), shift=y, axis=1), shift=z, axis=2 ) )
                        
        features.extend(images[primary_features:-1]) # add the rest
        app_features=primary_features*(patch_size*2+1)*(patch_size*2+1)*(patch_size*2+1)+(image_no-primary_features)
        primary_features=primary_features*(patch_size*2+1)*(patch_size*2+1)*(patch_size*2+1)
    else:
        features.extend(images)
        app_features=image_no
     
    # add joint features
    if use_joint and use_coord:
        for i in range(primary_features):
            # multiply apparance features by coordinate features 
            for j in range(3):
                # multiply apparance features by coordinate features 
                features.append( features[i+image_idx] * features[j] )

    # extract only what's needed
    if mask is not None:
      return [  i[ mask>0 ] for i in features ]
    else:
      return [  i for i in features ]


def convert_image_list(images):
    '''
    convert array of images into a single matrix
    '''
    s=[]
    for (i,k) in enumerate(images):
        s.append(np.column_stack( tuple( np.ravel( j ) for j in k ) ) )
        print(s[-1].shape)

    return np.vstack( tuple( i for i in s ) )


def extract_part(img, partition, part, border):
    '''
    extract slice of the image for parallelized execution
    '''
    if partition is None or part is None :
        return img
    else:
        strip=img.shape[2]//partition
        beg=strip*part
        end=strip*(part+1)

        if part>0:
            beg-=border
        if part<(partition-1):
            end+=border
        else :
            end=img.shape[2]
        return img[:,:,beg:end]


def pad_data(img, shape, partition, part, border):
    if partition is None or part is None :
        return img
    else:
        out = np.zeros(shape, dtype=img.dtype)
        strip = shape[2]//partition
        
        beg=strip*part
        end=strip*(part+1)
        
        _beg=0
        _end=img.shape[2]
        
        if part>0:
            beg-=border
            
        if part<(partition-1):
            end+=border
        else :
            end=shape[2]

        out[:,:,beg:end]=img[:,:,_beg:_end]
        return out
        

def merge_segmentations(inputs, output, partition, parameters):
    patch_size=parameters.get('patch_size',1)
    border=patch_size*2
    out=None
    strip=None
    for i in range(len(inputs)):
        d = minc2_file(inputs[i]).data
        if out is None:
            out = np.zeros(d.shape,dtype=np.int32)
            strip = d.shape[2]/partition
            
        beg = strip*i
        end = strip*(i+1)
        
        if i==(partition-1):
            end=d.shape[2]
            
        out[:,:,beg:end]=d[:,:,beg:end]
        
    out_i = minc2_file()
    out_i.imitate(inputs[0], path=output)
    out_i.data = out


def errorCorrectionTrain(input_images, 
                         output, 
                         parameters=None, 
                         debug=False, 
                         partition=None, 
                         part=None, 
                         multilabel=1):
    try:
        use_coord  = parameters.get('use_coord',True)
        use_joint  = parameters.get('use_joint',True)
        patch_size = parameters.get('patch_size',1)

        border=patch_size*2

        if patch_size==0:
            border=2

        normalize_input=parameters.get('normalize_input',True)

        method        = parameters.get('method','lSVC')
        method2       = parameters.get('method2',method)
        method_n      = parameters.get('method_n',15)
        method2_n     = parameters.get('method2_n',method_n)
        method_random = parameters.get('method_random',None)
        method_max_features=parameters.get('method_max_features','auto')
        method_n_jobs=parameters.get('method_n_jobs',1)
        primary_features=parameters.get('primary_features',1)

        training_images = []
        training_diff   = []
        training_images_direct = []
        training_direct = []

        if debug:
            print("errorCorrectionTrain use_coord={} use_joint={} patch_size={} normalize_input={} method={} output={} partition={} part={}".\
                    format(repr(use_coord),repr(use_joint),repr(patch_size),repr(normalize_input),method,output,partition,part))

        coords = None
        total_mask_size = 0
        total_diff_mask_size = 0
        
        for (i,inp) in enumerate(input_images):
            mask = None
            diff = None
            mask_diff = None
            
            if inp[-2] is not None:
                mask = extract_part(minc2_file(inp[-2]).data, partition, part, border)
            
            ground_data  = minc2_file(inp[-1]).data
            auto_data    = minc2_file(inp[-3]).data
            
            ground_shape = ground_data.shape
            ground = extract_part(ground_data, partition, part, border)
            auto   = extract_part(auto_data, partition, part, border)
            
            shape = ground_shape
            if coords is None and use_coord:
                c = np.mgrid[ 0:shape[0], 0:shape[1], 0: shape[2] ]
                coords = [ extract_part( (c[j]-shape[j]/2.0)/(shape[j]/2.0),   partition, part, border ) for j in range(3) ]
            
            features   = [ extract_part( minc2_file(k).data, partition, part, border ) for k in inp[0:-3] ]

            mask_size = shape[0] * shape[1] * shape[2]
            
            if debug:
                print("Training data size:{}".format(len(features)))
                if mask is not None:
                    mask_size = np.sum(mask)
                    print("Mask size:{}".format(mask_size))
                else:
                    print("Mask absent")
            total_mask_size += mask_size
            
            if multilabel>1:
                diff = (ground != auto)
                total_diff_mask_size += np.sum(mask)
                
                if mask is not None:
                    mask_diff = diff & ( mask > 0 )
                    print("Sample {} mask_diff={} diff={}".format(i,np.sum(mask_diff),np.sum(diff)))
                    #print(mask_diff)
                    training_diff.append( diff [ mask>0 ] ) 
                    training_direct.append( ground[ mask_diff ] ) 
                else:
                    mask_diff = diff
                    training_diff.append( diff ) 
                    training_direct.append( ground[ diff ] ) 
                
                training_images.append( prepare_features( 
                                    features, 
                                    coords, 
                                    mask=mask,
                                    use_coord=use_coord, 
                                    use_joint=use_joint,
                                    patch_size=patch_size, 
                                    primary_features=primary_features ) )
                
                training_images_direct.append( prepare_features( 
                                    features, 
                                    coords, 
                                    mask=mask_diff,
                                    use_coord=use_coord, 
                                    use_joint=use_joint,
                                    patch_size=patch_size, 
                                    primary_features=primary_features ) )
                
            else:
                mask_diff=mask
                if mask is not None:
                    training_diff.append( ground[ mask>0 ] ) 
                else:
                    training_diff.append( ground ) 

                
            
                training_images.append( prepare_features( 
                                    features, 
                                    coords, 
                                    mask=mask,
                                    use_coord=use_coord, 
                                    use_joint=use_joint,
                                    patch_size=patch_size, 
                                    primary_features=primary_features ) )

            if debug:
                print("feature size:{}".format(len(training_images[-1])))
            
            if i == 0 and parameters.get('dump',False):
                print("Dumping feature images...")
                for (j,k) in enumerate( training_images[-1] ):
                    test=np.zeros_like( images[0] )
                    test[ mask>0 ]=k
                    out = minc2_file()
                    out.imitate(inp[0], path="dump_{}.mnc".format(j))
                    out.data = test

        # calculate normalization coeffecients
        
        if debug: print("Done")

        clf=None
        clf2=None 

        if total_mask_size>0:
            training_X = convert_image_list( training_images )
            training_Y = np.ravel( np.concatenate( tuple(j for j in training_diff ) ) )

            if debug: print("Fitting 1st...")
            
            if method   == "xgb":
                clf = None
            elif method   == "SVM":
                clf = svm.SVC()
            elif method == "nuSVM":
                clf = svm.NuSVC()
            elif method == 'NC':
                clf = neighbors.NearestCentroid()
            elif method == 'NN':
                clf = neighbors.KNeighborsClassifier(method_n)
            elif method == 'RanForest':
                clf = ensemble.RandomForestClassifier(n_estimators=method_n,
                        n_jobs=method_n_jobs,
                        max_features=method_max_features,
                        random_state=method_random)
            elif method == 'AdaBoost':
                clf = ensemble.AdaBoostClassifier(n_estimators=method_n,random_state=method_random)
            elif method == 'AdaBoostPP':
                clf = Pipeline(steps=[('normalizer', Normalizer()), 
                                      ('AdaBoost', ensemble.AdaBoostClassifier(n_estimators=method_n,random_state=method_random))
                                     ])
            elif method == 'tree':
                clf = tree.DecisionTreeClassifier(random_state=method_random)
            elif method == 'ExtraTrees':
                clf = ensemble.ExtraTreesClassifier(n_estimators=method_n,
                        max_features=method_max_features,
                        n_jobs=method_n_jobs,
                        random_state=method_random)
            elif method == 'Bagging':
                clf = ensemble.BaggingClassifier(n_estimators=method_n,
                        max_features=method_max_features,
                        n_jobs=method_n_jobs,
                        random_state=method_random)
            elif method == 'dumb':
                clf = dummy.DummyClassifier(strategy="constant",constant=0)
            else:
                clf = svm.LinearSVC()

            #scores = cross_validation.cross_val_score(clf, training_X, training_Y)
            #print scores
            if method   == "xgb":
                xg_train = xgb.DMatrix( training_X, label=training_Y)
                param = {}
                num_round = 100
                # use softmax multi-class classification
                param['objective'] = 'multi:softmax'
                # scale weight of positive examples
                param['eta'] = 0.1
                param['max_depth'] = 8
                param['silent'] = 1
                param['nthread'] = 4
                param['num_class'] = 2
                clf = xgb.train(param, xg_train, num_round)
            elif method != 'dumb':
                clf.fit( training_X, training_Y )
            
            if  multilabel>1 and method != 'dumb':
                if debug: print("Fitting direct...")
                
                training_X = convert_image_list( training_images_direct )
                training_Y = np.ravel( np.concatenate( tuple(j for j in training_direct ) ) )
                
                if method2   == "xgb":
                    clf2 = None
                if method2   == "SVM":
                    clf2 = svm.SVC()
                elif method2 == "nuSVM":
                    clf2 = svm.NuSVC()
                elif method2 == 'NC':
                    clf2 = neighbors.NearestCentroid()
                elif method2 == 'NN':
                    clf2 = neighbors.KNeighborsClassifier(method_n)
                elif method2 == 'RanForest':
                    clf2 = ensemble.RandomForestClassifier(n_estimators=method_n,
                            n_jobs=method_n_jobs,
                            max_features=method_max_features,
                            random_state=method_random)
                elif method2 == 'AdaBoost':
                    clf2 = ensemble.AdaBoostClassifier(n_estimators=method_n,random_state=method_random)
                elif method2 == 'AdaBoostPP':
                    clf2 = Pipeline(steps=[('normalizer', Normalizer()), 
                                           ('AdaBoost', ensemble.AdaBoostClassifier(n_estimators=method_n,random_state=method_random))
                                          ])
                elif method2 == 'tree':
                    clf2 = tree.DecisionTreeClassifier(random_state=method_random)
                elif method2 == 'ExtraTrees':
                    clf2 = ensemble.ExtraTreesClassifier(n_estimators=method_n,
                            max_features=method_max_features,
                            n_jobs=method_n_jobs,
                            random_state=method_random)
                elif method2 == 'Bagging':
                    clf2 = ensemble.BaggingClassifier(n_estimators=method_n,
                            max_features=method_max_features,
                            n_jobs=method_n_jobs,
                            random_state=method_random)
                elif method2 == 'dumb':
                    clf2 = dummy.DummyClassifier(strategy="constant",constant=0)
                else:
                    clf2 = svm.LinearSVC()
                
                if method2   == "xgb" :
                    xg_train = xgb.DMatrix( training_X, label=training_Y)
                    
                    param = {}
                    num_round = 100
                    # use softmax multi-class classification
                    param['objective'] = 'multi:softmax'
                    # scale weight of positive examples
                    param['eta'] = 0.1
                    param['max_depth'] = 8
                    param['silent'] = 1
                    param['nthread'] = 4
                    param['num_class'] = multilabel
                    
                    clf2 = xgb.train(param, xg_train, num_round)
                    
                elif method != 'dumb':
                    clf2.fit( training_X, training_Y )

            #print(clf.score(training_X,training_Y))

            if debug: 
                print( clf )
                print( clf2 )
        else:
            print("Warning : zero total mask size!, using null classifier")
            clf = dummy.DummyClassifier(strategy="constant",constant=0)
        
        if method == 'xgb' and method2 == 'xgb':
            #save
            clf.save_model(output)
            clf2.save_model(output+'_2')
        else:
            with open(output,'wb') as f:
                pickle.dump( [clf, clf2] , f, -1)
    
    except mincError as e:
        print("Exception in linear_registration:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in linear_registration:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

def errorCorrectionApply(input_images, 
                         output, 
                         input_mask=None,
                         parameters=None, 
                         debug=False, 
                         history=None,
                         input_auto=None, 
                         partition=None, 
                         part=None,
                         multilabel=1,
                         debug_files=None ):
    try:
        use_coord=parameters.get('use_coord',True)
        use_joint=parameters.get('use_joint',True)
        patch_size=parameters.get('patch_size',1)
        normalize_input=parameters.get('normalize_input',True)
        primary_features=parameters.get('primary_features',1)
        
        method       =parameters.get('method','lSVC')
        method2      =parameters.get('method2',method)

        training=parameters['training']
        
        clf=None
        clf2=None
        
        border=patch_size*2

        if patch_size==0:
            border=2

        if debug: print( "Running error-correction, input_image:{} trining:{} partition:{} part:{} output:{} input_auto:{}".
                        format(repr(input_images), training, partition,part,output,input_auto) )

        if method == 'xgb' and method2 == 'xgb':
            # need to convert from Unicode
            _training=str(training)
            clf = xgb.Booster(model_file=_training)
            if multilabel>1:
                clf2 = xgb.Booster(model_file=_training+'_2')
        else:
            with open(training, 'rb') as f:
                c    = pickle.load(f)
                clf  = c[0]
                clf2 = c[1]

        if debug:
            print( clf  )
            print( clf2 )
            print( "Loading input images..." )

        input_data=[ minc2_file(k).load_complete_volume('float32') for k in input_images ]
        shape=input_data[0].shape
        
        #features = [ extract_part( minc.Image(k, dtype=np.float32).data, partition, part, border) for k in inp[0:-3] ]
        #if normalize_input:
            #features = [ extract_part( preprocessing.scale( k ), partition, part, border) for k in input_data ]
        #else:
        features = [ extract_part( k, partition, part, border) for k in input_data ]

        coords=None

        if use_coord:
            c=np.mgrid[ 0:shape[0] , 0:shape[1] , 0: shape[2] ]
            coords=[ extract_part( (c[j]-shape[j]/2.0)/(shape[j]/2.0), partition, part, border )  for j in range(3) ]

        if debug:
            print("Features data size:{}".format(len(features)))

        mask=None
        
        mask_size=shape[0]*shape[1]*shape[2]
        
        if input_mask is not None:
            mask=extract_part( minc2_file(input_mask).data, partition, part, border )
            mask_size=np.sum( mask )

        out_cls  = None
        out_corr = None

        test_x=convert_image_list ( [ prepare_features( 
                                        features, 
                                        coords,
                                        mask=mask, 
                                        use_coord=use_coord, 
                                        use_joint=use_joint,
                                        patch_size=patch_size, 
                                        primary_features=primary_features ) 
                                  ] )

        if input_auto is not None:
            out_corr = np.copy( extract_part( minc2_file( input_auto ).data, partition, part, border) ) # use input data
            out_cls  = np.copy( extract_part( minc2_file( input_auto ).data, partition, part, border) ) # use input data
        else:
            out_corr = np.zeros( shape, dtype=np.int32 )
            out_cls  = np.zeros( shape, dtype=np.int32 )

        if mask_size>0 and not isinstance(clf, dummy.DummyClassifier):
            if debug:
                print("Running classifier 1 ...")
            
            if method!='xgb':
                pred = np.asarray( clf.predict( test_x ), dtype=np.int32 ) 
            else:
                xg_predict = xgb.DMatrix(test_x)
                pred = np.array( clf.predict( xg_predict ), dtype=np.int32 )

            if debug_files is not None:
                out_dbg = np.zeros( shape, dtype=np.int32 )
                if mask is not None:
                    out_dbg[ mask > 0 ] = pred
                else:
                    out_dbg = pred
                    
                out_dbg_m = minc2_file()
                out_dbg_m.imitate(input_images[0], path=debug_files[0])
                out_dbg_m.data = pad_data(out_dbg, shape, partition, part, border)
            
            if mask is not None:
                out_corr[ mask > 0 ] = pred
            else:
                out_corr = pred
            
            if multilabel > 1 and clf2 is not None:
                if mask is not None:
                    mask=np.logical_and(mask>0, out_corr>0)
                else:
                    mask=(out_corr>0)
                    
                if debug:
                    print("Running classifier 2 ...")

                test_x = convert_image_list ( [ prepare_features( 
                                                features, 
                                                coords,
                                                mask=mask , 
                                                use_coord=use_coord, 
                                                use_joint=use_joint,
                                                patch_size=patch_size, 
                                                primary_features=primary_features ) 
                                        ] )
                if method2!='xgb':
                    pred = np.asarray( clf2.predict( test_x ), dtype=np.int32 ) 
                else:
                    xg_predict = xgb.DMatrix(test_x)
                    pred = np.array( clf2.predict( xg_predict ), dtype=np.int32 )
                
                out_cls[mask > 0] = pred
                
                if debug_files is not None:
                    out_dbg = np.zeros( shape, dtype=np.int32 )
                    if mask is not None:
                        out_dbg[ mask > 0 ] = pred
                    else:
                        out_dbg = pred

                    out_dbg_m = minc2_file()
                    out_dbg_m.imitate(input_images[0], path=debug_files[1])
                    out_dbg_m.data = pad_data(out_dbg, shape, partition, part, border)
                
                
            else:
                out_cls=out_corr
                
        else:
            pass # nothing to do!

        if debug:
            print("Saving output...")

        out = minc2_file()
        out.imitate(input_images[0], path=output)
        out.data = pad_data(out_cls, shape, partition, part, border)

    except mincError as e:
        print("Exception in errorCorrectionApply:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in errorCorrectionApply:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Perform error-correction learning and application')
    
    parser.add_argument('--train',
                    help="Training library in json format, list of lists: <img1>[,img2],<mask>,<auto>,<groud_truth>")
    
    parser.add_argument('--train_csv',
                    help="Training library in CSV format, format <img1>[,img2],<mask>,<auto>,<groud_truth>")
    
    parser.add_argument('--input',
                    help="Automatic seg to be corrected")
    
    parser.add_argument('--output',
                    help="Output image, required for application of method")
    
    parser.add_argument('--param',
                    help="Load error-correction parameters from file")
    
    parser.add_argument('--mask', 
                    help="Region for correction, required for application of method" )
                        
    parser.add_argument('--method',
                    choices=['SVM','lSVM','nuSVM','NN','RanForest','AdaBoost','tree'],
                    default='lSVM',
                    help='Classification algorithm')
    
    parser.add_argument('-n',
                    type=int,
                    help="nearest neighbors",
                    default=15)
    
    parser.add_argument('--debug', 
                    action="store_true",
                    dest="debug",
                    default=False,
                    help='Print debugging information' )
                    
    parser.add_argument('--dump', 
                    action="store_true",
                    dest="dump",
                    default=False,
                    help='Dump first sample features (for debugging)' )
    
    parser.add_argument('--coord', 
                    action="store_true",
                    dest="coord",
                    default=False,
                    help='Use image coordinates as additional features' )
                    
    parser.add_argument('--joint', 
                    action="store_true",
                    dest="joint",
                    default=False,
                    help='Produce joint features between appearance and coordinate' )
    
    parser.add_argument('--random', 
                    type=int,
                    dest="random",
                    help='Provide random state if needed' )
    
    parser.add_argument('--save', 
                    help='Save training results in a file')
    
    parser.add_argument('--load', 
                    help='Load training results from a file')
    
    parser.add_argument('image',
                    help='Input images', nargs='*')
    
    options = parser.parse_args()
    
    return options


if __name__ == "__main__":
    history = minc.format_history(sys.argv)
    
    options = parse_options()
    
    parameters={}
    if options.param is None:
        parameters['method']=options.method
        parameters['method_n']=options.n
        parameters['method_random']=options.random

        parameters['use_coord']=options.coord
        parameters['use_joint']=options.joint
        
    
    # load training images
    if ( (options.train is not None or \
          options.train_csv is not None)  and \
          options.save is not None) :
        
        if options.debug: print("Loading training images...")
        
        train=None
        
        if options.train is not None:
            with open(options.train,'rb') as f:
                train=json.load(f)
        else:
            with open(options.train_csv,'rb') as f:
                train=list(csv.reader(f))

        errorCorrectionTrain(train,options.save,
                             parameters=parameters,
                             debug=options.debug)

    
    elif options.input  is not None and \
         options.image  is not None and \
         options.output is not None:

        if options.load is not None:
            parameters['training']=options.load

        errorCorrectionApply(
           [options.image],options.input,
           options.output,
           input_mask=options.mask,
           debug=options.debug,
           history=history)
   
    else:
        print("Error in arguments, run with --help")


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

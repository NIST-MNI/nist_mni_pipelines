# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 
#

import shutil
import os
import sys
import csv
import copy
import random
import traceback

# MINC stuff
from ipl.minc_tools import mincTools,mincError

# scoop parallel execution
from scoop import futures

from .structures       import *
from .fuse             import *
from .train            import *
from .filter           import *
from .error_correction import *


def train_ec_loo( segmentation_library,
                  segmentation_parameters=None, 
                  ec_parameters=None,
                  debug=False,
                  fuse_variant='fuse',
                  regularize_variant='gc',
                  ec_variant='ec',
                  cleanup=False,
                  ext=False,
                  train_list=None):
    '''Train error correction using leave-one-out cross-validation'''
    # for each N subjects run segmentation and compare
    
    try:
        ec_variant = ec_parameters.get( 'variant'  , ec_variant)
        work_dir   = ec_parameters.get( 'work_dir' , segmentation_library.prefix + os.sep + fuse_variant )
        ec_output  = ec_parameters.get( 'output'   , work_dir + os.sep + ec_variant + '.pickle' )
            
        ec_border_mask          = ec_parameters.get( 'border_mask' , True )
        ec_border_mask_width    = ec_parameters.get( 'border_mask_width' , 3 )
        ec_antialias_labels     = ec_parameters.get( 'antialias_labels' , True )
        ec_blur_labels          = ec_parameters.get( 'blur_labels', 1.0 )
        ec_expit_labels         = ec_parameters.get( 'expit_labels', 1.0 )
        ec_normalize_labels     = ec_parameters.get( 'normalize_labels', True )
        ec_use_raw              = ec_parameters.get( 'use_raw', False )
        ec_split                = ec_parameters.get( 'split', None )
        
        ec_train_rounds         = ec_parameters.get( 'train_rounds', -1 )
        ec_train_cv             = ec_parameters.get( 'train_cv', 1 )
        ec_sample_pick_strategy = ec_parameters.get( 'train_pick', 'random' )
        ec_max_samples          = ec_parameters.get( 'max_samples', -1 )
        modalities              = ec_parameters.get( 'train_modalities', segmentation_library.modalities ) - 1
        
        print("\n\n")
        print("EC modalities:{}".format(modalities))
        print("train_list={}".format(repr(train_list)))
        print("ext={}".format(repr(ext)))
        print("\n\n")
        
        try:
          if not os.path.exists(work_dir):
              os.makedirs(work_dir)
        except: 
          pass
        
        if (train_list is not None) and not isinstance(train_list, list):
            print(repr(train_list))
            with open(train_list,'r') as f:
                train_list=list(csv.reader(f))
        
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        
        # setup parameters to stop early
        local_model_mask=segmentation_library.local_model_mask
        
        # disable EC options if present
        segmentation_parameters['ec_options']=None
        
        ec_train = []
        ec_train_file = work_dir+os.sep+'train_ec_'+ec_variant+'.json'
        #ec_train_library = segmentation_library.library
        ec_work_dirs=[]
        
        
        if not os.path.exists( ec_train_file ):
            results=[]
            
            _train_list = []
            # if we have pre-segmented scans, then we should pre-process training library again (!) and train on pre-segmented scans
            if ext and train_list :
                results2 = []
                
                for (i,j) in enumerate( train_list ):
                    n = os.path.basename( j[0] ).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
                    
                    output_pre_seg =work_dir+os.sep+'pre_'+n
                    ec_work_dir = work_dir+os.sep+'work_pre_'+n
                    
                    
                    #TODO: find out how to select appropriate segmentation
                    train_sample =  j[0]
                    train_segment = j[1]
                    train_add=[]
                    
                    train_presegment = None

                    train_presegment = j[2]
                    train_add = j[ 3: 3 + modalities ]
                    
                    experiment_segmentation_library = copy.deepcopy(segmentation_library)
                    print("Running pre-processing on {} - {}".format(train_sample,train_presegment))
                    
                    results2.append( futures.submit( 
                            fusion_segment,
                            train_sample, 
                            experiment_segmentation_library,
                            work_dir+os.sep+n,
                            parameters=segmentation_parameters,
                            debug=True,
                            work_dir=ec_work_dir,
                            ec_variant='noec',
                            fuse_variant=fuse_variant,
                            regularize_variant=regularize_variant,
                            add=train_add,
                            cleanup=cleanup,
                            presegment=train_presegment,
                            preprocess_only=True
                        ))
                    ###
                print("waiting for {} jobs".format(len(results2)))
                futures.wait(results2, return_when=futures.ALL_COMPLETED)
                print("Finished!")
                #train_list=range()
                
                # now pre-fill training library with freshly pre-processed samples
                for (_i, _j) in enumerate(results2):
                    print("{} - done ".format(_j.result()[1]['bbox_sample'].seg))
                    # raise("Not FINISHED!")
                    sample_id = os.path.basename(train_list[_i][0]).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
                    # include into the training list
                    train_list_i = [i for i, j in enumerate(segmentation_library.library) if j[0].find(sample_id) >= 0]
                    # the output should be either one or two samples, if symmetrized version is used 
                    
                    if len(train_list_i) == 1:
                        # we have a single match!
                        match = segmentation_library.library[train_list_i[0]]
                        
                        train = match[0:2]
                        train.append(_j.result()[1]['bbox_sample'].seg)
                        train.extend(match[2:len(match)])
                        _train_list.append(train)
                    elif len(train_list_i) == 2:
                        # we have left and right samples
                        # we assume that straight is first and flipped is second

                        match = segmentation_library.library[train_list_i[0]]
                        
                        train = match[0:2]
                        train.append(_j.result()[1]['bbox_sample'].seg)
                        train.extend(match[2:len(match)])
                        _train_list.append(train)
                        
                        # flipped version
                        match = segmentation_library.library[train_list_i[1]]
                        
                        train = match[0:2]
                        train.append(_j.result()[1]['bbox_sample'].seg_f)
                        train.extend(match[2:len(match)])
                        _train_list.append(train)
                    else:
                        raise "Unexpected number of matches encountered!"
                    
            else:
                _train_list = segmentation_library.library
            
            segmentation_parameters['run_in_bbox']=True
            if ec_train_cv == 1:
                print("_train_list={}".format(repr(_train_list)))
                if ec_train_rounds > 0 and ec_train_rounds < len(_train_list):

                    if ec_sample_pick_strategy=='random' and ec_max_samples>0:
                        ec_train_library = random.sample(_train_list, ec_max_samples)
                    else:
                        ec_train_library = _train_list[0:ec_max_samples]
                else:
                    ec_train_library = _train_list
                    
                for (_i, _j) in enumerate(ec_train_library):
                    n = os.path.basename(_j[0]).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
                    
                    output_loo_seg=work_dir+os.sep+n
                    ec_work_dir=work_dir+os.sep+'work_ec_'+n
                    
                    #TODO: find out how to select appropriate segmentation
                    train_sample = _j[0]
                    train_segment = _j[1]
                    train_add = []
                    
                    train_presegment = None
                    print(train_sample)

                    if ext:
                        train_presegment=_j[2]
                        train_add = _j[3:3+modalities]
                    else:
                        train_add = _j[2:2+modalities]
                    
                    experiment_segmentation_library = copy.deepcopy(segmentation_library)
                    # remove sample
                    experiment_segmentation_library.library = [ i for i in segmentation_library.library if i[0].find(n)<0 ]
                    
                    results.append( futures.submit( 
                            fusion_segment,
                            train_sample, 
                            experiment_segmentation_library,
                            work_dir+os.sep+n,
                            parameters=segmentation_parameters,
                            debug=debug,
                            work_dir=ec_work_dir,
                            ec_variant='noec',
                            fuse_variant=fuse_variant,
                            regularize_variant=regularize_variant,
                            add=train_add,
                            cleanup=cleanup,
                            presegment=train_presegment
                        ))
                    
                    ec_work_dirs.append(ec_work_dir)
            else:
                validation_library_idx=range(len(_train_list))
                ec_train_library=[]
                for i in range( ec_train_rounds ):
                    ran_file = work_dir + os.sep + ('random_{}_{}.json'.format(ec_variant,i))
                    if not os.path.exists( ran_file ):
                        rem_list=random.sample( validation_library_idx, ec_train_cv )
                        with open( ran_file,'w') as f:
                            json.dump(rem_list,f)
                    else:
                        with open( ran_file,'r') as f:
                            rem_list=json.load(f)
                            
                    # ec_sample_pick_strategy=='random'
                    
                    # list of subjects 
                    rem_items=[ _train_list[j] for j in rem_list ]
                    
                    rem_n=[os.path.basename(j[0]).rsplit('.gz',1)[0].rsplit('.mnc',1)[0] for j in rem_items]
                    rem_lib=[]
                    
                    for j in rem_n:
                        rem_lib.extend( [ k for (k,t) in enumerate( _train_list ) if t[0].find(j)>=0 ] )

                    if debug: print(repr(rem_lib))
                    rem_lib=set(rem_lib)
                    #prepare exclusion list
                    experiment_segmentation_library = copy.deepcopy(segmentation_library)
                    
                    experiment_segmentation_library.library = \
                        [k for j, k in enumerate(segmentation_library['library']) if j not in rem_lib]
                    
                    for j, k in enumerate(rem_items):
                        
                        output_experiment = work_dir+os.sep+'{}_{}_{}'.format(i, rem_n[j], 'ec')
                        ec_work_dir = work_dir+os.sep+'work_{}_{}_{}'.format(i, rem_n[j], fuse_variant)
                        
                        # ???
                        sample = [k[0], k[1]]
                        presegment = None

                        if ext:
                            presegment = k[2]
                            sample.extend(k[3:3+modalities])
                        else:
                            sample.extend(k[2:2+modalities])
                        
                        ec_train_library.append(sample)
                        
                        results.append(futures.submit(
                                fusion_segment,
                                k[0], 
                                experiment_segmentation_library,
                                output_experiment,
                                parameters=segmentation_parameters,
                                debug=debug,
                                work_dir=ec_work_dir,
                                ec_variant='noec',
                                fuse_variant=fuse_variant,
                                regularize_variant=regularize_variant,
                                add=k[2:2+modalities],
                                cleanup=cleanup,
                                presegment=presegment
                            ))
                        ec_work_dirs.append(ec_work_dir)

            futures.wait(results, return_when=futures.ALL_COMPLETED)

            results2=[]
            results3=[]
            
            for (i,j) in enumerate( ec_train_library ):
                train_sample=j[0]
                train_segment=j[1]
                train_add=j[2:2+modalities]
                train_mask=local_model_mask
                auto_segment=results[i].result()[0]
                
                # TODO: use the subject-specific mask somehow?
                if ec_border_mask:
                    train_mask=auto_segment.rsplit( '.mnc',1 )[0] + '_' + ec_variant+'_train_mask.mnc'
                    results2.append( 
                        futures.submit( make_border_mask, 
                                        auto_segment,  
                                        train_mask, 
                                        width=ec_border_mask_width, 
                                        labels=experiment_segmentation_library[ 'classes_number' ]
                        ) )

                # need to split up multilabel segmentation for training
                if experiment_segmentation_library[ 'classes_number' ]>2 and ( not ec_use_raw ) :
                    print("Splitting into individual files: class_number={} use_raw={}".format(experiment_segmentation_library[ 'classes_number' ],ec_use_raw))
                    labels_prefix=auto_segment.rsplit('.mnc', 1)[0]

                    results3.append( futures.submit( split_labels, auto_segment, 
                                                    experiment_segmentation_library.classes_number,
                                                    labels_prefix,
                                                    antialias=ec_antialias_labels,
                                                    blur=ec_blur_labels,
                                                    expit=ec_expit_labels,
                                                    normalize=ec_normalize_labels ) )

                    ec_input=[ train_sample ]
                    ec_input.extend(train_add)

                    ec_input.extend(['{}_{:02d}.mnc'.format(labels_prefix,i) for i in range(experiment_segmentation_library.classes_number) ])
                    ec_input.extend([ auto_segment, train_mask, train_segment ])
                    ec_train.append( ec_input )

                else : # binary label
                    ec_input=[ train_sample ]
                    ec_input.extend(train_add)
                    ec_input.extend([ auto_segment, auto_segment, train_mask, train_segment ])
                    ec_train.append( ec_input )

            if ec_border_mask:
                futures.wait(results2, return_when=futures.ALL_COMPLETED)

            if experiment_segmentation_library.classes_number>2 :
                futures.wait(results3, return_when=futures.ALL_COMPLETED)

            # TODO run Error correction here
            with open(ec_train_file ,'w') as f:
                json.dump(ec_train, f ,indent=1)
        else:
            with open(ec_train_file,'r') as r:
                ec_train = json.load(r)

        if ec_split is None :
            if not os.path.exists( ec_output ) :
                errorCorrectionTrain( ec_train, ec_output , 
                                    parameters=ec_parameters, debug=debug, 
                                    multilabel=segmentation_library[ 'classes_number' ] )
        else:
            results=[]
            for s in range(ec_split):
                
                out=ec_output.rsplit('.pickle',1)[0] + '_' + str(s) + '.pickle'
                
                if not os.path.exists(out):
                    results.append( futures.submit(
                        errorCorrectionTrain, ec_train, out , 
                        parameters=ec_parameters, debug=debug, partition=ec_split, part=s, 
                        multilabel=segmentation_library[ 'classes_number' ] ) )

            futures.wait(results, return_when=futures.ALL_COMPLETED)

        # TODO: cleanup not-needed files here!
        if cleanup:
            for i in ec_work_dirs:
                shutil.rmtree(i)
    except mincError as e:
        print("Exception in train_ec_loo:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in train_ec_loo:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise
     
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

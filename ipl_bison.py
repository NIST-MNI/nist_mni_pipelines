#! /usr/bin/env python

# standard libraries
import string
import os
import argparse
import sys
import csv
import json
import math
# MINC stuff
from minc2_simple import minc2_file,minc2_error

from   ipl.minc_tools    import mincTools,mincError

try:
    import joblib
except ModuleNotFoundError:
    # for old scikit-learn
    from sklearn.externals import joblib
# numpy
import numpy as np

# using joblib to parallelize jobs (?)
from multiprocessing import Pool


class Error(Exception):
    def __init__(self, message):
            self.message = message    


def load_labels( infile ):
    #with minc2_file(infile) as m:
    try:
        if infile is None:
            print("Trying to load None")
        m=minc2_file(infile)
        m.setup_standard_order()
        data=m.load_complete_volume(minc2_file.MINC2_INT)
        return data
    except minc2_error: 
        print("Error loading:",infile)
        raise


def load_image( infile ):
    #with minc2_file(infile) as m:
    try:
        if infile is None:
            print("Trying to load None")
        m=minc2_file(infile)
        m.setup_standard_order()
        data=m.load_complete_volume(minc2_file.MINC2_FLOAT)
        return data
    except minc2_error: 
        print("Error loading:",infile)
        raise

def save_labels( outfile, reference, data, mask=None, history=None ):
    # TODO: add history
    ref=minc2_file(reference)
    out=minc2_file()
    out.define(ref.store_dims(), minc2_file.MINC2_BYTE, minc2_file.MINC2_BYTE)
    out.create(outfile)
    out.copy_metadata(ref)
    out.setup_standard_order()
    if mask is None:
        out.save_complete_volume(data)
    else:
        shape = tuple( out.shape[2-i] for i in range(3) )
        _out = np.zeros(shape, dtype=np.int8)
        print(f'out.shape={out.shape} shape={shape}')
        _out[mask>0] = data
        out.save_complete_volume(_out)


def save_cnt( outfile, reference, data, mask=None, history=None ):
    # TODO: add history
    ref=minc2_file(reference)
    out=minc2_file()
    out.define(ref.store_dims(), minc2_file.MINC2_SHORT, minc2_file.MINC2_FLOAT)
    out.create(outfile)
    out.copy_metadata(ref)
    out.setup_standard_order()

    if mask is None:
        out.save_complete_volume(np.ascontiguousarray(data, dtype=np.float32))
    else:
        shape = tuple( out.shape[2-i] for i in range(3) )
        _out = np.zeros(shape, dtype=np.float32)
        print(f'out.shape={out.shape} shape={shape}')
        _out[mask>0] = data
        out.save_complete_volume(_out)

def load_cnt_volumes(vol_files, mask=None):
    out=[]
    for v,m in zip(vol_files,mask):
        vol=load_image(v)
        out+=[vol[m>0]]
    return out

def load_bin_volumes(vol_files, mask=None):
    out=[]

    if mask is None:
        mask=[None]*len(vol_files)
    for v,m in zip(vol_files, mask):
        vol=load_labels(v)
        if m is not None:
            out+=[vol[m>0]]
        else:
            out+=[vol]
    return out


def read_csv_dict(fname):
    train={}
    # read csv
    with open(fname, newline='') as csvfile: 
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k,v in row.items():
                _k=k.lower()
                if _k in train:
                    train[_k].append(v)
                else:
                    train[_k]=[v]
    return train


def resample_job(in_mnc, out_mnc, ref, xfm, invert_xfm):
    with mincTools() as m:
        m.resample_smooth(in_mnc, out_mnc, order=2, like=ref, transform=xfm, invert_transform=invert_xfm)
    return out_mnc

def load_all_volumes(train, n_cls, modalities=('t1','t2','pd','flair','ir','mp2t1', 'mp2uni'),
    resample=False, atlas_pfx=None, n_jobs=1, inverse_xfm=False,ran_subset=1.0):
    sample_vol={}

    sample_vol["subject"] = np.array(train["subject"])
    sample_vol["mask"]    = load_bin_volumes(train["mask"])
    if ran_subset<1.0:
        #remove random voxels from the mask
        for i,_ in enumerate(sample_vol["mask"]):
           sample_vol["mask"][i] = np.logical_and(sample_vol["mask"][i]>0, np.random.rand(*sample_vol["mask"][i].shape)<=ran_subset)

    if "labels" in train:
        sample_vol["labels"] = load_bin_volumes(train["labels"], mask=sample_vol["mask"])
    
    for m in modalities:
        if m in train:
            sample_vol[m] = load_cnt_volumes(train[m], mask=sample_vol['mask'])

    if resample: # assume that we need to apply transformation to the atlas first
        if "xfm" not in train:
            raise Error("Need to provide XFM files in xfm column")

        with mincTools() as minc:
            with Pool(processes=n_jobs) as pool: 
                jobs={}
                for m in modalities:
                    if m in train:
                        jobs[f'av_{m}']=[]

                        for i,xfm in enumerate(train["xfm"]):
                            jobs[f'av_{m}'].append(
                                pool.apply_async(
                                    resample_job, (f"{atlas_pfx}{m}.mnc", minc.tmp(f"{i}_avg_{m}.mnc"),
                                         train["mask"][i],xfm,inverse_xfm)
                                    )
                            )
                
                for c in range(n_cls):
                    jobs[f'p{c+1}']=[]
                    for i,xfm in enumerate(train["xfm"]):
                        jobs[f'p{c+1}'].append(
                                pool.apply_async(
                                    resample_job, (f"{atlas_pfx}{c+1}.mnc",minc.tmp(f"{i}_p{c+1}.mnc"),
                                         train["mask"][i], xfm, inverse_xfm)
                                    )
                            )
                # collect results of all jobs
                for m in modalities:
                    if m in train:
                        r=[i.get() for i in jobs[f'av_{m}']]
                        sample_vol[f'av_{m}'] = load_cnt_volumes(r, mask=sample_vol['mask'])
                for c in range(n_cls):
                    r=[i.get() for i in jobs[f'p{c+1}']]
                    sample_vol[f'p{c+1}'] = load_cnt_volumes(r, mask=sample_vol['mask'])
        # here all temp files should be removed    
    else:
        for m in modalities:
            if m in train:
                if not f'av_{m}' in train: # TODO: move to sanity check
                    raise Error(f"Missing av_{m}")
                sample_vol[f'av_{m}'] = load_cnt_volumes(train[f'av_{m}'], mask=sample_vol['mask'])
        
        for c in range(n_cls):
            sample_vol[f'p{c+1}'] = load_cnt_volumes(train[f'p{c+1}'], mask=sample_vol['mask'])

    return sample_vol


def estimate_histograms(scans, labels, n_cls, n_bins,subset=None):
    global_hist = np.zeros(shape=(n_bins , n_cls),dtype=float)
    #print("Estimating histogram:",end=' ',flush=True)
    if subset is None:
        subset=np.arange(len(scans))
    
    for i in subset:
        img = scans[i]
        lab = labels[i]
        # TODO: sanity check
        
        # BISON style 
        # img = np.round(img)
        # for nl in range(0 , n_cls):
        #     for j in range(1 , image_range):
        #         PDF_Label[j,nl] = PDF_Label[j,nl] + np.sum((image_vol * (manual_segmentation==(nl+1)) * brain_mask) == j)
        #     PDF_Label[:,nl] = PDF_Label[:,nl] / np.sum(PDF_Label[:,nl])

        # faster with numpy function 
        for c in range(n_cls):
            label_mask = (lab==(c+1))
            (hist,_) = np.histogram(img[label_mask], bins=n_bins, range=(0.0, n_bins), density=True) 
            global_hist[:,c] += hist
        #print('.',end='',flush=True)
    # Done
    # renormalize 
    for c in range(n_cls):
        global_hist[:,c]/=len(subset)
    #print("Done",flush=True)
    return global_hist


def draw_histograms(hist,out,modality='',dpi=100 ):
    import matplotlib
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt
    import matplotlib.cm  as cmx
    import matplotlib.colors as colors

    fig = plt.figure()

    x=np.arange(hist.shape[0])
    for c in range(hist.shape[1]):
        # Plot some data on the (implicit) axes.
        plt.plot(x, hist[:,c], label=f'{c+1}')  
    
    plt.xlabel('Intensity')
    plt.ylabel('Density')
    plt.legend()
    if modality is not None:
        plt.title(modality)

    plt.savefig(out, bbox_inches='tight', dpi=dpi)
    plt.close()
    plt.close('all')


def estimate_all_histograms(sample_vol, n_cls, n_bins, modalities=('t1','t2','pd','flair','ir','mp2t1', 'mp2uni'),subset=None):
    hist={}
    for m in modalities:
        if m in sample_vol:
            hist[m] = estimate_histograms(sample_vol[m], sample_vol['labels'], n_cls, n_bins, subset=subset)
    return hist


def load_XY_item(i, sample_vol, hist,n_cls, n_bins, modalities=('t1','t2','pd','flair','ir','mp2t1', 'mp2uni')):
    if 'labels' in sample_vol:
        Y__ = sample_vol['labels'][i]
    else:
        Y__ = None

    X__=[]
    for m in modalities:
        if m in sample_vol:
            f1 = sample_vol[m][i] # intensity feature
            X__ += [ f1[:, np.newaxis] ]
            f1_i = np.round(f1).astype(int).clip(0,n_bins-1)

            for c in range(n_cls):
                X__ += [ hist[m][ f1_i, c ][:, np.newaxis]  ] # p_int

            X__ += [ sample_vol[f'av_{m}'][i][:, np.newaxis] ] # intensity feature
    ###
    for c in range(n_cls):
        X__ += [ sample_vol[f'p{c+1}'][i][:, np.newaxis] ] # intensity feature

    X__ = np.concatenate(X__, axis=1)
    return X__,Y__


def load_XY(sample_vol, hist, n_cls, n_bins,
            modalities=('t1','t2','pd','flair','ir','mp2t1', 'mp2uni'),
            subset=None, noconcat=False):
    X_=[]
    if 'labels' in train:
        Y_=[]
    else:
        Y_=None
    
    if subset is None:
        subset=np.arange(len(sample_vol['mask']))

    for i in subset:
        (X__,Y__) = load_XY_item(i, sample_vol, hist, n_cls, n_bins, modalities)
        X_.append(X__)
        if 'labels' in train:
            Y_.append(Y__)
    if noconcat:
        return (X_,Y_)
    else:
        return (np.concatenate(X_, axis=0),np.concatenate(Y_, axis=0))


def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Run tissue classifier ')
    
    parser.add_argument('--train',
                        help="training csv")

    parser.add_argument('--infer',
                        help="inference csv, need to specify trained model")
    
    parser.add_argument('--output', 
                        help="Output prefix")

    parser.add_argument('--load', 
                        help="Directiry with pretrained classifier and other files")

    parser.add_argument('--prob', action="store_true",
                        dest="prob",
                        default=False,
                        help='Output probabilities' )

    parser.add_argument('--method',
                        choices=['RF-','RF0','RF1','RF2','RF3','NB','SVC','oSVC','LDA','QDA'],
                        default='RF1',
                        help='Classification algorithm')

    # parser.add_argument('--occ', action="store_true",
    #                     dest="occ",
    #                     default=False,
    #                     help='Apply output code classifier' )

    parser.add_argument('--debug', action="store_true",
                        dest="debug",
                        default=False,
                        help='Print debugging information' )
        
    parser.add_argument('--random', type=int,default=None,
                        dest="random",
                        help='Provide random state if needed for shuffling' )

    parser.add_argument('--n_cls', type=int,
                        dest="n_cls",
                        help='number of non BG classes', default=1 )

    parser.add_argument('--n_jobs', type=int,
                        dest="n_jobs",
                        help='number jobs for classifier', default=1 )
    
    parser.add_argument('--CV', type=int,
                        dest="CV",
                        help='Run cross-validation loop' )

    parser.add_argument('--batch', type=int,
                        dest="batch",
                        help='Batch size for inference', default=1 )

    parser.add_argument('--atlas_pfx', 
                        help='Atlas prefix, if on-line prior resampling is needed', default=1 )

    parser.add_argument('--resample', action="store_true",
                        dest="resample",
                        default=False,
                        help='Resample priors on line, need "xfm" column' )

    parser.add_argument('--inverse_xfm', action="store_true",
                        dest="inverse_xfm",
                        default=False,
                        help='Use invers of the xfm files for resampling (faster for nonlinear xfm)' )

    parser.add_argument('--ran_subset', type=float,
                        dest="ran_subset",
                        help='Random subset (fraction)', default=1.0 )

    parser.add_argument('--subset_seed', type=int,
                        dest="subset_seed",
                        help='Seed for RNG to perform random subset', default=1 )

    options = parser.parse_args()
    
    return options


def get_batch(dict_array,batch,batch_size):
    out={}
    for i in dict_array.keys():
        n=len(dict_array[i])
        out[i]=dict_array[i][batch*batch_size:min((batch+1)*batch_size,n)]
    return out

if __name__ == "__main__":
    options = parse_options()
    n_cls = options.n_cls
    n_bins = 256
    modalities = ('t1', 't2', 'pd', 'flair', 'ir','mp2t1', 'mp2uni')

    if options.train is not None and options.output is not None:
        train = read_csv_dict(options.train)
        # recognized headers:
        # t1,t2,pd,flair,ir,mp2t1,mp2uni
        # pCls<n>,labels,mask
        # minimal set: one modality, p<n>, av_modality, labels, mask  for training 

        if 'labels' not in train:
            print("labels are missing")
            exit(1)
        elif 'mask' not in train: # TODO: train with whole image?
            print('mask is missing')
            exit(1)

        if options.resample:
            if 'xfm' not in train:
                print("Need xfm column")
                exit(1)
        else:
            for i in range(n_cls):
                if f'p{i+1}' not in train:
                    print(f'p{i+1} is missing')
                    exit(1)

        print("Loading all volumes")
        if options.ran_subset<1.0: print("Using random subset:",options.ran_subset)

        _state = np.random.get_state()
        np.random.seed(options.subset_seed)

        sample_vol=load_all_volumes(train, n_cls, modalities=modalities,
            resample=options.resample,atlas_pfx=options.atlas_pfx,inverse_xfm=options.inverse_xfm,
            n_jobs=options.n_jobs,ran_subset=options.ran_subset)

        np.random.set_state(_state)
        n_feat = n_cls # p_spatial 

        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn import svm
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.naive_bayes import GaussianNB

        #TODO: make a proper name for method!!!

        if   options.method == 'RF-':
            clf = RandomForestClassifier(n_estimators = 64,  max_depth=10, verbose=False, random_state=options.random)
        elif options.method == 'RF0':
            clf = RandomForestClassifier(n_estimators = 128,  max_depth=10, verbose=False, random_state=options.random)
        elif options.method == 'RF1':
            clf = RandomForestClassifier(n_estimators = 128,  max_depth=20, verbose=True, random_state=options.random)
        elif options.method == 'RF2':
            clf = RandomForestClassifier(n_estimators = 128,  max_depth=30, verbose=False, random_state=options.random)
        elif options.method == 'RF3':
            clf = RandomForestClassifier(n_estimators = 128,  max_depth=40, verbose=False, random_state=options.random)
        elif options.method == 'SVC':
            clf = make_pipeline(StandardScaler(),svm.LinearSVC(max_iter=1000, C=0.1, dual=False))
        elif options.method == 'oSVC':
            from sklearn.multiclass import OutputCodeClassifier
            clf = make_pipeline(StandardScaler(),svm.LinearSVC(max_iter=1000, C=0.1, dual=False))
            clf = OutputCodeClassifier(estimator=clf, random_state=options.random)
        elif options.method == 'NB':
            clf = make_pipeline(StandardScaler(),GaussianNB(priors=None))
        elif options.method == 'LDA':
            clf = LinearDiscriminantAnalysis(solver='eigen')
        elif options.method == 'QDA':
            clf = QuadraticDiscriminantAnalysis(reg_param=1e-6) # new version!
        else:
            raise  Error(f"Unsupported classifier: {options.method}")
        
        if options.n_jobs is not None:
            clf.n_jobs = options.n_jobs

        print("Classifier:", clf)

        if options.CV is not None:
            folds = options.CV
            # create subset
            subject     = sample_vol['subject']
            unique_subj = np.unique(subject)

            _state = None
            if options.random is not None:
                _state = np.random.get_state()
                np.random.seed(options.random)

            np.random.shuffle( unique_subj )

            if options.random is not None:
                np.random.set_state(_state)

            n_samples = len( unique_subj )
            cv_res={'fold':[], 'cls':[], 'kappa':[], 'subject':[], 'method':[], 'gt_vol':[], 'sa_vol':[] }

            for fold in range(folds):
                print(f"Fold:{fold}")
                training = set( unique_subj[0:(fold * n_samples // folds)]).union(set(unique_subj[((fold + 1) * n_samples // folds):n_samples] ))
                testing  = set( unique_subj[(fold * n_samples // folds): ((fold + 1) * n_samples // folds)] )

                train_subset = np.array([ i for i,s in enumerate(subject) if s in training ],dtype=int)
                test_subset  = np.array([ i for i,s in enumerate(subject) if s in testing  ],dtype=int)
                print("Estimating histogram")
                hist = estimate_all_histograms(sample_vol, n_cls, n_bins, modalities=modalities, subset=train_subset)

                tr_X,tr_Y = load_XY(sample_vol, hist, n_cls, n_bins,subset=train_subset)
                te_X,te_Y = load_XY(sample_vol, hist, n_cls, n_bins,subset=test_subset, noconcat=True)
                te_s      = subject[test_subset]

                print("Training classifier")
                clf = clf.fit(tr_X , tr_Y)
                
                print("Classifying test set:",te_s)
                for x,y,s in zip(te_X,te_Y,te_s):
                    te_out  = clf.predict(x)
                    for c in range(n_cls):
                        gt = (y      == (c+1))
                        sa = (te_out == (c+1))
                        kappa = 2.0 * (gt * sa).sum()/(gt.sum() + sa.sum())
                        gt_vol = float(gt.sum())
                        sa_vol = float(sa.sum())
                        #res[s][str(c)] = kappa
                        print(f"\t{s},{c+1},{kappa},{gt_vol},{sa_vol}")
                        cv_res['fold'].append(fold)
                        cv_res['cls'].append(c+1)
                        cv_res['kappa'].append(kappa)
                        cv_res['subject'].append(s)
                        cv_res['method'].append(options.method)
                        cv_res['gt_vol'].append(gt_vol)
                        cv_res['sa_vol'].append(sa_vol)

            with open(options.output + os.sep + f'{options.method}_cv.json','w') as f:
                json.dump(cv_res, f, indent=2)
        else:
            # 1st stage : estimage intensity histograms
            hist = estimate_all_histograms(sample_vol, n_cls, n_bins, modalities=modalities)
            for i,j in hist.items():
                joblib.dump(j, options.output + os.sep + f'{i}_Label.pkl')
                draw_histograms(j,options.output + os.sep + f'{i}_hist.png', modality=i)

            print("Loading features")
            # load image features
            X,Y = load_XY(sample_vol, hist, n_cls, n_bins)

            # X = np.concatenate(X_, axis=0)
            # Y = np.concatenate(Y_, axis=0)

            clf = clf.fit(X , Y)
            
            path_save_classifier = options.output + os.sep + f'{options.method}.pkl' # TODO: use appropriate name 
            print("Saving results to ", path_save_classifier)
            joblib.dump(clf, path_save_classifier)

    elif options.infer is not None and options.output is not None and options.load is not None:
        infer = read_csv_dict(options.infer)
        # recoginized headers:
        # t1,t2,pd,flair,ir
        # pCls<n>,labels,mask
        # minimal set: one modality, p<n>, av_modality, labels, mask  for training 
        # sanity check
        if 'subject' not in infer:
            raise Error('subject is missing')
        
        if 'mask' not in infer: # TODO: train with whole image?
            raise Error('mask is missing')
        
        present_modalities = []
        hist = {}
        for m in modalities:
            if m in infer:
                if not options.resample and f'av_{m}' not in infer:
                    raise Error(f'missing av_{m}')
                present_modalities.append(m)
                hist[m] = joblib.load(options.load + os.sep + f'{m}_Label.pkl')
        
        # load classifier
        clf = joblib.load(options.load + os.sep + f'{options.method}.pkl') # TODO: use appropriate name
        if options.n_jobs is not None:
            clf.n_jobs = options.n_jobs

        for i in range(n_cls):
            if not options.resample and f'p{i+1}' not in infer:
                raise Error(f'p{i+1} is missing')
        
        nsamp=len(infer['subject'])
        print(f"Processing {nsamp} volumes, batch size:{options.batch}...",flush=True)
        for b in range(math.ceil(nsamp/options.batch)):
            infer_sub = get_batch(infer, b, options.batch)
            infer_vol = load_all_volumes(infer_sub, n_cls, modalities=modalities,
                            resample=options.resample, 
                            atlas_pfx=options.atlas_pfx, 
                            inverse_xfm=options.inverse_xfm,
                            n_jobs=options.n_jobs)
            
            for i,subj in enumerate( infer_vol['subject'] ):
                X_, _  = load_XY_item(i, infer_vol, hist, n_cls, n_bins)
                out  = clf.predict(X_)
                outf = options.output + os.sep + subj +f'_{options.method}.mnc'
                print("Saving:", outf, flush=True)
                save_labels(outf, infer['mask'][i], out, mask=infer_vol['mask'][i])
                if options.prob:
                    # saving probabilites
                    out_p = clf.predict_proba(X_)
                    for c in range(out_p.shape[1]):
                        outf = options.output + os.sep + infer['subject'][i]+f'_p{c}_{options.method}.mnc'
                        print("Saving:", outf, flush=True)
                        save_cnt(outf, infer['mask'][i], out_p[:,c], mask=infer_vol['mask'][i])
            print(f"{b}\t",flush=True,end='')
        print("")
    else:
        print("Error in arguments")
        exit(1)


# kate: indent-width 4; replace-tabs on; remove-trailing-space on; hl python; show-tabs on

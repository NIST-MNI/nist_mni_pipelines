#! /usr/bin/env python

# standard libraries
import string
import os
import argparse
import sys
import csv
import json
import math

from ipl.bison import init_clasifierr, train, run_cv, infer, read_csv_dict,load_all_volumes 

import numpy as np

def parse_options():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
                        choices=['RF-','RF0','RF1','RF2','RF3','NB','SVC','oSVC','LDA','QDA','HGB1','HGB2'],
                        default='RF1',
                        help='Classification algorithm')

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

    parser.add_argument('--symmetric', action="store_true",
                        dest="symmetric",
                        default=False,
                        help='Produce two outputs in inference mode, one regular, another is flipped' )


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


if __name__ == "__main__":
    options = parse_options()
    n_cls = options.n_cls
    
    #modalities = ('t1', 't2', 'pd', 'flair', 'ir','mp2t1', 'mp2uni')

    clf = init_clasifierr(options.method, n_jobs=options.n_jobs, random=options.random)

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

        sample_vol=load_all_volumes(train, n_cls, 
            resample=options.resample,
            atlas_pfx=options.atlas_pfx,
            inverse_xfm=options.inverse_xfm,
            n_jobs=options.n_jobs,ran_subset=options.ran_subset)

        np.random.set_state(_state)
        n_feat = n_cls # p_spatial 

        print("Classifier:", clf)

        if options.CV is not None:
            run_cv(options.CV, sample_vol, random=options.random, 
                   method=options.method,output=options.output,
                   clf=clf, n_cls=n_cls )
        else:
            train(sample_vol, random=options.random, method=options.method,
                  output=options.output, clf=clf, n_cls=n_cls )

    elif options.infer is not None and options.output is not None and options.load is not None:
        input = read_csv_dict(options.infer)

        infer(input,  n_cls=n_cls,  
          resample=options.resample, n_jobs=options.n_jobs,   
          method=options.method, batch=options.batch,
          load_pfx=options.load, atlas_pfx=options.atlas_pfx, 
          inverse_xfm=options.inverse_xfm,
          output=options.output, prob=options.prob,
          progress=True)
    else:
        print("Error in arguments, run with --help")
        exit(1)

# kate: indent-width 4; replace-tabs on; remove-trailing-space on; hl python; show-tabs on

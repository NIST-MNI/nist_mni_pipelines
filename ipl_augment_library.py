#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 12/10/2014
#
# Run fusion segmentation

from __future__ import print_function

import shutil
import os
import sys
import csv
import traceback
import argparse
import json
import tempfile
import re
import copy
import random

# MINC stuff
from ipl.minc_tools import mincTools,mincError

# internal funcions
from ipl.segment import *
from ipl.segment.resample import *
from ipl.segment.structures import *

# scoop parallel execution
from scoop import futures, shared

# 
import numpy as np

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Create augmented dataset for training deep nets')
    
    
    parser.add_argument('source',
                    help="Library source")
    
    parser.add_argument('library',
                    help="Library directory")
    
    parser.add_argument('output',
                    help="Output directory")

    parser.add_argument('-n',type=int,
                        default=10,
                        help="Aplification factor (i.e number of augmented samples per each input",
                        dest='n')
    
    parser.add_argument('--shift',type=float,
                        default=1.0,
                        help="Shift magnitude (mm)")
    
    parser.add_argument('--rot',type=float,
                        default=4.0,
                        help="rotation magnitude (degree)")
    
    parser.add_argument('--scale',type=float,
                        default=2.0,
                        help="Scale magnitude (percent)")
    
    parser.add_argument('--order',type=int,
                        default=2,
                        help="Intensity resample order")
    
    parser.add_argument('--label_order',type=int,
                        default=2,
                        help="Labels resample order")
    
    parser.add_argument('--debug',
                        action="store_true",
                        default=False,
                        help="Debug")
    
    ### TODO
    #parser.add_argument('--samples',
                        #default=None,
                        #help="Provide alternative samples (TODO)")
    
    
    options = parser.parse_args()
    
    if options.debug:
        print(repr(options))
    
    return options


def gen_sample(library, options, source_parameters, sample, idx=0, flip=False):
  try:
    with mincTools() as m:
        
        pre_filters  =        source_parameters.get('pre_filters', None )
        post_filters =        source_parameters.get('post_filters', source_parameters.get( 'filters', None ))
        
        build_symmetric     = source_parameters.get( 'build_symmetric',False)
        build_symmetric_flip= source_parameters.get( 'build_symmetric_flip',False)
        use_fake_masks      = source_parameters.get( 'fake_mask', False )
        
        use_fake_masks      = source_parameters.get( 'fake_mask', False )
        op_mask             = source_parameters.get( 'op_mask','E[2] D[4]')
        lib_sample          = library['library'][idx]
        
        lut                 = library['map']
        if flip:
            lut               = library['flip_map']
        
        # inverse lut
        lut=[ [ _i[1], _i[0] ] for _i in lut.items() ]
        
        
        model      = library['local_model']
        model_mask = library['local_model_mask']
        model_seg  = library.get('local_model_seg',None)
        
        
        mask = None
        
        sample_name=os.path.basename(sample[0]).rsplit('.mnc',1)[0]
        
        if flip:
            sample_name+='_f'
        
        if use_fake_masks:
            mask  = m.tmp('mask.mnc')
            create_fake_mask(sample[1], mask, op=op_mask)
        
        input_dataset = MriDataset(scan=sample[0], seg=sample[1], mask=mask, protect=True)
        filtered_dataset = input_dataset
        # preprocess sample
        # code from train.py
        if pre_filters is not None:
            # apply pre-filtering before other stages
            filtered_dataset = MriDataset( prefix=m.tempdir, name=sample_name )
            filter_sample( input_dataset, filtered_dataset, pre_filters, model=model)
            filtered_dataset.seg =input_samples[j].seg
            filtered_dataset.mask=input_samples[j].mask
        
        m.param2xfm(m.tmp('flip_x.xfm'), scales=[-1.0, 1.0, 1.0])
        
        out_=[]
        for r in range(options.n):
            out_suffix="_{:03d}".format(r)
            
            out_vol  = options.output+ os.sep+ sample_name+ out_suffix+ '_scan.mnc'
            out_seg  = options.output+ os.sep+ sample_name+ out_suffix+ '_seg.mnc'
            out_mask = options.output+ os.sep+ sample_name+ out_suffix+ '_mask.mnc'
            out_xfm  = options.output+ os.sep+ sample_name+ out_suffix+ '.xfm'
            
            if not os.path.exists(out_vol) or not os.path.exists(out_seg) or not os.path.exists(out_mask) or not os.path.exists(out_xfm):
                
                ran_xfm=m.tmp('random_{}.xfm'.format(r))
                
                m.param2xfm(ran_xfm,
                            scales=     ((np.random.rand(3)-0.5)*2*options.scale/100.0+1.0).tolist(),
                            translation=((np.random.rand(3)-0.5)*2*options.shift).tolist(),
                            rotations=  ((np.random.rand(3)-0.5)*2*options.rot).tolist())
                
                if flip:
                    m.xfmconcat([lib_sample[-1], m.tmp('flip_x.xfm'), ran_xfm], out_xfm)
                else:
                    m.xfmconcat([lib_sample[-1], ran_xfm], out_xfm)
                
                # TODO: add nonlinear XFM
                

                if mask is not None:
                    m.resample_labels(mask, out_mask, 
                                    transform=out_xfm, like=model)
                else:
                    out_mask=None
                    
                m.resample_labels(filtered_dataset.seg, out_seg, 
                                transform=out_xfm, order=options.label_order, remap=lut, like=model, baa=True)

                if post_filters is not None:
                    output_scan=m.tmp('scan_{}.mnc'.format(r))
                else:
                    output_scan=out_vol
                # create a file in temp dir first
                m.resample_smooth(filtered_dataset.scan, output_scan, 
                                order=options.order, transform=out_xfm,like=model)

                if post_filters is not None:
                    apply_filter(output_scan, out_vol, post_filters, model=model, 
                                input_mask=out_mask, input_labels=out_seg, model_labels=model_seg)
            out_.append([out_vol, out_seg, out_xfm ])
            
        return out_
  except:
    print("Exception:{}".format(sys.exc_info()[0]))
    traceback.print_exc( file=sys.stdout)
    raise
      
    
if __name__ == '__main__':
    options = parse_options()
    
    if options.source  is not None and \
       options.library is not None and \
       options.output  is not None:
           
        source_parameters={}
        try:
            with open(options.source,'r') as f:
                source_parameters=json.load(f)
        except :
            print("Error loading configuration:{} {}\n".format(options.source, sys.exc_info()[0]),file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            exit( 1)
        
        library=load_library_info( options.library )
        
        samples      =        source_parameters[ 'library' ]
        build_symmetric     = source_parameters.get( 'build_symmetric',False)
        
        # load csv file
        if samples is not list:
            with open(samples,'r') as f:
                samples=list(csv.reader(f))
        
        
        n_samples    =        len(samples)
        #
        if not os.path.exists(options.output):
            os.makedirs(options.output)
        
        outputs=[]
        print(repr(samples))
        for i,j in enumerate( samples ):
            # submit jobs to produce augmented dataset
            outputs.append( futures.submit( 
                gen_sample, library, options, source_parameters, j , idx=i  ) )
            # flipped (?)
            if build_symmetric:
                outputs.append( futures.submit( 
                    gen_sample, library, options, source_parameters, j , idx=i , flip=True ) )
                    
        #
        futures.wait(outputs, return_when=futures.ALL_COMPLETED)
        # generate a new library for augmented samples
        augmented_library=copy.deepcopy(library)
        # wipe all the samples
        augmented_library['library']=[]
        
        for j in outputs:
            augmented_library['library'].extend(j.result())
        
        # save new library description
        save_library_info(augmented_library, options.output)
    else:
        print("Run with --help")
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

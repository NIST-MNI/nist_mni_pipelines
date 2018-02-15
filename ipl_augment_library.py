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
                        help="Resample order")
    
    parser.add_argument('--debug',
                        action="store_true",
                        default=False,
                        help="Debug")
    
    options = parser.parse_args()
    
    if options.debug:
        print(repr(options))
    
    return options


def gen_sample(library, options, sample, i, r):
    with mincTools() as m:
        vol_name=os.path.basename(sample[0]).split('.mnc',1)[0]
        seg_name=os.path.basename(sample[1]).split('.mnc',1)[0]
        
        out_suffix="_{:03d}".format(r)
        out_vol=options.output+os.sep+vol_name+out_suffix+'.mnc'
        out_seg=options.output+os.sep+seg_name+out_suffix+'.mnc'
        out_xfm=options.output+os.sep+vol_name+out_suffix+'_rnd.xfm'
        
        m.param2xfm(out_xfm,
                    scales=     ((np.random.rand(3)-0.5)*2*options.scale/100.0+1.0).tolist(),
                    translation=((np.random.rand(3)-0.5)*2*options.shift).tolist(),
                    rotations=  ((np.random.rand(3)-0.5)*2*options.rot).tolist())
        
        m.resample_labels(sample[1], out_seg, order=options.order, transform=out_xfm)
        m.resample_smooth(sample[0], out_vol, order=options.order, transform=out_xfm)
        
        return [out_vol, out_seg, out_xfm ]
        
        
      
    
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
        
        pre_filters=          source_parameters.get('pre_filters', None )
        post_filters=         source_parameters.get('post_filters', parameters.get( 'filters', None ))
        
        
        #
        if not os.path.exists(options.output):
            os.makedirs(options.output)
        #
        
        outputs=[]
        for i,j in enumerate(library['library']):
            # submit jobs to produce augmented dataset
            for r in range(options.n):
                outputs.append( futures.submit( 
                    gen_sample,library,options,j,i,r ) )
        #
        futures.wait(outputs, return_when=futures.ALL_COMPLETED)
        # generate a new library for augmented samples
        augmented_library=copy.deepcopy(library)
        # wipe all the samples
        augmented_library['library']=[]
        
        augmented_library['library']=[j.result() for j in outputs]
        
        # save new library description
        save_library_info(augmented_library, options.output)
    else:
        print("Run with --help")
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

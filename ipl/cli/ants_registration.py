#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 29/06/2015
#
# registration tools


from __future__ import print_function

import os
import sys
import shutil
import tempfile
import subprocess
import re
import fcntl
import traceback
import collections
import math

# command-line interface
import argparse

# local stuff
from   ipl.minc_tools    import mincTools,mincError
import ipl.ants_registration


def parse_options():
    parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    description="Run ANTs registration" )
    
    parser.add_argument("source",
                    help="Source file")
    
    parser.add_argument("target",
                    help="Target file")
    
    parser.add_argument("--output",
                    help="Output transformation file, MINC xfm format",
                    default=None)
    
    parser.add_argument("--source_mask",
                        default= None,
                        help="Source mask")
    
    parser.add_argument("--target_mask",
                        default= None,
                        help="Target mask")
    
    parser.add_argument("--init",
                        default   = None,
                        help="Initial transformation, minc format")

    parser.add_argument("--downsample",
                        default = None,
                        help="Downsample to given voxel size ",
                        type=float)
    
    parser.add_argument("--start",
                        default = 32,
                        help="Start level ",
                        type=float)
    
    parser.add_argument("--level",
                        default = 2,
                        help="Final level ",
                        type=float)

    parser.add_argument("--iter",
                        default = '20x20x20x20x20',
                        help="Non-linear iterations ")

    
    parser.add_argument("--cost",
                        default="Mattes",
                        help="Cost Function",
                        choices=[    "Mattes",
                                     "CC",
                                     "MI",
                                     "MeanSquares",
                                     "Demons",
                                     "GC"])
    
    parser.add_argument("--par",
                        default="1,32,regular,0.3",
                        help="Cost Function parameters",
                        )
    
    parser.add_argument("--nl",
                    dest="nl",
                    action="store_true",
                    help="Use nonlinear mode",
                    default=False)
    
    parser.add_argument("--lin",
                    dest="nl",
                    action="store_false",
                    help="Use linear mode",
                    default=False)
    
    parser.add_argument("--close",
                    dest="close",
                    action="store_true",
                    help="Start close",
                    default=False)
    
    parser.add_argument("--verbose",
                        default = 0,
                        help="Verbosity level ",
                        type=int)
    
    parser.add_argument("--transform",
                        default=None,
                        help="Transform options, default affine[0.1] for linear and SyN[.25,2,0.5] for nonlinear")
    
    options = parser.parse_args()
    return options


def main():
    options = parse_options()

    if options.source is None or options.target is None:
         print("Error in arguments, run with --help")
         print(repr(options))
    else:

        parameters= { 'conf':  {},
                      'blur':  {}, 
                      'shrink':{}, 
                      'convergence':'1.e-8,20',
                      'cost_function':options.cost,
                      'cost_function_par':options.par,
                      'use_histogram_matching':False,
                      'transformation':'affine[ 0.1 ]'
                    }
        
        if options.nl:
            
            conf=options.iter.split('x')
            
            for (i,j) in zip(range(int(math.log(options.start)/math.log(2)),-1,-1),conf):
                res=2**i
                if res>=options.level:
                    parameters['conf'][str(res)]=j
                    
            if options.transform is not None:
                parameters['transformation']=options.transform
            else:
                parameters['transformation']='SyN[.25,2,0.5]'
            
            ipl.ants_registration.non_linear_register_ants2(
                options.source, options.target, 
                options.output,
                source_mask= options.source_mask,
                target_mask= options.target_mask,
                init_xfm   = options.init,
                parameters = parameters,
                downsample = options.downsample,
                start      = options.start,
                level      = options.level,
                verbose    = options.verbose
                )
        else:
            if options.transform is not None:
                parameters['transformation']=options.transform
                
            ipl.ants_registration.linear_register_ants2( 
                    options.source, options.target, 
                    options.output,
                    source_mask= options.source_mask,
                    target_mask= options.target_mask,
                    init_xfm   = options.init,
                    parameters = parameters,
                    downsample = options.downsample,
                    close      = options.close,
                    verbose    = options.verbose
                    )

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

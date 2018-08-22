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
import argparse

# local stuff
from   ipl.minc_tools import mincTools,mincError
import ipl.registration


def parse_options():
    parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    description="Run minctracc-based registration")
    
    parser.add_argument("--verbose",
                    action="store_true",
                    default=False,
                    help="Be verbose",
                    dest="verbose")
                    
    parser.add_argument("source",
                    help="Source file")
    
    parser.add_argument("target",
                    help="Target file")
    
    parser.add_argument("output_xfm",
                    help="Output transformation file, xfm format")
    
    parser.add_argument("--source_mask",
                        default= None,
                        help="Source mask")
    
    parser.add_argument("--target_mask",
                        default= None,
                        help="Target mask")
    
    parser.add_argument("--init_xfm",
                        default   = None,
                        help="Initial transformation, minc format")
    
    parser.add_argument("--work_dir",
                        default   = None,
                        help="Work directory")
    
    parser.add_argument("--downsample",
                        default = None,
                        help="Downsample to given voxel size ",
                        type=float)

    parser.add_argument("--start",
                        default = None,
                        help="Start level of registration 32 for nonlinear, 16 for linear",
                        type=float)
    
    parser.add_argument("--level",
                        default = 4.0,
                        help="Final level of registration (nl)",
                        type=float)
    
    parser.add_argument("--nl",
                    action="store_true",
                    dest='nl',
                    help="Use nonlinear mode",
                    default=False)
    
    parser.add_argument("--lin",
                    help="Linear mode, default lsq6",
                    default='lsq6')
    
    parser.add_argument("--objective",
                    default="xcorr",
                    help="Registration objective function (linear)")
    
    parser.add_argument("--conf",
                    default="bestlinreg_s2",
                    help="Linear registrtion configuration")
    

    options = parser.parse_args()
    return options


def main():
    options = parse_options()

    if options.source is None or options.target is None:
         print("Error in arguments, run with --help")
         print(repr(options))
    else:
        
        if options.nl :
            if options.start is None:
                options.start=32.0
            
            ipl.registration.non_linear_register_full( 
                        options.source, options.target, options.output_xfm,
                        source_mask= options.source_mask,
                        target_mask= options.target_mask,
                        init_xfm   = options.init_xfm,
                        start      = options.start,
                        level      = options.level,
                        work_dir   = options.work_dir,
                        downsample = options.downsample)
        else:
            if options.start is None:
                options.start=16.0
            _verbose=0
            if options.verbose: _verbose=2
            
            ipl.registration.linear_register(
                        options.source, options.target, options.output_xfm,
                        source_mask= options.source_mask,
                        target_mask= options.target_mask,
                        init_xfm   = options.init_xfm,
                        #start      = options.start,
                        work_dir   = options.work_dir,
                        downsample = options.downsample,
                        objective  = '-'+options.objective,
                        conf       = options.conf,
                        parameters = '-'+options.lin,
                        verbose    = _verbose
                        )
            
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

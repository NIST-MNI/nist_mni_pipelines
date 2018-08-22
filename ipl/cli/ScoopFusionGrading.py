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
from ipl.grading import *

# scoop parallel execution
from scoop import futures, shared


def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Build fusion grading library')
    
    parser.add_argument('--create',
                    help="Create new library with parameters in json format",
                    dest='create')
    
    parser.add_argument('--cv',
                    help="Run cross-validation using existing library with parameters in json format",
                    dest='cv')

    parser.add_argument('--cv-iter',
                    help="Run one iteration from cross validation, use -1 to aggregate all CVs",
                    type=int,
                    dest='cv_iter')

    parser.add_argument('--grade',
                    help="Apply segmentation using provided library",
                    dest='grade')
    
    parser.add_argument('--library',
                    help="Specify library",
                    dest='library')
    
    parser.add_argument('--input',
                    help="input file, required for application of method",
                    dest='input')
    
    parser.add_argument('--options',
                    help="Segmentation options in json format",
                    dest='options')
    
    parser.add_argument('--output',
                    help="Output directory/file, required for application of method",
                    dest='output')
    
    parser.add_argument('--work',
                    help="Work directory, place to store temporary files",
                    dest='work')
    
    parser.add_argument('--mask',
                    help="Input mask",
                    dest='mask')
    
    parser.add_argument('--debug', 
                    action="store_true",
                    dest="debug",
                    default=False,
                    help='Print debugging information' )
    
    parser.add_argument('--cleanup', 
                    action="store_true",
                    dest="cleanup",
                    default=False,
                    help='Remove most temporary files' )
    
    parser.add_argument('--variant_fuse',
                        default='fuse',
                        dest='variant_fuse')
    
    parser.add_argument('--variant_reg',
                        default='ec',
                        dest='variant_reg')

    parser.add_argument('--exclude',
                        dest='exclude',
                        help='exclude based on this regex pattern from the library')
    
    options = parser.parse_args()
    
    if options.debug:
        print(repr(options))
    
    return options


if __name__ == '__main__':
    options = parse_options()
    
    if options.create is not None and options.output is not None:
        create_parameters={}
        try:
            with open(options.create,'r') as f:
                create_parameters=json.load(f)
        except :
            print("Error loading configuration:{} {}".format(options.create,sys.exc_info()[0]),file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            exit( 1)
        try:
            generate_library(create_parameters, options.output, debug=options.debug,
                            cleanup=options.cleanup)
        except :
            print("Error in library generation {}".format(sys.exc_info()[0]),file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            exit(1)
        
    elif options.cv      is not None and \
         options.grade   is not None and \
         options.output  is not None:
 
        cv_parameters={}
        try:
            with open(options.cv,'r') as f:
                cv_parameters=json.load(f)
        except :
            print("Error loading configuration:{}\n{}".format(options.cv,sys.exc_info()[0]),file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            exit(1)

        library=load_library_info( options.grade )

        grading_parameters={}

        if options.options is not None:
            try:
                with open(options.options,'r') as f:
                    grading_parameters=json.load(f)
            except :
                print("Error loading configuration:{}\n{}".format(options.options,sys.exc_info()[0]),file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                exit(1)

        cv_fusion_grading(cv_parameters,
                          library,
                          options.output,
                          grading_parameters,
                          debug=options.debug,
                          cleanup=options.cleanup,
                          cv_iter=options.cv_iter)
            
    elif options.grade is not None and options.input is not None:
        library=load_library_info(options.grade)
        grading_parameters={}
        
        if options.options is not None:
            try:
                with open(options.options,'r') as f:
                    grading_parameters=json.load(f)
            except :
                print("Error loading configuration:{}\n{}".format(options.options,sys.exc_info()[0]),file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                exit(1)
        
        fusion_grading(options.input, library,
                       options.output,
                       input_mask=options.mask,
                       parameters=grading_parameters,
                       debug=options.debug,
                       fuse_variant=options.variant_fuse,
                       regularize_variant=options.variant_reg,
                       work_dir=options.work,
                       cleanup=options.cleanup,
                       exclude_re=options.exclude)
    
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

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

# YAML stuff
import yaml


# MINC stuff
from ipl.minc_tools import mincTools,mincError

# internal funcions
from ipl.segment import *

# scoop parallel execution
from scoop import futures, shared

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Build fusion segmentation library')
    
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
    
    parser.add_argument('--train-ec',
                    help="Train error correction using specified parametrs",
                    dest='train_ec')
    
    parser.add_argument('--ext', 
                    action="store_true",
                    dest="ext",
                    default=False,
                    help='Assume external segmentation is done' )
    
    parser.add_argument('--extlib', 
                    dest="extlib",
                    help='Externally segmented samples, for training : <im1>[,img2..],<auto>,<ground>' )

    parser.add_argument('--presegment', 
                    help='Externally segmented sample, will apply error correction on top' )
    
    parser.add_argument('--library',
                    help="Specify library for error correction",
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
    
    parser.add_argument('--variant_ec',
                        default='ec',
                        dest='variant_ec')
    
    parser.add_argument('--variant_reg',
                        default='ec',
                        dest='variant_reg')
    
    options = parser.parse_args()
    
    if options.debug:
        print(repr(options))
    
    return options


def main():
    options = parse_options()
    
    if options.create is not None and options.output is not None:
        create_parameters={}
        try:
            with open(options.create,'r') as f:
                create_parameters = yaml.safe_load(f)
        except:
            print("Error loading configuration:{} {}\n".format(options.create,sys.exc_info()[0]),file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            exit(1)
        try:
            generate_library(create_parameters, options.output, debug=options.debug,
                             cleanup=options.cleanup)
        except:
            print("Error in library generation {}".format(sys.exc_info()[0]),file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            exit(1)

    elif options.cv is not None and \
         options.library is not None and \
         options.output is not None:

        cv_parameters={}
        try:
            with open(options.cv, 'r') as f:
                cv_parameters = yaml.safe_load(f)
        except:
            print("Error loading configuration:{}\n{}".format(options.cv,sys.exc_info()[0]),file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            exit(1)

        ec_parameters = None
        if options.train_ec is not None:
            try:
                with open(options.train_ec, 'r') as f:
                    ec_parameters = yaml.safe_load(f)
            except:
                print("Error loading configuration:{}\n{}".format(options.train_ec,sys.exc_info()[0]),file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                exit(1)

        library = SegLibrary(options.library)

        segmentation_parameters = {}

        if options.options is not None:
            try:
                with open(options.options, 'r') as f:
                    segmentation_parameters = yaml.safe_load(f)
            except :
                print("Error loading configuration:{}\n{}".format(options.options,sys.exc_info()[0]),file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                exit(1)

        cv_fusion_segment(cv_parameters,
                          library,
                          options.output,
                          segmentation_parameters,
                          ec_parameters=ec_parameters,
                          debug=options.debug,
                          cleanup=options.cleanup,
                          ext=options.ext,
                          extlib=options.extlib,
                          cv_iter=options.cv_iter)

    elif options.train_ec is not None:
        library = SegLibrary(options.library)

        ec_parameters={}
        segmentation_parameters = {}

        with open(options.train_ec,'r') as f:
            ec_parameters = yaml.safe_load(f)

        if options.options is not None:
            with open(options.options,'r') as f:
                segmentation_parameters = yaml.load(f)
        
        train_ec_loo(library,
                     segmentation_parameters=segmentation_parameters, 
                     ec_parameters=ec_parameters,
                     debug=options.debug,
                     cleanup=options.cleanup,
                     ext=options.ext,
                     train_list=options.extlib,
                     fuse_variant=options.variant_fuse,
                     ec_variant=options.variant_ec,
                     regularize_variant=options.variant_reg)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

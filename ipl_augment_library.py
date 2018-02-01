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

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Build fusion segmentation library')
    
    
    parser.add_argument('library',
                    help="Specify input library for error correction",
                    dest='library')
    
    parser.add_argument('output',
                    help="Output directory",
                    dest='output')
    
    parser.add_argument('-n',type=int,
                        default=10,
                        help="Aplification factor (i.e number of augmented samples per each input",
                        dest='n')
    
    options = parser.parse_args()
    
    if options.debug:
        print(repr(options))
    
    return options


if __name__ == '__main__':
    options = parse_options()
    
    if library is not None and output os not None:
        library=load_library_info( options.library )


    else:
        print("Run with --help")
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

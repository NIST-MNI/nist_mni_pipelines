#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 10/07/2011
#
# Generate average model

import shutil
import os
import sys
import csv
import traceback
import argparse

from ipl.minc_tools import mincTools,mincError

from scoop import futures, shared

# high level functions
from ipl.model.generate_linear             import generate_linear_average
from ipl.model.generate_linear             import generate_linear_model
from ipl.model.generate_linear             import generate_linear_model_csv

from ipl.model.generate_nonlinear          import generate_nonlinear_average
from ipl.model.generate_nonlinear          import generate_nonlinear_model_csv
from ipl.model.generate_nonlinear          import generate_nonlinear_model

from ipl.model_ldd.generate_nonlinear_ldd  import generate_ldd_average
from ipl.model_ldd.generate_nonlinear_ldd  import generate_ldd_model_csv
from ipl.model_ldd.generate_nonlinear_ldd  import generate_ldd_model

from ipl.model_ldd.regress_ldd             import regress_ldd
from ipl.model_ldd.regress_ldd             import regress_ldd_csv
from ipl.model_ldd.regress_ldd             import regress_ldd_simple
from ipl.model_ldd.regress_ldd             import build_estimate as build_estimate_ldd

from ipl.model.regress                     import regress
from ipl.model.regress                     import regress_csv
from ipl.model.regress                     import regress_simple
from ipl.model.regress                     import build_estimate as build_estimate_std


def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Build model estimate')

    parser.add_argument('--ldd_estimate',
                    help="Model estimation description (for example results_final.json)",
                    dest='ldd_estimate')
    
    parser.add_argument('--model',
                    help="Model estimation description (for example results_final.json)",
                    dest='ldd_estimate')

    parser.add_argument('--output',
                    help="Output prefix",
                    dest='output')
                    
    parser.add_argument('--int_par_count',
                    help="Intensity paramters count",
                    dest='int_par_count',
                    type=int,default=None
                    )
    
    parser.add_argument('--debug',
                    action="store_true",
                    dest="debug",
                    default=False,
                    help='Print debugging information' )

    parser.add_argument('parameters',
                    help='regression paramters', nargs='*')
                    
    options = parser.parse_args()

    if options.debug:
        print(repr(options))

    return options


def main():
    options = parse_options()

    if options.output is None or options.parameters is None or options.ldd_estimate is None:
         print("Error in arguments, run with --help")
         print(repr(options))
    else:
        parameters=[float(i) for i in options.parameters]
        build_estimate_ldd(options.ldd_estimate,parameters, options.output, int_par_count=options.int_par_count)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

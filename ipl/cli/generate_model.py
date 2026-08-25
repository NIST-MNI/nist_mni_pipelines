#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 10/07/2011
#
# Generate average model

# import shutil
# import os
# import sys
# import csv
# import traceback
import argparse

# from ipl.minc_tools import mincTools,mincError

# high level functions
from ipl.model.generate_linear             import generate_linear_model_csv
from ipl.model.generate_nonlinear          import generate_nonlinear_model_csv
#from ipl.model_ldd.generate_nonlinear_ldd  import generate_ldd_model_csv

# parallel processing
import ray

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Build model average')

    parser.add_argument('--list',
                    help="List of comma separated files: scan,mask",
                    dest='list')

    parser.add_argument('--output',
                    help="Output prefix",
                    dest='output')

    parser.add_argument('--model',
                    help="initial model")
    
    parser.add_argument('--mask',
                    help="initial model mask")

    parser.add_argument('--sym',
                    action="store_true",
                    dest="symmetric",
                    default=False,
                    help='Make symmetric model' )

    parser.add_argument('--nl',
                    action="store_true",
                    dest="nonlinear",
                    default=False,
                    help='Make nonlinear model' )

    parser.add_argument('--debug',
                    action="store_true",
                    dest="debug",
                    default=False,
                    help='Print debugging information' )
    
    parser.add_argument(
        '-q',
        '--quiet',
        help='Suppress some logging messages',
        action='store_true',
        default=False,
        )
    
    parser.add_argument('--ray_start',type=int,
                        help='start local ray instance')
    parser.add_argument('--ray_local',action='store_true',
                        help='local ray (single process)')
    parser.add_argument('--ray_host',
                        help='ray host address')

    parser.add_argument('parameters',
                    help='regression paramters', nargs='*')
                    
    options = parser.parse_args()

    if options.debug:
        print(repr(options))

    return options


def main():
    options = parse_options()

    if options.output is None or options.parameters is None or options.list is None:
         print("Error in arguments, run with --help")
    else:
        if options.ray_start is not None: # HACK?
            ray.init(num_cpus=options.ray_start,log_to_driver=not options.quiet)
        elif options.ray_local:
            ray.init(local_mode=True,log_to_driver=not options.quiet)
        elif options.ray_host is not None:
            ray.init(address=options.ray_host+':6379',log_to_driver=not options.quiet)
        else:
            ray.init(address='auto',log_to_driver=not options.quiet)

    if options.nonlinear:
        generate_nonlinear_model_csv(options.list,
            work_prefix=options.output,
            options={'symmetric':options.symmetric,
                     'protocol': [  {'iter':4,'level':16},
                                    {'iter':4,'level':8},
                                    {'iter':4,'level':4},
                                    {'iter':4,'level':2},
                                ],
                     'cleanup': True,
                     'refine': True
                    },
            model=options.model,
            mask=options.mask,
        )
    else:
        generate_linear_model_csv(options.list,
            work_prefix=options.output,
            options={'symmetric':options.symmetric,
                    'iterations':4,
                    'cleanup': True,
                     'refine': True
                    },
            model=options.model,
            mask=options.mask,
        )

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

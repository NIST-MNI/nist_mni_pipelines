#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 19/01/2016
#
from __future__ import print_function

import shutil
import os
import sys
import csv
import traceback
import argparse
import json
import math

from   ipl.minc_tools    import mincTools,mincError

import ipl.registration
import ipl.ants_registration
import ipl.elastix_registration

from ipl.minc_qc    import qc,qc_field_contour

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Run nonlinear registration')

    parser.add_argument("--debug",
                    action="store_true",
                    dest="debug",
                    default=False,
                    help="Print debugging information" )

    parser.add_argument("--param",
                    help="Registration parameters in json format",
                    dest="param")

    parser.add_argument("--level",
                    help="Final registration level",
                    dest="level",
                    type=float, 
                    default=2.0)

    parser.add_argument("--downsample",
                    help="Downsample input files to lower resolution",
                    dest="downsample",
                    type=float, 
                    default=None)

    parser.add_argument("--start",
                    help="initial registration level",
                    dest="start",
                    type=float, 
                    default=32.0)

    parser.add_argument("--method",
                    help="registration method",
                    dest="method",
                    choices=['std','elastix','ants'],
                    default='std')
    
    parser.add_argument("--metric",
                    help="Metric function (elastix)",
                    dest="metric",
                    default=None)

    parser.add_argument("--objective",
                    help="objectiuve function (std)",
                    dest="objective",
                    default='xcorr')

    parser.add_argument("--init_xfm",
                    help="Initial transform",
                    dest="init_xfm",
                    default=None)

    parser.add_argument("source",
                        help="Source scan")

    parser.add_argument("--source_mask",
                    help="Source mask",
                    dest="source_mask",
                    default=None)
    
    parser.add_argument("target",
                        help="Target scan")

    parser.add_argument("--target_mask",
                    help="Target mask",
                    dest="target_mask",
                    default=None)

    parser.add_argument("output",
                        help="Output transform")

    parser.add_argument("output_inverse",
                        help="Output inverse transform",
                        nargs='?')
    
    parser.add_argument("--qc",
                    help="Generate QC image",
                    dest="qc",
                    default=None)
    
    parser.add_argument("--clobber",
                    help="Overwrite output files",
                    dest="clobber",
                    action="store_true",
                    default=False)
    

    options = parser.parse_args()

    if options.debug:
        print(repr(options))

    return options

def create_qc(source,target,xfm,qc_file):
    with mincTools() as m:
        m.resample_smooth(source,m.tmp('source.mnc'),order=1,like=target,transform=xfm)
        qc(m.tmp('source.mnc'),
            qc_file,
            mask=target,
            image_cmap='red',
            mask_cmap='green',
            use_max=True)# ,ialpha=0.5,oalpha=0.5


def main():
    options = parse_options()

    try:
        if os.path.exists(options.output) and not options.clobber:
            raise mincError('File exists : {}'.format(options.output))
        
        parameters=None
        
        if options.param is not None:
            with open( options.param  ,'r') as f:
                parameters=json.load(f)
        
        if options.method=='ants':
            ipl.ants_registration.non_linear_register_ants2(
                options.source,
                options.target,
                options.output,
                source_mask=options.source_mask,
                target_mask=options.target_mask,
                init_xfm=options.init_xfm,
                parameters=parameters,
                downsample=options.downsample,
                level=options.level,
                start=options.start,
                )
        
        elif options.method=='elastix':
            if parameters is None:
                levels=''
                for i in range(int(math.log(options.start)/math.log(2)),-1,-1):
                    res=2**i
                    if res>=options.level:
                        levels+='{} {} {} '.format(res,res,res)
                parameters={'pyramid':levels,'grid_spacing':options.level*3.0 }

            ipl.elastix_registration.register_elastix(
                options.source,
                options.target,
                output_xfm=options.output,
                source_mask=options.source_mask,
                target_mask=options.target_mask,
                init_xfm=options.init_xfm,
                parameters=parameters,
                downsample=options.downsample,
                downsample_grid=options.level,
                nl=True
                )
        else:
            objective='-'+options.objective
            if parameters is not None:
                objective=parameters.get('objective')
        
            ipl.registration.non_linear_register_full(
                options.source,
                options.target,
                options.output,
                source_mask=options.source_mask,
                target_mask=options.target_mask,
                init_xfm=options.init_xfm,
                parameters=parameters,
                downsample=options.downsample,
                level=options.level,
                start=options.start
                )
        
        if options.qc is not None:
            # create QC image for deformation field
            create_qc(options.source, options.target, options.output, options.qc)
        
    except mincError as e:
        print(str(e),file=sys.stderr)
        traceback.print_exc( file=sys.stderr)
    except :
        print("Exception :{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stderr)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

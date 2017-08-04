#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 
#

from __future__ import print_function

import argparse
import shutil
import os
import sys
import csv
import copy
import json

#import minc

import ipl.elastix_registration
import ipl.minc_tools as minc_tools

import numpy as np 

def parse_options():
    parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    description="Run registration metric")
    
    parser.add_argument("--verbose",
                    action="store_true",
                    default=False,
                    help="Be verbose",
                    dest="verbose")
                    
    parser.add_argument("source",
                    help="Source file")
    
    parser.add_argument("target",
                    help="Target file")
    
    parser.add_argument("output",
                    help="Save output in a file")
    
    parser.add_argument("--exact",
                    action="store_true",
                    default=False,
                    help="Use exact metric",
                    dest="exact")
    
    parser.add_argument("--xfm",
                    help="Apply transform to source before running metric",
                    default=None)
    
    parser.add_argument("--random",
                    help="Apply random transform to source before running metric",
                    default=False,
                    action="store_true"
                    )

    options = parser.parse_args()
    return options


def extract_part(inp,outp,info, x=None, y=None, z=None,parts=None):
    #
    with minc_tools.mincTools() as minc:
        ranges=[
                'zspace={},{}'.format( info['zspace'].length/parts*z , info['zspace'].length/parts ),
                'yspace={},{}'.format( info['yspace'].length/parts*y , info['yspace'].length/parts ),
                'xspace={},{}'.format( info['xspace'].length/parts*x , info['xspace'].length/parts )
                ]
        minc.reshape(inp, outp, dimrange=ranges )

if __name__ == "__main__":
    options = parse_options()
    metric  = 'NormalizedMutualInformation'
    sampler = 'Grid'
    if options.source is None or options.target is None or options.output is None:
         print("Error in arguments, run with --help")
         print(repr(options))
    else:
        #src =  minc.Image(options.source, dtype=np.float32).data
        #trg =  minc.Image(options.target, dtype=np.float32).data
        measures=[]
        with minc_tools.mincTools() as minc:
            # 
            _source=options.source
            if options.xfm is not None:
                _source=minc.tmp("source.mnc")
                minc.resample_smooth(options.source,_source,transform=options.xfm,like=options.target)


            measures={
              'source':options.source,
              'target':options.target,
              
              }

             
            if options.random:
                xfm=minc.tmp('random.xfm')
                rx=np.random.random_sample()*20.0-10.0
                ry=np.random.random_sample()*20.0-10.0
                rz=np.random.random_sample()*20.0-10.0
                tx=np.random.random_sample()*20.0-10.0
                ty=np.random.random_sample()*20.0-10.0
                tz=np.random.random_sample()*20.0-10.0
                minc.param2xfm(xfm,translation=[tx,ty,tz],rotations=[rx,ry,rz])
                measures['rot']=[rx,ry,rz]
                measures['tran']=[tx,ty,tz]
                _source=minc.tmp("source.mnc")
                minc.resample_smooth(options.source,_source,transform=xfm,like=options.target)
                
            src_info=minc.mincinfo(_source)
            trg_info=minc.mincinfo(options.target)
            #
            parts=3
            os.environ['MINC_COMPRESS']='0'
            
            parameters={'metric':metric,
                        'resolutions':1, 
                        'pyramid': '1 1 1',
                        'measure': True,
                        'sampler': sampler,
                        'grid_spacing': '3 3 3',
                        'exact_metric': options.exact,
                        'iterations': 1,
                        'new_samples': False,
                        'optimizer': "AdaptiveStochasticGradientDescent",
                        }
            #
            measures['sim']={'whole':ipl.elastix_registration.register_elastix(_source, options.target, parameters=parameters, nl=False)}
            
            for z in range(parts):
                for y in range(parts):
                    for x in range(parts):
                        # 
                        # extract part
                        src=minc.tmp("src_{}_{}_{}.mnc".format(x,y,z))
                        trg=minc.tmp("trg_{}_{}_{}.mnc".format(x,y,z))
                        #print(1)
                        extract_part(_source,src,src_info,x=x,y=y,z=z,parts=parts)
                        #print(2)
                        extract_part(options.target,trg,trg_info,x=x,y=y,z=z,parts=parts)
                        # run elastix measurement
                        k="{}_{}_{}".format(x,y,z)
                        measures['sim'][k]=ipl.elastix_registration.register_elastix(src, trg, parameters=parameters, nl=False)
            # TODO: parallelize?
            with open(options.output,'w') as f:
                json.dump(measures,f,indent=2)
                
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

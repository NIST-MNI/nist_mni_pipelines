#!/usr/bin/env python
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

# local stuff
import ipl.minc_tools


# hack to make it work on Python 3
try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    str = str
    unicode = str
    bytes = bytes
    basestring = (str,bytes)
else:
    # 'unicode' exists, must be Python 2
    str = str
    unicode = unicode
    bytes = str
    basestring = basestring

def non_linear_register_ldd(
    source, target,
    output_velocity,
    output_xfm=None,
    source_mask=None,
    target_mask=None,
    init_xfm=   None,
    init_velocity=None,
    level=2,
    start=32,
    parameters=None,
    work_dir=None,
    downsample=None
    ):
    """Use log-diffeomorphic demons to run registration"""
    
    with ip.minc_tools.mincTools() as minc:
        if not minc.checkfiles(inputs=[source,target],
                                outputs=[output_velocity]):
            return
        if parameters is None:
            parameters={'conf':{},
                        'smooth_update':2,
                        'smooth_field':2,
                        'update_rule':1,
                        'grad_type':0,
                        'max_step':2.0,
                        'hist_match':True,
                        'LCC': False } 

        LCC=parameters.get('LCC',False)
        
        source_lr=source
        target_lr=target
        source_mask_lr=source_mask
        target_mask_lr=target_mask

        if downsample is not None:
            s_base=os.path.basename(source).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
            t_base=os.path.basename(target).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
            source_lr=minc.tmp(s_base+'_'+str(downsample)+'.mnc')
            target_lr=minc.tmp(t_base+'_'+str(downsample)+'.mnc')

            minc.resample_smooth(source,source_lr,unistep=downsample)
            minc.resample_smooth(target,target_lr,unistep=downsample)

            if target_mask is not None:
                target_mask_lr=minc.tmp(s_base+'_mask_'+str(downsample)+'.mnc')
                minc.resample_labels(target_mask,target_mask_lr,unistep=downsample,datatype='byte')
            if target_mask is not None:
                target_mask_lr=minc.tmp(s_base+'_mask_'+str(downsample)+'.mnc')
                minc.resample_labels(target_mask,target_mask_lr,unistep=downsample,datatype='byte')


        prog=''

        for i in range(int(math.log(start)/math.log(2)),-1,-1):
            res=2**i
            if res>=level:
                prog+=str(parameters['conf'].get(res,20))
            else:
                prog+='0'
            if i>0:
                prog+='x'
                
        inputs=[source,target]
        cmd=None
        
        if LCC:
            cmd=['rpiLCClogDemons',
                    '-f',source_lr,'-m', target_lr,
                    '--output-transform', output_velocity,
                    '-S',str(parameters.get('tradeoff',0.15)),
                    '-u',str(parameters.get('smooth_update',2)),
                    '-d',str(parameters.get('smooth_field',2)),
                    '-C',str(parameters.get('smooth_similarity',3)),
                    '-b',str(parameters.get('bending_weight',1)),
                    '-x',str(parameters.get('harmonic_weight',0)),
                    '-r',str(parameters.get('update_rule',2)),
                    '-g',str(parameters.get('grad_type',0)),
                    '-l',str(parameters.get('max_step',2.0)),
                    '-a',prog ]
            
            if parameters.get('hist_match',True):
                cmd.append('--use-histogram-matching')
            
            # generate programm
            if source_mask_lr is not None:
                cmd.extend(['--mask-image', source_mask_lr])
                inputs.append(source_mask_lr)
            
            if init_velocity is not None:
                cmd.extend(['--initial-transform',init_velocity])
                inputs.append(init_velocity)
        else:
            cmd=['LogDomainDemonsRegistration',
                    '-f',source_lr,'-m', target_lr,
                    '--outputVel-field', output_velocity,
                    '-g',str(parameters.get('smooth_update',2)),
                    '-s',str(parameters.get('smooth_field',2)),
                    '-a',str(parameters.get('update_rule',1)),
                    '-t',str(parameters.get('grad_type',0)),
                    '-l',str(parameters.get('max_step',2.0)),
                    '-i',prog ]

            if parameters.get('hist_match',True):
                cmd.append('--use-histogram-matching')
            
            # generate programm
            if source_mask_lr is not None:
                cmd.extend(['--fixed-mask', source_mask_lr])
                inputs.append(source_mask_lr)
            
            if target_mask_lr is not None:
                cmd.extend(['--moving-mask', target_mask_lr])
                inputs.append(target_mask_lr)

            if init_velocity is not None:
                cmd.extend(['--input-field',init_velocity])
                inputs.append(init_velocity)
            
            if init_xfm is not None:
                cmd.extend(['--input-transform',init_xfm])
                inputs.append(init_xfm)
            
            if output_xfm is not None:
                cmd.extend(['--outputDef-field',output_xfm])
                outputs.append(output_xfm)

        outputs=[output_velocity]
            
        minc.command(cmd, inputs=inputs, outputs=outputs)
        # todo add dependency for masks

def non_linear_register_dd(
    source,
    target,
    output_xfm,
    source_mask=None,
    target_mask=None,
    init_xfm=None,
    level=4,
    start=32,
    parameters=None,
    work_dir=None,
    downsample=None
    ):
    """perform incremental non-linear registration with diffeomorphic demons"""
    
    with ip.minc_tools.mincTools() as minc:
        if not minc.checkfiles(inputs=[source,target],
                                outputs=[output_xfm]):
            return
        
        if parameters is None:
            parameters={'conf':{},
                        'smooth_update':2,
                        'smooth_field':2,
                        'update_rule':0,
                        'grad_type':0,
                        'max_step':2.0,
                        'hist_match':True } 
            
            
        source_lr=source
        target_lr=target
        source_mask_lr=source_mask
        target_mask_lr=target_mask

        if downsample is not None:
            s_base=os.path.basename(source).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
            t_base=os.path.basename(target).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
            source_lr=minc.tmp(s_base+'_'+str(downsample)+'.mnc')
            target_lr=minc.tmp(t_base+'_'+str(downsample)+'.mnc')

            minc.resample_smooth(source,source_lr,unistep=downsample)
            minc.resample_smooth(target,target_lr,unistep=downsample)

            if target_mask is not None:
                target_mask_lr=minc.tmp(s_base+'_mask_'+str(downsample)+'.mnc')
                minc.resample_labels(target_mask,target_mask_lr,unistep=downsample,datatype='byte')
            if target_mask is not None:
                target_mask_lr=minc.tmp(s_base+'_mask_'+str(downsample)+'.mnc')
                minc.resample_labels(target_mask,target_mask_lr,unistep=downsample,datatype='byte')
            
        prog=''

        for i in range(int(math.log(start)/math.log(2)),-1,-1):
            res=2**i
            if res>=level:
                prog+=str(parameters['conf'].get(res,20))
            else:
                prog+='0'
            if i>0:
                prog+='x'
                
        inputs=[source_lr,target_lr]
        cmd=['DemonsRegistration',
                '-f',source_lr,'-m', target_lr,
                '--outputDef-field', output_xfm,
                '-g',str(parameters.get('smooth_update',2)),
                '-s',str(parameters.get('smooth_field',2)),
                '-a',str(parameters.get('update_rule',0)),
                '-t',str(parameters.get('grad_type',0)),
                '-l',str(parameters.get('max_step',2.0)),
                '-i',prog ]

        if parameters.get('hist_match',True):
            cmd.append('--use-histogram-matching')
        # generate programm

        if source_mask_lr is not None:
            cmd.extend(['--fixed-mask', source_mask_lr])
            inputs.append(source_mask_lr)
            
        if target_mask_lr is not None:
            cmd.extend(['--moving-mask', target_mask_lr])
            inputs.append(target_mask_lr)

        if init_xfm is not None:
            cmd.extend(['--input-transform',init_xfm])
            inputs.append(init_xfm)

        outputs=[output_xfm]
            
        minc.command(cmd, inputs=inputs, outputs=outputs)
        # todo add dependency for masks
    
    
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

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

# 
import numpy as np

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
    
    parser.add_argument('--shift',type=float,
                        default=1.0,
                        help="Shift magnitude (mm)")
    
    parser.add_argument('--rot',type=float,
                        default=4.0,
                        help="rotation magnitude (degree)")
    
    parser.add_argument('--scale',type=float,
                        default=2.0,
                        help="Scale magnitude (percent)")
    
    parser.add_argument('--order',type=int,
                        default=2,
                        help="Resample order")
    
    options = parser.parse_args()
    
    if options.debug:
        print(repr(options))
    
    return options


def gen_sample(library, options, sample, i, r):
    with mincTools() as m:
        vol_name=os.path.basename(sample[0]).split('.mnc',1)[0]
        seg_name=os.path.basename(sample[1]).split('.mnc',1)[0]
        
        out_suffix="_{:03d}".format(r)
        out_vol=options.output+os.sep+vol_name+out_suffix+'.mnc'
        out_seg=options.output+os.sep+seg_name+out_suffix+'.mnc'
        out_xfm=options.output+os.sep+vol_name+out_suffix+'_rnd.xfm'
        
        m.param2xfm(out_xfm,
                    scales=((np.random.rand(3)-0.5)*2*options.scale).tolist(),
                    translation=((np.random.rand(3)-0.5)*2*options.shift).tolist(),
                    rotations=((np.random.rand(3)-0.5)*2*options.rot).tolist())
        
        m.resample_labels(sample[1],out_seg,order=options.order,transform=out_xfm)
        m.resample_smooth(sample[0],out_vol,order=options.order,transform=out_xfm)
        
        return [out_vol, out_seg, out_xfm ]
        
        
        
    
    # TODO: generate random non-linear transformations
    # generate sample name (?) 
    # TODO: there should be a way to specify sample IDs and keep the associated with original data
                #my_execute(string.format('param2xfm -clob -translation %f %f %f -rotations %f %f %f -scales %f %f %f %s',
                                        #torch.uniform(-opt.shift,opt.shift),torch.uniform(-opt.shift,opt.shift),torch.uniform(-opt.shift,opt.shift),
                                        #torch.uniform(-opt.rot,opt.rot),torch.uniform(-opt.rot,opt.rot),torch.uniform(-opt.rot,opt.rot),
                                        #1.0+torch.uniform(-opt.scale,opt.scale),1.0+torch.uniform(-opt.scale,opt.scale),1.0+torch.uniform(-opt.scale,opt.scale),
                                        #xfm))

                #-- apply transformation
                #local j

                #local baa=""
                #if opt.order[features+1]>0 then baa="--baa" end
                
                #if opt.classes then
                  #local infile_seg =s[features+1]
                  #local outfile_seg=o[features+1]
                  #my_execute(string.format('itk_resample --byte --labels --order %d  %s %s --transform %s --clob %s', opt.order[features+1], infile_seg, outfile_seg, xfm, baa))
                #end

                #for j=1,features do
                    #local _infile  = s[j]
                    #local _outfile = o[j]
                    
                    #if not opt.discrete[j] then
                        #if not opt.randomize or opt.randomize[j] then
                         #if opt.gain==nil or opt.gain==0.0 or model_opts.mean_sd==nil then
                            #my_execute(string.format('itk_resample --order %d %s %s --transform %s  --clob',opt.order[j], _infile, _outfile, xfm))
                         #else
                            #local tmp_res=paths.tmpname()..'.mnc'
                            #local tmp_random=paths.tmpname()..'.mnc'
                            
                            #local rval=torch.uniform(-opt.intensity, opt.intensity)*model_opts.mean_sd.sd[j]
                            
                            #my_execute(string.format('itk_resample --order %d %s %s --transform %s  --clob', opt.order[j], _infile, tmp_res, xfm ))
                            #my_execute(string.format('random_volume --float --gauss %f %s %s --clob',opt.gain*model_opts.mean_sd.sd[j], tmp_res, tmp_random ))
                            #my_execute(string.format("minccalc -q -clob -express 'A[0]*(1+%f)+A[1]' %s %s %s",rval, tmp_res, tmp_random, _outfile ))
                            
                            #os.remove(tmp_res)
                            #os.remove(tmp_random)
                         #end
                        #else
                         #-- no need to resample this feature
                        #end
                    #else
                        #my_execute(string.format('itk_resample --labels --order %d %s %s --transform %s --clob %s',opt.order[j], _infile, _outfile, xfm,baa))    
    
    


if __name__ == '__main__':
    options = parse_options()
    
    if library is not None and output os not None:
        library=load_library_info( options.library )
        #
        if not os.path.exists(options.output):
            os.makedirs(options.output)
        #
        
        outputs=[]
        for i,j in enumerate(library['library']):
            # submit jobs to produce augmented dataset
            for r in range(options.n):
                outputs.append( futures.submit( 
                    gen_sample,library,options,j,i,r ) )
        #
        futures.wait(outputs, return_when=futures.ALL_COMPLETED)
        # generate a new library for augmented samples
        augmented_library=copy.deepcopy(library)
        # wipe all the samples
        augmented_library['library']=[]
        
        augmented_library['library']=[j.result() for j in outputs]
        
        # save new library description
        save_library_info(augmented_library,options.output)
    else:
        print("Run with --help")
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

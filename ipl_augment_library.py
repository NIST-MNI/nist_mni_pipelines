#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 12/10/2014
#
# Run data augmentation

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
from ipl.segment.resample import *
from ipl.segment.structures import *

# scoop parallel execution
from scoop import futures, shared

# 
import numpy as np

class pca_lib:
  def __init__(self,loc=None):
    self.pca_lib_dir=None
    self.lib=[]
    self.var=[]
    if loc is not None:
      self.load(loc)
    
  def load(self, loc):
    self.pca_lib_dir=os.path.dirname(loc)
    
    self.lib=[]
    self.var=[]
    with open(loc,'r') as f:
        for l in f:
            l=l.rstrip("\n")
            (_v,_f)=l.split(',')
            self.var.append(float(_v))
            self.lib.append(pca_lib_dir+os.sep+_f)
            
    # remove mean
    self.lib.pop(0)
    self.var.pop(0)
    return self.lib
    
  def __len__(self):
    return len(self.lib)
  
  def __getitem__(self,key):
    return self.lib[key]

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Create augmented dataset for training deep nets')
    
    parser.add_argument('source',
                    help="Library source")
    
    parser.add_argument('library',
                    help="Library directory")
    
    parser.add_argument('output',
                    help="Output directory")

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
                        help="Intensity resample order")
    
    parser.add_argument('--label_order',type=int,
                        default=2,
                        help="Labels resample order")
    
    parser.add_argument('--debug',
                        action="store_true",
                        default=False,
                        help="Debug")
    
    parser.add_argument('--intpca',
                    help="Apply intensity variance using PCA library of log fields")
                    
    parser.add_argument('--gridpca',
                    help="Apply nonlinar tranformation variance using PCA library of log fields")
    
    parser.add_argument('--intvar',
                    default=0.1,type=float,
                    help="Intensity variance (log space)")
    
    parser.add_argument('--int_n',
                    default=3,type=int,
                    help="Number of Intensity PCA components (log space)")
                    
    parser.add_argument('--grid_n',
                    default=10,type=int,
                    help="Number of grid PCA components to use")
    ### TODO: augment samples that were segmented using something else
    #parser.add_argument('--samples',
                        #default=None,
                        #help="Provide alternative samples (TODO)")
    
    
    options = parser.parse_args()
    
    if options.debug:
        print(repr(options))
    
    return options


def gen_sample(library, options, source_parameters, sample, idx=0, flip=False, pca_int=None, pca_grid=None):
  try:
    with mincTools() as m:
        
        pre_filters  =        source_parameters.get('pre_filters', None )
        post_filters =        source_parameters.get('post_filters', source_parameters.get( 'filters', None ))
        
        build_symmetric     = source_parameters.get( 'build_symmetric',False)
        build_symmetric_flip= source_parameters.get( 'build_symmetric_flip',False)
        use_fake_masks      = source_parameters.get( 'fake_mask', False )
        
        use_fake_masks      = source_parameters.get( 'fake_mask', False )
        op_mask             = source_parameters.get( 'op_mask','E[2] D[4]')
        
        # Using linear XFM from the library
        # TODO: make route to estimate when not available
        lib_sample          = library['library'][idx]
        
        lut                 = library['map']
        if flip:
            lut               = library['flip_map']
        
        # inverse lut
        lut=[ [ _i[1], _i[0] ] for _i in lut.items() ]
        
        
        model      = library['local_model']
        model_mask = library['local_model_mask']
        model_seg  = library.get('local_model_seg',None)
        
        
        mask = None
        
        sample_name=os.path.basename(sample[0]).rsplit('.mnc',1)[0]
        
        if flip:
            sample_name+='_f'
        
        if use_fake_masks:
            mask  = m.tmp('mask.mnc')
            create_fake_mask(sample[1], mask, op=op_mask)
        
        input_dataset = MriDataset(scan=sample[0], seg=sample[1], mask=mask, protect=True)
        filtered_dataset = input_dataset
        # preprocess sample
        # code from train.py
        if pre_filters is not None:
            # apply pre-filtering before other stages
            filtered_dataset = MriDataset( prefix=m.tempdir, name=sample_name )
            filter_sample( input_dataset, filtered_dataset, pre_filters, model=model)
            filtered_dataset.seg =input_samples[j].seg
            filtered_dataset.mask=input_samples[j].mask
        
        m.param2xfm(m.tmp('flip_x.xfm'), scales=[-1.0, 1.0, 1.0])
        
        out_=[]
        for r in range(options.n):
            out_suffix="_{:03d}".format(r)
            
            out_vol  = options.output+ os.sep+ sample_name+ out_suffix+ '_scan.mnc'
            out_seg  = options.output+ os.sep+ sample_name+ out_suffix+ '_seg.mnc'
            out_mask = options.output+ os.sep+ sample_name+ out_suffix+ '_mask.mnc'
            out_xfm  = options.output+ os.sep+ sample_name+ out_suffix+ '.xfm'
            
            if not os.path.exists(out_vol) or not os.path.exists(out_seg) or not os.path.exists(out_mask) or not os.path.exists(out_xfm):
                
                # apply random linear xfm
                ran_xfm=m.tmp('random_{}.xfm'.format(r))
                
                if pca_grid is None:
                  m.param2xfm(ran_xfm,
                              scales=     ((np.random.rand(3)-0.5)*2*float(options.scale)/100.0+1.0).tolist(),
                              translation=((np.random.rand(3)-0.5)*2*float(options.shift)).tolist(),
                              rotations=  ((np.random.rand(3)-0.5)*2*float(options.rot))  .tolist())
                else:
                  # create a random transform
                  ran_grid=ran_xfm.rsplit('.xfm',1)[0]+'_grid_0.mnc'
                
                  _files=[]
                  cmd=[]
                  _par=((np.random.rand(options.grid_n)-0.5)*2.0*float(options.shift)).tolist()
                  # resample fields first
                  for i in range(options.grid_n):
                      _files.append(pca_grid[i])
                      cmd.append('A[{}]*{}'.format(i+1,_par[i]))
                  cmd='+'.join(cmd)
                  # apply to the output
                  m.calc(_files,cmd,ran_grid)
                  with open(ran_xfm,'w') as f:
                    f.write("MNI Transform File\n\nTransform_Type = Grid_Transform;\nInvert_Flag = True;\nDisplacement_Volume = {};\n".\
                      format(os.path.basename(ran_grid)))
                if flip:
                    m.xfmconcat([lib_sample[-1], m.tmp('flip_x.xfm'), ran_xfm], out_xfm)
                else:
                    m.xfmconcat([lib_sample[-1], ran_xfm], out_xfm)
                                
                if mask is not None:
                    m.resample_labels(mask, out_mask, 
                                    transform=out_xfm, like=model)
                else:
                    out_mask=None
                    
                m.resample_labels(filtered_dataset.seg, out_seg, 
                                transform=out_xfm, order=options.label_order, remap=lut, like=model, baa=True)

                output_scan=m.tmp('scan_{}.mnc'.format(r))
                    
                # create a file in temp dir first
                m.resample_smooth(filtered_dataset.scan, output_scan, 
                                order=options.order, transform=out_xfm,like=model)

                if post_filters is not None:
                    output_scan2=m.tmp('scan2_{}.mnc'.format(r))
                    apply_filter(output_scan, output_scan2, 
                                post_filters, model=model, 
                                input_mask=out_mask, 
                                input_labels=out_seg, 
                                model_labels=model_seg)
                    output_scan=output_scan2
                
                # apply itensity variance
                if pca_int is not None and options.int_n>0:
                    output_scan2=m.tmp('scan3_{}.mnc'.format(r))
                    
                    _files=[output_scan]
                    cmd='A[0]'
                    _par=((np.random.rand(options.int_n)-0.5)*2.0*float(options.intvar)).tolist()
                    # resample fields first
                    for i in range(options.int_n):
                        fld=m.tmp('field_{}_{}.mnc'.format(r,i))
                        m.resample_smooth(pca_int[i], fld, order=1, transform=out_xfm,like=model)
                        _files.append(fld)
                        cmd+='*exp(A[{}]*{})'.format(i+1,_par[i])
                    # apply to the output
                    m.calc(_files,cmd,output_scan2)
                    output_scan=output_scan2
                
                # finally copy to putput
                shutil.copyfile(output_scan,out_vol)
                
            # end of loop    
            out_.append([out_vol, out_seg, out_xfm ])
            
        return out_
  except:
    print("Exception:{}".format(sys.exc_info()[0]))
    traceback.print_exc( file=sys.stdout)
    raise
      
    
if __name__ == '__main__':
    options = parse_options()
    
    if options.source  is not None and \
       options.library is not None and \
       options.output  is not None:
           
        source_parameters={}
        try:
            with open(options.source,'r') as f:
                source_parameters=json.load(f)
        except :
            print("Error loading configuration:{} {}\n".format(options.source, sys.exc_info()[0]),file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            exit( 1)
        
        library=load_library_info( options.library )
        
        samples      =        source_parameters[ 'library' ]
        build_symmetric     = source_parameters.get( 'build_symmetric',False)
        
        # load csv file
        if samples is not list:
            with open(samples,'r') as f:
                samples=list(csv.reader(f))
        
        
        n_samples    =        len(samples)
        #
        if not os.path.exists(options.output):
            os.makedirs(options.output)
        
        pca_int=None
        pca_grid=None
        
        if options.intpca is not None:
            pca_int=pca_lib(options.intpca)
            
        if options.gridpca is not None:
            pca_grid=pca_lib(options.gridpca)
        
        
        outputs=[]
        print(repr(samples))
        print(repr(pca_grid.lib))
        for i,j in enumerate( samples ):
            # submit jobs to produce augmented dataset
            outputs.append( futures.submit( 
                gen_sample, library, options, source_parameters, j , idx=i,pca_grid=pca_grid,pca_int=pca_int ) )
            # flipped (?)
            if build_symmetric:
                outputs.append( futures.submit( 
                    gen_sample, library, options, source_parameters, j , idx=i , flip=True,pca_grid=pca_grid,pca_int=pca_int) )
                    
        #
        futures.wait(outputs, return_when=futures.ALL_COMPLETED)
        # generate a new library for augmented samples
        augmented_library=copy.deepcopy(library)
        # wipe all the samples
        augmented_library['library']=[]
        
        for j in outputs:
            augmented_library['library'].extend(j.result())
        
        # save new library description
        save_library_info(augmented_library, options.output)
    else:
        print("Run with --help")
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

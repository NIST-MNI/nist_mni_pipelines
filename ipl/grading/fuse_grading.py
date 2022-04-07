# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 
#

import shutil
import os
import sys
import csv
import copy
import re
import json

# MINC stuff
from ipl.minc_tools import mincTools,mincError
import ipl.minc_hl as hl

import ray

from .filter           import *
from .structures       import *
from .registration     import *
from .resample         import *
from .preselect        import *
from .qc               import *

import traceback

def fuse_grading( sample, output, library,
                        fuse_options={},
                        flip=False,
                        classes_number=2,
                        model=None,
                        debug=False,
                        fuse_variant='',
                        work_dir=None,
                        groups=None):
    try:
        final_out_seg=output.seg
        final_out_grad=output.scan
        
        scan=sample.scan
        add_scan=sample.add
        output_info={}

        if flip:
            scan=sample.scan_f
            add_scan=sample.add_f
            final_out_seg=output.seg_f
            final_out_grad=output.scan_f
                                                                            
        if not os.path.exists( final_out_grad ):
            with mincTools( verbose=2 ) as m:
                patch=0
                search=0
                threshold=0
                iterations=0
                gco_optimize=False
                nnls=False
                gco_diagonal=False
                label_norm=None
                select_top=None
                if fuse_options is not None:
                    
                    patch=         fuse_options.get('patch',           0)
                    search=        fuse_options.get('search',          0)
                    threshold=     fuse_options.get('threshold',       0.0)
                    iterations=    fuse_options.get('iter',            3)
                    weights=       fuse_options.get('weights',         None)
                    nnls   =       fuse_options.get('nnls',            False)
                    label_norm   = fuse_options.get('label_norm',      None)
                    select_top   = fuse_options.get('top',             None)
                    beta         = fuse_options.get('beta',            None)
                
                if work_dir is None:
                    work_dir=os.path.dirname(output.seg)
                
                dataset_name=sample.name
                
                if flip:
                    dataset_name+='_f'
                    
                output_info['work_dir']=work_dir
                output_info['dataset_name']=work_dir
                
                
                ##out_seg_fuse  = work_dir+os.sep+dataset_name+'_'+fuse_variant+'.mnc'
                out_dist      = work_dir+os.sep+dataset_name+'_'+fuse_variant+'_dist.mnc'
                out_grading   = final_out_grad
                
                output_info['out_seg']=final_out_seg
                output_info['out_grading']=out_grading
                output_info['out_dist']=out_dist
                
                if label_norm is not None:
                    print("Using label_norm:{}".format(repr(label_norm)))
                    # need to create rough labeling  and average
                    segs=['multiple_volume_similarity']
                    segs.extend([ i.seg for i in library ])
                    segs.extend(['--majority', m.tmp('maj_seg.mnc'), '--bg'] )
                    m.execute(segs)
                    
                    scans=[ i.scan for i in library ]
                    m.median(scans,m.tmp('median.mnc'))
                
                    norm_order=label_norm.get('order',3)
                    norm_median=label_norm.get('median',True)
                    
                    n_scan=work_dir+os.sep+dataset_name+'_'+fuse_variant+'_norm.mnc'
                    
                    if flip:
                        n_scan=work_dir+os.sep+dataset_name+'_'+fuse_variant+'_f_norm.mnc'
                    
                    hl.label_normalize(scan,m.tmp('maj_seg.mnc'),m.tmp('median.mnc'),m.tmp('maj_seg.mnc'),out=n_scan,order=norm_order,median=norm_median)
                    scan=n_scan

                if patch==0 and search==0: # perform simple majority voting
                    # create majority voted model segmentation, for ANIMAL segmentation if needed
                    # TODO: figure out what it means for grading
                    segs=['multiple_volume_similarity']
                    segs.extend([ i.seg for i in library ])
                    segs.extend(['--majority', out_seg_fuse, '--bg'] )
                    m.execute(segs)
                else:
                    # create text file for the training library
                    train_lib=os.path.dirname(library[0].seg)+os.sep+sample.name+'.lst'

                    if flip:
                        train_lib=os.path.dirname(library[0].seg)+os.sep+sample.name+'_f.lst'
                    
                    output_info['train_lib']=train_lib
                    
                    with open(train_lib,'w') as f:
                        for i in library:
                            ss=[ os.path.basename(i.scan) ]
                            ss.extend([os.path.basename(j) for j in i.add])
                            ss.append(os.path.basename(i.seg))
                            ss.append(str(i.grading))
                            ss.append(str(i.group))
                            f.write(",".join(ss))
                            f.write("\n")
                    
                    outputs=[]
                    
                    if len(add_scan)>0:
                        segs=['itk_patch_morphology_mc', 
                              scan,
                            '--train',    train_lib, 
                            '--search',   str(search), 
                            '--patch',    str(patch),
                            '--discrete', str(classes_number),
                            '--adist',    out_dist,
                            '--grading',  out_grading]
                        
                        if weights is not None:
                            segs.extend(['--weights',weights])
                            
                        segs.extend(add_scan)
                        segs.extend(['--output', final_out_seg])
                    else:
                        segs=['itk_patch_morphology', scan,
                            '--train',    train_lib, 
                            '--search',   str(search), 
                            '--patch',    str(patch),
                            '--discrete', str(classes_number),
                            '--iter',     str(iterations),
                            '--adist',    out_dist,
                            '--threshold', str(threshold),
                            '--grading',  out_grading,
                            '--verbose' ]
                        segs.append(final_out_seg)
                        
                    if beta is not None: 
                        segs.extend(['--beta',str(beta)])
                    if sample.mask is not None:
                        segs.extend(['--mask', sample.mask])
                    if select_top is not None:
                        segs.extend(['--top',str(select_top)])
                    if groups is not None:
                        segs.extend(['--groups',str(groups)])
                    
                    outputs=[ final_out_seg, out_grading, out_dist  ]

                    m.command(segs, inputs=[sample.scan], outputs=outputs)
                    print(' '.join(segs))
        return output_info
    except mincError as e:
        print("Exception in fuse_segmentations:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in fuse_segmentations:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise

def join_left_right(sample,output_seg,output_grad=None,datatype=None):
    with mincTools() as m:
        cmd=['itk_merge_discrete_labels',sample.seg,sample.seg_f,output]
        if datatype is not None:
            cmd.append('--'+datatype)
        m.command(cmd,inputs=[sample.seg,sample.seg_f],outputs=[output])
        if output_grad is not None:
            # TODO:figure out how to merge gradings
            print("Can't merge gradings yet!")

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

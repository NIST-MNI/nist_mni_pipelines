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

# scoop parallel execution
from scoop import futures, shared

from .filter           import *
from .structures       import *
from .registration     import *
from .resample         import *
from .error_correction import *
from .preselect        import *
from .qc               import *

import traceback

def fuse_segmentations( sample, output, library,
                        fuse_options={},
                        flip=False,
                        classes_number=2,
                        gco_energy=None,
                        ec_options=None,
                        model=None,
                        debug=False,
                        ec_variant='',
                        fuse_variant='',
                        regularize_variant='',
                        work_dir=None ):
    try:
        final_out_seg=output.seg
        scan=sample.scan
        add_scan=sample.add
        output_info={}
        preseg=sample.seg

        if flip:
            scan=sample.scan_f
            add_scan=sample.add_f
            final_out_seg=output.seg_f
            preseg=sample.seg_f
        
        if not os.path.exists( final_out_seg ):
            with mincTools( verbose=2 ) as m:
                if work_dir is None:
                    work_dir=os.path.dirname(output.seg)
                
                dataset_name=sample.name
                
                if flip:
                    dataset_name+='_f'

                out_seg_fuse  = work_dir+os.sep+dataset_name+'_'+fuse_variant+'.mnc'
                out_prob_base = work_dir+os.sep+dataset_name+'_'+fuse_variant+'_prob'
                out_dist      = work_dir+os.sep+dataset_name+'_'+fuse_variant+'_dist.mnc'
                out_seg_reg   = work_dir+os.sep+dataset_name+'_'+fuse_variant+'_'+regularize_variant+'.mnc'
                
                out_seg_ec    = final_out_seg

                output_info['work_dir']=work_dir
                output_info['dataset_name']=work_dir

                if ec_options is None: # skip error-correction part
                    out_seg_reg=out_seg_ec
                    print("ec_options={}".format(repr(ec_options)))
                                
                output_info['out_seg_reg']=out_seg_reg
                output_info['out_seg_fuse']=out_seg_fuse
                output_info['out_dist']=out_dist
                
                probs=[ '{}_{:02d}.mnc'.format(out_prob_base, i) for i in range(classes_number) ]
                
                output_info['probs']=probs
                
                
                if preseg is None:
                    patch=0
                    search=0
                    threshold=0
                    iterations=0
                    gco_optimize=False
                    nnls=False
                    gco_diagonal=False
                    label_norm=None
                    ext_tool=None
                    
                    if fuse_options is not None:
                        # get parameters
                        patch=         fuse_options.get('patch',           0)
                        search=        fuse_options.get('search',          0)
                        threshold=     fuse_options.get('threshold',       0.0)
                        iterations=    fuse_options.get('iter',            3)
                        weights=       fuse_options.get('weights',         None)
                        nnls   =       fuse_options.get('nnls',            False)
                        label_norm   = fuse_options.get('label_norm',      None)
                        beta         = fuse_options.get('beta',            None)
                        new_prog     = fuse_options.get('new',             True)
                        ext_tool     = fuse_options.get('ext',             None)

                        # graph-cut based segmentation
                        gco_optimize = fuse_options.get('gco',             False)
                        gco_diagonal = fuse_options.get('gco_diagonal',    False)
                        gco_wlabel=    fuse_options.get('gco_wlabel',      1.0)
                        gco_wdata =    fuse_options.get('gco_wdata',       1.0)
                        gco_wintensity=fuse_options.get('gco_wintensity',  0.0)
                        gco_epsilon   =fuse_options.get('gco_epsilon',     1e-4)
                    
                    
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
                    if ext_tool is not None: # will run external segmentation tool!
                        # ext_tool is expected to be a string with format language specs
                        segs=ext_tool.format(sample=sample.scan, 
                                              mask=sample.mask, 
                                              output=out_seg_fuse, 
                                              prob_base=out_prob_base,
                                              model_mas=model.mask,
                                              model_atlas=model.seg)
                        outputs=[out_seg_fuse]
                        m.command(segs, inputs=[sample.scan], outputs=outputs)

                        pass #TODO: finish this
                    elif patch==0 and search==0: # perform simple majority voting
                        # create majority voted model segmentation, for ANIMAL segmentation if needed
                        segs=['multiple_volume_similarity']
                        segs.extend([ i.seg for i in library ])
                        segs.extend(['--majority', out_seg_fuse, '--bg'] )
                        m.execute(segs)
                        
                        #TODO:Output fake probs ?
                        
                        if gco_energy is not None and gco_optimize:
                            # todo place this into parameters
                            split_labels( out_seg_fuse,
                                        classes_number,
                                        out_prob_base,
                                        antialias=True,
                                        blur=1.0,
                                        expit=1.0,
                                        normalize=True )
                    else: # run patc-based label fusion
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
                                '--prob',     out_prob_base ]
                            
                            if weights is not None:
                                segs.extend(['--weights',weights])
                                
                            segs.extend(add_scan)
                            segs.extend(['--output', out_seg_fuse])
                        else:
                            if nnls:
                                segs=['itk_patch_segmentation', scan,
                                    '--train',    train_lib, 
                                    '--search',   str(search), 
                                    '--patch',    str(patch),
                                    '--discrete', str(classes_number),
                                    '--iter',     str(iterations),
                                    '--prob',     out_prob_base,
                                    '--adist',    out_dist,
                                    '--nnls',
                                    '--threshold', str(threshold) ]
                            else:
                                if new_prog:
                                    segs=['itk_patch_segmentation','--exp']
                                else:
                                    segs=['itk_patch_morphology']
                                    
                                segs.extend([scan,
                                    '--train',    train_lib, 
                                    '--search',   str(search), 
                                    '--patch',    str(patch),
                                    '--discrete', str(classes_number),
                                    '--iter',     str(iterations),
                                    '--prob',     out_prob_base,
                                    '--adist',    out_dist,
                                    '--threshold', str(threshold) ])
                                if beta is not None:
                                    segs.extend(['--beta',str(beta)])
                                    
                            segs.append(out_seg_fuse)
                        # plug in additional modalities

                        outputs=[ out_seg_fuse ]
                        outputs.extend(probs)
                            
                        if sample.mask is not None:
                            segs.extend(['--mask', sample.mask])
                        print("*****")
                        print(repr(segs))
                        print("*****")
                        m.command(segs, inputs=[sample.scan], outputs=outputs)
                        print(' '.join(segs))
                    
                    if gco_energy is not None and gco_optimize:
                        gco=  [ 'gco_classify', '--cooc', gco_energy ]

                        gco.extend( probs )
                        gco.extend([out_seg_reg, 
                                        '--iter', '1000', 
                                        '--wlabel', str(gco_wlabel), 
                                        '--wdata',  str(gco_wdata), 
                                        '--epsilon', str(gco_epsilon)])

                        if gco_diagonal:
                            gco.append('--diagonal')

                        if gco_wintensity > 0.0:
                            gco.extend( ['--intensity',scan,
                                        '--wintensity',str(gco_wintensit)] )

                        if sample.mask is not None:
                            gco.extend(['--mask', sample.mask])

                        m.command(gco, inputs=probs, outputs=[ out_seg_reg ] )
                    else:
                        shutil.copyfile(out_seg_fuse, out_seg_reg)
                else:
                    #shutil.copyfile(preseg, out_seg_reg)
                    
                    
                    if ec_options is None:
                        shutil.copyfile(preseg,final_out_seg)
                        out_seg_reg=final_out_seg
                    else:
                        out_seg_reg=preseg
                    
                    output_info['out_seg_reg']=out_seg_reg
                    output_info['out_seg_fuse']=out_seg_reg
                    output_info['out_dist']=None
                    output_info['prob']=None
                    #out_seg_reg = preseg
                
                if ec_options is not None:
                    # create ec mask
                    ec_border_mask       = ec_options.get( 'border_mask' , True )
                    ec_border_mask_width = ec_options.get( 'border_mask_width' , 3 )
                    
                    ec_antialias_labels = ec_options.get( 'antialias_labels' , True )
                    ec_blur_labels      = ec_options.get( 'blur_labels', 1.0 )
                    ec_expit_labels     = ec_options.get( 'expit_labels', 1.0 )
                    ec_normalize_labels = ec_options.get( 'normalize_labels', True )
                    ec_use_raw          = ec_options.get( 'use_raw', False )
                    ec_split            = ec_options.get( 'split',   None )
                    
                    train_mask = model.mask
                    ec_input_prefix = out_seg_reg.rsplit('.mnc',1)[0]+'_'+ec_variant
                    
                    if ec_border_mask :
                        train_mask = ec_input_prefix + '_train_mask.mnc'
                        make_border_mask( out_seg_reg,  train_mask, 
                                            width=ec_border_mask_width, labels=classes_number )
                    
                    ec_input=[ scan ]
                    ec_input.extend(sample.add)
                    
                    if classes_number>2 and (not ec_use_raw ):
                        split_labels( out_seg_reg, classes_number, ec_input_prefix,
                                    antialias=ec_antialias_labels,
                                    blur=ec_blur_labels,
                                    expit=ec_expit_labels,
                                    normalize=ec_normalize_labels )
                    
                        ec_input.extend([ '{}_{:02d}.mnc'.format(ec_input_prefix,i) for i in range(classes_number) ]) # skip background feature ?
                    else:
                        ec_input.append( out_seg_reg )# the auto segmentation is 
                    
                    output_info['out_seg_ec']=out_seg_ec
                    
                    if ec_split is None:
                        if ec_variant is not None:
                            out_seg_ec_errors1  = work_dir + os.sep + dataset_name + '_' + fuse_variant+'_'+regularize_variant+'_'+ec_variant+'_error1.mnc'
                            out_seg_ec_errors2  = work_dir + os.sep + dataset_name + '_' + fuse_variant+'_'+regularize_variant+'_'+ec_variant+'_error2.mnc'
                            
                            output_info['out_seg_ec_errors1']=out_seg_ec_errors1
                            output_info['out_seg_ec_errors2']=out_seg_ec_errors2
                        
                        errorCorrectionApply(ec_input, 
                                             out_seg_ec, 
                                             input_mask=train_mask, 
                                             parameters=ec_options, 
                                             input_auto=out_seg_reg, 
                                             debug=debug,
                                             multilabel=classes_number,
                                             debug_files=[out_seg_ec_errors1, out_seg_ec_errors2 ] )
                    else:
                        results=[]
                        parts=[]
                        
                        for s in range(ec_split):
                            out='{}_part_{:d}.mnc'.format(ec_input_prefix,s)
                            train_part=ec_options['training'].rsplit('.pickle',1)[0] + '_' + str(s) + '.pickle'
                            ec_options_part=copy.deepcopy(ec_options)
                            ec_options_part['training']=train_part
                            
                            if ec_variant is not None:
                                out_seg_ec_errors1  = work_dir+os.sep+dataset_name+'_'+fuse_variant+'_'+regularize_variant+'_'+ec_variant+'_error1_'+str(s)+'.mnc'
                                out_seg_ec_errors2  = work_dir+os.sep+dataset_name+'_'+fuse_variant+'_'+regularize_variant+'_'+ec_variant+'_error2_'+str(s)+'.mnc'
                            
                            parts.append(out)
                            results.append( futures.submit(
                                errorCorrectionApply, 
                                                     ec_input, out, 
                                                     input_mask=train_mask, 
                                                     parameters=ec_options_part,
                                                     input_auto=out_seg_reg, 
                                                     debug=debug, 
                                                     partition=ec_split, 
                                                     part=s,
                                                     multilabel=classes_number,
                                                     debug_files=[out_seg_ec_errors1,out_seg_ec_errors2] ))
                        
                        futures.wait(results, return_when=futures.ALL_COMPLETED)
                        merge_segmentations(parts, out_seg_ec, ec_split, ec_options)
        
        return output_info

    except mincError as e:
        print("Exception in fuse_segmentations:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in fuse_segmentations:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise

def join_left_right(sample,output,datatype=None):
    with mincTools() as m:
        cmd=['itk_merge_discrete_labels',sample.seg,sample.seg_f,output]
        if datatype is not None:
            cmd.append('--'+datatype)
        m.command(cmd,inputs=[sample.seg,sample.seg_f],outputs=[output])

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

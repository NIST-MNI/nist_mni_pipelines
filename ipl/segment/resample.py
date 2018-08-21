# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 
#

import shutil
import os
import sys
import csv
import traceback

# MINC stuff
from ipl.minc_tools import mincTools,mincError

from .filter import *


# scoop parallel execution
from scoop import futures, shared

def create_fake_mask(in_seg, out_mask, op=None ):
    try:
        with mincTools() as m :
            if op is None :
                m.calc([in_seg], 'A[0]>0.5?1:0', out_mask, labels=True)
            else :
                m.binary_morphology(in_seg, op, out_mask, binarize_threshold=0.5)
    except mincError as e:
           print("Exception in create_fake_mask:{}".format(repr(e)))
           traceback.print_exc(file=sys.stdout)
           raise
    except : 
           print("Exception in create_fake_mask:{}".format(sys.exc_info()[0]))
           traceback.print_exc(file=sys.stdout)
           raise



def resample_file(input,output,xfm=None,like=None,order=4,invert_transform=False):
    '''resample input file using proveded transformation'''
    try:
        with mincTools() as m:
            m.resample_smooth(input,output,xfm=xfm,like=like,order=order,invert_transform=invert_transform)
    except mincError as e:
        print("Exception in resample_file:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in resample_file:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise


def resample_split_segmentations(input, output,xfm=None, like=None, order=4, invert_transform=False, symmetric=False):
    '''resample individual segmentations, using parallel execution'''
    results=[]
    base=input.seg.rsplit('.mnc',1)[0]
    for (i,j) in input.seg_split.items():
        if not output.seg_split.has_key(i):
            output.seg_split[i]='{}_{:03d}.mnc'.format(base,i)
            
        results.append(futures.submit(
            resample_file,j,output.seg_split[i],xfm=xfm,like=like,order=order,invert_transform=invert_transform
        ))
    if symmetric:
        base=input.seg_f.rsplit('.mnc',1)[0]
        for (i,j) in input.seg_f_split.items():
            if not output.seg_f_split.has_key(i):
                output.seg_split[i]='{}_{:03d}.mnc'.format(base,i)

            results.append(futures.submit(
                resample_file,j,output.seg_f_split[i],xfm=xfm,like=like,order=order,invert_transform=invert_transform
            ))
    futures.wait(results, return_when=futures.ALL_COMPLETED)


def warp_rename_seg( sample, model, output, 
                    transform=None, 
                    symmetric=False, 
                    symmetric_flip=False,
                    lut=None, 
                    flip_lut=None, 
                    resample_order=2,
                    resample_aa=None,
                    resample_baa=False,
                    invert_transform=False,
                    use_flipped=False,
                    datatype=None,
                    create_mask=False,
                    op_mask=None):
    #TODO: should i warp mask if present too?
    try:
        print("warp_rename_seg sampl={} output={} lut={} flip_lut={}".format(repr(sample),repr(output),repr(lut),repr(flip_lut)))
        with mincTools() as m:
            xfm=None
            if transform is not None:
                xfm=transform.xfm
                
                if symmetric:
                    xfm_f=transform.xfm_f

            m.resample_labels(sample.seg, output.seg,
                              transform=xfm,
                              aa=resample_aa,
                              order=resample_order,
                              remap=lut,
                              like=model.scan,
                              invert_transform=invert_transform,
                              datatype=datatype,
                              baa=resample_baa)
            
            if create_mask:
                create_fake_mask(output.seg, output.mask, op=op_mask)
            elif sample.mask is not None:
                m.resample_labels(sample.mask, output.mask,
                                transform=xfm,
                                order=resample_order,
                                like=model.scan,
                                invert_transform=invert_transform,
                                datatype=datatype )
                
            if symmetric:

                seg_f=sample.seg

                if use_flipped:
                    seg_f=sample.seg_f
                
                if symmetric_flip:
                    m.param2xfm(m.tmp('flip_x.xfm'), scales=[-1.0, 1.0, 1.0])
                    xfm_f=m.tmp('flip_x.xfm')

                    if transform is not None: 
                        m.xfmconcat( [m.tmp('flip_x.xfm'), transform.xfm_f ], m.tmp('transform_flip.xfm') )
                        xfm_f=m.tmp('transform_flip.xfm')

                m.resample_labels(seg_f,  output.seg_f,
                                  transform=xfm_f,
                                  aa=resample_aa,
                                  order=resample_order,
                                  remap=flip_lut,
                                  like=model.scan,
                                  invert_transform=invert_transform,
                                  datatype=datatype,
                                  baa=resample_baa)
                if create_mask:
                    create_fake_mask(output.seg_f, output.mask_f, op=op_mask)
                elif sample.mask_f is not None:
                    m.resample_labels(sample.mask_f, output.mask_f,
                                    transform=xfm,
                                    order=resample_order,
                                    like=model.scan,
                                    invert_transform=invert_transform,
                                    datatype=datatype )
    except mincError as e:
        print("Exception in warp_rename_seg:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    
    except :
        print("Exception in warp_rename_seg:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout) 
        raise


def warp_sample( sample,
                 model, 
                 output,
                 transform=None,
                 symmetric=False,
                 symmetric_flip=False,
                 resample_order=None,
                 use_flipped=False,
                 filters=None):
    # TODO: add filters here
    try:
        with mincTools() as m:
            xfm=None
            xfm_f=None
            seg_output=output.seg
            seg_output_f=output.seg_f
            
            if transform is not None:
                xfm=transform.xfm
                if symmetric:
                    xfm_f=transform.xfm_f

            output_scan=output.scan
            
            if filters is not None:
                output_scan=m.tmp('sample.mnc')
            
            m.resample_smooth(sample.scan, output_scan, transform=xfm, like=model.scan, order=resample_order)
            
            if filters is not None:
                # TODO: maybe move it to a separate stage?
                # HACK: assuming that segmentation was already warped!
                apply_filter(output_scan, output.scan, filters, model=model.scan, input_mask=output.mask, input_labels=seg_output, model_labels=model.seg)
            
            for (i,j) in enumerate( sample.add ):
                output_scan = output.add[i]
                if filters is not None:
                    output_scan=m.tmp('sample_{}.mnc').format(i)

                m.resample_smooth(sample.add[i], output_scan, transform=xfm, like=model.scan, order=resample_order)
                
                if filters is not None:
                    # TODO: maybe move it to a separate stage?
                    # TODO: apply segmentations for seg-based filtering
                    apply_filter(output_scan, output.add[i], filters, model=model.scan, input_mask=output.mask, input_labels=seg_output, model_labels=model.seg)

            if symmetric:
                scan_f=sample.scan
                
                if use_flipped:
                    scan_f=sample.scan #TODO: figure out what is it 
                
                if symmetric_flip:
                    m.param2xfm(m.tmp('flip_x.xfm'), scales=[-1.0, 1.0, 1.0])
                    xfm_f=m.tmp('flip_x.xfm')

                    if transform is not None: 
                        m.xfmconcat( [m.tmp('flip_x.xfm'), transform.xfm_f ], m.tmp('transform_flip.xfm') )
                        xfm_f=m.tmp('transform_flip.xfm')

                output_scan_f=output.scan_f
                
                if filters is not None:
                    output_scan_f=m.tmp('sample_f.mnc')
                    
                m.resample_smooth(scan_f, output_scan_f, transform=xfm_f, like=model.scan, order=resample_order)
                
                if filters is not None:
                    # TODO: maybe move it to a separate stage?
                    apply_filter(output_scan_f, output.scan_f, filters, model=model.scan, input_mask=output.mask_f, input_labels=seg_output_f, model_labels=model.seg)

                for (i,j) in enumerate( sample.add_f ):
                    output_scan_f = output.add_f[i]
                    if filters is not None:
                        output_scan_f=m.tmp('sample_f_{}.mnc').format(i)

                    m.resample_smooth( sample.add_f[i], output_scan_f, transform=xfm_f, like=model.scan, order=resample_order)

                    if filters is not None:
                        apply_filter( output_scan_f, output.add_f[i], filters, model=model.scan, input_mask=output.mask_f, input_labels=seg_output_f, model_labels=model.seg)

            output.mask=None
            output.mask_f=None

    except mincError as e:
        print("Exception in warp_sample:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in warp_sample:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise


def warp_model_mask( model,
                 output,
                 transform=None,
                 symmetric=False,
                 symmetric_flip=False,
                 resample_order=None):
    # TODO: add filters here
    try:
        with mincTools() as m:
            xfm=None
            xfm_f=None
            
            if transform is not None:
                xfm=transform.xfm
                if symmetric:
                    xfm_f=transform.xfm_f

            m.resample_labels(model.mask, output.mask, transform=xfm, like=output.scan, invert_transform=True)
            
            if symmetric:
                if symmetric_flip:
                    m.param2xfm(m.tmp('flip_x.xfm'), scales=[-1.0, 1.0, 1.0])
                    xfm_f=m.tmp('flip_x.xfm')

                    if transform is not None: 
                        m.xfmconcat( [m.tmp('flip_x.xfm'), transform.xfm_f ], m.tmp('transform_flip.xfm') )
                        xfm_f=m.tmp('transform_flip.xfm')

                m.resample_labels(model.mask, output.mask_f, transform=xfm_f, like=output.scan_f, invert_transform=True)

    except mincError as e:
        print("Exception in warp_sample:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in warp_sample:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise



def concat_resample(lib_scan,
                    xfm_lib,
                    xfm_sample,
                    output,
                    model=None,
                    resample_aa=None,
                    resample_order=2,
                    resample_baa=False,
                    flip=False ):
    '''Cocnatenate inv(xfm2) and inv(xfm1) and resample scan'''
    try:
        
        if not os.path.exists(output.seg) or \
           not os.path.exists(output.scan) :
            with mincTools() as m:
                _model=None

                if model is not None:
                    _model=model.scan

                full_xfm=None

                if xfm_lib is not None and xfm_sample is not None:
                    if flip:
                        m.xfmconcat([ xfm_sample.xfm_f, xfm_lib.xfm_inv ], m.tmp('Full.xfm') )
                    else:
                        m.xfmconcat([ xfm_sample.xfm,   xfm_lib.xfm_inv ], m.tmp('Full.xfm') )
                    full_xfm=m.tmp('Full.xfm')
                elif xfm_lib is not None:
                    full_xfm=xfm_lib.xfm_inv
                elif xfm_sample is not None:
                    if flip:
                        full_xfm=xfm_sample.xfm_f
                    else:
                        full_xfm=xfm_sample.xfm

                m.resample_labels(lib_scan.seg, output.seg,
                                transform=full_xfm,
                                aa=resample_aa,
                                order=resample_order,
                                like=_model,
                                invert_transform=True,
                                baa=resample_baa )

                m.resample_smooth(lib_scan.scan, output.scan,
                                transform=full_xfm,
                                order=resample_order,
                                like=_model,
                                invert_transform=True)

                for (i,j) in enumerate(lib_scan.add):
                    m.resample_smooth(lib_scan.add[i], output.add[i],
                                transform=full_xfm,
                                order=resample_order,
                                like=_model,
                                invert_transform=True)
    except mincError as e:
        print("Exception in concat_resample:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in concat_resample:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

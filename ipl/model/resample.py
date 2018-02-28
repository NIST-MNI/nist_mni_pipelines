import shutil
import os
import sys
import csv
import traceback

# MINC stuff
from ipl.minc_tools import mincTools,mincError
from .filter import *
from .structures import *

# scoop parallel execution
from scoop import futures, shared


def concat_resample(
    input_mri,
    input_transform,
    corr_transform,
    output_mri,
    output_transform,
    model,
    symmetric=False,
    qc=False,
    bias=None
    ):
    """apply correction transformation and resample input"""
    try:
        with mincTools() as m:
            
            if not ( os.path.exists(output_mri.scan) and os.path.exists(output_transform.xfm) ):
                scan=input_mri.scan
            
                if bias is not None:
                    m.calc([input_mri.scan,bias.scan],'A[0]*A[1]',m.tmp('corr.mnc'))
                    scan=m.tmp('corr.mnc')
            
                m.xfmconcat([input_transform.xfm, corr_transform.xfm], output_transform.xfm)
                m.resample_smooth(scan, output_mri.scan, transform=output_transform.xfm,like=model.scan)

                if input_mri.mask is not None and output_mri.mask is not None:
                    m.resample_labels(input_mri.mask, 
                                    output_mri.mask,
                                    transform=output_transform.xfm,
                                    like=model.scan)
                    if qc:
                        m.qc(output_mri.scan,output_mri.scan+'.jpg',mask=output_mri.mask)
                else:
                    if qc:
                        m.qc(output_mri.scan,output_mri.scan+'.jpg')

                
                if symmetric:
                    scan_f=input_mri.scan_f
                
                    if bias is not None:
                        m.calc([input_mri.scan_f,bias.scan_f],'A[0]*A[1]',m.tmp('corr_f.mnc'))
                        scan_f=m.tmp('corr_f.mnc')
                    
                    m.xfmconcat([input_transform.xfm_f, corr_transform.xfm], output_transform.xfm_f)
                    m.resample_smooth(scan_f, output_mri.scan_f, transform=output_transform.xfm_f,like=model.scan)

                    if input_mri.mask is not None and output_mri.mask is not None:
                        m.resample_labels(input_mri.mask_f, 
                                        output_mri.mask_f,
                                        transform=output_transform.xfm_f,
                                        like=model.scan)
                        if qc:
                            m.qc(output_mri.scan_f,output_mri.scan_f+'.jpg',mask=output_mri.mask_f)
                    else:
                        if qc:
                            m.qc(output_mri.scan_f,output_mri.scan_f+'.jpg')
    except mincError as e:
        print("Exception in concat_resample:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in concat_resample:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

def concat_resample_nl(
    input_mri,
    input_transform,
    corr_transform,
    output_mri,
    output_transform,
    model,
    level,
    symmetric=False,
    qc=False,
    invert_transform=False
    ):
    """apply correction transformation and resample input"""
    try:
        with mincTools() as m:
            tfm=input_transform.xfm
            if corr_transform is not None:
                m.xfmconcat([input_transform.xfm, corr_transform.xfm], m.tmp('transform.xfm'))
                tfm=m.tmp('transform.xfm')
            ref=None
            if isinstance(model, MriDatasetRegress): ref=model.volume[0]
            else: ref=model.scan
            
            m.xfm_normalize( tfm, ref, output_transform.xfm, 
                             step=level)
            
            m.resample_smooth(input_mri.scan, output_mri.scan, 
                              transform=output_transform.xfm, 
                              like=ref,
                              invert_transform=invert_transform)
            
            if input_mri.mask and output_mri.mask:
                m.resample_labels(input_mri.mask, 
                                output_mri.mask,
                                transform=output_transform.xfm,
                                like=ref,
                                invert_transform=invert_transform)
                if qc:
                    m.qc(output_mri.scan,output_mri.scan+'.jpg',
                         mask=output_mri.mask)
            else:
                if qc:
                    m.qc(output_mri.scan,output_mri.scan+'.jpg')
            
            if symmetric:
                tfm_f=input_transform.xfm_f
                if corr_transform is not None:
                    m.xfmconcat( [input_transform.xfm_f, corr_transform.xfm], m.tmp('transform_f.xfm') )
                    tfm_f=m.tmp('transform_f.xfm')
                m.xfm_normalize( tfm_f, ref, output_transform.xfm_f, step=level )
                m.resample_smooth(input_mri.scan_f, output_mri.scan_f, transform=output_transform.xfm_f, 
                                  like=ref,
                                  invert_transform=invert_transform )
                
                if input_mri.mask and output_mri.mask:
                    m.resample_labels(input_mri.mask_f, 
                                      output_mri.mask_f,
                                      transform=output_transform.xfm_f,
                                      like=ref,
                                      invert_transform=invert_transform)
                    
                    if qc:
                        m.qc(output_mri.scan_f, output_mri.scan_f+'.jpg',
                             mask=output_mri.mask_f)
                else:
                    if qc:
                        m.qc(output_mri.scan_f, output_mri.scan_f+'.jpg')
                
                    
        return True
    except mincError as e:
        print("Exception in concat_resample_nl:{}".format(str(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in concat_resample_nl:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

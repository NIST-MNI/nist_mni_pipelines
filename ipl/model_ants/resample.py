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
import ray




@ray.remote
def concat_resample_nl_inv(
    input_mri,
    input_transform,
    corr_transform,
    output_mri,
    output_transform,
    model,
    level,
    symmetric=False,
    qc=False
    ):
    """apply correction transformation and resample input"""
    try:
        with mincTools() as m:

            if corr_transform is not None:
                m.xfmconcat(
                    [corr_transform.fw, corr_transform.fw, 
                     corr_transform.fw, corr_transform.fw,
                     corr_transform.lin_fw,
                     input_transform.lin_fw,
                     input_transform.fw
                     ], 
                    m.tmp('transform.xfm'))
                tfm=m.tmp('transform.xfm')
            else:
                m.xfmconcat(
                    [input_transform.lin_fw , input_transform.fw ], 
                    m.tmp('transform.xfm'))
                tfm=m.tmp('transform.xfm')

            ref=model.scan
            # TODO: decide if needed?
            m.xfm_normalize( tfm, ref, output_transform.fw,
                             step=level)
            
            m.resample_smooth(input_mri.scan, output_mri.scan, 
                              transform=output_transform.fw, 
                              like=ref,
                              invert_transform=True)
            
            if input_mri.mask and output_mri.mask:
                m.resample_labels(input_mri.mask, 
                                output_mri.mask,
                                transform=output_transform.fw,
                                like=ref,
                                invert_transform=True)
                if qc:
                    m.qc(output_mri.scan,output_mri.scan+'.jpg',
                         mask=output_mri.mask)
            else:
                if qc:
                    m.qc(output_mri.scan,output_mri.scan+'.jpg')
            
            if symmetric:
                # TODO: fix symmetric
                tfm_f=input_transform.fw_f
                if corr_transform is not None:
                    m.xfmconcat( [corr_transform.fw, input_transform.fw_f], m.tmp('transform_f.xfm') )
                    tfm_f=m.tmp('transform_f.xfm')
                m.xfm_normalize( tfm_f, ref, output_transform.fw_f, step=level )
                m.resample_smooth(input_mri.scan_f, output_mri.scan_f, transform=output_transform.xfm_f, 
                                  like=ref,
                                  invert_transform=True )
                
                if input_mri.mask and output_mri.mask:
                    m.resample_labels(input_mri.mask_f, 
                                      output_mri.mask_f,
                                      transform=output_transform.fw_f,
                                      like=ref,
                                      invert_transform=True)
                    
                    if qc:
                        m.qc(output_mri.scan_f, output_mri.scan_f+'.jpg',
                             mask=output_mri.mask_f)
                else:
                    if qc:
                        m.qc(output_mri.scan_f, output_mri.scan_f+'.jpg')
                
                    
        return True
    except mincError as e:
        print("Exception in concat_resample_nl_inv:{}".format(str(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in concat_resample_nl_inv:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

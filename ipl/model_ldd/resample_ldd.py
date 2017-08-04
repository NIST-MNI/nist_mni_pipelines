import shutil
import os
import sys
import csv
import traceback

# MINC stuff
from ipl.minc_tools import mincTools,mincError
from .filter_ldd import *


# scoop parallel execution
from scoop import futures, shared


def concat_resample_ldd(
    input_mri,
    input_transform,
    corr_transform,
    output_mri,
    output_transform,
    model,
    symmetric=False,
    qc=False,
    bias=None,
    invert_transform=False,
    datatype='short'
    ):
    """apply correction transformation and resample input"""
    try:
        with mincTools() as m:
            
            if not ( os.path.exists(output_mri.scan) and os.path.exists(output_transform.vel) ):
                scan=input_mri.scan
            
                if bias is not None:
                    m.calc([input_mri.scan,bias.scan],'A[0]*A[1]',m.tmp('corr.mnc'))
                    scan=m.tmp('corr.mnc')
                
                if corr_transform is not None:
                    m.calc([input_transform.vel, corr_transform.vel],'A[0]+A[1]', output_transform.vel, datatype='-'+datatype)
                else:
                    # TODO: copy?
                    m.calc([input_transform.vel ],'A[0]', output_transform.vel, datatype='-'+datatype)

                m.resample_smooth_logspace(scan, output_mri.scan,
                                           velocity=output_transform.vel,
                                           like=model,
                                           invert_transform=invert_transform,
                                           datatype=datatype)

                if input_mri.mask is not None and output_mri.mask is not None:
                    m.resample_labels_logspace(input_mri.mask, 
                                    output_mri.mask,
                                    velocity=output_transform.vel,
                                    like=model,
                                    invert_transform=invert_transform)
                    if qc:
                        m.qc(output_mri.scan, output_mri.scan+'.jpg',
                             mask=output_mri.mask)
                else:
                    if qc:
                        m.qc(output_mri.scan, output_mri.scan+'.jpg')

                if qc:
                    m.grid_magnitude(output_transform.vel,
                                    m.tmp('velocity_mag.mnc'))

                    m.qc(m.tmp('velocity_mag.mnc'), output_mri.scan+'_vel.jpg')
                
                if symmetric:
                    scan_f=input_mri.scan_f
                
                    if bias is not None:
                        m.calc([input_mri.scan_f,bias.scan_f],'A[0]*A[1]', 
                               m.tmp('corr_f.mnc'),datatype='-'+datatype)
                        scan_f=m.tmp('corr_f.mnc')
                        
                    if corr_transform is not None:
                        m.calc([input_transform.vel_f, corr_transform.vel],'A[0]+A[1]', output_transform.vel_f, datatype='-'+datatype)
                    else:
                        m.calc([input_transform.vel_f],'A[0]', output_transform.vel_f, datatype='-'+datatype)

                    m.resample_smooth_logspace(scan_f, output_mri.scan_f,
                                              velocity=output_transform.vel_f,
                                              like=model,
                                              invert_transform=invert_transform,
                                              datatype=datatype)

                    if input_mri.mask is not None and output_mri.mask is not None:
                        m.resample_labels_logspace(input_mri.mask_f, 
                                        output_mri.mask_f,
                                        velocity=output_transform.vel_f,
                                        like=model,
                                        invert_transform=invert_transform)
                        if qc:
                            m.qc(output_mri.scan_f,output_mri.scan_f+'.jpg',
                                 mask=output_mri.mask_f)
                    else:
                        if qc:
                            m.qc(output_mri.scan_f,output_mri.scan_f+'.jpg')
                            
                    if qc:
                        m.grid_magnitude(output_transform.vel_f,
                                        m.tmp('velocity_mag_f.mnc'))

                        m.qc(m.tmp('velocity_mag_f.mnc'), output_mri.scan_f+'_vel.jpg' )
                        
    except mincError as e:
        print "Exception in concat_resample_ldd:{}".format(str(e))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print "Exception in concat_resample_ldd:{}".format(sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
        raise

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

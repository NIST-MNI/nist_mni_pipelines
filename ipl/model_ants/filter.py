import shutil
import os
import sys
import csv
import traceback
import shutil

# MINC stuff
from ipl.minc_tools import mincTools,mincError

try:
    from minc2_simple import minc2_file
    import numpy as np
    have_minc2_simple=True
except ImportError:
    # minc2_simple not available :(
    have_minc2_simple=False

import ray

def faster_average(infiles, out_avg, out_sd=None, binary=False, threshold=0.5):
    # faster then mincaverage for large number of samples
    ref=minc2_file(infiles[0])
    dims_ref=ref.store_dims()
    o_avg=minc2_file()
    
    if binary:
        o_avg.define(dims_ref, minc2_file.MINC2_BYTE,  minc2_file.MINC2_BYTE)
    else:
        o_avg.define(dims_ref, minc2_file.MINC2_FLOAT, minc2_file.MINC2_FLOAT)
        
    o_avg.create(out_avg)
    o_avg.copy_metadata(ref)
    o_avg.setup_standard_order()
    
    if out_sd is not None:
        o_sd=minc2_file()
        o_sd.define(dims_ref, minc2_file.MINC2_FLOAT, minc2_file.MINC2_FLOAT)
        o_sd.create(out_sd)
        o_sd.copy_metadata(ref)
        o_sd.setup_standard_order()

    ref.setup_standard_order()
    
    # iterate over all input files and update avg and sd
    vol_avg=ref.load_complete_volume(minc2_file.MINC2_FLOAT).astype(np.float64)
    ref.close()
    
    if out_sd is not None:
        vol_sd=vol_avg*vol_avg

    lll=1.0

    for i in range(1,len(infiles)):
        in_minc=minc2_file(infiles[i])
        #TODO: check dimensions
        in_minc.setup_standard_order()
        v=in_minc.load_complete_volume(minc2_file.MINC2_FLOAT).astype(np.float64)
        in_minc.close()
        vol_avg+=v
        lll+=1.0

        if out_sd is not None:
            vol_sd+=v*v

    #Averaging
    vol_avg/=lll
    
    if binary:
        # binarize:
        vol_avg=np.greater(vol_avg,threshold).astype('int8')
    else:
        vol_avg=vol_avg.astype(np.float64)
    
    o_avg.save_complete_volume(vol_avg)
    
    if out_sd is not None:
        vol_sd/=lll
        vol_sd-=vol_avg*vol_avg
        vol_sd=np.sqrt(vol_sd).astype(np.float64)
        o_sd.save_complete_volume(vol_sd)

@ray.remote
def generate_flip_sample(input):
    '''generate flipped version of sample'''
    with mincTools() as m:
        m.flip_volume_x(input.scan,input.scan_f)
        
        if input.mask is not None:
            m.flip_volume_x(input.mask,input.mask_f,labels=True)
            
    return True
    
@ray.remote 
def normalize_sample(input,
                     output,
                     model,
                     bias_field=None,
                    ):
    """Normalize sample intensity"""

    with mincTools() as m:
        m.apply_n3_vol_pol(
            input.scan,
            model.scan,
            output.scan,
            source_mask=input.mask,
            target_mask=model.mask,
            bias=bias_field,
            )
    output.mask=input.mask
    return output

@ray.remote
def average_samples(
    samples,
    output,
    #upd=None,
    output_sd=None,
    symmetric=False,
    symmetrize=False,
    #median=False,
    average_mode='std'
    ):
    """average individual samples"""
    try:
        with mincTools() as m:
            avg = []

            out_scan=output.scan
            out_mask=output.mask
            
            if symmetrize: #TODO: fix this
                out_scan=m.tmp('avg.mnc')
                out_mask=m.tmp('avg_mask.mnc')

            # if upd is not None:
            #     corr_xfm=m.tmp("correction.xfm")
            #     m.xfmconcat([ upd.lin_fw, upd.fw, upd.fw, upd.fw, upd.fw], corr_xfm)
                
            for s in samples:
                avg.append(s.scan)
                
            if symmetric:
                for s in samples:
                    avg.append(s.scan_f)

            out_sd=None
            if output_sd:
                out_sd=output_sd.scan

            out_scan_mean=None

            if average_mode == 'median':
                m.median(avg, out_scan, madfile=out_sd)
            elif average_mode == 'ants':
                cmd = ['AverageImages', '3', out_scan,'1']
                cmd.extend(avg)
                m.command(cmd, inputs=avg, outputs=[out_scan])

                #HACK to create a "standard" average
                out_scan_mean=output.scan.rsplit(".mnc",1)[0]+'_mean.mnc'
                if have_minc2_simple:
                    faster_average(avg, out_scan_mean,out_sd=out_sd)
                else:
                    m.average(avg, out_scan_mean, sdfile=out_sd)
            else:
                if have_minc2_simple:
                    faster_average(avg, out_scan,out_sd=out_sd)
                else:
                    m.average(avg, out_scan, sdfile=out_sd)

            # if upd is not None:
            #     m.resample_smooth(out_scan, output.scan, transform=corr_xfm, invert_transform=True)
            #     if out_sd is not None: 
            #         m.resample_smooth(out_sd,output_sd.scan,transform=corr_xfm,invert_transform=True)
            #     if out_scan_mean is not None:
            #         m.resample_smooth(m.tmp('avg_mean.mnc'),out_scan_mean, transform=corr_xfm, invert_transform=True)
            #     # DEBUG
            #     shutil.copyfile(out_scan, output.scan.rsplit(".mnc",1)[0]+'_nc.mnc')
                
            # if symmetrize: # TODO: fix this
            #     # TODO: replace flipping of averages with averaging of flipped 
            #     # some day
            #     m.flip_volume_x(out_scan,m.tmp('flip.mnc'))
            #     m.average([out_scan,m.tmp('flip.mnc')],output.scan)
            
            # average masks
            if output.mask is not None:
                avg = []
                for s in samples:
                    avg.append(s.mask)

                # if symmetric:
                #     for s in samples:
                #         avg.append(s.mask_f)

                if not os.path.exists(output.mask):                    
                    if symmetrize:
                        if have_minc2_simple:
                            faster_average(avg,m.tmp('avg_mask.mnc'))
                        else:
                            m.average(avg,m.tmp('avg_mask.mnc'),datatype='-float')
                        
                        m.flip_volume_x(m.tmp('avg_mask.mnc'),m.tmp('flip_avg_mask.mnc'))
                        m.average([m.tmp('avg_mask.mnc'),m.tmp('flip_avg_mask.mnc')],m.tmp('sym_avg_mask.mnc'),datatype='-float')
                        m.calc([m.tmp('sym_avg_mask.mnc')],'A[0]>=0.5?1:0',output.mask, datatype='-byte',labels=True)
                    else:
                        if have_minc2_simple:
                            faster_average(avg,m.tmp('avg_mask.mnc'))
                        else:
                            m.average(avg,m.tmp('avg_mask.mnc'),datatype='-float')

                    # if upd is not None:
                    #     m.resample_smooth(m.tmp('avg_mask_.mnc'),m.tmp('avg_mask.mnc'), transform=corr_xfm,invert_transform=True)
                    # else:
                    #    m.resample_smooth(m.tmp('avg_mask_.mnc'),m.tmp('avg_mask.mnc')) # HACK
                    # apply correction!
                    m.calc([m.tmp('avg_mask.mnc')],'A[0]>=0.5?1:0',output.mask, datatype='-byte',labels=True)
        return  True
    except mincError as e:
        print("Exception in average_samples:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in average_samples:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote
def average_stats(
    avg,
    sd,
    upd_xfm
    ):
    
    """calculate iteration summary: median sd of intensity and bias deformations"""
    try:
        median_sd=0.0
        median_def=[0.0,0.0,0.0]
        median_mag=0.0
        with mincTools(verbose=0) as m:
            m.grid_magnitude(upd_xfm.fw_grid,m.tmp("mag.mnc"))

            if avg.mask is not None:
                m.resample_labels(avg.mask,m.tmp("mag_mask.mnc"),like=m.tmp("mag.mnc"))

            if avg.mask is not None:
                median_sd=float(m.stats(sd.scan,'-median', mask=avg.mask))
                median_mag=float(m.stats(m.tmp("mag.mnc"),'-median', mask=m.tmp("mag_mask.mnc")))
            else:
                median_sd=float(m.stats(sd.scan,'-median'))
                median_mag=float(m.stats(m.tmp("mag.mnc"),'-median'))

            for i in range(3):
                m.reshape(upd_xfm.fw_grid, m.tmp(f"dim_{i}.mnc"),dimrange=f"vector_dimension={i}")
                if avg.mask is not None:
                    if i==0: m.resample_labels(avg.mask,m.tmp("dim_mask.mnc"),like=m.tmp(f"dim_{i}.mnc"))
                    median_def[i]=float(m.stats(m.tmp(f"dim_{i}.mnc"), '-median', mask=m.tmp("dim_mask.mnc")))
                else:
                    median_def[i]=float(m.stats(m.tmp(f"dim_{i}.mnc"),'-median'))

            
        return {'intensity':median_sd, 'dx':median_def[0], 'dy':median_def[1], 'dz':median_def[2], 'mag':median_mag}
    except mincError as e:
        print("mincError in average_stats:{}".format(repr(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in average_stats:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

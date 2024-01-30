import shutil
import os
import sys
import csv
import traceback

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
    output_sd=None,
    symmetric=False,
    symmetrize=False,
    median=False
    ):
    """average individual samples"""
    try:
        with mincTools() as m:
            avg = []

            out_scan=output.scan
            out_mask=output.mask
            
            if symmetrize:
                out_scan=m.tmp('avg.mnc')
                out_mask=m.tmp('avg_mask.mnc')
                
            for s in samples:
                avg.append(s.scan)
                
            if symmetric:
                for s in samples:
                    avg.append(s.scan_f)
            out_sd=None
            
            if output_sd:
                out_sd=output_sd.scan

            if median:
                m.median(avg, out_scan,madfile=out_sd)
            else:
                if have_minc2_simple:
                    faster_average(avg,out_scan,out_sd=out_sd)
                else:
                    m.average(avg, out_scan,sdfile=out_sd)

            if symmetrize:
                # TODO: replace flipping of averages with averaging of flipped 
                # some day
                m.flip_volume_x(out_scan,m.tmp('flip.mnc'))
                m.average([out_scan,m.tmp('flip.mnc')],output.scan)
            
            # average masks
            if output.mask is not None:
                avg = []
                for s in samples:
                    avg.append(s.mask)

                if symmetric:
                    for s in samples:
                        avg.append(s.mask_f)

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
    ):
    """calculate median sd within mask"""
    try:
        st=0
        with mincTools(verbose=2) as m:
            if avg.mask is not None:
                st=float(m.stats(sd.scan,'-median', mask=avg.mask))
            else:
                st=float(m.stats(sd.scan,'-median'))
        return st
    except mincError as e:
        print("mincError in average_stats:{}".format(repr(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in average_stats:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote
def calculate_diff_bias_field(sample, model, output, symmetric=False, distance=100, n4=False ):
    try:
        with mincTools() as m:
            if n4:
                if model.mask is not None:
                    m.difference_n4(sample.scan, model.scan, output.scan, mask=model.mask, distance=distance)
                else:
                    m.difference_n4(sample.scan, model.scan, output.scan, distance=distance )
                if symmetric:
                    if model.mask is not None:
                        m.difference_n4(sample.scan_f, model.scan, output.scan_f, mask=model.mask, distance=distance)
                    else:
                        m.difference_n4(sample.scan_f, model.scan, output.scan_f, distance=distance )
            else:
                if model.mask is not None:
                    m.difference_n3(sample.scan, model.scan, output.scan, mask=model.mask, distance=distance, normalize=True)
                else:
                    m.difference_n3(sample.scan, model.scan, output.scan, distance=distance, normalize=True )
                if symmetric:
                    if model.mask is not None:
                        m.difference_n3(sample.scan_f, model.scan, output.scan_f, mask=model.mask, distance=distance, normalize=True)
                    else:
                        m.difference_n3(sample.scan_f, model.scan, output.scan_f, distance=distance, normalize=True )
        return True
    except mincError as e:
        print("mincError in average_stats:{}".format(repr(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in average_stats:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote
def average_bias_fields(samples, output, symmetric=False ):
    try:
        with mincTools() as m:
            
            avg = []
            
            for s in samples:
                avg.append(s.scan)
            
            if symmetric:
                for s in samples:
                    avg.append(s.scan_f)
            
            m.log_average(avg, output.scan)
        return True
    except mincError as e:
        print("mincError in average_stats:{}".format(repr(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in average_stats:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote
def resample_and_correct_bias(
        sample,
        transform,
        avg_bias,
        output,
        previous=None,
        symmetric=False, 
        ):
    # resample bias field and apply previous estimate
    try:
        with mincTools() as m:
            
            m.calc([sample.scan, avg_bias.scan],
                    'A[1]>0.1&&A[0]>0.1?log(A[0]/A[1]):0.0', 
                    m.tmp('corr_bias.mnc'))

            m.resample_smooth(m.tmp('corr_bias.mnc'), 
                              m.tmp('corr_bias2.mnc'), 
                              like=sample.scan,
                              order=1,
                              transform=transform.xfm,
                              invert_transform=True )
            if previous:
                m.calc([previous.scan, m.tmp('corr_bias2.mnc') ], 
                    'A[0]*exp(A[1])',
                    output.scan, datatype='-float')
            else:
                m.calc([previous.scan], 'exp(A[0])',
                    output.scan, datatype='-float')
                
            if symmetric:
                m.calc([sample.scan_f, avg_bias.scan],
                        'A[1]>0.1&&A[0]>0.1?log(A[0]/A[1]):0.0', 
                        m.tmp('corr_bias_f.mnc'))

                m.resample_smooth(m.tmp('corr_bias_f.mnc'), 
                                  m.tmp('corr_bias2_f.mnc'), 
                                  like=sample.scan,
                                  order=1,
                                  transform=transform.xfm,
                                  invert_transform=True)
                if previous:
                    m.calc([previous.scan_f, m.tmp('corr_bias2_f.mnc')], 
                        'A[0]*exp(A[1])',
                        output.scan_f, datatype='-float')
                else:
                    m.calc([m.tmp('corr_bias2_f.mnc')], 
                        'exp(A[0])',
                        output.scan_f, datatype='-float')
            
        return True
    except mincError as e:
        print("Exception in resample_and_correct_bias:{}".format(str(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in resample_and_correct_bias:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote
def apply_linear_model(
    lin_model,
    parameters,
    output_volume
    ):
    """build a volume, for a given regression model and parameters"""
    try:
        with mincTools() as m:
            
            if lin_model.N!=len(parameters):
                raise mincError("Expected: {} parameters, got {}".format(lin_model.N,len(parameters)))
            # create minccalc expression
            _exp=[]
            for i in range(0,lin_model.N):
                _exp.append('A[{}]*{}'.format(i,parameters[i]))
            exp='+'.join(_exp)
            m.calc(lin_model.volume,exp,output_volume)
        return  True
    except mincError as e:
        print( "Exception in apply_linear_model:{}".format(str(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print( "Exception in apply_linear_model:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise


@ray.remote
def build_approximation(int_model, 
                        geo_model ,
                        parameters_int, 
                        parameters_def, 
                        output_scan, 
                        output_transform,
                        noresample=False):
    try:
        with mincTools() as m:

            intensity=m.tmp('int_model.mnc')
            if noresample:
                intensity=output_scan.scan
            #geometry=m.tmp('geometry_model.mnc')
            
            # TODO: paralelelize?
            if int_model.N>0:
                apply_linear_model(int_model,parameters_int,intensity)
            else: # not modelling intensity
                intensity=int_model.volume[0]
            
            # if we have geometry information
            if geo_model is not None and geo_model.N>0 :
                apply_linear_model(geo_model, parameters_def, output_transform.grid )
                # create appropriate .xfm file
                with open(output_transform.xfm,'w') as f:
                    f.write(
"""
MNI Transform File
Transform_Type = Linear;
Linear_Transform =
 1 0 0 0
 0 1 0 0
 0 0 1 0;
Transform_Type = Grid_Transform;
Displacement_Volume = {};
""".format(os.path.basename(output_transform.grid))
                    )
                
                if not noresample:
                    m.resample_smooth(intensity, output_scan.scan,
                            transform=output_transform.xfm,
                            like=int_model.volume[0])
                                        
                if int_model.mask is not None:
                    if noresample:
                        shutil.copyfile(int_model.mask,
                                        output_scan.mask)
                    else:
                        m.resample_labels(int_model.mask,
                            output_scan.mask,
                            transform=output_transform.xfm,
                            like=int_model.volume[0])
                else:
                    output_scan.mask=None
            else: # not modelling shape!
                shutil.copyfile(intensity,output_scan.scan)
                if int_model.mask is not None: 
                    shutil.copyfile(int_model.mask,
                                    output_scan.mask)
                else:
                    output_scan.mask=None
                output_transform=None
        return  (output_scan, output_transform)
    except mincError as e:
        print( "Exception in build_approximation:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print( "Exception in build_approximation:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote
def voxel_regression(int_design_matrix,
                def_design_matrix,
                int_estimate,
                def_estimate,
                next_int_model,
                next_def_model,
                int_residual,
                def_residual,
                blur_int_model=None,
                blur_def_model=None,
                qc=False):
    """Perform voxel-wise regression using given design matrix"""
    try:
        with mincTools() as m:
            #print(repr(next_int_model))
            
            # a small hack - assume that input directories are the same
            _prefix=def_estimate[0].prefix
            _design_vel=_prefix+os.sep+'regression_vel.csv'
            _design_int=_prefix+os.sep+'regression_int.csv'

            #nomask=False
            #for i in for i in int_estimate:
            #    if i.mask is None:
            #    nomask=True
            _masks=[i.mask for i in int_estimate]
            _inputs=[]
            _outputs=[]
            _outputs.extend(next_int_model.volume)
            _outputs.extend(next_def_model.volume)
            
            with open(_design_vel,'w') as f:
                for (i, l ) in enumerate(def_design_matrix):
                    f.write(os.path.basename(def_estimate[i].grid))
                    f.write(',')
                    f.write(','.join([str(qq) for qq in l]))
                    f.write("\n")
                    _inputs.append(def_estimate[i].grid)
            
            with open(_design_int,'w') as f:
                for (i, l ) in enumerate(int_design_matrix):
                    f.write(os.path.basename(int_estimate[i].scan))
                    f.write(',')
                    f.write(','.join([str(qq) for qq in l]))
                    f.write("\n")
                    _inputs.append(int_estimate[i].scan)
        
            if not m.checkfiles(inputs=_inputs, outputs=_outputs):
                return
            
            int_model=next_int_model
            def_model=next_def_model
            
            if blur_int_model is not None:
                int_model=MriDatasetRegress(prefix=m.tempdir,  name='model_int',N=next_int_model.N,nomask=(next_int_model.mask is None))

            if blur_def_model is not None:
                def_model=MriDatasetRegress(prefix=m.tempdir,name='model_def', N=next_def_model.N, nomask=(next_def_model.mask is None))
                
            
            # regress deformations
            m.command(['volumes_lm',_design_vel, def_model.volume[0].rsplit('_0.mnc',1)[0]],
                      inputs=[_design_vel],
                      outputs=def_model.volume, 
                      verbose=2)
                      

            # regress intensity
            m.command(['volumes_lm',_design_int, int_model.volume[0].rsplit('_0.mnc',1)[0]],
                      inputs=[_design_int], 
                      outputs=int_model.volume, 
                      verbose=2)
            
            if blur_def_model is not None:
                # blur estimates
                for (i,j) in  enumerate(def_model.volume):
                    m.blur_vectors(def_model.volume[i],next_def_model.volume[i],blur_def_model)
                # a hack preserve unfiltered RMS volume
                shutil.copyfile(def_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc',
                                next_def_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc')

            if blur_int_model is not None:
                for (i,j) in  enumerate(int_model.volume):
                    m.blur(int_model.volume[i],next_int_model.volume[i],blur_int_model)
                # a hack preserve unfiltered RMS volume
                shutil.copyfile(int_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc',
                                next_int_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc')
            
            # average masks
            if next_int_model.mask is not None:
                m.average(_masks,m.tmp('avg_mask.mnc'),datatype='-float')
                m.calc([m.tmp('avg_mask.mnc')],'A[0]>0.5?1:0',m.tmp('avg_mask_.mnc'),datatype='-byte')
                m.reshape(m.tmp('avg_mask_.mnc'),next_int_model.mask,image_range=[0,1],valid_range=[0,1])
            
            if qc:
                m.qc(next_int_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc', 
                     next_int_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.jpg' )
                
                m.grid_magnitude(next_def_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc',
                                 m.tmp('def_RMS_mag.mnc'))
                m.qc(m.tmp('def_RMS_mag.mnc'), 
                     next_def_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.jpg')
            
            #cleanup
            #os.unlink(_design_vel)
            #os.unlink(_design_int)
            

    except mincError as e:
        print( "Exception in voxel_regression:{}".format(str(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print( "Exception in voxel_regression:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise
        
@ray.remote    
def average_stats_regression(
    current_int_model, current_def_model,
    int_residual, def_residual,
    ):
    """calculate median sd within mask for intensity and velocity"""
    try:
        sd_int=0.0
        sd_def=0.0
        with mincTools(verbose=2) as m:
            m.grid_magnitude(def_residual.scan, m.tmp('mag.mnc'))
            if current_int_model.mask is not None:
                sd_int=float(m.stats(int_residual.scan,'-median',mask=current_int_model.mask))
                m.resample_smooth(m.tmp('mag.mnc'),m.tmp('mag_.mnc'),like=current_int_model.mask)
                sd_def=float(m.stats(m.tmp('mag_.mnc'),'-median',mask=current_int_model.mask))
            else:
                sd_int=float(m.stats(int_residual.scan,'-median'))
                sd_def=float(m.stats(m.tmp('mag.mnc'),'-median'))

        return (sd_int,sd_def)
    except mincError as e:
        print("mincError in average_stats_regression:{}".format(repr(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in average_stats_regression:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

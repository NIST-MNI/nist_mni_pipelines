import shutil
import os
import sys
import csv
import traceback
import copy

# MINC stuff
from ipl.minc_tools import mincTools,mincError
from .structures_ldd       import MriDataset, LDDMriTransform, LDDMRIEncoder,MriDatasetRegress

import ray

@ray.remote
def generate_flip_sample(input):
    '''generate flipped version of sample'''
    with mincTools() as m:
        m.flip_volume_x(input.scan,input.scan_f)

        if input.mask is not None:
            m.flip_volume_x(input.mask,input.mask_f,labels=True)

    return True


def normalize_sample(
    input,
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
    ):
    """average individual samples"""
    try:
        with mincTools() as m:
            avg = []
            
            for s in samples:
                avg.append(s.scan)
                
            if symmetric:
                for s in samples:
                    avg.append(s.scan_f)
            
            if output_sd:
                m.average(avg, output.scan, sdfile=output_sd.scan)
            else:
                m.average(avg, output.scan)

            # average masks
            if output.mask is not None:
                avg = []
                for s in samples:
                    avg.append(s.mask)

                if symmetric:
                    for s in samples:
                        avg.append(s.mask_f)

                if not os.path.exists(output.mask):
                    m.average(avg,m.tmp('avg_mask.mnc'),datatype='-float')
                    m.calc([m.tmp('avg_mask.mnc')],'A[0]>0.5?1:0',m.tmp('avg_mask_.mnc'),datatype='-byte')
                    m.reshape(m.tmp('avg_mask_.mnc'),output.mask,image_range=[0,1],valid_range=[0,1])
                    

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
                st=float(m.stats(sd.scan,'-median',mask=avg.mask))
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


def average_stats_regression(
    current_intensity_model, current_velocity_model,
    intensity_residual, velocity_residual,
    ):
    """calculate median sd within mask for intensity and velocity"""
    try:
        sd_int=0.0
        sd_vel=0.0
        with mincTools(verbose=2) as m:
            m.grid_magnitude(velocity_residual.scan,m.tmp('mag.mnc'))
            if current_intensity_model.mask is not None:
                sd_int=float(m.stats(intensity_residual.scan,'-median',mask=current_intensity_model.mask))
                m.resample_smooth(m.tmp('mag.mnc'),m.tmp('mag_.mnc'),like=current_intensity_model.mask)
                sd_vel=float(m.stats(m.tmp('mag_.mnc'),'-median',mask=current_intensity_model.mask))
            else:
                sd_int=float(m.stats(intensity_residual.scan,'-median'))
                sd_vel=float(m.stats(m.tmp('mag.mnc'),'-median'))

        return (sd_int,sd_vel)
    except mincError as e:
        print("mincError in average_stats_regression:{}".format(repr(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in average_stats_regression:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise
    
        

def calculate_diff_bias_field(sample, model, output, symmetric=False, distance=100 ):
    try:
        with mincTools() as m:
            if model.mask is not None:
                m.difference_n3(sample.scan, model.scan, output.scan, mask=model.mask, distance=distance, normalize=True)
            else:
                m.difference_n3(sample.scan, model.scan, output.scan, distance=distance, normalize=True )
            if symmetric:
                if model.mask is not None:
                    m.difference_n3(sample.scan_f, model.scan, output.scan, mask=model.mask_f, distance=distance, normalize=True)
                else:
                    m.difference_n3(sample.scan_f, model.scan, output.scan, distance=distance, normalize=True )
        return True
    except mincError as e:
        print("mincError in calculate_diff_bias_field:{}".format(repr(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in calculate_diff_bias_field:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise


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
        print("mincError in average_bias_fields:{}".format(repr(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in average_bias_fields:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise


def resample_and_correct_bias_ldd(
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
                        'A[1]>0.1?A[0]/A[1]:1.0', m.tmp('corr_bias.mnc'))

            m.resample_smooth_logspace(m.tmp('corr_bias.mnc'), 
                              m.tmp('corr_bias2.mnc'), 
                              like=sample.scan,
                              transform=transform.vel,
                              invert_transform=True)
            if previous:
                m.calc([previous.scan, m.tmp('corr_bias2.mnc') ], 'A[0]*A[1]',
                    output.scan, datatype='-float')
            else:
                shutil.copy(m.tmp('corr_bias2.mnc'), output.scan)
                
            if symmetric:
                m.calc([sample.scan_f, avg_bias.scan],
                            'A[1]>0.1?A[0]/A[1]:1.0', m.tmp('corr_bias_f.mnc'))

                m.resample_smooth_logspace(m.tmp('corr_bias_f.mnc'), 
                                  m.tmp('corr_bias2_f.mnc'), 
                                  like=sample.scan,
                                  transform=transform.vel,
                                  invert_transform=True)
                if previous:
                    m.calc([previous.scan_f, m.tmp('corr_bias2_f.mnc')], 
                        'A[0]*A[1]',
                        output.scan_f, datatype='-float')
                else:
                    shutil.copy(m.tmp('corr_bias2_f.mnc'), output.scan)
            
        return True
    except mincError as e:
        print("Exception in resample_and_correct_bias_ldd:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in resample_and_correct_bias_ldd:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

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


def build_approximation(intensity_model, geo_model , 
                        parameters_intensity, parameters_velocity, 
                        output_scan, output_transform,
                        noresample=False,
                        remove0=False):
    try:
        with mincTools() as m:

            intensity=m.tmp('intensity_model.mnc')
            if noresample:
                intensity=output_scan.scan
            #geometry=m.tmp('geometry_model.mnc')
            
            # TODO: paralelelize?
            if intensity_model.N>0:
                apply_linear_model(intensity_model,parameters_intensity,intensity)
            else: # not modelling intensity
                intensity=intensity_model.volume[0]
            
            # if we have geometry information
            if geo_model is not None and geo_model.N>0 :
                _parameters_velocity=copy.deepcopy(parameters_velocity)
                if remove0:_parameters_velocity[0]=0
                apply_linear_model(geo_model, _parameters_velocity, output_transform.vel)
                
                if not noresample:
                    m.resample_smooth_logspace(intensity, output_scan.scan,
                            velocity=output_transform.vel,
                            like=intensity_model.volume[0])
                                        
                if intensity_model.mask is not None:
                    if noresample:
                        shutil.copyfile(intensity_model.mask,
                                        output_scan.mask)
                    else:
                        m.resample_labels_logspace(intensity_model.mask,
                            output_scan.mask,
                            velocity=output_transform.vel,
                            like=intensity_model.volume[0])
                else:
                    output_scan.mask=None
            else: # not modelling shape!
                shutil.copyfile(intensity,output_scan.scan)
                if intensity_model.mask is not None: 
                    shutil.copyfile(intensity_model.mask,
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


def voxel_regression(intensity_design_matrix,
                velocity_design_matrix,
                intensity_estimate,      velocity_estimate,
                next_intensity_model,    next_velocity_model,
                intensity_residual,      velocity_residual,
                blur_int_model=None,     blur_vel_model=None,
                qc=False):
    """Perform voxel-wise regression using given design matrix"""
    try:
        with mincTools() as m:
            #print(repr(next_intensity_model))
            
            # a small hack - assume that input directories are the same
            _prefix=velocity_estimate[0].prefix
            _design_vel=_prefix+os.sep+'regression_vel.csv'
            _design_int=_prefix+os.sep+'regression_int.csv'

            #nomask=False
            #for i in for i in intensity_estimate:
            #    if i.mask is None:
            #    nomask=True
            _masks=[i.mask for i in intensity_estimate]
            _inputs=[]
            _outputs=[]
            _outputs.extend(next_intensity_model.volume)
            _outputs.extend(next_velocity_model.volume)
            
            with open(_design_vel,'w') as f:
                for (i, l ) in enumerate(velocity_design_matrix):
                    f.write(os.path.basename(velocity_estimate[i].vel))
                    f.write(',')
                    f.write(','.join([str(qq) for qq in l]))
                    f.write("\n")
                    _inputs.append(velocity_estimate[i].vel)
            
            with open(_design_int,'w') as f:
                for (i, l ) in enumerate(intensity_design_matrix):
                    f.write(os.path.basename(intensity_estimate[i].scan))
                    f.write(',')
                    f.write(','.join([str(qq) for qq in l]))
                    f.write("\n")
                    _inputs.append(intensity_estimate[i].scan)
        
            if not m.checkfiles(inputs=_inputs, outputs=_outputs):
                return
            
            intensity_model=next_intensity_model
            velocity_model=next_velocity_model
            
            if blur_int_model is not None:
                intensity_model=MriDatasetRegress(prefix=m.tempdir,  name='model_intensity',N=next_intensity_model.N,nomask=(next_intensity_model.mask is None))

            if blur_vel_model is not None:
                velocity_model=MriDatasetRegress(prefix=m.tempdir,name='model_velocity', N=next_velocity_model.N, nomask=(next_velocity_model.mask is None))
                
            
            # regress velocity
            m.command(['volumes_lm',_design_vel, velocity_model.volume[0].rsplit('_0.mnc',1)[0]],
                      inputs=[_design_vel],
                      outputs=velocity_model.volume, 
                      verbose=2)

            # regress intensity
            m.command(['volumes_lm',_design_int, intensity_model.volume[0].rsplit('_0.mnc',1)[0]],
                      inputs=[_design_int], 
                      outputs=intensity_model.volume, 
                      verbose=2)
            
            if blur_vel_model is not None:
                # blur estimates
                for (i,j) in  enumerate(velocity_model.volume):
                    m.blur_vectors(velocity_model.volume[i],next_velocity_model.volume[i],blur_vel_model)
                # a hack preserve unfiltered RMS volume
                shutil.copyfile(velocity_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc',
                                next_velocity_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc')

            if blur_int_model is not None:
                for (i,j) in  enumerate(intensity_model.volume):
                    m.blur(intensity_model.volume[i],next_intensity_model.volume[i],blur_int_model)
                # a hack preserve unfiltered RMS volume
                shutil.copyfile(intensity_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc',
                                next_intensity_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc')
            
            # average masks
            if next_intensity_model.mask is not None:
                m.average(_masks,m.tmp('avg_mask.mnc'),datatype='-float')
                m.calc([m.tmp('avg_mask.mnc')],'A[0]>0.5?1:0',m.tmp('avg_mask_.mnc'),datatype='-byte')
                m.reshape(m.tmp('avg_mask_.mnc'),next_intensity_model.mask,image_range=[0,1],valid_range=[0,1])
            
            if qc:
                m.qc(next_intensity_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc', 
                     next_intensity_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.jpg' )

                m.grid_magnitude(next_velocity_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc',
                                 m.tmp('velocity_RMS_mag.mnc'))

                m.qc(m.tmp('velocity_RMS_mag.mnc'), 
                     next_velocity_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.jpg')
            
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
        
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

import shutil
import os
import sys
import csv
import traceback
import json

# MINC stuff
from iplMincTools          import mincTools,mincError

from .structures_ldd       import MriDataset, LDDMriTransform, LDDMRIEncoder,MriDatasetRegress
from .filter_ldd           import generate_flip_sample, normalize_sample
from .filter_ldd           import average_samples,average_stats_regression
from .filter_ldd           import calculate_diff_bias_field
from .filter_ldd           import average_bias_fields
from .filter_ldd           import resample_and_correct_bias_ldd
from .filter_ldd           import build_approximation
from .filter_ldd           import voxel_regression
from .registration_ldd     import non_linear_register_step_ldd
from .registration_ldd     import average_transforms_ldd
from .registration_ldd     import non_linear_register_step_regress_ldd
from .resample_ldd         import concat_resample_ldd

from scoop import futures, shared


def regress_ldd(
    samples,
    initial_model=None,
    initial_intensity_model=None,
    initial_velocity_model=None,
    output_intensity_model=None,
    output_velocity_model=None,
    output_residuals_int=None,
    output_residuals_vel=None,
    prefix='.',
    options={}
    ):
    """ perform iterative model creation"""
    try:


        # make sure all input scans have parameters
        N_int=None
        N_vel=None
        
        intensity_design_matrix=[]
        velocity_design_matrix=[]
        nomask=False
        
        for s in samples:
            
            if N_int is None:
                N_int=len(s.par_int)
            elif N_int!=len(s.par_int):
                raise mincError("Sample {} have inconsisten number of intensity paramters: {} expected {}".format(repr(s),len(s),N_int))
            
            if N_vel is None:
                N_vel=len(s.par_vel)
            elif N_vel!=len(s.par_vel):
                raise mincError("Sample {} have inconsisten number of intensity paramters: {} expected {}".format(repr(s),len(s),N_vel))
            
            intensity_design_matrix.append(s.par_int)
            velocity_design_matrix.append(s.par_vel)

            if s.mask is None:
                nomask=True

        #print("Intensity design matrix=\n{}".format(repr(intensity_design_matrix)))
        #print("Velocity design matrix=\n{}".format(repr(velocity_design_matrix)))

        ref_model=None
        # current estimate of template
        if initial_model is not None:
            current_intensity_model     = initial_model
            current_velocity_model      = None
            ref_model=initial_model.scan
        else:
            current_intensity_model     = initial_intensity_model
            current_velocity_model      = initial_velocity_model
            ref_model=initial_intensity_model.volume[0]

        transforms=[]
        
        full_transforms=[]

        protocol=options.get(
                              'protocol', [{'iter':4,'level':32, 'blur_int': None, 'blur_vel': None },
                                           {'iter':4,'level':16, 'blur_int': None, 'blur_vel': None }]
                            )

        cleanup=    options.get('cleanup',False)
        cleanup_intermediate=    options.get('cleanup_intermediate',False)
        
        parameters= options.get('parameters',None)
        refine=     options.get('refine',False)
        qc=         options.get('qc',False)
        downsample =options.get('downsample',None)
        start_level=options.get('start_level',None)
        debug      =options.get('debug',False)
        debias     =options.get('debias',True)
        incremental=options.get('incremental',True)
        remove0    =options.get('remove0',False)
        sym        =options.get('sym',False)

        if parameters is None:
            parameters={
                        'conf':{},
                        'smooth_update':2,
                        'smooth_field':2,
                        'update_rule':1,
                        'grad_type': 0,
                        'max_step':  2.0, # This paramter is probably domain specific 
                        'hist_match':True # this turns out to be very important!
                        }

        intensity_models=[]
        velocity_models=[]
        intensity_residuals=[]
        velocity_residuals=[]
        
        intensity_residual=None
        velocity_residual=None
        
        prev_velocity_estimate=None
        # go through all the iterations
        it=0
        residuals=[]
        
        for (i,p) in enumerate(protocol):
            blur_int_model=p.get('blur_int',None)
            blur_vel_model=p.get('blur_vel',None)
            for j in range(1,p['iter']+1):
                it+=1
                _start_level=None
                if it==1:
                    _start_level=start_level
                # this will be a model for next iteration actually
                
                it_prefix=prefix+os.sep+str(it)
                if not os.path.exists(it_prefix):
                    os.makedirs(it_prefix)
                    
                next_intensity_model=MriDatasetRegress(prefix=prefix,  name='model_intensity',iter=it, N=N_int,nomask=nomask)
                next_velocity_model=MriDatasetRegress(prefix=prefix, name='model_velocity', iter=it, N=N_vel, nomask=True)
                

                intensity_residual=MriDataset(prefix=prefix, scan= next_intensity_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc')
                                              #name=next_intensity_model.name, iter=it )

                velocity_residual =MriDataset(prefix=prefix, scan= next_velocity_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc')
                                              #name=next_velocity_model.name, iter=it )

                # skip over existing models here!
                
                if not next_intensity_model.exists() or \
                   not next_velocity_model.exists() or \
                   not intensity_residual.exists() or \
                   not velocity_residual.exists():
                    
                    intensity_estimate=[]
                    velocity_estimate=[]
                    r=[]
                    
                    
                    # 1 for each sample generate current approximation
                    # 2. perform non-linear registration between each sample and sample-specific approximation
                    # 3. update transformation
                    # 1+2+3 - all together 
                    for (i, s) in enumerate(samples):
                        sample_velocity=  LDDMriTransform(name=s.name,prefix=it_prefix,iter=it)
                        sample_intensity= MriDataset(name=s.name,prefix=it_prefix,iter=it)

                        previous_velocity=None

                        if refine and it>1 and (not remove0):
                            previous_velocity=prev_velocity_estimate[i]
                        
                        r.append(
                            futures.submit(
                                non_linear_register_step_regress_ldd,
                                s,
                                current_intensity_model,
                                current_velocity_model,
                                None,
                                sample_velocity,
                                parameters=parameters,
                                level=p['level'],
                                start_level=_start_level,
                                work_dir=prefix,
                                downsample=downsample,
                                debug=debug,
                                previous_velocity=previous_velocity,
                                incremental=incremental,
                                remove0=remove0,
                                sym=sym
                                )
                            )
                        velocity_estimate.append(sample_velocity)
                        #intensity_estimate.append(sample_intensity)

                    # wait for jobs to finish
                    futures.wait(r, return_when=futures.ALL_COMPLETED)
                    avg_inv_transform=None
                    
                    if debias:
                        # here all the transforms should exist
                        avg_inv_transform=LDDMriTransform(name='avg_inv',prefix=it_prefix,iter=it)
                        # 2 average all transformations
                        average_transforms_ldd(velocity_estimate, avg_inv_transform, symmetric=False, invert=True)

                    corr=[]
                    corr_transforms=[]
                    corr_samples=[]
                    
                    # 3 concatenate correction and resample
                    for (i, s) in enumerate(samples):
                        c=MriDataset(prefix=it_prefix,iter=it,name=s.name)
                        x=LDDMriTransform(name=s.name+'_corr',prefix=it_prefix,iter=it)

                        corr.append(futures.submit(concat_resample_ldd, 
                            s, velocity_estimate[i], avg_inv_transform, 
                            c, x,
                            model=ref_model,
                            symmetric=False,
                            qc=qc,
                            invert_transform=True ))
                        corr_transforms.append(x)
                        corr_samples.append(c)

                    futures.wait(corr, return_when=futures.ALL_COMPLETED)

                    # 4. perform regression and create new estimate
                    # 5. calculate residulas (?)
                    # 4+5
                    result=futures.submit(voxel_regression,
                                        intensity_design_matrix, velocity_design_matrix,
                                        corr_samples,            corr_transforms,    
                                        next_intensity_model,    next_velocity_model,     
                                        intensity_residual,      velocity_residual,
                                        blur_int_model=blur_int_model,
                                        blur_vel_model=blur_vel_model,
                                        qc=qc
                                        )
                    futures.wait([result], return_when=futures.ALL_COMPLETED)

                    # 6. cleanup
                    if cleanup :
                        print("Cleaning up iteration: {}".format(it))
                        for i in velocity_estimate:
                            i.cleanup()
                        for i in corr_samples:
                            i.cleanup()
                        if prev_velocity_estimate is not None:
                            for i in prev_velocity_estimate:
                                i.cleanup()
                        if debias:
                            avg_inv_transform.cleanup()
                else:
                    # files were there, reuse them
                    print("Iteration {} already performed, skipping".format(it))
                    corr_transforms=[]
                    # this is a hack right now
                    for (i, s) in enumerate(samples):
                        x=LDDMriTransform(name=s.name+'_corr',prefix=it_prefix,iter=it)
                        corr_transforms.append(x)
                        
                intensity_models.append(current_intensity_model)
                velocity_models.append(current_velocity_model)
                intensity_residuals.append(intensity_residual)
                velocity_residuals.append(velocity_residual)
                    
                current_intensity_model=next_intensity_model
                current_velocity_model=next_velocity_model
                
               
                result=futures.submit(average_stats_regression,
                                      current_intensity_model, current_velocity_model,
                                      intensity_residual, velocity_residual  )
                residuals.append(result)
                
                regression_results={
                        'intensity_model':     current_intensity_model,
                        'velocity_model':      current_velocity_model,
                        'intensity_residuals': intensity_residual.scan,
                        'velocity_residuals':  velocity_residual.scan,
                        }
                with open(prefix+os.sep+'results_{:03d}.json'.format(it),'w') as f:
                    json.dump(regression_results,f,indent=1, cls=LDDMRIEncoder)

                # save for next iteration
                # TODO: regularize?
                prev_velocity_estimate=corr_transforms # have to use adjusted velocity estimate

        # copy output to the destination
        futures.wait(residuals, return_when=futures.ALL_COMPLETED)
        with open(prefix+os.sep+'stats.txt','w') as f:
            for s in residuals:
                f.write("{}\n".format(s.result()))


        with open(prefix+os.sep+'results_final.json','w') as f:
            json.dump(regression_results, f, indent=1, cls=LDDMRIEncoder)


        if cleanup_intermediate:
            for i in range(len(intensity_models)-1):
                intensity_models[i].cleanup()
                velocity_models[i].cleanup()
                intensity_residuals[i].cleanup()
                velocity_residuals[i].cleanup()
            # delete unneeded models
            #shutil.rmtree(prefix+os.sep+'reg')

        return regression_results
    except mincError as e:
        print "Exception in generate_ldd_average:{}".format(str(e))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print "Exception in generate_ldd_average:{}".format(sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
        raise

def regress_ldd_csv(input_csv, 
                    int_par_count=None, 
                    model=None, 
                    mask=None, 
                    work_prefix=None, options={}, 
                    regress_model=None):
    """convinience function to run model generation using CSV input file and a fixed init"""
    internal_sample=[]
    
    with open(input_csv, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            
            par=[ float(i) for i in row[2:] ]
            par_vel=par
            par_int=par
            
            if int_par_count is not None:
                par_int=par[:int_par_count]
                par_vel=par[int_par_count:]
            _mask=row[1]
            if _mask=='':
                _mask=None
            internal_sample.append( MriDataset(scan=row[0], mask=_mask, par_int=par_int, par_vel=par_vel) )
    
    internal_model=None
    initial_intensity_model=None
    initial_velocity_model=None
    
    if regress_model is None:
        if model is not None:
            internal_model=MriDataset(scan=model, mask=mask)
    else:
        # assume that regress_model is an array
        initial_intensity_model=MriDatasetRegress(prefix=work_prefix,  name='initial_model_intensity', N=len(regress_model))
        initial_intensity_model.volume=regress_model
        initial_intensity_model.mask=mask
        
        initial_intensity_model.protect=True
        initial_velocity_model=None
        

    if work_prefix is not None and not os.path.exists(work_prefix):
        os.makedirs(work_prefix)

    return regress_ldd( internal_sample, 
                        initial_model=internal_model,
                        prefix=work_prefix, 
                        options=options,
                        initial_intensity_model=initial_intensity_model,
                        initial_velocity_model=initial_velocity_model)


def regress_ldd_simple(input_samples, 
                       int_design_matrix,
                       geo_design_matrix,
                       model=None, 
                       mask=None, 
                       work_prefix=None, options={}, 
                       regress_model=None):
    """convinience function to run model generation using CSV input file and a fixed init"""
    internal_sample=[]
    
    for (i,j) in enumerate(input_samples):
        
        internal_sample.append( MriDataset(scan=j[0], mask=j[1], 
                                           par_int=int_design_matrix[i], 
                                           par_vel=geo_design_matrix[i]) 
        )
    
    internal_model=None
    initial_intensity_model=None
    initial_velocity_model=None
    
    if regress_model is None:
        if model is not None:
            internal_model=MriDataset(scan=model, mask=mask)
    else:
        # assume that regress_model is an array
        initial_intensity_model=MriDatasetRegress(prefix=work_prefix,  name='initial_model_intensity', N=len(regress_model))
        initial_intensity_model.volume=regress_model
        initial_intensity_model.mask=mask

        initial_intensity_model.protect=True
        initial_velocity_model=None

    if work_prefix is not None and not os.path.exists(work_prefix):
        os.makedirs(work_prefix)

    return regress_ldd( internal_sample, 
                        initial_model=internal_model,
                        prefix=work_prefix, 
                        options=options,
                        initial_intensity_model=initial_intensity_model,
                        initial_velocity_model=initial_velocity_model)



def build_estimate(description_json, parameters, output_prefix, int_par_count=None):
    desc=None
    with open(description_json, 'r') as f:
        desc=json.load(f)
    intensity_parameters=parameters
    velocity_parameters=parameters
    
    if int_par_count is not None:
        intensity_parameters=parameters[:int_par_count]
        velocity_parameters=parameters[int_par_count:]
        
    if len(velocity_parameters)!=len(desc["velocity_model"]["volume"]) or \
       len(intensity_parameters)!=len(desc["intensity_model"]["volume"]):
           
       print(desc["intensity_model"]["volume"])
       print("intensity_parameters={}".format(repr(intensity_parameters)))
       
       print(desc["velocity_model"]["volume"])
       print("velocity_parameters={}".format(repr(velocity_parameters)))
       
       raise mincError("{} inconsisten number of paramters, expected {}". 
                       format(repr(intensity_parameters),
                              len(desc["velocity_model"]["volume"])))

    velocity=MriDatasetRegress(from_dict=desc["velocity_model"])
    intensity=MriDatasetRegress(from_dict=desc["intensity_model"])
    
    output_scan=MriDataset(prefix=os.path.dirname(output_prefix),name=os.path.basename(output_prefix))
    output_transform=LDDMriTransform(prefix=os.path.dirname(output_prefix),name=os.path.basename(output_prefix))

    build_approximation(intensity, velocity,
                        intensity_parameters, velocity_parameters,
                        output_scan, output_transform)
    
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

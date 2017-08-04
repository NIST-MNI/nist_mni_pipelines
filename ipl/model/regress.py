import shutil
import os
import sys
import csv
import traceback
import json

# MINC stuff
from iplMincTools      import mincTools,mincError

from .structures       import MriDataset, MriTransform, MRIEncoder, MriDatasetRegress
from .filter           import generate_flip_sample, normalize_sample
from .filter           import average_samples,average_stats
from .filter           import calculate_diff_bias_field,average_bias_fields
from .filter           import resample_and_correct_bias
from .filter           import build_approximation
from .filter           import average_stats_regression
from .filter           import voxel_regression

from .registration     import non_linear_register_step
from .registration     import dd_register_step
from .registration     import ants_register_step
from .registration     import average_transforms
from .registration     import non_linear_register_step_regress_std
from .resample         import concat_resample_nl

from scoop import futures, shared

def regress(
    samples,
    initial_model=None,
    initial_int_model=None,
    initial_def_model=None,
    output_int_model=None,
    output_def_model=None,
    output_residuals_int=None,
    output_residuals_def=None,
    prefix='.',
    options={}
    ):
    """ perform iterative model creation"""
    try:

        # make sure all input scans have parameters
        N_int=None
        N_def=None
        
        int_design_matrix=[]
        def_design_matrix=[]
        nomask=False
        
        for s in samples:
            
            if N_int is None:
                N_int=len(s.par_int)
            elif N_int!=len(s.par_int):
                raise mincError("Sample {} have inconsisten number of int paramters: {} expected {}".format(repr(s),len(s),N_int))
            
            if N_def is None:
                N_def=len(s.par_def)
            elif N_def!=len(s.par_def):
                raise mincError("Sample {} have inconsisten number of int paramters: {} expected {}".format(repr(s),len(s),N_def))
            
            int_design_matrix.append(s.par_int)
            def_design_matrix.append(s.par_def)

            if s.mask is None:
                nomask=True

        #print("Intensity design matrix=\n{}".format(repr(int_design_matrix)))
        #print("Velocity design matrix=\n{}".format(repr(def_design_matrix)))

        ref_model=None
        # current estimate of template
        if initial_model is not None:
            current_int_model     = initial_model
            current_def_model     = None
            ref_model=initial_model.scan
        else:
            current_int_model     = initial_int_model
            current_def_model     = initial_def_model
            ref_model=initial_int_model.volume[0]
        transforms=[]

        full_transforms=[]

        protocol=options.get(
                              'protocol', [{'iter':4,'level':32, 'blur_int': None, 'blur_def': None },
                                           {'iter':4,'level':16, 'blur_int': None, 'blur_def': None }]
                            )

        cleanup=    options.get('cleanup',False)
        cleanup_intermediate=    options.get('cleanup_intermediate',False)
        
        parameters=  options.get('parameters',None)
        refine=      options.get('refine',False)
        qc=          options.get('qc',False)
        downsample = options.get('downsample',None)
        start_level= options.get('start_level',None)
        debug      = options.get('debug',False)
        debias     = options.get('debias',True)
        nl_mode    = options.get('nl_mode','animal')

        if parameters is None:
            pass
            #TODO: make sensible parameters?

        int_models=[]
        def_models=[]
        int_residuals=[]
        def_residuals=[]
        
        int_residual=None
        def_residual=None
        
        prev_def_estimate=None
        # go through all the iterations
        it=0
        residuals=[]
        
        for (i,p) in enumerate(protocol):
            blur_int_model=p.get('blur_int',None)
            blur_def_model=p.get('blur_def',None)
            for j in range(1,p['iter']+1):
                it+=1
                _start_level=None
                if it==1:
                    _start_level=start_level
                # this will be a model for next iteration actually
                
                it_prefix=prefix+os.sep+str(it)
                if not os.path.exists(it_prefix):
                    os.makedirs(it_prefix)
                    
                next_int_model=MriDatasetRegress(prefix=prefix, name='model_int', iter=it, N=N_int, nomask=nomask)
                next_def_model=MriDatasetRegress(prefix=prefix, name='model_def', iter=it, N=N_def, nomask=True)
                print("next_int_model={}".format( next_int_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc') )
                
                int_residual=MriDataset(prefix=prefix, scan=next_int_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc')
                                              #name=next_int_model.name, iter=it )

                def_residual=MriDataset(prefix=prefix, scan=next_def_model.volume[0].rsplit('_0.mnc',1)[0]+'_RMS.mnc')
                                              #name=next_def_model.name, iter=it )

                # skip over existing models here!
                
                if not next_int_model.exists() or \
                   not next_def_model.exists() or \
                   not int_residual.exists() or \
                   not def_residual.exists():
                    
                    int_estimate=[]
                    def_estimate=[]
                    r=[]
                    
                    
                    # 1 for each sample generate current approximation
                    # 2. perform non-linear registration between each sample and sample-specific approximation
                    # 3. update transformation
                    # 1+2+3 - all together 
                    for (i, s) in enumerate(samples):
                        sample_def= MriTransform(name=s.name,prefix=it_prefix,iter=it)
                        sample_int= MriDataset(name=s.name,  prefix=it_prefix,iter=it)

                        previous_def=None

                        if refine and it>1:
                            previous_def=prev_def_estimate[i]
                        
                        r.append(
                            futures.submit(
                                non_linear_register_step_regress_std,
                                s,
                                current_int_model,
                                current_def_model,
                                None,
                                sample_def,
                                parameters=parameters,
                                level=p['level'],
                                start_level=_start_level,
                                work_dir=prefix,
                                downsample=downsample,
                                debug=debug,
                                previous_def=previous_def,
                                nl_mode=nl_mode
                                )
                            )
                        def_estimate.append(sample_def)
                        #int_estimate.append(sample_int)

                    # wait for jobs to finish
                    futures.wait(r, return_when=futures.ALL_COMPLETED)
                    avg_inv_transform=None

                    if debias:
                        # here all the transforms should exist
                        avg_inv_transform=MriTransform(name='avg_inv',prefix=it_prefix,iter=it)
                        # 2 average all transformations
                        average_transforms(def_estimate, avg_inv_transform, symmetric=False, invert=True,nl=True)

                    corr=[]
                    corr_transforms=[]
                    corr_samples=[]
                    
                    # 3 concatenate correction and resample
                    for (i, s) in enumerate(samples):
                        c=MriDataset(prefix=it_prefix,iter=it,name=s.name)
                        x=MriTransform(name=s.name+'_corr',prefix=it_prefix,iter=it)
                        
                        corr.append(futures.submit(concat_resample_nl, 
                            s, def_estimate[i], avg_inv_transform, 
                            c, x,
                            current_int_model,
                            p['level'],
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
                                        int_design_matrix, def_design_matrix,
                                        corr_samples,      corr_transforms,    
                                        next_int_model,    next_def_model,     
                                        int_residual,      def_residual,
                                        blur_int_model=blur_int_model,
                                        blur_def_model=blur_def_model,
                                        qc=qc
                                        )
                    
                    futures.wait([result], return_when=futures.ALL_COMPLETED)

                    # 6. cleanup
                    if cleanup :
                        print("Cleaning up iteration: {}".format(it))
                        for i in def_estimate:
                            i.cleanup()
                        for i in corr_samples:
                            i.cleanup()
                        if prev_def_estimate is not None:
                            for i in prev_def_estimate:
                                i.cleanup()
                        avg_inv_transform.cleanup()
                else:
                    # files were there, reuse them
                    print("Iteration {} already performed, skipping".format(it))
                    corr_transforms=[]
                    # this is a hack right now
                    for (i, s) in enumerate(samples):
                        x=MriTransform(name=s.name+'_corr',prefix=it_prefix,iter=it)
                        corr_transforms.append(x)
                        
                int_models.append(current_int_model)
                def_models.append(current_def_model)
                int_residuals.append(int_residual)
                def_residuals.append(def_residual)
                    
                current_int_model=next_int_model
                current_def_model=next_def_model
                
               
                result=futures.submit(average_stats_regression,
                                      current_int_model, current_def_model,
                                      int_residual, def_residual  )
                residuals.append(result)
                
                regression_results={
                        'int_model':     current_int_model,
                        'def_model':      current_def_model,
                        'int_residuals': int_residual.scan,
                        'def_residuals':  def_residual.scan,
                        }
                with open(prefix+os.sep+'results_{:03d}.json'.format(it),'w') as f:
                    json.dump(regression_results,f,indent=1, cls=MRIEncoder)

                # save for next iteration
                # TODO: regularize?
                prev_def_estimate=corr_transforms # have to use adjusted def estimate

        # copy output to the destination
        futures.wait(residuals, return_when=futures.ALL_COMPLETED)
        with open(prefix+os.sep+'stats.txt','w') as f:
            for s in residuals:
                f.write("{}\n".format(s.result()))


        with open(prefix+os.sep+'results_final.json','w') as f:
            json.dump(regression_results, f, indent=1, cls=MRIEncoder)


        if cleanup_intermediate:
            for i in range(len(int_models)-1):
                int_models[i].cleanup()
                def_models[i].cleanup()
                int_residuals[i].cleanup()
                def_residuals[i].cleanup()
            # delete unneeded models
            #shutil.rmtree(prefix+os.sep+'reg')

        return regression_results
    except mincError as e:
        print "Exception in regress:{}".format(str(e))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print "Exception in regress:{}".format(sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
        raise

def regress_csv(input_csv, 
                    int_par_count=None, 
                    model=None, 
                    mask=None, 
                    work_prefix=None, 
                    options={}, 
                    regress_model=None):
    """convinience function to run model generation using CSV input file and a fixed init"""
    internal_sample=[]
    
    with open(input_csv, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            
            par=[ float(i) for i in row[2:] ]
            par_def=par
            par_int=par
            
            if int_par_count is not None:
                par_int=par[:int_par_count]
                par_def=par[int_par_count:]
            _mask=row[1]
            if _mask=='':
                _mask=None
            internal_sample.append( MriDataset(scan=row[0], mask=_mask, par_int=par_int, par_def=par_def) )
    
    internal_model=None
    initial_int_model=None
    initial_def_model=None
    
    if regress_model is None:
        if model is not None:
            internal_model=MriDataset(scan=model, mask=mask)
    else:
        # assume that regress_model is an array
        initial_int_model=MriDatasetRegress(prefix=work_prefix,  name='initial_model_int', N=len(regress_model))
        initial_int_model.volume=regress_model
        initial_int_model.mask=mask
        
        initial_int_model.protect=True
        initial_def_model=None
        

    if work_prefix is not None and not os.path.exists(work_prefix):
        os.makedirs(work_prefix)

    return regress( internal_sample, 
                    initial_model=internal_model,
                    prefix=work_prefix, 
                    options=options,
                    initial_int_model=initial_int_model,
                    initial_def_model=initial_def_model)


def regress_simple(input_samples, 
                       int_design_matrix,
                       geo_design_matrix,
                       model=None, 
                       mask=None, 
                       work_prefix=None, 
                       options={}, 
                       regress_model=None):
    """convinience function to run model generation using CSV input file and a fixed init"""
    internal_sample=[]
    
    for (i,j) in enumerate(input_samples):
        internal_sample.append( MriDataset(scan=j[0], mask=j[1], 
                                            par_int=int_design_matrix[i], 
                                            par_def=geo_design_matrix[i]) 
        )
    
    internal_model=None
    initial_int_model=None
    initial_def_model=None
    
    if regress_model is None:
        if model is not None:
            internal_model=MriDataset(scan=model, mask=mask)
    else:
        # assume that regress_model is an array
        initial_int_model=MriDatasetRegress(prefix=work_prefix,  name='initial_model_int', N=len(regress_model))
        initial_int_model.volume=regress_model
        initial_int_model.mask=mask

        initial_int_model.protect=True
        initial_def_model=None

    if work_prefix is not None and not os.path.exists(work_prefix):
        os.makedirs(work_prefix)

    return regress( internal_sample, 
                    initial_model=internal_model,
                    prefix=work_prefix, 
                    options=options,
                    initial_int_model=initial_int_model,
                    initial_def_model=initial_def_model)



def build_estimate(description_json, parameters, output_prefix, int_par_count=None):
    desc=None
    with open(description_json, 'r') as f:
        desc=json.load(f)
        
    int_parameters=parameters
    def_parameters=parameters
    
    if int_par_count is not None:
        int_parameters=parameters[:int_par_count]
        def_parameters=parameters[int_par_count:]
        
    if len(def_parameters)!=len(desc["def_model"]["volume"]) or \
       len(int_parameters)!=len(desc["int_model"]["volume"]):
           
       print(desc["int_model"]["volume"])
       print("int_parameters={}".format(repr(int_parameters)))
       
       print(desc["def_model"]["volume"])
       print("def_parameters={}".format(repr(def_parameters)))
       
       raise mincError("{} inconsisten number of paramters, expected {}". 
                       format(repr(int_parameters),
                              len(desc["def_model"]["volume"])))

    deformation=MriDatasetRegress(from_dict=desc["def_model"])
    intensity=MriDatasetRegress(from_dict=desc["int_model"])
    
    output_scan=MriDataset(prefix=os.path.dirname(output_prefix),name=os.path.basename(output_prefix))
    output_transform=MriTransform(prefix=os.path.dirname(output_prefix),name=os.path.basename(output_prefix))

    build_approximation(intensity, deformation,
                        int_parameters, def_parameters,
                        output_scan, output_transform)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

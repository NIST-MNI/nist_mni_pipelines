import shutil
import os
import sys
import csv
import traceback
import json

# MINC stuff
from ipl.minc_tools import mincTools,mincError

from ipl.model.structures       import MriDataset, MriTransform,MRIEncoder
from ipl.model_ants.filter           import generate_flip_sample
from ipl.model_ants.filter           import average_samples, average_stats

from ipl.model_ants.registration     import ants_register_step
from ipl.model_ants.registration     import generate_update_transform
from ipl.model_ants.resample         import concat_resample_nl_inv

import ray


def generate_nonlinear_average(
    samples,
    initial_model  =None,
    output_model   =None,
    output_model_sd=None,
    prefix='.',
    options={},
    skip=0,
    stop_early=100000
    ):
    """ perform iterative model creation"""

    # use first sample as initial model
    if not initial_model:
        initial_model = samples[0]

    # current estimate of template
    current_model = initial_model
    current_model_sd = None

    transforms=[]
    corr=[]

    corr_transforms=[]
    sd=[]
    corr_samples=[]

    protocol=options.get('protocol', [{'iter':4,'level':32},
                                      {'iter':4,'level':32}] )

    cleanup=       options.get('cleanup',False)
    symmetric=     options.get('symmetric',False)
    parameters=    options.get('parameters',
            {'convergence':'1.e-9,10',
             'cost_function': 'CC',
             'transformation': 'SyN[ .25, 2, 0.5 ]',
             'use_histogram_matching': True,
             'winsorize_intensity':None,

             }
            )
    refine=        options.get('refine',True)
    qc=            options.get('qc',False)
    downsample_=   options.get('downsample',None)
    start_level=   options.get('start_level',32)
    use_median=    options.get('median',False)
    grad_step =    options.get('grad_step',0.25)

    models=[]
    models_sd=[]

    if symmetric:
        flipdir=prefix+os.sep+'flip'
        if not os.path.exists(flipdir):
            os.makedirs(flipdir)

        flip_all=[]
        # generate flipped versions of all scans
        for (i, s) in enumerate(samples):
            _s_name=os.path.basename(s.scan).rsplit('.gz',1)[0]
            s.scan_f=prefix+os.sep+'flip'+os.sep+_s_name

            if s.mask is not None:
                s.mask_f=prefix+os.sep+'flip'+os.sep+'mask_'+_s_name

            flip_all.append( generate_flip_sample.remote(s )  )

        ray.wait(flip_all, num_returns=len(flip_all))
    # go through all the iterations
    it=0
    for (i,p) in enumerate(protocol):
        downsample=p.get('downsample',downsample_)
        for j in range(1,p['iter']+1):
            it+=1
            if it>stop_early:
                break
            # this will be a model for next iteration actually

            # 1 register all subjects to current template
            next_model   =MriDataset(prefix=prefix,iter=it,name='avg',
                has_mask=current_model.has_mask())
            next_model_sd=MriDataset(prefix=prefix,iter=it,name='sd' ,
                has_mask=current_model.has_mask())
            transforms=[]

            it_prefix=prefix+os.sep+str(it)
            if not os.path.exists(it_prefix):
                os.makedirs(it_prefix)

            inv_transforms=[]
            fwd_transforms=[]
            
            start=32 # TODO:make this a parameter

            for (i, s) in enumerate(samples):
                sample_xfm=MriTransform(name=s.name,prefix=it_prefix,iter=it)
                sample_inv_xfm=MriTransform(name=s.name+'_inv',prefix=it_prefix,iter=it)

                prev_transform = None

#                if it > 1:
                    # if refine:
                    #     prev_transform = corr_transforms[i]
                    # else:
                    #     start=start_level # TWEAK?

                if it>skip and it<stop_early:
                    transforms.append(
                            ants_register_step.remote(
                            s,
                            current_model,
                            sample_xfm,
                            output_invert=sample_inv_xfm,
                            init_xfm=prev_transform,
                            symmetric=symmetric,
                            parameters=parameters,
                            level=p['level'],
                            start=start_level,
                            work_dir=prefix,
                            downsample=downsample)
                        )
                inv_transforms.append(sample_inv_xfm)
                fwd_transforms.append(sample_xfm)

            # wait for jobs to finish
            if it>skip and it<stop_early:
                ray.wait(transforms, num_returns=len(transforms))

            if cleanup and it>1 :
                # remove information from previous iteration
                for s in corr_samples:
                    s.cleanup(verbose=True)
                for x in corr_transforms:
                    x.cleanup(verbose=True)

            # here all the transforms should exist
            update_group_transform = MriTransform(name='upd_group', 
                prefix=it_prefix, iter=it)

            # 2 average all transformations
            if it>skip and it<stop_early:
                result=generate_update_transform.remote(inv_transforms, 
                    update_group_transform, nl=True, 
                    symmetric=symmetric, grad_step=grad_step)
                ray.wait([result])

            corr=[]
            corr_transforms=[]
            corr_samples=[]
            # 3 concatenate correction and resample
            for (i, s) in enumerate(samples):
                c=MriDataset(prefix=it_prefix,iter=it,name=s.name)
                x=MriTransform(name=s.name+'_corr',prefix=it_prefix,iter=it)

                if it>skip and it<stop_early:
                    corr.append(
                        concat_resample_nl_inv.remote(
                        s, 
                        inv_transforms[i], 
                        update_group_transform, 
                        c, 
                        x, 
                        current_model, 
                        level=p['level'], 
                        symmetric=symmetric, qc=qc ))
                corr_transforms.append(x)
                corr_samples.append(c)

            if it>skip and it<stop_early:
                ray.wait(corr, num_returns=len(corr))

            # cleanup transforms
            if cleanup :
                for x in inv_transforms:
                    x.cleanup()
                for x in fwd_transforms:
                    x.cleanup()
                update_group_transform.cleanup()
                
            # 4 average resampled samples to create new estimate
            if it>skip and it<stop_early:
                result = average_samples.remote( corr_samples, next_model, 
                    next_model_sd, symmetric=symmetric, 
                    symmetrize=symmetric, 
                    median=use_median)
                # TODO: add sharpening here
                ray.wait([result])

            if cleanup and it>1:
                # remove previous template estimate
                models.append(next_model)
                models_sd.append(next_model_sd)

            current_model=next_model
            current_model_sd=next_model_sd

            if it>skip and it<stop_early:
                result = average_stats.remote( next_model, next_model_sd)
                sd.append(result)
    
    # copy output to the destination
    ray.wait(sd, num_returns=len(sd))
    with open(prefix+os.sep+'stats.txt','w') as f:
        for s in sd:
            f.write("{}\n".format(ray.get(s)))
            
    results={
            'model':      current_model,
            'model_sd':   current_model_sd,
            'xfm':        corr_transforms,
            'biascorr':   None,
            'scan':       corr_samples,
            'symmetric':  symmetric,
            'samples':    samples
            }
            
    with open(prefix+os.sep+'results.json','w') as f:
         json.dump(results, f, indent=1, cls=MRIEncoder)

    if cleanup and stop_early==100000:
        # keep the final model
        models.pop()
        models_sd.pop()
        
        # delete unneeded models
        # for m in models:
        #     m.cleanup()
        # for m in models_sd:
        #     m.cleanup()

    return results

def generate_nonlinear_model_csv(input_csv, model=None, mask=None, work_prefix=None, options={},skip=0,stop_early=100000):
    internal_sample=[]

    with open(input_csv, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row)>=2:
                internal_sample.append(MriDataset(scan=row[0],mask=row[1]))
            else:
                internal_sample.append(MriDataset(scan=row[0]))

    internal_model=None
    if model is not None:
        internal_model=MriDataset(scan=model,mask=mask)

    if work_prefix is not None and not os.path.exists(work_prefix):
        os.makedirs(work_prefix)

    return generate_nonlinear_average(internal_sample,internal_model,prefix=work_prefix,options=options,skip=skip,stop_early=stop_early)

def generate_nonlinear_model(samples, model=None, mask=None, work_prefix=None, options={},skip=0,stop_early=100000):
    internal_sample=[]
    try:
        for i in samples:
            s=MriDataset(scan=i[0],mask=i[1])
            internal_sample.append(s)

        internal_model=None
        if model is not None:
            internal_model=MriDataset(scan=model,mask=mask)

        if work_prefix is not None and not os.path.exists(work_prefix):
            os.makedirs(work_prefix)

        return generate_nonlinear_average(internal_sample,internal_model,prefix=work_prefix,options=options,skip=skip,stop_early=stop_early)

    except mincError as e:
        print("Exception in generate_nonlinear_model:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in generate_nonlinear_model:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

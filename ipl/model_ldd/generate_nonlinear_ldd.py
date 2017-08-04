import shutil
import os
import sys
import csv
import traceback
import json

# MINC stuff
from ipl.minc_tools import mincTools,mincError

from .structures_ldd       import MriDataset, LDDMriTransform, LDDMRIEncoder
from .filter_ldd           import generate_flip_sample, normalize_sample
from .filter_ldd           import average_samples,average_stats
from .filter_ldd           import calculate_diff_bias_field
from .filter_ldd           import average_bias_fields
from .filter_ldd           import resample_and_correct_bias_ldd
from .registration_ldd     import non_linear_register_step_ldd
from .registration_ldd     import average_transforms_ldd
from .resample_ldd         import concat_resample_ldd

from scoop import futures, shared

def generate_ldd_average(
    samples,
    initial_model=None,
    output_model=None,
    output_model_sd=None,
    prefix='.',
    options={}
    ):
    """ perform iterative model creation"""
    try:
        #print(repr(options))
        # use first sample as initial model
        if not initial_model:
            initial_model = samples[0]

        # current estimate of template
        current_model = initial_model
        current_model_sd = None

        transforms=[]
        corr=[]

        bias_fields=[]
        corr_transforms=[]
        sd=[]
        corr_samples=[]

        protocol=options.get('protocol', [
                                          {'iter':4,'level':32},
                                          {'iter':4,'level':16}]
                            )

        cleanup=options.get('cleanup',False)
        symmetric=options.get('symmetric',False)
        parameters=options.get('parameters',None)
        refine=options.get('refine',True)
        qc=options.get('qc',False)
        downsample=options.get('downsample',None)

        models=[]
        models_sd=[]

        if symmetric:
            flipdir=prefix+os.sep+'flip'
            if not os.path.exists(flipdir):
                os.makedirs(flipdir)

            flip_all=[]
            # generate flipped versions of all scans
            for (i, s) in enumerate(samples):
                s.scan_f=prefix+os.sep+'flip'+os.sep+os.path.basename(s.scan)

                if s.mask is not None:
                    s.mask_f=prefix+os.sep+'flip'+os.sep+'mask_'+os.path.basename(s.scan)

                flip_all.append( futures.submit( generate_flip_sample,s )  )

            futures.wait(flip_all, return_when=futures.ALL_COMPLETED)
        # go through all the iterations
        it=0
        for (i,p) in enumerate(protocol):
            for j in xrange(1,p['iter']+1):
                it+=1
                # this will be a model for next iteration actually

                # 1 register all subjects to current template
                next_model=MriDataset(prefix=prefix,iter=it,name='avg')
                next_model_sd=MriDataset(prefix=prefix,iter=it,name='sd')
                transforms=[]

                it_prefix=prefix+os.sep+str(it)
                if not os.path.exists(it_prefix):
                    os.makedirs(it_prefix)

                inv_transforms=[]
                fwd_transforms=[]

                for (i, s) in enumerate(samples):
                    sample_xfm=LDDMriTransform(name=s.name,prefix=it_prefix,iter=it)

                    prev_transform = None
                    prev_bias_field = None

                    if it > 1 and refine:
                        prev_transform = corr_transforms[i]

                    transforms.append(
                        futures.submit(
                            non_linear_register_step_ldd,
                            s,
                            current_model,
                            sample_xfm,
                            init_vel=prev_transform,
                            symmetric=symmetric,
                            parameters=parameters,
                            level=p['level'],
                            work_dir=prefix,
                            downsample=downsample)
                        )
                    fwd_transforms.append(sample_xfm)

                # wait for jobs to finish
                futures.wait(transforms, return_when=futures.ALL_COMPLETED)

                if cleanup and it>1 :
                    # remove information from previous iteration
                    for s in corr_samples:
                        s.cleanup()
                    for x in corr_transforms:
                        x.cleanup()

                # here all the transforms should exist
                avg_inv_transform=LDDMriTransform(name='avg_inv',prefix=it_prefix,iter=it)

                # 2 average all transformations
                average_transforms_ldd(fwd_transforms, avg_inv_transform, symmetric=symmetric, invert=True)

                corr=[]
                corr_transforms=[]
                corr_samples=[]

                # 3 concatenate correction and resample
                for (i, s) in enumerate(samples):
                    c=MriDataset(prefix=it_prefix,iter=it,name=s.name)
                    x=LDDMriTransform(name=s.name+'_corr',prefix=it_prefix,iter=it)

                    corr.append(futures.submit(concat_resample_ldd, s,
                        fwd_transforms[i], avg_inv_transform, c, x, current_model.scan,
                        symmetric=symmetric, qc=qc ))

                    corr_transforms.append(x)
                    corr_samples.append(c)

                futures.wait(corr, return_when=futures.ALL_COMPLETED)

                # 4 average resampled samples to create new estimate

                result=futures.submit(average_samples, corr_samples, next_model, next_model_sd, symmetric=symmetric)
                futures.wait([result], return_when=futures.ALL_COMPLETED)
                

                if cleanup:
                    for s in fwd_transforms:
                        s.cleanup()

                if cleanup and it>1 :
                    # remove previous template estimate
                    models.append(next_model)
                    models_sd.append(next_model_sd)

                current_model=next_model
                current_model_sd=next_model_sd

                result=futures.submit(average_stats, next_model, next_model_sd)
                sd.append(result)

        # copy output to the destination
        futures.wait(sd, return_when=futures.ALL_COMPLETED)
        with open(prefix+os.sep+'stats.txt','w') as f:
            for s in sd:
                f.write("{}\n".format(s.result()))

        results={
                'model':      current_model,
                'model_sd':   current_model_sd,
                'vel':        corr_transforms,
                'biascorr':   None,
                'scan':       corr_samples,
                'symmetric':  symmetric,
                }

        with open(prefix+os.sep+'results.json','w') as f:
            json.dump(results,f,indent=1,cls=LDDMRIEncoder)

        if cleanup:
            # delete unneeded models
            for m in models:
                m.cleanup()
            for m in models_sd:
                m.cleanup()

        return results
    except mincError as e:
        print "Exception in generate_ldd_average:{}".format(str(e))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print "Exception in generate_ldd_average:{}".format(sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
        raise
    
def generate_ldd_model_csv(input_csv,model=None,mask=None,work_prefix=None,options={}):
    internal_sample=[]

    with open(input_csv, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            internal_sample.append(MriDataset(scan=row[0],mask=row[1]))

    internal_model=None
    if model is not None:
        internal_model=MriDataset(scan=model,mask=mask)

    if work_prefix is not None and not os.path.exists(work_prefix):
        os.makedirs(work_prefix)

    return generate_ldd_average(internal_sample,internal_model,
                                prefix=work_prefix,options=options)
                                

def generate_ldd_model(samples,model=None,mask=None,work_prefix=None,options={}):
    internal_sample=[]
    try:
        #print(repr(options))
        for i in samples:
            s=MriDataset(scan=i[0],mask=i[1])
            internal_sample.append(s)

        internal_model=None
        if model is not None:
            internal_model=MriDataset(scan=model,mask=mask)

        if work_prefix is not None and not os.path.exists(work_prefix):
            os.makedirs(work_prefix)

        return generate_ldd_average(internal_sample,internal_model,
                                    prefix=work_prefix,options=options)

    except mincError as e:
        print "Exception in generate_ldd_model:{}".format(str(e))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print "Exception in generate_ldd_model:{}".format(sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
        raise

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

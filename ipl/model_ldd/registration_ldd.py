import os
import sys
import csv
import traceback

# MINC stuff
from ipl.minc_tools import mincTools,mincError

import ipl.dd_registration

# internal stuff
from .filter_ldd     import build_approximation
from .structures_ldd import *

import ray

@ray.remote
def non_linear_register_step_ldd(
    sample,
    model,
    output,
    init_vel=None,
    level=32,
    start=None,
    symmetric=False,
    parameters=None,
    work_dir=None,
    downsample=None,
    ):
    """perform linear registration to the model, and calculate inverse"""

    try:
        _init_vel=None
        _init_vel_f=None
        
        if start is None:
            start=level
            
        if init_vel is not None:
            _init_vel=init_vel.vel
            if symmetric:
                _init_vel_f=init_vel.vel_f
        
        with mincTools() as m:
            
            if symmetric:

                if m.checkfiles(inputs=[sample.scan,model.scan,sample.scan_f],
                                outputs=[output.vel,output.vel_f]):
                                    
                    ipl.dd_registration.non_linear_register_ldd(
                        sample.scan,
                        model.scan,
                        output.vel,
                        source_mask=sample.mask,
                        target_mask=model.mask,
                        init_velocity=_init_vel,
                        parameters=parameters,
                        start=level,
                        level=level,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    ipl.dd_registration.non_linear_register_ldd(
                        sample.scan_f,
                        model.scan,
                        output.vel_f,
                        source_mask=sample.mask_f,
                        target_mask=model.mask,
                        init_velocity=_init_vel_f,
                        parameters=parameters,
                        start=level,
                        level=level,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                        
              
            else:                
                if m.checkfiles(inputs=[sample.scan,model.scan],
                                outputs=[output.vel]):

                    ipl.dd_registration.non_linear_register_ldd(
                        sample.scan,
                        model.scan,
                        output.vel,
                        source_mask=sample.mask,
                        target_mask=model.mask,
                        init_velocity=_init_vel,
                        parameters=parameters,
                        start=level,
                        level=level,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
    except mincError as e:
        print("Exception in non_linear_register_step_ldd:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in non_linear_register_step_ldd:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise
    
def average_transforms_ldd(
    samples,
    output,
    symmetric=False,
    invert=False
    ):
    """average given transformations"""
    try:
        with mincTools() as m:
            avg = []
            if not os.path.exists(output.vel):
                out=output.vel

                if invert:
                    out=m.tmp('avg.mnc')

                for i in samples:
                    avg.append(i.vel)

                if symmetric:
                    for i in samples:
                        avg.append(i.vel_f)
                m.average(avg, out)

                if invert:
                    m.calc([out],'-A[0]',output.vel)
                
    except mincError as e:
        print("Exception in average_transforms_ldd:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in average_transforms_ldd:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

def non_linear_register_step_regress_ldd(
    sample,
    model_intensity,
    model_velocity,
    output_intensity,
    output_velocity,
    level=32,
    start_level=None,
    parameters=None,
    work_dir=None,
    downsample=None,
    debug=False,
    previous_velocity=None,
    datatype='short',
    incremental=True,
    remove0=False,
    sym=False
    ):
    """perform linear registration to the model, and calculate new estimate"""
    try:

        with mincTools() as m:
            #print(repr(sample))
            
            if m.checkfiles(inputs=[sample.scan],
                            outputs=[output_velocity.vel]):

                #velocity_approximate  = LDDMriTransform(prefix=m.tempdir,name=sample.name+'_velocity')
                #intensity_approximate = MriDataset(prefix=m.tempdir,name=sample.name+'_intensity')
                intensity_approximate = None
                velocity_approximate = None
                velocity_update = None

                if debug:
                    intensity_approximate = MriDataset(      prefix=output_velocity.prefix,
                                                                name=output_velocity.name +'_int_approx',
                                                                iter=output_velocity.iter )

                    velocity_approximate  = LDDMriTransform( prefix=output_velocity.prefix,
                                                            name=output_velocity.name  +'_approx',
                                                            iter=output_velocity.iter )

                    velocity_update  = LDDMriTransform( prefix=output_velocity.prefix,
                                                            name=output_velocity.name  +'_update',
                                                            iter=output_velocity.iter )
                else:
                    intensity_approximate = MriDataset(      prefix=m.tempdir,
                                                            name=output_velocity.name +'_int_approx')

                    velocity_approximate  = LDDMriTransform( prefix=m.tempdir,
                                                            name=output_velocity.name  +'_approx' )

                    velocity_update  = LDDMriTransform( prefix=m.tempdir,
                                                            name=output_velocity.name  +'_update')

                # A hack! assume that if initial model is MriDataset it means zero regression coeff
                if isinstance(model_intensity, MriDataset):
                    intensity_approximate=model_intensity
                    velocity_approximate=None

                else:
                    (intensity_approximate, velocity_approximate) = \
                        build_approximation(model_intensity,
                                        model_velocity,
                                        sample.par_int,
                                        sample.par_vel,
                                        intensity_approximate,
                                        velocity_approximate,
                                        noresample=(not incremental),
                                        remove0=remove0)
                    if model_velocity is None:
                        velocity_approximate=None

                if start_level is None:
                    start_level=level

                # we are refining previous estimate
                init_velocity=None
                #if velocity_approximate is not None:
                    #init_velocity=velocity_approximate.vel
                if incremental:
                    if previous_velocity is not None:
                    ## have to adjust it based on the current estimate
                        if velocity_approximate is not None:
                            init_velocity=m.tmp('init_velocity.mnc')
                            m.calc( [previous_velocity.vel, velocity_approximate.vel ], 
                                    'A[0]-A[1]', init_velocity)

                        else:
                            init_velocity=previous_velocity.vel
                else:
                    if previous_velocity is not None:
                        init_velocity=previous_velocity.vel
                    elif velocity_approximate is not None:
                        init_velocity=velocity_approximate.vel
                if sym:
                    print("Using symmetrization!")
                    # TODO: parallelalize this
                    update1=m.tmp('update1.mnc')
                    m.non_linear_register_ldd(
                        intensity_approximate.scan,
                        sample.scan,
                        update1,
                        source_mask=intensity_approximate.mask,
                        target_mask=sample.mask,
                        init_velocity=init_velocity,
                        parameters=parameters,
                        start=start_level,
                        level=level,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    update2=m.tmp('update2.mnc')
                    m.non_linear_register_ldd(
                        sample.scan,
                        intensity_approximate.scan,
                        update2,
                        source_mask=sample.mask,
                        target_mask=intensity_approximate.mask,
                        init_velocity=init_velocity,
                        parameters=parameters,
                        start=start_level,
                        level=level,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    m.calc([update1,update2],'(A[0]-A[1])/2.0',velocity_update.vel)
                else:
                    m.non_linear_register_ldd(
                        intensity_approximate.scan,
                        sample.scan,
                        velocity_update.vel,
                        source_mask=intensity_approximate.mask,
                        target_mask=sample.mask,
                        init_velocity=init_velocity,
                        parameters=parameters,
                        start=start_level,
                        level=level,
                        downsample=downsample,
                        #work_dir=work_dir
                        )

                # update estimate, possibility to use link function?
                if incremental and velocity_approximate is not None:
                    m.calc( [velocity_approximate.vel, velocity_update.vel ], 'A[0]+A[1]', output_velocity.vel,
                            datatype='-'+datatype)
                else:
                    m.calc( [velocity_update.vel ], 'A[0]', output_velocity.vel, 
                        datatype='-'+datatype)

                if output_intensity is not None:
                    # resample intensity
                    m.resample_smooth_logspace(sample.scan, output_intensity.scan,
                                               velocity=output_velocity.vel,
                                               invert_transform=True,
                                               datatype='-'+datatype
                                            )

                    if sample.mask is not None:
                        m.resample_labels_logspace(sample.mask, output_intensity.mask,
                                                velocity=output_velocity.vel,
                                                invert_transform=True)
        # done
        
    except mincError as e:
        print("Exception in non_linear_register_step_ldd:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in non_linear_register_step_ldd:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

import shutil
import os
import sys
import csv
import traceback


# MINC stuff
from ipl.minc_tools import mincTools,mincError

import ipl.registration
import ipl.dd_registration
import ipl.ants_registration
import ipl.elastix_registration

import numpy as np

from .structures import *
from .filter import build_approximation

try:
    # needed to read and write XFM files
    from minc2_simple import minc2_xfm
    # needed for matrix log and exp
    import scipy.linalg
except:
    pass


import ray

def xfmavg(inputs, output, verbose=False):
    # TODO: handle inversion flag correctly
    all_linear=True
    all_nonlinear=True
    input_xfms=[]
    input_grids=[]
    
    for j in inputs:
        x=minc2_xfm(j)
        if x.get_n_concat()==1 and x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR:
            # this is a linear matrix
            input_xfms.append(np.asmatrix(x.get_linear_transform()))
        else:
            all_linear&=False
            # strip identity matrixes
            nl=[]
            _identity=np.asmatrix(np.identity(4))
            _eps=1e-6
            if x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR and x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
                if scipy.linalg.norm(_identity-np.asmatrix(x.get_linear_transform(0)) )>_eps: # this is non-identity matrix
                    all_nonlinear&=False
                else:
                    # TODO: if grid have to be inverted!
                    (grid_file,grid_invert)=x.get_grid_transform(1)
                    input_grids.append(grid_file)
            elif x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
                # TODO: if grid have to be inverted!
                (grid_file,grid_invert)=x.get_grid_transform(0)
                input_grids.append(grid_file)
                
    if all_linear:
        acc=np.asmatrix(np.zeros([4,4],dtype=complex))
        for i in input_xfms:
            print(i)
            acc+=scipy.linalg.logm(i)
            
        acc/=len(input_xfms)
        acc=np.asarray(scipy.linalg.expm(acc).real,'float64','C')
        
        x=minc2_xfm()
        x.append_linear_transform(acc)
        x.save(output)
        
    elif all_nonlinear:
        
        output_grid=output.rsplit('.xfm',1)[0]+'_grid_0.mnc'
        
        with mincTools(verbose=2) as m:
            m.average(input_grids,output_grid)
        
        x=minc2_xfm()
        x.append_grid_transform(output_grid, False)
        x.save(output)
    else:
        raise Exception("Mixed XFM files provided as input")

@ray.remote
def linear_register_step(
    sample,
    model,
    output,
    output_invert=None,
    init_xfm=None,
    symmetric=False,
    reg_type='-lsq12',
    objective='-xcorr',
    linreg=None,
    work_dir=None,
    bias=None,
    downsample=None,
    avg_symmetric=True
    ):
    """perform linear registration to the model, and calculate inverse"""

    try:
        _init_xfm=None
        _init_xfm_f=None
        
        if init_xfm is not None:
            _init_xfm=init_xfm.xfm
            if symmetric:
                _init_xfm_f=init_xfm.xfm_f
        
        with mincTools() as m:
            scan=sample.scan
            
            if bias is not None:
                m.calc([sample.scan,bias.scan],'A[0]*A[1]',m.tmp('corr.mnc'))
                scan=m.tmp('corr.mnc')
            
            if symmetric:
                scan_f=sample.scan_f
                
                if bias is not None:
                    m.calc([sample.scan_f,bias.scan_f],'A[0]*A[1]',m.tmp('corr_f.mnc'))
                    scan_f=m.tmp('corr_f.mnc')
                    
                _out_xfm=output.xfm
                _out_xfm_f=output.xfm_f
                
                if avg_symmetric:
                    _out_xfm=m.tmp('straight.xfm')
                    _out_xfm_f=m.tmp('flipped.xfm')
                    
                ipl.registration.linear_register(
                    scan,
                    model.scan,
                    _out_xfm,
                    source_mask=sample.mask,
                    target_mask=model.mask,
                    init_xfm=_init_xfm,
                    objective=objective,
                    parameters=reg_type,
                    conf=linreg,
                    downsample=downsample,
                    #work_dir=work_dir
                    )
                ipl.registration.linear_register(
                    scan_f,
                    model.scan,
                    _out_xfm_f,
                    source_mask=sample.mask,
                    target_mask=model.mask,
                    init_xfm=_init_xfm_f,
                    objective=objective,
                    parameters=reg_type,
                    conf=linreg,
                    downsample=downsample,
                    #work_dir=work_dir
                    )
                    
                if avg_symmetric:
                    m.param2xfm(m.tmp('flip_x.xfm'), scales=[-1.0,1.0,1.0] )
                    m.xfmconcat([m.tmp('flip_x.xfm'), _out_xfm_f , m.tmp('flip_x.xfm')], m.tmp('double_flipped.xfm'))
                    
                    xfmavg([_out_xfm,m.tmp('double_flipped.xfm')],output.xfm)
                    m.xfmconcat([m.tmp('flip_x.xfm'), output.xfm , m.tmp('flip_x.xfm')], output.xfm_f )
                    
            else:
                ipl.registration.linear_register(
                    scan,
                    model.scan,
                    output.xfm,
                    source_mask=sample.mask,
                    target_mask=model.mask,
                    init_xfm=_init_xfm,
                    objective=objective,
                    parameters=reg_type,
                    conf=linreg,
                    downsample=downsample
                    #work_dir=work_dir
                    )
            if output_invert is not None:
                m.xfminvert(output.xfm, output_invert.xfm)
                
                if symmetric:
                    m.xfminvert(output.xfm_f, output_invert.xfm_f)

        return True
    except mincError as e:
        print("Exception in linear_register_step:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in linear_register_step:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote   
def non_linear_register_step(
    sample,
    model,
    output,
    output_invert=None,
    init_xfm=None,
    level=32,
    start=None,
    symmetric=False,
    parameters=None,
    work_dir=None,
    downsample=None,
    avg_symmetric=True,
    verbose=2   
    ):
    """perform linear registration to the model, and calculate inverse"""

    try:
        _init_xfm=None
        _init_xfm_f=None
        
        if start is None:
            start=level
        
        if init_xfm is not None:
            _init_xfm=init_xfm.xfm
            if symmetric:
                _init_xfm_f=init_xfm.xfm_f
        
        with mincTools(verbose=verbose) as m:
            
            if symmetric:

                if m.checkfiles(inputs=[sample.scan,model.scan,sample.scan_f],
                                outputs=[output.xfm,output.xfm_f]):
                    
                    ipl.registration.non_linear_register_full(
                        sample.scan,
                        model.scan,
                        m.tmp('forward.xfm'),
                        source_mask=sample.mask,
                        target_mask=model.mask,
                        init_xfm=_init_xfm,
                        parameters=parameters,
                        level=level,
                        start=start,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    
                    ipl.registration.non_linear_register_full(
                        sample.scan_f,
                        model.scan,
                        m.tmp('forward_f.xfm'),
                        source_mask=sample.mask_f,
                        target_mask=model.mask,
                        init_xfm=_init_xfm_f,
                        parameters=parameters,
                        level=level,
                        start=start,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    
                    if avg_symmetric:
                        m.param2xfm(m.tmp('flip_x.xfm'), scales=[-1.0,1.0,1.0] )
                        m.xfmconcat([m.tmp('flip_x.xfm'), m.tmp('forward_f.xfm') , m.tmp('flip_x.xfm')], m.tmp('forward_f_f.xfm'))
                        
                        m.xfm_normalize(m.tmp('forward.xfm'),model.scan,m.tmp('forward_n.xfm'),step=level)
                        m.xfm_normalize(m.tmp('forward_f_f.xfm'),model.scan,m.tmp('forward_f_f_n.xfm'),step=level)
                        
                        xfmavg([m.tmp('forward_n.xfm'),m.tmp('forward_f_f_n.xfm')],output.xfm)
                        m.xfmconcat([m.tmp('flip_x.xfm'), output.xfm , m.tmp('flip_x.xfm')], m.tmp('output_f.xfm' ))
                        m.xfm_normalize(m.tmp('output_f.xfm'),model.scan,output.xfm_f,step=level)
                        
                    else:
                        m.xfm_normalize(m.tmp('forward.xfm'),model.scan,output.xfm,step=level)
                        m.xfm_normalize(m.tmp('forward_f.xfm'),model.scan,output.xfm_f,step=level)
                
            else:
                if m.checkfiles(inputs=[sample.scan,model.scan],
                                outputs=[output.xfm]):

                    ipl.registration.non_linear_register_full(
                        sample.scan,
                        model.scan,
                        m.tmp('forward.xfm'),
                        source_mask=sample.mask,
                        target_mask=model.mask,
                        init_xfm=_init_xfm,
                        parameters=parameters,
                        level=level,
                        start=start,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    m.xfm_normalize(m.tmp('forward.xfm'),model.scan,output.xfm,step=level)

            if output_invert is not None and m.checkfiles(inputs=[], outputs=[output_invert.xfm]):
                m.xfm_normalize(m.tmp('forward.xfm'),model.scan,output_invert.xfm,step=level,invert=True)
                if symmetric:
                    m.xfm_normalize(m.tmp('forward_f.xfm'),model.scan,output_invert.xfm_f,step=level,invert=True)
                    
        return True
    except mincError as e:
        print("Exception in non_linear_register_step:{}".format(str(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in non_linear_register_step:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote
def dd_register_step(
    sample,
    model,
    output,
    output_invert=None,
    init_xfm=None,
    level=32,
    start=None,
    symmetric=False,
    parameters=None,
    work_dir=None,
    downsample=None,
    avg_symmetric=True,
    verbose=2
    ):
    """perform linear registration to the model, and calculate inverse"""

    try:
        _init_xfm=None
        _init_xfm_f=None
        
        if start is None:
            start=level
        
        if init_xfm is not None:
            _init_xfm=init_xfm.xfm
            if symmetric:
                _init_xfm_f=init_xfm.xfm_f
        
        with mincTools(verbose=verbose) as m:
            
            if symmetric:

                if m.checkfiles(inputs=[sample.scan,model.scan,sample.scan_f],
                                outputs=[output.xfm,output.xfm_f]):
                    
                    ipl.dd_registration.non_linear_register_dd(
                        sample.scan,
                        model.scan,
                        m.tmp('forward.xfm'),
                        source_mask=sample.mask,
                        target_mask=model.mask,
                        init_xfm=_init_xfm,
                        parameters=parameters,
                        level=level,
                        start=start,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    
                    ipl.dd_registration.non_linear_register_dd(
                        sample.scan_f,
                        model.scan,
                        m.tmp('forward_f.xfm'),
                        source_mask=sample.mask_f,
                        target_mask=model.mask,
                        init_xfm=_init_xfm_f,
                        parameters=parameters,
                        level=level,
                        start=start,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    
                    if avg_symmetric:
                        m.param2xfm(m.tmp('flip_x.xfm'), scales=[-1.0,1.0,1.0] )
                        m.xfmconcat([m.tmp('flip_x.xfm'), m.tmp('forward_f.xfm') , m.tmp('flip_x.xfm')], m.tmp('forward_f_f.xfm'))
                        
                        m.xfm_normalize(m.tmp('forward.xfm'),model.scan,m.tmp('forward_n.xfm'),step=level)
                        m.xfm_normalize(m.tmp('forward_f_f.xfm'),model.scan,m.tmp('forward_f_f_n.xfm'),step=level)
                        
                        xfmavg([m.tmp('forward_n.xfm'),m.tmp('forward_f_f_n.xfm')],output.xfm)
                        m.xfmconcat([m.tmp('flip_x.xfm'), output.xfm , m.tmp('flip_x.xfm')], m.tmp('output_f.xfm' ))
                        m.xfm_normalize(m.tmp('output_f.xfm'),model.scan,output.xfm_f,step=level)
                        
                    else:
                        m.xfm_normalize(m.tmp('forward.xfm'),model.scan,output.xfm,step=level)
                        m.xfm_normalize(m.tmp('forward_f.xfm'),model.scan,output.xfm_f,step=level)
                
            else:
                if m.checkfiles(inputs=[sample.scan,model.scan],
                                outputs=[output.xfm]):

                    ipl.dd_registration.non_linear_register_dd(
                        sample.scan,
                        model.scan,
                        m.tmp('forward.xfm'),
                        source_mask=sample.mask,
                        target_mask=model.mask,
                        init_xfm=_init_xfm,
                        parameters=parameters,
                        level=level,
                        start=start,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    m.xfm_normalize(m.tmp('forward.xfm'),model.scan,output.xfm,step=level)

            if output_invert is not None and m.checkfiles(inputs=[], outputs=[output_invert.xfm]):
                m.xfm_normalize(m.tmp('forward.xfm'),model.scan,output_invert.xfm,step=level,invert=True)
                if symmetric:
                    m.xfm_normalize(m.tmp('forward_f.xfm'),model.scan,output_invert.xfm_f,step=level,invert=True)
                    
        return True
    except mincError as e:
        print("Exception in dd_register_step:{}".format(str(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in dd_register_step:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote
def ants_register_step(
    sample,
    model,
    output,
    output_invert=None,
    init_xfm=None,
    level=32,
    start=None,
    symmetric=False,
    parameters=None,
    work_dir=None,
    downsample=None,
    avg_symmetric=True,
    verbose=2 
    ):
    """perform linear registration to the model, and calculate inverse"""

    try:
        _init_xfm=None
        _init_xfm_f=None
        
        if start is None:
            start=level
        
        if init_xfm is not None:
            _init_xfm=init_xfm.xfm
            if symmetric:
                _init_xfm_f=init_xfm.xfm_f
        
        with mincTools(verbose=verbose) as m:
            out=m.tmp('forward')
            out_f=m.tmp('forward_f')
            if symmetric:

                if m.checkfiles(inputs=[sample.scan,model.scan,sample.scan_f],
                                outputs=[output.xfm, output.xfm_f]):
                    
                    ipl.ants_registration.non_linear_register_ants2(
                        sample.scan,
                        model.scan,
                        out+'.xfm',
                        source_mask=sample.mask,
                        target_mask=model.mask,
                        init_xfm=_init_xfm,
                        parameters=parameters,
                        level=level,
                        start=start,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    
                    ipl.ants_registration.non_linear_register_ants2(
                        sample.scan_f,
                        model.scan,
                        out_f+'.xfm',
                        source_mask=sample.mask_f,
                        target_mask=model.mask,
                        init_xfm=_init_xfm_f,
                        parameters=parameters,
                        level=level,
                        start=start,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    
                    if avg_symmetric:
                        m.param2xfm(m.tmp('flip_x.xfm'), scales=[-1.0,1.0,1.0] )
                        m.xfmconcat([m.tmp('flip_x.xfm'), out_f+'.xfm', m.tmp('flip_x.xfm')], m.tmp('forward_f_f.xfm'))
                        
                        m.xfm_normalize(out+'.xfm', model.scan, m.tmp('forward_n.xfm'),step=level)
                        m.xfm_normalize(m.tmp('forward_f_f.xfm'),model.scan,m.tmp('forward_f_f_n.xfm'),step=level)
                        
                        xfmavg([m.tmp('forward_n.xfm'),m.tmp('forward_f_f_n.xfm')],output.xfm)
                        m.xfmconcat([m.tmp('flip_x.xfm'), output.xfm , m.tmp('flip_x.xfm')], m.tmp('output_f.xfm' ))
                        m.xfm_normalize(out_f+'.xfm',model.scan,output.xfm_f,step=level)
                        
                    else:
                        m.xfm_normalize(out+'.xfm',model.scan,output.xfm,step=level)
                        m.xfm_normalize(out_f+'.xfm',model.scan,output.xfm_f,step=level)
                
            else:
                if m.checkfiles(inputs=[sample.scan,model.scan],
                                outputs=[output.xfm]):

                    ipl.ants_registration.non_linear_register_ants2(
                        sample.scan,
                        model.scan,
                        out+'.xfm',
                        source_mask=sample.mask,
                        target_mask=model.mask,
                        init_xfm=_init_xfm,
                        parameters=parameters,
                        level=level,
                        start=start,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    m.xfm_normalize(out+'.xfm',model.scan,output.xfm,step=level)

            if output_invert is not None and m.checkfiles(inputs=[], outputs=[output_invert.xfm]):
                m.xfm_normalize(out+'_inverse.xfm',model.scan,output_invert.xfm,step=level)
                if symmetric:
                    m.xfm_normalize(out_f+'_inverse.xfm',model.scan,output_invert.xfm_f,step=level)
                    
        return True
    except mincError as e:
        print( "Exception in ants_register_step:{}".format(str(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in ants_register_step:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote
def elastix_register_step(
    sample,
    model,
    output,
    output_invert=None,
    init_xfm=None,
    level=32,
    start=None,
    symmetric=False,
    parameters=None,
    work_dir=None,
    downsample=None,
    avg_symmetric=True,
    verbose=2   
    ):
    """perform linear registration to the model, and calculate inverse"""

    try:
        _init_xfm=None
        _init_xfm_f=None
        
        if start is None:
            start=level
        
        if init_xfm is not None:
            _init_xfm=init_xfm.xfm
            if symmetric:
                _init_xfm_f=init_xfm.xfm_f
        
        # setup parameters appropriate for given level
        elx_parameters=parameters.get(str(level),{})
        downsample_grid=elx_parameters.get('downsample_grid',level/2.0)
        
        with mincTools(verbose=verbose) as m:
            out=m.tmp('forward')
            out_f=m.tmp('forward_f')
            if symmetric:

                if m.checkfiles(inputs=[sample.scan,model.scan,sample.scan_f],
                                outputs=[output.xfm, output.xfm_f]):
                    
                    ipl.elastix_registration.register_elastix(
                        sample.scan,
                        model.scan,
                        output_xfm=out+'.xfm',
                        source_mask=sample.mask,
                        target_mask=model.mask,
                        init_xfm=_init_xfm,
                        parameters=elx_parameters,
                        downsample_grid=downsample_grid,
                        downsample=downsample,
                        nl=True
                        )
                    
                    ipl.elastix_registration.register_elastix(
                        sample.scan_f,
                        model.scan,
                        output_xfm=out_f+'.xfm',
                        source_mask=sample.mask_f,
                        target_mask=model.mask,
                        init_xfm=_init_xfm_f,
                        parameters=elx_parameters,
                        downsample_grid=downsample_grid,
                        downsample=downsample,
                        nl=True
                        )
                    
                    if avg_symmetric:
                        m.param2xfm(m.tmp('flip_x.xfm'), scales=[-1.0,1.0,1.0] )
                        m.xfmconcat([m.tmp('flip_x.xfm'), out_f+'.xfm', m.tmp('flip_x.xfm')], m.tmp('forward_f_f.xfm'))
                        
                        m.xfm_normalize(out+'.xfm', model.scan, m.tmp('forward_n.xfm'),step=level)
                        m.xfm_normalize(m.tmp('forward_f_f.xfm'),model.scan,m.tmp('forward_f_f_n.xfm'),step=level)
                        
                        xfmavg([m.tmp('forward_n.xfm'),m.tmp('forward_f_f_n.xfm')],output.xfm)
                        m.xfmconcat([m.tmp('flip_x.xfm'), output.xfm , m.tmp('flip_x.xfm')], m.tmp('output_f.xfm' ))
                        m.xfm_normalize(out_f+'.xfm',model.scan,output.xfm_f,step=level)
                        
                    else:
                        m.xfm_normalize(out+'.xfm',  model.scan,output.xfm,  step=level)
                        m.xfm_normalize(out_f+'.xfm',model.scan,output.xfm_f,step=level)

            else:
                if m.checkfiles(inputs=[sample.scan,model.scan],
                                outputs=[output.xfm]):

                    ipl.elastix_registration.register_elastix(
                        sample.scan,
                        model.scan,
                        output_xfm=out+'.xfm',
                        source_mask=sample.mask,
                        target_mask=model.mask,
                        init_xfm=_init_xfm,
                        parameters=elx_parameters,
                        downsample_grid=downsample_grid,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    m.xfm_normalize(out+'.xfm',model.scan,output.xfm,step=level)

            if output_invert is not None and m.checkfiles(inputs=[], outputs=[output_invert.xfm]):
                m.xfm_normalize(out+'.xfm',model.scan,output_invert.xfm,step=level,invert=True)
                if symmetric:
                    m.xfm_normalize(out_f+'.xfm',model.scan,output_invert.xfm_f,step=level,invert=True)
                    
        return True
    except mincError as e:
        print("Exception in elastix_register_step:{}".format(str(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in elastix_register_step:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote
def average_transforms(
    samples,
    output,
    nl=False,
    symmetric=False,
    invert=False
    ):
    """average given transformations"""
    try:
        with mincTools() as m:
            avg = []
            out_xfm=output.xfm

            for i in samples:
                avg.append(i.xfm)

            if symmetric:
                for i in samples:
                    avg.append(i.xfm_f)
            if invert:
                out_xfm=m.tmp("average.xfm")
            xfmavg(avg, out_xfm)

            if invert:
                m.xfminvert(out_xfm, output.xfm)
        return True
    except mincError as e:
        print("Exception in average_transforms:{}".format(str(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in average_transforms:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote
def non_linear_register_step_regress_std(
    sample,
    model_int,
    model_def,
    output_int,
    output_def,
    level=32,
    start_level=None,
    parameters=None,
    work_dir=None,
    downsample=None,
    debug=False,
    previous_def=None,
    datatype='short',
    nl_mode='animal',
    verbose=2
    ):
    """perform linear registration to the model, and calculate new estimate"""
    try:

        with mincTools(verbose=verbose) as m:
            if m.checkfiles(inputs=[sample.scan],
                            outputs=[output_def.xfm]):

                int_approximate = None
                def_approximate = None
                def_update      = None

                if debug:
                    int_approximate  = MriDataset(  prefix=output_def.prefix,
                                                    name=output_def.name +'_int_approx',
                                                    iter=output_def.iter )

                    def_approximate  = MriTransform( prefix=output_def.prefix,
                                                     name=output_def.name  +'_approx',
                                                     iter=output_def.iter )

                    def_update       = MriTransform( prefix=output_def.prefix,
                                                     name=output_def.name  +'_update',
                                                     iter=output_def.iter )
                else:
                    int_approximate  = MriDataset(   prefix=m.tempdir,
                                                     name=output_def.name +'_int_approx')

                    def_approximate  = MriTransform( prefix=m.tempdir,
                                                     name=output_def.name  +'_approx' )

                    def_update       = MriTransform( prefix=m.tempdir,
                                                     name=output_def.name  +'_update')

                # A hack! assume that if initial model is MriDataset it means zero regression coeff
                if isinstance(model_int, MriDataset):
                    int_approximate=model_int
                    def_approximate=None
                else:
                    (int_approximate, def_approximate) = \
                        build_approximation(model_int,
                                        model_def,
                                        sample.par_int,
                                        sample.par_def,
                                        int_approximate,
                                        def_approximate,
                                        noresample=False)
                    if model_def is None:
                        def_approximate=None

                if start_level is None:
                    start_level=level

                init_xfm=None

                # we are refining previous estimate
                if previous_def is not None:
                    ## have to adjust it based on the current estimate
                    if def_approximate is not None:
                        init_xfm=m.tmp('init_def.xfm')
                        m.xfminvert(def_approximate.xfm, m.tmp('approx_inv.xfm'))
                        m.xfmconcat(previous_def.xfm,m.tmp('approx_inv.xfm'))
                        m.xfm_normalize(m.tmp('approx_inv.xfm'),int_approximate.scan,m.tmp('init.xfm'),step=level)
                        init_xfm=m.tmp('init.xfm')
                    else:
                        init_xfm=previous_def.xfm
                
                print("level={} start={}".format(level,start_level))
                print("parameters={}".format(repr(parameters)))
                
                if nl_mode=='animal':
                    ipl.registration.non_linear_register_full(
                        int_approximate.scan,
                        sample.scan,
                        def_update.xfm,
                        source_mask=int_approximate.mask,
                        target_mask=sample.mask,
                        init_xfm=init_xfm,
                        parameters=parameters,
                        start=start_level,
                        level=level,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                elif nl_mode=='ants':
                    ipl.ants_registration.non_linear_register_ants2(
                        int_approximate.scan,
                        sample.scan,
                        def_update.xfm,
                        source_mask=int_approximate.mask,
                        target_mask=sample.mask,
                        init_xfm=init_xfm,
                        parameters=parameters,
                        start=start_level,
                        level=level,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                elif nl_mode=='dd':
                    ipl.dd_registration.non_linear_register_dd(
                        int_approximate.scan,
                        sample.scan,
                        def_update.xfm,
                        source_mask=int_approximate.mask,
                        target_mask=sample.mask,
                        init_xfm=init_xfm,
                        parameters=parameters,
                        start=start_level,
                        level=level,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                else: #elastix
                    ipl.elatix_registration.register_elastix(
                        int_approximate.scan,
                        sample.scan,
                        def_update.xfm,
                        source_mask=int_approximate.mask,
                        target_mask=sample.mask,
                        init_xfm=init_xfm,
                        parameters=parameters,
                        downsample_grid=level,
                        downsample=downsample,
                        #work_dir=work_dir
                        )
                    

                # update estimate, 
                if def_approximate is not None:
                    m.xfmconcat([def_approximate.xfm,def_update.xfm],m.tmp('output_def.xfm'))
                    m.xfm_normalize(m.tmp('output_def.xfm'),int_approximate.scan,output_def.xfm,  step=level)
                else:
                    m.xfm_normalize(def_update.xfm,int_approximate.scan,output_def.xfm,  step=level)
                    
                if output_int is not None:
                    # resample intensity
                    m.resample_smooth(sample.scan, output_int.scan,
                                               transform=output_def.xfm,
                                               invert_transform=True,
                                               datatype='-'+datatype
                                              )
                    if sample.mask is not None:
                        m.resample_labels(sample.mask, output_int.mask,
                                                transform=output_def.xfm,
                                                invert_transform=True)
        # done
        
    except mincError as e:
        print("Exception in non_linear_register_step_regress_std:{}".format(str(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in non_linear_register_step_regress_std:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise
        

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

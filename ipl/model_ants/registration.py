import shutil
import os
import sys
import csv
import traceback


# MINC stuff
from ipl.minc_tools import mincTools,mincError

import ipl.ants_registration

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

def xfmavg(inputs, output, verbose=False,mult_grid=None):
    # TODO: handle inversion flag correctly
    all_linear=True
    all_nonlinear=True
    input_xfms=[]
    input_grids=[]
    input_inv_grids=[]
    
    for j in inputs:
        x=minc2_xfm(j)
        if x.get_n_concat()==1 and x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR:
            # this is a linear matrix
            input_xfms.append(np.asmatrix(x.get_linear_transform()))
        else:
            all_linear &= False
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
                    input_inv_grids.append(grid_invert)
            elif x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
                # TODO: if grid have to be inverted!
                (grid_file, grid_invert)=x.get_grid_transform(0)
                input_grids.append(grid_file)
                input_inv_grids.append(grid_invert)
                
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
        # check if all transform have the same inverse flag
        if any(input_inv_grids) != all(input_inv_grids):
            raise Exception("Mixed XFM inversion flag in nonlinear transforms")
        
        output_grid=output.rsplit('.xfm',1)[0]+'_grid_0.mnc'
        
        with mincTools(verbose=2) as m:
            if mult_grid is not None:
                m.average(input_grids, m.tmp("avg_grid.mnc"))
                m.calc([m.tmp("avg_grid.mnc")],f"A[0]*{mult_grid}",output_grid)
            else:    
                m.average(input_grids, output_grid)
        
        x=minc2_xfm()
        x.append_grid_transform(output_grid, all(input_inv_grids))
        x.save(output)
    else:
        raise Exception("Mixed XFM files provided as input")


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
def generate_update_transform(
    samples,
    output,
    symmetric=False,
    invert=False,
    grad_step=0.25,
    ):
    """create update transform ANTs style"""
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
            # TODO: implement Affine part 
            xfmavg(avg, out_xfm, mult_grid=-grad_step)

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

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

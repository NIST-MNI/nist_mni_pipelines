#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from minc2_simple import minc2_file
from minc2_simple import minc2_xfm
from minc2_simple import minc2_dim
from minc2_simple import minc2_error

from time import gmtime, strftime

import numpy as np

from .geo import decompose,compose

""" 
    Create minc-style history entry
"""
def format_history(argv):
    stamp=strftime("%a %b %d %T %Y>>>", gmtime())
    return stamp+(' '.join(argv))

""" 
    Convert minc file header int voxel to world affine matrix
"""
def hdr_to_affine(hdr):
    rot=np.zeros((3,3))
    scales=np.zeros((3,3))
    start=np.zeros(3)

    ax = np.array([h.id for h in hdr])

    for i in range(3):
        aa=np.where(ax == (i+1))[0][0] # HACK, assumes DIM_X=1,DIM_Y=2 etc
        if hdr[aa].have_dir_cos:
            rot[:,i] = hdr[aa].dir_cos
        else:
            rot[i,i] = 1

        scales[i,i] = hdr[aa].step
        start[i] = hdr[aa].start
    
    origin = rot@start
    out = np.eye(4)

    out[0:3,0:3] = rot@scales
    out[0:3,3]   = origin
    return out


"""
    Convert affine matrix into minc file dimension description
"""
def affine_to_dims(aff, shape):
    # convert to minc2 sampling format
    start, step, dir_cos = decompose(aff)
    if len(shape) == 3: # this is a 3D volume
        dims=[
                minc2_dim(id=i+1, length=shape[2-i], start=start[i], step=step[i], have_dir_cos=True, dir_cos=np.ascontiguousarray(dir_cos[0:3,i])) for i in range(3)
            ]
    elif len(shape) == 4: # this is a 3D grid volume, vector space is the last one
        dims=[
                minc2_dim(id=i+1, length=shape[2-i], start=start[i], step=step[i], have_dir_cos=True, dir_cos=np.ascontiguousarray(dir_cos[0:3,i])) for i in range(3)
             ] + [ 
                minc2_dim(id=minc2_file.MINC2_DIM_VEC, length=shape[3], start=0, step=1, have_dir_cos=False, dir_cos=[0,0,0])
             ]
    else:
        assert False, f"Unsupported number of dimensions: {len(shape)}"
    return dims


""" 
    Load minc volume into tensor and return voxel2world matrix too
"""
def load_minc_volume(fname, as_byte=False):
    mm=minc2_file(fname)
    mm.setup_standard_order()

    d = mm.load_complete_volume_tensor(minc2_file.MINC2_UBYTE if as_byte else minc2_file.MINC2_DOUBLE)
    aff = np.asmatrix(hdr_to_affine(mm.representation_dims()))

    mm.close()
    return d, aff


""" 
    Load minc volume into numpy volume and return voxel2world matrix too
"""
def load_minc_volume_np(fname, as_byte=False, dtype=None):
    mm=minc2_file(fname)
    mm.setup_standard_order()

    if as_byte:
        dtype='uint8'
    elif dtype is None:
        dtype='float64'

    d = mm.load_complete_volume(dtype)
    aff=np.asmatrix(hdr_to_affine(mm.representation_dims()))

    mm.close()
    return d, aff


"""
    Save torch tensor on numpy volume into minc file
"""
def save_minc_volume(fn, data, aff, ref_fname=None, history=None):
    dims=affine_to_dims(aff, data.shape)
    out=minc2_file()
    if data.dtype == np.uint8: 
        out.define(dims, minc2_file.MINC2_UBYTE, minc2_file.MINC2_UBYTE)
    elif data.dtype == np.uint16:
        out.define(dims, minc2_file.MINC2_USHORT, minc2_file.MINC2_USHORT)
    elif data.dtype == np.int8: 
        out.define(dims, minc2_file.MINC2_BYTE, minc2_file.MINC2_BYTE)
    elif data.dtype == np.int16:
        out.define(dims, minc2_file.MINC2_SHORT, minc2_file.MINC2_SHORT)
    elif data.dtype == np.float32:
        out.define(dims, minc2_file.MINC2_SHORT, minc2_file.MINC2_FLOAT)
    elif data.dtype == np.float64:
        out.define(dims, minc2_file.MINC2_SHORT, minc2_file.MINC2_DOUBLE)
    else:
        assert(False) # unsupported type

    out.create(fn)
    
    if ref_fname is not None:
        ref=minc2_file(ref_fname)
        out.copy_metadata(ref)

    if history is not None:
        try:
            old_history=out.read_attribute("","history")
            old_history=old_history+"\n"
        except minc2_error: # assume no history available
            old_history=""
            
        new_history=old_history+history
        out.write_attribute("","history",new_history)

    out.setup_standard_order()
    if isinstance(data, np.ndarray):
        out.save_complete_volume(np.ascontiguousarray(data))
    else:
        out.save_complete_volume_tensor(data)
    out.close()


"""
    WIP: load a nonlinear only transform
"""
def load_nl_xfm(fn):
    x=minc2_xfm(fn)
    if x.get_n_concat()==1 and x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR:
        assert(False)
    else:
        _identity=np.asmatrix(np.identity(4))
        _eps=1e-6
        if x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR and x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            assert(np.linalg.norm(_identity-np.asmatrix(x.get_linear_transform(0)) )<=_eps)
            grid_file, grid_invert=x.get_grid_transform(1)
        elif x.get_n_type(0)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            # TODO: if grid have to be inverted!
            grid_file, grid_invert =x.get_grid_transform(0)
        else:
            # probably unsupported type
            assert(False)

        # load grid file into 4D memory
        grid, v2w = load_minc_volume(grid_file, as_byte=False)
        return grid, v2w, grid_invert

"""
    WIP: load a linear only transform
"""
def load_lin_xfm(fn):
    _identity=np.asmatrix(np.identity(4))
    _eps=1e-6
    x=minc2_xfm(fn)

    if x.get_n_concat()==1 and x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR:
        # this is a linear matrix
        lin_xfm=np.asmatrix(x.get_linear_transform())
        return lin_xfm
    else:
        if x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR and x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            # is this identity matrix
            assert(np.linalg.norm(_identity-np.asmatrix(x.get_linear_transform(0)) )<=_eps)
            # TODO: if grid have to be inverted!
            grid_file, grid_invert=x.get_grid_transform(1)
        elif x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            # TODO: if grid have to be inverted!
            (grid_file, grid_invert)=x.get_grid_transform(0)
        assert(False) # TODO
        return None


"""
    Resample volume to different sampling
"""
def resample_volume(in_data, in_v2w, out_shape, out_v2w, order=1, fill=0.0):
    import scipy
    
    # voxel storage matrix
    xyz_to_zyx = np.array([[0,0,1,0],
                        [0,1,0,0],
                        [1,0,0,0],
                        [0,0,0,1]])
    
    # have to account for the shift of the voxel center

    full_xfm = xyz_to_zyx @ np.linalg.inv(in_v2w) @ out_v2w @ xyz_to_zyx

    new_data = scipy.ndimage.affine_transform(in_data, full_xfm, output_shape=out_shape, order=order, mode='constant',cval=fill)
    
    return new_data, out_v2w


"""
    Resample volume to the uniform sampling, if needed
"""
def uniformize_volume(data,v2w,tolerance=0.1,order=1,step=1.0):

    start, step_, dir_cos = decompose(v2w)

    # check if we need to resample
    if np.any(np.abs(step_ - step) > tolerance):
      
        # voxel storage matrix
        xyz_to_zyx = np.array([[0,0,1,0],
                            [0,1,0,0],
                            [1,0,0,0],
                            [0,0,0,1]])
        # need to account for the different order of dimensions
        new_shape = np.ceil(np.array(data.shape) * step_[[2,1,0]]).astype(int)
        # have to account for the shift of the voxel center
        new_start = start - step_*0.5 + np.ones(3)*step*0.5
        new_v2w = compose(new_start, np.ones(3)*step, dir_cos)

        return resample_volume(data,v2w,new_shape,new_v2w,order=order,fill=0.0)
    else:
        return data, v2w




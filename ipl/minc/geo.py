import numpy as np
import math as m

"""
    decompose affine matrix into start, step and direction cosines
"""
def decompose(aff):
    (u,s,vh) = np.linalg.svd(aff[0:3,0:3])
    # remove scaling
    dir_cos = u @ vh
    step  = np.diag( np.linalg.inv(dir_cos) @ aff[0:3,0:3])
    start = np.squeeze(np.asarray(np.linalg.inv(dir_cos) @ aff[0:3,3]))
    return start, step, dir_cos


"""
create affine matrix from start, step and direction cosines
"""
def compose(start, step, dir_cos):
    aff=np.eye(4)

    aff[0:3,0:3] = dir_cos @ np.diag(step)
    aff[0:3,3]   = dir_cos @ start

    return aff

"""
    create voxel to pytorch matrix
"""
def create_v2p_matrix(shape):
    v2p = np.diag( [2/shape[2],   2/shape[1],   2/shape[0], 1])
    v2p[0:3,3] = (  1/shape[2]-1, 1/shape[1]-1, 1/shape[0]-1  ) # adjust for half a voxel shift
    return v2p

"""
    create rotation matrix
"""
def create_rotation_matrix(rot):
    affine_x = np.eye(4)
    # rotate around x
    sin_, cos_ = m.sin(rot[0]), m.cos(rot[0])
    affine_x[1, 1], affine_x[1, 2] = cos_, -sin_
    affine_x[2, 1], affine_x[2, 2] = sin_, cos_
    # rotate around y
    affine_y = np.eye(4)
    sin_, cos_ = m.sin(rot[1]), m.cos(rot[1])
    affine_y[0, 0], affine_y[0, 2] = cos_, sin_
    affine_y[2, 0], affine_y[2, 2] = -sin_, cos_
    # rotate around z
    sin_, cos_ = m.sin(rot[2]), m.cos(rot[2])
    affine_z = np.eye(4)
    affine_z[0, 0], affine_z[0, 1] = cos_, -sin_
    affine_z[1, 0], affine_z[1, 1] = sin_, cos_
    return  affine_x @ affine_y @ affine_z

"""
    create scale matrix
"""
def create_scale_matrix(scale):
    return np.diag( [*scale, 1.0])

"""
    create translation matrix
"""
def create_translation_matrix(shift):
    affine=np.eye(4)
    affine[0:3,3]=shift
    return affine

"""
    create shear matrix
"""
def create_shear_matrix(shear):
    affine = np.eye(4)
    affine[0, 1], affine[0, 2] = shear[0], shear[1]
    affine[1, 0], affine[1, 2] = shear[2], shear[3]
    affine[2, 0], affine[2, 1] = shear[4], shear[5]
    return affine

"""
    create full transform matrix from parameters
"""
def create_transform(rot, scale, shift, shear):
    Mrot=create_rotation_matrix(rot)
    Mscale=create_scale_matrix(scale)
    Mshear=create_shear_matrix(shear)
    Mtrans=create_translation_matrix(shift)

    return Mtrans @ Mshear @ Mscale @ Mrot

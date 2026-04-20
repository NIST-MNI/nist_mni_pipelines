#! /usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import nibabel as nib
    have_nibabel=True
except ImportError:
    # nibabel not available :(
    have_nibabel=False



""" 
    Load minc volume into numpy volume and return voxel2world matrix too
"""
def load_nifti_volume_np(fname, as_byte=False, dtype=None):

    assert(have_nibabel) # nibabel not available

    x = nib.load(fname)
    volume = x.get_fdata().squeeze().transpose([2,1,0]).copy()
    aff = x.affine

    header = x.header ### not used

    if as_byte:
        dtype='uint8'
    elif dtype is None:
        dtype='float64'

    return volume.astype(dtype), aff


def save_nifti_volume(fn, data, aff, ref_fname=None, history=None):
    assert(have_nibabel) # nibabel not available

    #if ref_fname is not None:
    #    x = nib.load(ref_fname)
    #    header = x.header
    #else:
    #    header = None
    header = None

    out = nib.Nifti1Image(data.transpose([2,1,0]).copy(), aff, header=header)
    if history is not None:
        out.header['descrip'] = history

    nib.save(out, fn)


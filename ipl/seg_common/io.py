#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
I/O utilities for 3D MRI brain segmentation inference.
Dispatches between MINC and NIfTI formats.
"""

from ..minc.io import load_minc_volume_np, save_minc_volume, format_history
from ..nifti.io import load_nifti_volume_np, save_nifti_volume


def load_volume_np(fname, dtype=None, as_byte=False):
    """
    Load volume from MINC or NIfTI file.
    
    Args:
        fname: Path to volume file (.mnc or .nii.gz)
        dtype: Data type for numpy array
        as_byte: Load as uint8 (for labels/masks)
    
    Returns:
        tuple: (volume_array, affine_matrix)
    """
    if fname.endswith('.mnc'):
        return load_minc_volume_np(fname, as_byte=as_byte, dtype=dtype)
    elif fname.endswith('.nii.gz'):
        return load_nifti_volume_np(fname, dtype=dtype)
    else:
        raise ValueError(f"Unsupported file format: {fname}")


def save_volume(fname, data, aff, ref_fname=None, history=None):
    """
    Save volume to MINC or NIfTI file.
    
    Args:
        fname: Output path (.mnc or .nii.gz)
        data: numpy array with volume data
        aff: 4x4 affine matrix
        ref_fname: Optional reference file for metadata
        history: Optional history string to embed
    """
    if fname.endswith('.mnc'):
        save_minc_volume(fname, data, aff, ref_fname=ref_fname, history=history)
    elif fname.endswith('.nii.gz'):
        save_nifti_volume(fname, data, aff, ref_fname=ref_fname, history=history)
    else:
        raise ValueError(f"Unsupported file format: {fname}")

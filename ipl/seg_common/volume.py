#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Volume preparation utilities for 3D MRI brain segmentation inference.
All functions use NumPy only (no PyTorch dependency).
"""

import numpy as np


def autonorm_np(arr):
    """
    Normalize array using quantile-based scaling (0-1 range).
    
    Args:
        arr: Input numpy array
    
    Returns:
        Normalized array with values in [0, 1]
    """
    arr = arr - np.min(arr)
    p99 = np.percentile(arr, 99)
    if p99 > 0:
        arr = np.clip(arr / p99, 0.0, 1.0)
    return arr


def maxnorm_np(arr):
    """
    Normalize array using maximum value (0-1 range).
    
    Args:
        arr: Input numpy array
    
    Returns:
        Normalized array with values in [0, 1]
    """
    max_val = np.max(arr)
    if max_val > 0:
        arr = arr / max_val
    return arr


def mean_std_normalize_np(arr):
    """
    Normalize array using mean and standard deviation.
    
    Args:
        arr: Input numpy array
    
    Returns:
        Normalized array (subtract mean, divide by std)
    """
    mean = np.mean(arr)
    std = np.std(arr)
    if std > 0:
        arr = (arr - mean) / std
    return arr


def apply_cropvol(dset, cropvol):
    """
    Crop volume by removing border voxels.
    
    Args:
        dset: Input numpy array (B, C, X, Y, Z)
        cropvol: Number of voxels to crop from each border
    
    Returns:
        tuple: (cropped_dset, orig_size) where orig_size is the original shape
    """
    orig_size = dset.shape
    cropped = dset[:, :, cropvol: orig_size[2]-cropvol,
                        cropvol: orig_size[3]-cropvol,
                        cropvol: orig_size[4]-cropvol]
    return cropped, orig_size


def apply_padvol(dset, padvol, padfill=0.0):
    """
    Pad volume with constant border.
    
    Args:
        dset: Input numpy array (B, C, X, Y, Z)
        padvol: Number of voxels to pad on each border
        padfill: Value to fill padding with
    
    Returns:
        tuple: (padded_dset, orig_size) where orig_size is the original shape
    """
    orig_size = dset.shape
    padded = np.pad(dset, pad_width=((0,0), (0,0), 
                                    (padvol, padvol), 
                                    (padvol, padvol), 
                                    (padvol, padvol)), 
                    mode='constant', constant_values=padfill)
    return padded, orig_size


def undo_cropvol(dset_out, orig_size, cropvol, bck=0):
    """
    Restore cropped volume to original size (with background fill).
    
    Args:
        dset_out: Cropped output array
        orig_size: Original size before cropping
        cropvol: Crop amount used
        bck: Background value to fill
    
    Returns:
        Restored array at original size
    """
    restored = np.full(orig_size, bck, dtype=dset_out.dtype)
    restored[:, :, cropvol: orig_size[2]-cropvol,
                   cropvol: orig_size[3]-cropvol,
                   cropvol: orig_size[4]-cropvol] = dset_out
    return restored


def undo_padvol(dset_out, orig_size, padvol):
    """
    Remove padding from volume.
    
    Args:
        dset_out: Padded output array
        orig_size: Original size before padding
        padvol: Padding amount used
    
    Returns:
        Array with padding removed
    """
    return dset_out[:, :, padvol: orig_size[2]+padvol, 
                      padvol: orig_size[3]+padvol, 
                      padvol: orig_size[4]+padvol]


def parse_bracket_input(spec):
    """
    Parse bracket-format input spec [a,b,...] into list of filenames/floats.
    
    Args:
        spec: String like "[a,b,c]" where a,b,c are either filenames or constant numbers
    
    Returns:
        tuple: (inputs_list, ref_file) where inputs_list contains filenames/floats
    """
    import re
    m = re.match(r"\[(.*)\]", spec)
    if m is None:
        return None, None
    
    inp = m[1].split(",")
    inputs = []
    ref_file = None
    
    for i in inp:
        q = re.match(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", i)
        if q is not None:
            inputs.append(float(q[0]))
        else:
            inputs.append(i)
            if ref_file is None:
                ref_file = i
    
    return inputs, ref_file

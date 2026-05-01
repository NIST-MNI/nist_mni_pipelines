#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-processing utilities for 3D MRI brain segmentation inference.
All functions use NumPy/SciPy only (no PyTorch dependency).
"""

import csv
import json
import numpy as np


def find_largest_component(input):
    """
    Find the largest connected component in a segmentation.
    
    Args:
        input: Binary or labeled numpy array
    
    Returns:
        Boolean mask of the largest component
    """
    from scipy.ndimage import label
    structure = np.ones((3, 3, 3), dtype=np.int32)
    labeled, ncomponents = label(input, structure)
    if ncomponents == 0:
        return np.zeros_like(input, dtype=bool)
    largest = np.argmax([np.sum(labeled==i) for i in range(1, ncomponents+1)]) + 1
    return labeled == largest


def measure_volumes(seg, aff, labels_desc, out_seg_f=None, in_scan=None, load_output=False,scale=1.0):
    """
    Measure volumes of labels in segmentation.
    
    Args:
        seg: Segmentation array (can be None if loading from file)
        aff: Affine matrix (4x4)
        labels_desc: dict with label descriptions, or filename, or list
        out_seg_f: Output segmentation file path (for loading if seg is None)
        in_scan: Input scan identifier
        load_output: If True and seg is None, load segmentation from out_seg_f
    
    Returns:
        dict: Volume measurements per label
    """
    from .io import load_volume_np
    
    if isinstance(labels_desc, dict):
        labels = {int(i): j for i, j in labels_desc.items()}
    elif isinstance(labels_desc, str):
        with open(labels_desc, 'r') as f:
            labels = json.load(f)
        labels = {int(i): j for i, j in labels.items()}
    elif isinstance(labels_desc, list):
        labels = {i+1: j for i, j in enumerate(labels_desc)}
    else:
        raise ValueError("labels_desc must be dict, filename or list")
    
    if load_output and seg is None and out_seg_f is not None:
        if out_seg_f.endswith('.mnc'):
            seg, aff = load_volume_np(out_seg_f, dtype='int16')
        elif out_seg_f.endswith('.nii.gz'):
            seg, aff = load_volume_np(out_seg_f, dtype='int16')
        else:
            raise ValueError(f"Unsupported file format: {out_seg_f}")
    
    if seg is None:
        results = {'scan': in_scan, 'segmentation': out_seg_f}
        for il, l in labels.items():
            results[l] = float('NaN')
    else:
        voxel_volume = np.abs(np.linalg.det(aff[:3, :3]))
        results = {'scan': in_scan, 'segmentation': out_seg_f}
        for il, l in labels.items():
            count = np.sum(seg == il)
            volume = count * voxel_volume * scale
            results[l] = volume
    
    return results


def save_measurements(measure, all_measurements):
    """
    Save all measurements to CSV file.
    
    Args:
        measure: Output file path
        all_measurements: List of measurement dicts
    """
    with open(measure, 'w') as f:
        writers = csv.DictWriter(f, fieldnames=sorted(all_measurements[0].keys()))
        writers.writeheader()
        for m in all_measurements:
            writers.writerow(m)

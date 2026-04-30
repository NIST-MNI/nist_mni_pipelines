#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 29/01/2018

import argparse
import re
from time import gmtime, strftime
import sys
import math
import json
import os
import traceback

import numpy as np

# MINC IO
from minc2_simple import minc2_file

# seg_common utilities
from .seg_common.io import load_volume_np, save_volume, format_history
from .seg_common.volume import (autonorm_np, maxnorm_np, mean_std_normalize_np,
                                apply_cropvol, apply_padvol, undo_cropvol, undo_padvol,
                                parse_bracket_input)
from .seg_common.postprocess import find_largest_component, measure_volumes, save_measurements

# geo utilities (not moved to seg_common)
from .minc.geo import decompose, compose
from .minc.io import resample_volume, uniformize_volume

import onnxruntime
#from onnxruntime import ONNXRuntimeError
from onnx import numpy_helper

from scipy.special import softmax, log_softmax


def parse_options():
    parser = argparse.ArgumentParser(description='Apply pre-trained model using ONNX runtime',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config", 
                        type=str, 
                        help="Path to JSON configuration file")
    
    parser.add_argument("--model", 
                        type=str, 
                        nargs='+',
                        help="pretrained model(s) (ONNX), can specify multiple for majority voting. If not specified, will use models from config file.")

    parser.add_argument("--model_prefix", 
                        type=str, 
                        help="Add prefix to model names from config file")

    parser.add_argument("input", type=str, nargs='?',
                        help="Input minc file, or input spec in [a,b,...] where a,b is ether const number of file name")
    
    parser.add_argument("output", type=str, nargs='?',
                        help="Output minc file")

    parser.add_argument("--bi", type=str, nargs='+',
                        help="Batch inputs")

    parser.add_argument("--bo", type=str, nargs='+',
                        help="Batch outputs")

    parser.add_argument("--li", type=str, 
                        help="Batch inputs in file")

    parser.add_argument("--lo", type=str, 
                        help="Batch outputs in file")
    
    parser.add_argument("--minibatch_size", type=int, default=1,
                        help="Batch size for processing multiple inputs at once")
    
    parser.add_argument('--progress', action="store_true",
                        default=False,
                        help='Show progress bar' )

    parser.add_argument("--add", 
                        type=str,
                        nargs='+',
                        help="Input minc file")

    parser.add_argument("--patch_sz",
                        nargs='+',
                        type=int, default=[64, 64, 64],
                        help="Patch size")
    
    parser.add_argument("--quant", type=int, default=64,
                        help="Spatial quantization factor for whole volume processing")
                        
    parser.add_argument("--stride", type=int, default=None,
                        nargs='+',
                        help="Stride, default patch_sz-crop*2")

    parser.add_argument("--channels", type=int, default=1,
                        help="add more input channels, fill them with 38.81240207 for now")

    parser.add_argument("--crop", type=int, default=0,
                        help="Crop edges of patch (segment with overlapping patches)")

    parser.add_argument("--cropvol", type=int, default=0,
                        help="Crop edges of the input whole volume before applying the model")

    parser.add_argument("--padvol", type=int, default=0,
                        help="pad the input volume before applying the model")

    parser.add_argument("--padfill", type=float, default=0,
                        help="pad with this value")

    parser.add_argument("--mask", type=str,
                        help="Apply mask to result")

    parser.add_argument("--bck", type=int, default=0,
                        help="Background label")

    parser.add_argument('--cpu', action="store_true",
                        dest="cpu",
                        default=False,
                        help='Do everything in cpu' )
    
    parser.add_argument("--device_id", type=int, default=0,
                        help="Devide ID")

    parser.add_argument('-q','--quiet', action="store_true",
                        default=False,
                        help='Suppress warnings' )

    parser.add_argument('-F','--fuzzy',
                        help='Output fuzzy volume(s)' )

    parser.add_argument('-T','--threads',type=int, default=0,
                        help='Number of threads to use' )

    parser.add_argument('-n','--n_classes',type=int, default=0,
                        help='Number of segmentation classes, needed for overlap only' )
    
    parser.add_argument('-u','--use_classes',type=int, default=None,
                        help='Use only these classes (for models that produce something else in additional channels)' )
    
    parser.add_argument('-U','--uniformize',type=float,
                        help='Uniformize image resolution befor applying CNN' )

    parser.add_argument('-R','--reference',type=str,
                        help='Resample input as reference, usefull for small ROIs in stx space' )

    parser.add_argument('-S','--saveuniform', action="store_true",
                        default=False,
                        help='Output segmentation at uniform resolution or reference space' )
    
    parser.add_argument('--measure',
                        default=None,
                        help='Perform volumetric measurements, store in file' )
    
    parser.add_argument('--recover', action="store_true",
                        default=False,
                        help='Recover from incomplete batch mode' )
    
    parser.add_argument('--crash', action="store_true",
                        default=False,
                        help='Crash on error' )
    
    parser.add_argument('--distance', action="store_true",
                        default=False,
                        help='Uses Distance model' )

    parser.add_argument('--whole', action="store_true",
                        default=False,
                        help='Apply model to the whole image, without overlapping patches' )
    
    parser.add_argument('--trim',default=False, action='store_true',help='Trim instead of expand')

    parser.add_argument("--nibabel", action="store_true", default=False,
                        help="Use nibabel coordinate conversion in model")
    
    parser.add_argument('--freesurfer', action="store_true",
                        default=False,
                        help='Apply model using freesurfer coordinate convention' )
    
    parser.add_argument('--normalize', action="store_true",
                        default=False,
                        help='Apply intensity normalization between 0 and 1 using quantiles' )

    parser.add_argument('--max_normalize', action="store_true",
                        default=False,
                        help='Apply intensity normalization between 0 and 1 using maximum' )
    
    parser.add_argument('--mean_std_normalize', action="store_true",
                        default=False,
                        help='Subtract mean and devide by std, for nonzero voxels' )
    
    parser.add_argument('--largest', action="store_true",
                        default=False,
                        help='Apply largest component filtering' )

    parser.add_argument('--use_tf32', action="store_true",
                        default=False,
                        help='Use TF32 precision in CUDA execution provider' )

    parser.add_argument('--use_gaussian_weights', action="store_true",
                        default=False,
                        help='Use Gaussian weights for overlapping regions in sliding window inference' )

    parser.add_argument('--continuous', action="store_true",
                        default=False,
                        help='Model produces continuous single channel output' )

    parser.add_argument('--majority', action="store_true",
                        default=False,
                        help='Majority voting for multiple models' )
    
    parser.add_argument('--channel_last', action="store_true",
                        default=False,
                        help='Use channel last format for input and output' )

    params = parser.parse_args()
    
    # Handle stride default value
    if params.stride is None:
        if isinstance(params.patch_sz, list):
            params.stride = min(params.patch_sz[0]-params.crop*2,params.patch_sz[1]-params.crop*2,params.patch_sz[2]-params.crop*2)
        else:
            params.stride = params.patch_sz-params.crop*2

    return params

# def log_softmax(x, axis=1):
#     e_x = np.exp(x - np.max(x,axis=axis))
#     return np.log(e_x / e_x.sum(axis=axis))


# def softmax(x,axis=1):
#     e_x = np.exp(x - np.max(x,axis=axis))
#     return e_x / e_x.sum(axis=axis)



def segment_whole(
    dataset, model, 
    quant_size=64,
    normalize=False,
    normalize_max=False,
    normalize_mean_std=False,
    freesurfer=False,
    nibabel=False,
    largest_component=False,
    dist=False,
    continuous=False,
    trim=False,
    use_classes=None,
    channel_last=False
    ):
    """
    Apply model to dataset of arbitrary size 
    Args:
        dataset: Input data array
        model: ONNX model
        crop: Number of voxels to crop from patch edges
        n_classes: Number of output classes
        freesurfer: Whether to use FreeSurfer coordinate convention
        nibabel: Whether to use NIBabel coordinate convention
        normalize: Whether to normalize using quantiles
        normalize_max: Whether to normalize using max value
        normalize_mean_std: Whether to normalize using mean and std
        dist: Whether to use distance-based segmentation
        continuous: model should produce continuous output
        trim: Whether to trim the dataset to a multiple of quant_size
        use_classes: Use only these classes (for models that produce something else in additional channels)
        channel_last: Whether to use channel last format for input and output
    Returns:
        output_fuzzy: Fuzzy output 
    """

    out_name = "seg" if not dist else "dist"

    target_shape = np.ceil(np.array(dataset.shape[2:]) / quant_size).astype(int) * quant_size

    # normalize before padding with zeros
    if normalize:
        dataset_norm = dataset-dataset.min()
        dataset_norm = np.clip(dataset_norm / np.percentile(dataset_norm,99),0.0, 1.0)
    elif normalize_max:
        dataset_norm = dataset / np.max(dataset)
    else:
        out_name = "seg" 

    if np.any(target_shape != dataset_norm.shape[2:]):
        conformed = np.zeros( (1,1, *target_shape), dtype='float32')
        conformed[:,:, :dataset_norm.shape[2], :dataset_norm.shape[3], :dataset_norm.shape[4]] = dataset_norm   
    else:
        conformed = dataset_norm.astype('float32') # to be compatible with spatial expectation of the model

    if freesurfer:
        conformed=np.ascontiguousarray(conformed.transpose([0,1,4,3,2])[:,:,:,::-1,:]).copy()
    elif nibabel:
        conformed=np.ascontiguousarray(conformed.transpose([0,1,4,3,2])).copy()

    # MONAI-style normalization
    if normalize:
        # Quantile-based normalization (0-1 range)
        conformed = conformed - conformed.min()
        conformed = np.clip(conformed / np.percentile(conformed,99), 0.0, 1.0)
    elif normalize_max:
        # Max normalization (0-1 range)
        conformed = np.clip(conformed / np.max(conformed), 0.0, 1.0)
    elif normalize_mean_std:
        # Mean-std normalization for nonzero voxels
        mean = np.mean(conformed[conformed>0])
        std = np.std(conformed[conformed>0])
        conformed = (conformed - mean) / std

    # Run inference
    if channel_last:
        conformed = np.ascontiguousarray(conformed.transpose([0, 2, 3, 4, 1]))

    out = model.run([out_name],{'scan':conformed})[0]

    if channel_last:
        out = out.transpose([0, 4, 1, 2, 3])

    if use_classes is not None:
        out = out[:, 0:use_classes, :, :, :]

    if freesurfer:
        out=np.ascontiguousarray(out[:,:,:,::-1,:].transpose([0,1,4,3,2]))
    elif nibabel:
        out=np.ascontiguousarray(out.transpose([0,1,4,3,2]))
    
    print(f"{dataset.shape=} {out.shape=}")

    if np.any(target_shape != dataset_norm.shape[2:]):
        out=out[:,:,:dataset_norm.shape[2], :dataset_norm.shape[3], :dataset_norm.shape[4]]

    if out.shape[1]==1 : # single channel distance 
        output_fuzzy = out
        output_seg = (out < dist_border)
    elif dist:
        output_fuzzy = out
        output_seg = np.expand_dims( np.argmin(output_fuzzy, axis=1).astype(np.uint8), axis=1)
    else:
        output_fuzzy = log_softmax(out, axis=1)
        output_seg = np.expand_dims( np.argmax(output_fuzzy, axis=1).astype(np.uint8), axis=1)

    if largest_component:
        from scipy.ndimage import label
        # find largest CC 
        structure = np.ones((3, 3, 3), dtype=np.int32)
        if out.shape[1]==1 or out.shape[1]==2:
            # To be compatible
            output_seg= np.expand_dims(np.expand_dims(
                find_largest_component(output_seg.squeeze()),axis=0),axis=0) 
        else:
            out=out[:,:,:dataset.shape[2], :dataset.shape[3], :dataset.shape[4]]

    return out

def get_gaussian_weights(patch_size, sigma_scale=1.0/8):
    """
    Generate Gaussian weights for overlapping regions in sliding window inference.
    Args:
        patch_size: Size of the patch
        sigma_scale: Scale factor for sigma
        device: Device to place the weights on
    Returns:
        Gaussian weights tensor
    """
    if not isinstance(patch_size, (list, tuple)):
        patch_size = [patch_size] * 3
    
    sigma = [patch_size[i] * sigma_scale for i in range(3)]
    coords = [np.arange(patch_size[i]) for i in range(3)]
    mesh = np.meshgrid(*coords, indexing='ij')
    
    # Calculate distances from center
    center = [(patch_size[i] - 1) / 2 for i in range(3)]
    dist =sum(((mesh[i] - center[i]) / sigma[i]) ** 2 for i in range(3))
    
    # Calculate Gaussian weights
    weights = np.exp(-0.5 * dist)
    weights = weights / np.max(weights)
    # handle non-positive weights
    min_non_zero = max(np.min(weights), 1e-3)
    weights = np.clip(weights, min=min_non_zero)

    # Convert to tensor and add batch and channel dimensions
    weights = np.expand_dims(np.expand_dims(weights, 0), 0)

    return weights.astype(np.float32)

def segment_with_patches_overlap(
        dataset, model, 
        crop=0,
        patch_sz = None, 
        stride = None,
        n_classes=2,
        use_classes=None,
        bck = 0, 
        freesurfer=False,
        nibabel=False,
        normalize=False,
        normalize_max=False,
        normalize_mean_std=False,
        dist=False,
        use_gaussian_weights=False,
        continuous=False,
        orig_aff=None,
        channel_last=False):
    """
    Apply model to dataset of arbitrary size using sliding window inference
    Args:
        dataset: Input data array
        model: ONNX model
        crop: Number of voxels to crop from patch edges
        patch_sz: Size of patches to process
        stride: Step size between patches
        n_classes: Number of output classes
        bck: Background value
        out_fuzzy: Whether to output fuzzy results
        freesurfer: Whether to use FreeSurfer coordinate convention
        nibabel: Whether to use NIBabel coordinate convention
        normalize: Whether to normalize using quantiles
        normalize_max: Whether to normalize using max value
        normalize_mean_std: Whether to normalize using mean and std
        dist: Whether to use distance-based segmentation
        use_gaussian_weights: Whether to use Gaussian weights for overlapping regions
    """
    if continuous:
        out_name = "scan_out"
    elif dist:
        out_name = "dist"
    else:
        out_name = "seg" 

    out_classes = 1 if continuous or (dist and n_classes is None) else n_classes

    if not isinstance(patch_sz, list):
        patch_sz = [patch_sz, patch_sz, patch_sz]

    if not isinstance(stride, list):
        stride = [stride, stride, stride]

    if freesurfer:
        dataset=np.ascontiguousarray(dataset.transpose([0,1,4,3,2])[:,:,:,::-1,:]).copy()
    elif nibabel:
        dataset=np.ascontiguousarray(dataset.transpose([0,1,4,3,2])).copy()

    # MONAI-style normalization
    if normalize:
        dataset = dataset - dataset.min()
        dataset = np.clip(dataset / np.percentile(dataset,99), 0.0, 1.0)
    elif normalize_max:
        dataset = np.clip(dataset / np.max(dataset), min=0.0, max=1.0)
    elif normalize_mean_std:
        mean = np.mean(dataset[dataset>0])
        std = np.std(dataset[dataset>0])
        dataset = (dataset - mean) / std

    dsize = dataset.shape
    output_size = list(dsize)
    output_size[1] = 1
    output_size_fuzzy = list(dsize)
    output_size_fuzzy[1] = out_classes

    patch_sz_ = [patch_sz[0] - crop*2, patch_sz[1] - crop*2, patch_sz[2] - crop*2]
    
    out_roi = [dsize[2]-crop*2, dsize[3]-crop*2, dsize[4]-crop*2 ]

    patch_sz_ = [patch_sz[0] - crop*2, patch_sz[1] - crop*2, patch_sz[2] - crop*2]
    out_roi = [dsize[2]-crop*2, dsize[3]-crop*2, dsize[4]-crop*2]

    # Generate Gaussian weights if requested
    if use_gaussian_weights:
        gaussian_weights = get_gaussian_weights(patch_sz_, sigma_scale=0.25)
    else:
        gaussian_weights = np.ones((1, 1, *patch_sz_), dtype=np.float32)

    # Sliding window inference
    for k in range(math.ceil(out_roi[0]/stride[0])):
        for l in range(math.ceil(out_roi[1]/stride[1])):
            for m in range(math.ceil(out_roi[2]/stride[2])):
                c = [k*stride[0] + crop, l*stride[1] + crop, m*stride[2] + crop]

                for i in range(3):
                    c[i] = max(min( c[i], dsize[i+2] - patch_sz[i] + crop ), crop)

                # Extract patch
                in_data = np.ascontiguousarray(
                    dataset[:, :, c[0]-crop: c[0]-crop+patch_sz[0],
                           c[1]-crop: c[1]-crop+patch_sz[1],
                           c[2]-crop: c[2]-crop+patch_sz[2]]
                )

                if channel_last:
                    in_data = np.ascontiguousarray(in_data.transpose([0, 2, 3, 4, 1]))

                # Run inference
                out = model.run([out_name],{'scan':in_data})[0]

                if channel_last:
                    out = out.transpose([0, 4, 1, 2, 3])

                if continuous:
                    patch_output = out
                elif dist:
                    patch_output = out
                elif use_classes is not None:
                    patch_output = out[:, 0:use_classes, :, :, :]
                else:
                    patch_output = out

                # Apply Gaussian weights to the patch output
                weighted_output = patch_output[:, :, 
                                             crop: crop+patch_sz_[0], 
                                             crop: crop+patch_sz_[1], 
                                             crop: crop+patch_sz_[2]] * gaussian_weights

                # Accumulate results
                output_fuzzy [:, :, c[0]: c[0]+patch_sz_[0], c[1]: c[1]+patch_sz_[1], c[2]: c[2]+patch_sz_[2]] += weighted_output
                output_weight[:, :, c[0]: c[0]+patch_sz_[0], c[1]: c[1]+patch_sz_[1], c[2]: c[2]+patch_sz_[2]] += gaussian_weights

    # Normalize accumulated results
    invalid = output_weight < 1e-3
    output_weight[invalid] = 1.0
    output_fuzzy =  output_fuzzy/output_weight

    # Handle invalid regions
    if not continuous:
        for q in range(output_fuzzy.shape[1]):
            output_fuzzy[:,(q):(q+1),:,:,:][invalid] = 0.0
        output_fuzzy[:,bck:bck+1,:,:,:][invalid] = 1.0
     
    if freesurfer:
        output_fuzzy = output_fuzzy[:,:,:,::-1,:].transpose([0,1,4,3,2])

    elif nibabel:
        output_fuzzy=output_fuzzy.transpose([0,1,4,3,2]).copy()
    return output_fuzzy 

"""
High level function to apply segmentation
"""
def segment_with_onnx(in_scans, out_seg, settings,
    cpu=True, threads=0,
    history=None,device_id=None,use_tf32=False,
    measure=None):
    """
    High level function to apply segmentation
    """
    # Extract parameters from settings dictionary
    n_classes = settings.get('n_classes', 2)
    use_classes = settings.get('use_classes', None)
    mask = settings.get('mask', None)
    models = settings.get('models', None)
    patch_sz = settings.get('patch_sz', 64)
    stride = settings.get('stride', 32)
    crop = settings.get('crop', 0)
    cropvol = settings.get('cropvol', 0)
    padvol = settings.get('padvol', 0)
    padfill = settings.get('padfill', 0.0)
    bck = settings.get('bck', 0)
    history = settings.get('history', None)
    fuzzy = settings.get('fuzzy', None)
    whole = settings.get('whole', False)
    quant_size = settings.get('quant_size', 64)
    freesurfer = settings.get('freesurfer', False)
    nibabel = settings.get('nibabel', False)
    normalize = settings.get('normalize', False)
    normalize_max = settings.get('normalize_max', False)
    normalize_mean_std = settings.get('normalize_mean_std', False)
    largest = settings.get('largest', False)
    dist = settings.get('dist', False)
    uniformize = settings.get('uniformize', None)
    reference = settings.get('reference', None)
    use_gaussian_weights = settings.get('use_gaussian_weights', False)
    continuous = settings.get('continuous', False)
    trim = settings.get('trim', False)
    channel_last = settings.get('channel_last', False)
    majority = settings.get('majority', False)
    save_uniformized = settings.get('save_uniformized', False)
    labels_desc = settings.get('labels_desc', None)

    inputs=[]
    # load all inputs
    # TODO: deal with floating point values
    orig_aff = None
    orig_shape = None

    if reference is not None:
        if reference .endswith('.mnc'):
            ref_data, ref_aff = load_volume_np(reference, dtype='uint8', as_byte=True)
        else:
            ref_data, ref_aff = load_volume_np(reference, dtype='uint8', as_byte=True)
    else:
        ref_data = None
        ref_aff = None
    
    for i in in_scans:
        if isinstance(i, float):
            data = np.full(orig_shape, i, dtype='float32')
            aff = None
        else:
            ref_file = i
            data, aff = load_minc_volume_np(i, dtype='float32')

        # make sure all files have the same shape and orientation
        if orig_shape is not None:
            assert(np.all(orig_shape == np.array(data.shape)))
        else:
            orig_shape = np.array(data.shape)
        
        if orig_aff is not None and aff is not None:
            assert(np.all(np.abs(orig_aff - aff) < 1e-3))
        elif aff is not None:
            orig_aff = aff

        if uniformize is not None and aff is not None:
            data, new_aff = uniformize_volume(data, aff, step=uniformize)

        inputs+=[ np.expand_dims(data, axis=(0, 1))]
    # 
    dset = np.concatenate(inputs, axis=1)

    sess_options = onnxruntime.SessionOptions()
    if threads>0:
        sess_options.intra_op_num_threads = threads

    cuda_opts={"use_tf32": 1 if use_tf32 else 0}
    if device_id is not None:
        cuda_opts["device_id"] = device_id
        
    if cpu:
        providers=['CPUExecutionProvider']
    else:
        # Configure CUDA execution provider with TF32 precision control
        providers=[("CUDAExecutionProvider", cuda_opts)]

    # Handle multiple models
    if not isinstance(models, list):
        models = [models]

    if whole:
        patch_sz = np.clip(np.ceil((np.array(dset.shape[2:]) - cropvol*2 + padvol*2) / quant_size).astype(int) * quant_size, quant_size*2, quant_size*5).tolist()
        stride = patch_sz
    elif not isinstance(patch_sz, list):
        patch_sz = [patch_sz, patch_sz, patch_sz]
    
    if cropvol>0:
        orig_size = dset.shape
        orig_fuzzy_size = dset.shape
        orig_vae_size = dset.shape
        dset = dset[:, :, cropvol: orig_size[2]-cropvol, cropvol: orig_size[3]-cropvol, cropvol: orig_size[4]-cropvol]
    elif padvol>0:
        orig_size = dset.shape
        orig_fuzzy_size = dset.shape
        orig_vae_size = dset.shape
        
        dset = np.ascontiguousarray( np.pad(dset, pad_width=((0,0),(0,0),(padvol,padvol),(padvol,padvol),(padvol,padvol)), 
            mode='constant', constant_values = padfill))
    
    # Apply models and collect results
    all_outputs = []
    all_fuzzy_outputs = []
    dset_out = None
    for m in models:
        model=onnxruntime.InferenceSession(m, sess_options, providers=providers)

        if whole:
            dset_out_fuzzy = segment_whole(
                dset, model,
                freesurfer=freesurfer,
                nibabel=nibabel,
                normalize=normalize,
                normalize_max=normalize_max,
                normalize_mean_std=normalize_mean_std,
                dist=dist,
                continuous=continuous,
                trim=trim,
                use_classes=use_classes,
                channel_last=channel_last)
        else:
            dset_out_fuzzy = segment_with_patches_overlap(
                dset, model,
                n_classes=n_classes,use_classes=use_classes,
                patch_sz=patch_sz, crop=crop,
                bck=bck, stride=stride, 
                freesurfer=freesurfer,
                nibabel=nibabel,
                normalize=normalize,
                normalize_max=normalize_max,
                normalize_mean_std=normalize_mean_std,
                dist=dist,
                use_gaussian_weights=use_gaussian_weights,
                continuous=continuous,
                orig_aff=orig_aff,
                channel_last=channel_last)
        all_fuzzy_outputs.append(dset_out_fuzzy)

    if len(models) > 1:
        if majority and not continuous:
            if dist and all_fuzzy_outputs[0].shape[1]==1:
                stacked_outputs = np.stack([(i<1.0).astype(np.uint8) for i in all_fuzzy_outputs],axis=0)
            elif dist:
                stacked_outputs = np.stack([np.argmin(i, axis=1,keepdims=True).astype(np.uint8) for i in all_fuzzy_outputs],axis=0)
            else:
                stacked_outputs = np.stack([np.argmax(i, axis=1,keepdims=True).astype(np.uint8) for i in all_fuzzy_outputs],axis=0)
            dset_out = np.apply_along_axis(lambda x: np.bincount(x.astype(np.int32)).argmax(), 0, stacked_outputs.squeeze())
            dset_out_fuzzy = np.mean(all_fuzzy_outputs, axis=0)
        else:
            dset_out_fuzzy = np.mean(all_fuzzy_outputs, axis=0)
    else:
        dset_out_fuzzy = all_fuzzy_outputs[0]

        dset_out_[:, :, cropvol: orig_size[2]-cropvol, cropvol: orig_size[3]-cropvol, cropvol: orig_size[4]-cropvol]=\
            dset_out
        dset_out = dset_out_

    if cropvol>0: # unpcrop output
        if fuzzy is not None:
            orig_fuzzy_size=list(orig_fuzzy_size)
            orig_fuzzy_size[1] = dset_out_fuzzy.shape[1]
            dset_out_fuzzy_ = np.zeros(orig_fuzzy_size)
            dset_out_fuzzy_[:, :, cropvol: orig_size[2]-cropvol, cropvol: orig_size[3]-cropvol, cropvol: orig_size[4]-cropvol]=\
                dset_out_fuzzy
            dset_out_fuzzy = dset_out_fuzzy_
    elif padvol>0: # unpad output
        dset_out = dset_out[:, :, padvol: orig_size[2]+padvol, padvol: orig_size[3]+padvol, padvol: orig_size[4]+padvol]
        if fuzzy is not None:
            dset_out_fuzzy = dset_out_fuzzy[:, :, padvol: orig_size[2]+padvol, padvol: orig_size[3]+padvol, padvol: orig_size[4]+padvol]

    if not continuous:
        dset_out = np.ascontiguousarray(dset_out.squeeze(), dtype=np.uint8)
    else:
        dset_out = np.ascontiguousarray(dset_out.squeeze(), dtype=np.float32)

    if not save_uniformized and (uniformize is not None or ref_aff is not None) and np.any(np.array(dset_out.shape) != orig_shape):
        dset_out = resample_volume(dset_out, new_aff, orig_shape, orig_aff, order=0, fill=bck)[0]
        
    if save_uniformized and (uniformize is not None or ref_aff is not None):
        # adjust  for output
        orig_aff = new_aff
        orig_shape = dset_out.shape
    
    if fuzzy is not None:
        for f in range(dset_out_fuzzy.shape[1]):
            dset_out_f=dset_out_fuzzy[0,f,:,:,:]

            if not save_uniformized and (uniformize is not None or ref_aff is not None) and np.any(np.array(dset_out_f.shape) != orig_shape):
                dset_out_f = resample_volume(dset_out_f, new_aff, orig_shape, orig_aff, order=1, fill=bck)[0]
            else:
                dset_out_f = np.ascontiguousarray(dset_out_f)

            save_volume(fuzzy+'_{}.mnc'.format(f), 
                             dset_out_f, orig_aff, ref_fname=ref_file, history=history)

    if mask is not None:
        mask, mask_aff = load_volume_np(mask, dtype='uint8', as_byte=True)
        if np.any(np.array(mask.shape) != orig_shape) or np.any(orig_aff - aff > 1e-3): # resample mask
            mask = resample_volume(mask, mask_aff, orig_shape, orig_aff, order=0, fill=0)[0]

        dset_out[mask<1] = bck

    save_volume(out_seg, dset_out, orig_aff, ref_fname=ref_file, history=history)

    if labels_desc is not None and not continuous and measure is not None:
        # save label measurements as json
        vols=[measure_volumes(dset_out, orig_aff, labels_desc, out_seg_f=out_seg,in_scan=in_scans[0])]
        save_measurements(measure, vols)

    return out_seg



"""
High level function to apply segmentation
"""
def segment_with_onnx_batched(in_scans, out_segs, 
    settings,
    cpu=True, 
    threads=0,
    history=None,
    device_id=None,
    use_tf32=False,
    minibatch_size=1,
    fuzzy_output=False,
    progress=False,
    measure=None,
    recover=False,
    crash=False):
    """
    High level function to apply segmentation
    """
    # Extract parameters from settings dictionary
    n_classes = settings.get('n_classes', 2)
    use_classes = settings.get('use_classes', None)
    mask = settings.get('mask', None)
    models = settings.get('models', None)
    patch_sz = settings.get('patch_sz', 64)
    stride = settings.get('stride', 32)
    crop = settings.get('crop', 0)
    cropvol = settings.get('cropvol', 0)
    padvol = settings.get('padvol', 0)
    padfill = settings.get('padfill', 0.0)
    bck = settings.get('bck', 0)
    whole = settings.get('whole', False)
    quant_size = settings.get('quant_size', 64)
    freesurfer = settings.get('freesurfer', False)
    nibabel = settings.get('nibabel', False)
    normalize = settings.get('normalize', False)
    normalize_max = settings.get('normalize_max', False)
    normalize_mean_std = settings.get('normalize_mean_std', False)
    largest = settings.get('largest', False)
    dist = settings.get('dist', False)
    reference= settings.get('reference', None)
    uniformize = settings.get('uniformize', None)
    use_gaussian_weights = settings.get('use_gaussian_weights', False)
    continuous = settings.get('continuous', False)
    trim = settings.get('trim', False)
    channel_last = settings.get('channel_last', False)
    save_uniformized = settings.get('save_uniformized', False)
    labels_desc = settings.get('labels_desc', None)
    majority = settings.get('majority', False)
    augment_tta = settings.get('augment_tta', None)

    #assert len(models) == 1, "Batched inference is only supported for a single model"  
    assert len(out_segs) == len(in_scans), "Number of output segments must match number of input scans"

    ### for now only flip map augmentation is supported
    if augment_tta is not None:
        assert "flip_x" in augment_tta, "Only flip augmentation is supported for TTA now"
        if not continuous:
            flip_map = augment_tta["flip_x"]
            assert isinstance(flip_map, list),"Flip map should be a list of indexes which are remapped after flipping"
            flip_map = np.array(flip_map,dtype=int)
            assert np.all(np.sort(flip_map) == np.arange(len(flip_map))), "Flip map should be a permutation of [0, 1, ..., n_classes-1]"
            assert len(flip_map) == n_classes, "Flip map length should match number of classes"
            flip_axis=4

    sess_options = onnxruntime.SessionOptions()
    if threads>0:
        sess_options.intra_op_num_threads = threads

    cuda_opts={"use_tf32": 1 if use_tf32 else 0}
    if device_id is not None:
        cuda_opts["device_id"] = device_id
        
        m = re.match(r"\[(.*)\]", params.input)
        if m is not None:
            inp = m[1].split(",")
            inputs=[]
            for i in inp:
                q=re.match(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$",i)
                if q is not None:
                    inputs.append(float(q[0]))
                else:
                    inputs.append(i)
        else:
            ref_file=params.input
            inputs=[params.input]

            # attach additional channels
            if params.add is not None:
                for a in params.add:
                    inputs.append(a)
            if params.channels>1:
                for i in range(params.channels-1):
                    inputs.append(params.fill)

        segment_with_onnx(inputs, params.output, model=params.model,
                            n_classes=params.n_classes,
                            patch_sz=params.patch_sz, crop=params.crop,
                            bck=params.bck, stride=params.stride,
                            padvol=params.padvol, cropvol=params.cropvol,
                            mask=params.mask, fuzzy=params.fuzzy, 
                            threads=params.threads, cpu=params.cpu,
                            uniformize=params.uniformize,
                            history=_history,whole=params.whole,
                            freesurfer=params.freesurfer,
                            normalize=params.normalize,
                            normalize_max=params.max_normalize,
                            largest=params.largest,
                            quant_size=params.quant,
                            dist=params.distance,
                            padfill=params.padfill)


    else:
      print("Run with --help")
   


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

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

import numpy as np

from minc2_simple import minc2_file
from .minc.io  import *
from .minc.geo import *

import onnxruntime
from onnx import numpy_helper


def parse_options():

    parser = argparse.ArgumentParser(description='Apply pre-trained model using OpenVino',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("model", type=str, default=None,
                        help="pretrained model (ONNX or OpenVino)")
    
    parser.add_argument("input", type=str, 
                        help="Input minc file, or input spec in [a,b,...] where a,b is ether const number of file name")
    
    parser.add_argument("output", type=str, nargs='?',
                        help="Output minc file")
    
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

    parser.add_argument('-q','--quiet', action="store_true",
                        default=False,
                        help='Suppress warnings' )

    parser.add_argument('-F','--fuzzy',
                        help='Output fuzzy volume(s)' )

    parser.add_argument('-T','--threads',type=int,default=0,
                        help='Number of threads to use' )

    parser.add_argument('-n','--n_classes',type=int,default=0,
                        help='Number of segmentation classes, needed for overlap only' )
    
    parser.add_argument('-U','--uniformize',type=float,
                        help='Uniformize image resolution befor applying CNN' )
    
    # parser.add_argument('-V','--vae', 
    #                     help='Output vae volume(s)' )

    # parser.add_argument('--latent',
    #                     help='Save latent state vectors' )

    parser.add_argument("--fill", type=float, default=38.81240207,
                        help="Fill value for missing channel")

    # parser.add_argument("--loc", type=str,
    #                     help="latent space location")

    # parser.add_argument("--prec", type=str,
    #                     help="latent space precision (inv covariance)")

    # parser.add_argument('--likelihood',
    #                     help='Output likelihood volume(s)' )
    parser.add_argument('--distance', action="store_true",
                        default=False,
                        help='Uses Distance model' )

    parser.add_argument('--whole', action="store_true",
                        default=False,
                        help='Apply model to the whole image, without overlapping patches' )
    
    parser.add_argument('--freesurfer', action="store_true",
                        default=False,
                        help='Apply model using freesurfer coordinate convention' )
    
    parser.add_argument('--normalize', action="store_true",
                        default=False,
                        help='Apply intensity normalization between 0 and 1 using quantiles' )

    parser.add_argument('--max_normalize', action="store_true",
                        default=False,
                        help='Apply intensity normalization between 0 and 1 using maximum' )
    
    parser.add_argument('--largest', action="store_true",
                        default=False,
                        help='Apply largest component filtering' )

    params = parser.parse_args()
    
    params.instance_norm=False

    if params.stride is None:
        if isinstance(params.patch_sz, list):
            params.stride = min(params.patch_sz[0]-params.crop*2,params.patch_sz[1]-params.crop*2,params.patch_sz[2]-params.crop*2)
        else:
            params.stride = params.patch_sz-params.crop*2

    return params

def log_softmax(x, axis=1):
    e_x = np.exp(x - np.max(x,axis=axis))
    return np.log(e_x / e_x.sum(axis=axis))


def softmax(x,axis=1):
    e_x = np.exp(x - np.max(x,axis=axis))
    return e_x / e_x.sum(axis=axis)

def find_largest_component(input):
    from scipy.ndimage import label
    structure = np.ones((3, 3, 3), dtype=np.int32)
    labeled, ncomponents = label(input, structure)
    largest = np.argmax([np.sum(labeled==i) for i in range(1,ncomponents+1)])+1
    return labeled == largest


def segment_with_patches_whole(
    dataset, model, 
    quant_size=64,
    normalize=False,
    normalize_max=False,
    freesurfer=False,
    dist_border=1.0, 
    largest_component=False,
    out_fuzzy=False,
    dist=False
    ):

    out_name = "seg" if not dist else "dist"

    target_shape = np.ceil(np.array(dataset.shape[2:]) / quant_size).astype(int) * quant_size

    if np.any(target_shape != dataset.shape[2:]):
        conformed = np.zeros( (1,1, *target_shape), dtype='float32')
        conformed[:,:, :dataset.shape[2], :dataset.shape[3], :dataset.shape[4]] = dataset
    else:
        conformed = dataset.astype('float32') # to be compatible with spatial expectation of the model

    if freesurfer:
        conformed=conformed.transpose([0,1,4,2,3])[:,:,:,::-1,:].copy()

    if normalize:
        conformed -= conformed.min()
        conformed = np.clip(conformed / np.percentile(conformed,99),0.0, 1.0)
    elif normalize_max:
        conformed = conformed / np.max(conformed)

    out = model.run([out_name],{'scan':conformed})[0]

    if freesurfer:
        out=out[:,:,:,::-1,:].transpose([0,1,3,4,2])

    if np.any(target_shape != dataset.shape[2:]):
        out=out[:,:,:dataset.shape[2], :dataset.shape[3], :dataset.shape[4]]

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
            # need to iterate over all labels
            output_seg = output_seg.squeeze()
            output_seg_ = np.zeros_like(output_seg,dtype=np.uint8)

            for l in range(1,out.shape[1]):
                tmp=find_largest_component(output_seg==l)
                output_seg_[tmp ] = l
           
            output_seg = np.expand_dims(np.expand_dims(output_seg_,axis=0),axis=0) # To be compatible
    
    if out_fuzzy :
        return output_seg, output_fuzzy
    else:
        return output_seg 
    

def segment_with_patches_overlap_ov(
        dataset, model, 
        crop=0,
        patch_sz = None, 
        stride = None,
        n_classes=2,
        bck = 0, 
        out_fuzzy=False,
        freesurfer=False,
        dist_border=1.0, 
        normalize=False,
        normalize_max=False,
        dist=False):
    """
    Apply model to dataset of arbitrary size
    Arguments:
        dataset - numpy.array of input data, 5D
        model - openvino model
    Keyword arguments:
        crop - crop patches by this many voxels in all spatial dimensions for output
        patch_sz - size of patch to process with a model
        stride - step between patches, patches can overlap
        bck - background value, to be used for areas where model was not applied
        out_fuzzy - output fuzzy results instead of just discrete
    Returns: 
        3D segmentation , if out_fuzzy is False
        tuple: 3D segmentation, 4D fuzzy output if out_fuzzy is True
    """

    out_name = "seg" if not dist else "dist"
    out_classes = 1 if dist and n_classes is None else n_classes

    if not isinstance(patch_sz, list):
        patch_sz = [patch_sz, patch_sz, patch_sz]

    if not isinstance(stride, list):
        stride = [stride, stride, stride]

    if freesurfer:
        dataset=dataset.transpose([0,1,4,2,3])[:,:,:,::-1,:].copy()

    if normalize:
        dataset -= dataset.min()
        dataset = np.clip(dataset / np.percentile(dataset,99),0.0, 1.0)
    elif normalize_max:
        dataset = dataset / np.max(dataset)

    dsize = dataset.shape
    output_size = list( dsize )
    output_size[1] = 1 
    output_size_fuzzy = list( dsize )
    output_size_fuzzy[1] = out_classes

    output_fuzzy  = np.zeros( output_size_fuzzy, dtype=np.float32 )
    output_weight = np.zeros( output_size, dtype=np.float32 )

    patch_sz_ = [patch_sz[0] - crop*2,patch_sz[1] - crop*2,patch_sz[2] - crop*2]
    
    out_roi = [dsize[2]-crop*2, dsize[3]-crop*2, dsize[4]-crop*2 ]

    for k in range(math.ceil( out_roi[0]/stride[0] )):
        for l in range(math.ceil( out_roi[1]/stride[1] )):
            for m in range(math.ceil( out_roi[2]/stride[2] )):
                c = [k*stride[0] + crop, l*stride[1] + crop, m*stride[2] + crop]

                for i in range(3):
                    c[i] = max(min(c[i], dsize[i+2] - patch_sz[i] + crop - 1),crop)

                # extract a patch
                in_data = np.ascontiguousarray(
                        dataset[:, :, c[0]-crop: c[0]-crop+patch_sz[0], c[1]-crop: c[1]-crop+patch_sz[1], c[2]-crop: c[2]-crop+patch_sz[2]]
                    )

                out = model.run([out_name],{'scan':in_data})[0]

                if dist:
                    patch_output = out
                else:
                    patch_output = log_softmax(out, axis=1)

                # accumulate data
                output_fuzzy[:, :, c[0]: c[0]+patch_sz_[0], c[1]: c[1]+patch_sz_[1], c[2]: c[2]+patch_sz_[2]] += \
                        patch_output[:, :, crop: crop+patch_sz_[0], crop: crop+patch_sz_[1], crop: crop+patch_sz_[2]]

                output_weight[:, :, c[0]: c[0]+patch_sz_[0], c[1]: c[1]+patch_sz_[1], c[2]: c[2]+patch_sz_[2]] += 1.0
                    
    # aggregate weights
    invalid = output_weight<1.0
    output_weight[invalid] = 1.0
    output_fuzzy /= output_weight

    if not dist:
        output_fuzzy = softmax(output_fuzzy, axis=1)

    # set BG to 1 where mask was invalid
    output_fuzzy[:,0:1,:,:,:][invalid] = 1.0 

    # HACK , because i can't figure out equivalnet to masked_fill in torch
    for q in range(output_fuzzy.shape[1]-1):
        output_fuzzy[:,(q+1):(q+2),:,:,:][invalid]  = 0.0
     
    if n_classes==1 and dist:
        output_dataset = (output_fuzzy < dist_border)
    elif dist:
        output_dataset = np.expand_dims( np.argmin(output_fuzzy, axis=1).astype(np.uint8),axis=1)
    else:
        output_dataset = np.expand_dims( np.argmax(output_fuzzy, axis=1).astype(np.uint8),axis=1)

    if freesurfer:
        output_dataset=output_dataset[:,:,:,::-1,:].transpose([0,1,3,4,2])
        if out_fuzzy:
            output_fuzzy=output_fuzzy[:,:,:,::-1,:].transpose([0,1,3,4,2])

    if out_fuzzy :
        return output_dataset, output_fuzzy
    else:
        return output_dataset 

"""
High level function to apply segmentation
"""
def segment_with_onnx(  in_scans, 
                        out_seg, 
                        n_classes=None,
                        mask=None,
                        model=None,
                        patch_sz=64, stride=32, 
                        cpu=True, threads=0,
                        crop=0,cropvol=0,padvol=0,padfill=0.0,bck=0,
                        history=None,
                        fuzzy=None,
                        whole=False,
                        quant_size=64,
                        freesurfer=False,
                        normalize=False,
                        normalize_max=False,
                        largest=False,
                        dist=False,
                        uniformize=None):
    inputs=[]
    # load all inputs
    # TODO: deal with floating point values
    orig_aff = None
    orig_shape = None
    
    for i in in_scans:
        ref_file = i
        data, aff = load_minc_volume_np(i, dtype='float32')

        # make sure all files have the same shape and orientation
        if orig_shape is not None:
            assert(np.all(orig_shape == np.array(data.shape)))
        else:
            orig_shape = np.array(data.shape)
        
        if orig_aff is not None:
            assert(np.all(orig_aff - aff < 1e-3))
        else:
            orig_aff = aff

        if uniformize is not None:
            data, new_aff = uniformize_volume(data, aff, step=uniformize)

        inputs+=[np.expand_dims(data, axis=(0, 1))]
    # 
    dset = np.concatenate(inputs, axis=1)

    sess_options = onnxruntime.SessionOptions()
    if threads>0:
        sess_options.intra_op_num_threads = threads
    
    model = onnxruntime.InferenceSession(model, sess_options, providers=['CPUExecutionProvider'])

    if whole:
        patch_sz = np.clip(np.ceil((np.array(dset.shape[2:]) - cropvol*2 + padvol*2) / quant_size).astype(int) * quant_size, quant_size*2, quant_size*5).tolist()
        stride = patch_sz
    elif not isinstance(patch_sz, list):
        patch_sz = [patch_sz, patch_sz, patch_sz]
    
    if cropvol>0:
        orig_size = dset.shape
        orig_fuzzy_size = dset.shape
        orig_vae_size = dset.shape
        dset = dset[:, :, cropvol: orig_size[2]-cropvol*2, cropvol: orig_size[3]-cropvol*2, cropvol: orig_size[4]-cropvol*2]
    elif padvol>0:
        orig_size = dset.shape
        orig_fuzzy_size = dset.shape
        orig_vae_size = dset.shape
        
        dset = np.ascontiguousarray( np.pad(dset, pad_width=((0,0),(0,0),(padvol,padvol),(padvol,padvol),(padvol,padvol)), 
            mode='constant', constant_values = padfill))
    
    # apply model
    if fuzzy is not None : # or params.vae is not None or params.latent is not None
        if whole:
            dset_out, dset_out_fuzzy = segment_with_patches_whole(
                dset, model,
                largest_component=largest, 
                out_fuzzy=True,
                freesurfer=freesurfer,
                normalize=normalize,
                dist=dist) 
        else:
            dset_out, dset_out_fuzzy = segment_with_patches_overlap_ov(
                dset, model, 
                n_classes=n_classes,
                patch_sz=patch_sz, crop=crop, 
                bck=bck, stride=stride, 
                freesurfer=freesurfer,
                normalize=normalize,
                normalize_max=normalize_max,
                out_fuzzy=True,
                dist=dist)
    else:
        if whole:
            dset_out = segment_with_patches_whole(
                dset, model,
                # TODO: fix this
                #dist_border=(0 if freesurfer else None),
                largest_component=largest, 
                out_fuzzy=False, freesurfer=freesurfer, 
                normalize=normalize,
                normalize_max=normalize_max,
                dist=dist)
        else:
            dset_out = segment_with_patches_overlap_ov(
                dset, model, 
                patch_sz=patch_sz, crop=crop, 
                n_classes=n_classes,
                bck=bck, stride=stride, out_fuzzy=False,
                freesurfer=freesurfer,
                normalize=normalize,
                normalize_max=normalize_max,
                dist=dist)
    if cropvol>0:
        dset_out_ = np.full(orig_size, bck, dtype=np.uint8)

        dset_out_[:, :, cropvol: orig_size[2]-cropvol*2, cropvol: orig_size[3]-cropvol*2, cropvol: orig_size[4]-cropvol*2]=\
            dset_out
        dset_out = dset_out_

        if fuzzy is not None:
            orig_fuzzy_size[1] = dset_out_fuzzy.shape[1]
            dset_out_fuzzy_ = np.zeros(orig_fuzzy_size)
            dset_out_fuzzy_[:, :, cropvol: orig_size[2]-cropvol*2, cropvol: orig_size[3]-cropvol*2, cropvol: orig_size[4]-cropvol*2]=\
                dset_out_fuzzy
            dset_out_fuzzy = dset_out_fuzzy_

    elif padvol>0:
        dset_out = dset_out[:, :, padvol: orig_size[2]+padvol, padvol: orig_size[3]+padvol, padvol: orig_size[4]+padvol]
        if fuzzy is not None:
            dset_out_fuzzy = dset_out_fuzzy[:, :, padvol: orig_size[2]+padvol, padvol: orig_size[3]+padvol, padvol: orig_size[4]+padvol]

    dset_out = np.ascontiguousarray(dset_out.squeeze(), dtype=np.uint8)

    if uniformize is not None and np.any(np.array(dset_out.shape) != orig_shape):
        # potentially resize to original size
        dset_out = resample_volume(dset_out, new_aff, orig_shape, orig_aff, order=0, fill=bck)[0]
    
    if fuzzy is not None:
        #dset_out_fuzzy = np.ascontiguousarray(dset_out_fuzzy)
        
        for f in range(dset_out_fuzzy.shape[1]):
            dset_out_f=dset_out_fuzzy[0,f,:,:,:]

            if uniformize is not None and np.any(np.array(dset_out_f.shape) != orig_shape):
                dset_out_f = resample_volume(dset_out_f, new_aff, orig_shape, orig_aff, order=1, fill=bck)[0]
            else:
                dset_out_f = np.ascontiguousarray(dset_out_f)
            
            save_minc_volume(fuzzy+'_{}.mnc'.format(f), 
                             dset_out_f, orig_aff, ref_fname=ref_file, history=history)

    if mask is not None:
        mask, mask_aff = load_minc_volume_np(mask, as_byte=True)
        assert(np.all(orig_shape == np.array(mask.shape)))
        dset_out[mask<1] = bck

    save_minc_volume(out_seg,dset_out, orig_aff, ref_fname=ref_file, history=history)
    return out_seg

if __name__ == '__main__':
    _history=format_history(sys.argv)
    params = parse_options()
    
    if params.model is not None and \
       params.input is not None:
        
        m = re.match("\[(.*)\]", params.input)
        if m is not None:
            inp = m[1].split(",")
            shape = None
            inputs=[]
            for i in inp:
                q=re.match(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$",i)
                if q is not None:
                    inputs.append(float(q[0]))
                else:
                    inputs.append(i)
            ####
            dset=[]
            for i in inputs:
                if isinstance(i,np.ndarray):
                    dset+=[i]
                else:
                    dset+=[np.full(shape, i)]

            dset = np.concatenatecat(dset, axis=1)
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
                            dist=params.distance)


    else:
      print("Run with --help")
   


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

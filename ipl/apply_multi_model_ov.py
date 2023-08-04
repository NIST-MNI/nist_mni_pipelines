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

# deep models
# import torch
# from model.util import *

# debug
from minc2_simple import minc2_file
# import torch.nn.functional as F

#from openvino.inference_engine import IECore
from openvino.runtime import Core
from openvino.runtime.properties import inference_num_threads


def format_history(argv):
    stamp=strftime("%a %b %d %T %Y>>>", gmtime())
    return stamp+(' '.join(argv))

def load_input(fname, as_byte=False):
    mm=minc2_file(fname)
    mm.setup_standard_order()
    d = mm.load_complete_volume(minc2_file.MINC2_UBYTE if as_byte else minc2_file.MINC2_FLOAT)
    mm.close()
    return d

def save_output(arr, out, ref_fname=None, ref=None, history=None,as_byte=False):

    if ref_fname is not None:
        _ref=minc2_file(ref_fname)
        _ref.setup_standard_order()
    elif ref is not None:
        _ref=ref

    a=minc2_file()

    if as_byte:
        a.define(_ref.store_dims(), minc2_file.MINC2_UBYTE, minc2_file.MINC2_UBYTE)
    else:
        a.define(_ref.store_dims(), minc2_file.MINC2_SHORT, minc2_file.MINC2_FLOAT)
    a.create(out)
    a.setup_standard_order()

    a.copy_metadata(_ref)
    
    if history is not None:
        old_history=_ref.read_attribute("","history")
        new_history=old_history+"\n"+history
        a.write_attribute("","history",new_history)
    
    _arr=np.ascontiguousarray(arr) #.contiguous().clone()
    a.save_complete_volume(_arr)
    a.close()
    if ref_fname is not None:
        _ref.close()


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
                        type=int, default=64,
                        help="Patch size")
                        
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

    parser.add_argument('-T','--threads',
                        help='Number of threads to use' )

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

    params = parser.parse_args()
    
    params.instance_norm=False

    if params.stride is None:
        params.stride = params.patch_sz-params.crop*2

    return params

def log_softmax(x, axis=1):
    e_x = np.exp(x - np.max(x,axis=axis))
    return np.log(e_x / e_x.sum(axis=axis))


def softmax(x,axis=1):
    e_x = np.exp(x - np.max(x,axis=axis))
    return e_x / e_x.sum(axis=axis)

def segment_with_patches_overlap_ov(
        dataset, model, 
        crop=0,
        patch_sz = None, 
        stride = None,
        bck = 0, 
        out_fuzzy=False):
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

    # find seg output
    #input_layer=model.inputs[0] # expect single input
    seg_idx=next(i for i,v in enumerate(model.outputs) if v.any_name=="seg")
    out_shape = model.outputs[seg_idx].shape

    dsize = dataset.shape
    output_size = list( dsize )
    output_size[1] = 1 
    output_size_fuzzy = list( dsize )
    output_size_fuzzy[1] = out_shape[1]

    output_fuzzy = np.zeros( output_size_fuzzy, dtype=np.float32 )
    output_weight = np.zeros( output_size, dtype=np.float32 )

    patch_sz_ = patch_sz - crop*2 
    
    out_roi = [ dsize[2]-crop*2, dsize[3]-crop*2, dsize[4]-crop*2 ]

    for k in range(math.ceil( out_roi[0]/stride )):
        for l in range(math.ceil( out_roi[1]/stride )):
            for m in range(math.ceil( out_roi[2]/stride )):
                c = [k*stride + crop, l*stride + crop, m*stride + crop]

                for i in range(3):
                    c[i] = min(c[i], dsize[i+2] - patch_sz + crop-1)
                                
                # extract a patch
                in_data = dataset[:, :, c[0]-crop: c[0]-crop+patch_sz, c[1]-crop: c[1]-crop+patch_sz, c[2]-crop: c[2]-crop+patch_sz]
                
                out = model([in_data])[seg_idx]
                patch_output = log_softmax(out, axis=1)

                # accumulate data
                output_fuzzy[:, :, c[0]: c[0]+patch_sz_, c[1]: c[1]+patch_sz_, c[2]: c[2]+patch_sz_] += \
                        patch_output[:, :, crop: crop+patch_sz_, crop: crop+patch_sz_, crop: crop+patch_sz_]

                output_weight[:, :, c[0]: c[0]+patch_sz_, c[1]: c[1]+patch_sz_, c[2]: c[2]+patch_sz_] += 1.0
                    
    # aggregate weights
    invalid = output_weight<1.0
    output_weight[invalid] = 1.0
    output_fuzzy /= output_weight
    output_fuzzy = softmax(output_fuzzy, axis=1)

    # set BG to 1 where mask was invalid
    output_fuzzy[:,0:1,:,:,:][invalid] = 1.0 

    # HACK , because i can't figure out equivalnet to masked_fill in torch
    for q in range(output_fuzzy.shape[1]-1):
        output_fuzzy[:,(q+1):(q+2),:,:,:][invalid]  = 0.0
     
    output_dataset = np.expand_dims( np.argmax(output_fuzzy,axis=1).astype(np.uint8),axis=1)
    # output_dataset[invalid] = bck

    if out_fuzzy :
        return output_dataset, output_fuzzy
    else:
        return output_dataset 

"""
High level function to apply segmentation
"""
def segment_with_openvino(in_scans, out_seg, 
                        mask=None,
                        model=None,
                        patch_sz=64, stride=32, cpu=True, threads=None,
                        crop=0,cropvol=0,padvol=0,padfill=0.0,bck=0,
                        history=None,fuzzy=None):
    inputs=[]
    # load all inputs
    # TODO: deal with floating point values
    for i in in_scans:
        ref_file = i
        d = np.expand_dims(load_input(i,as_byte=False), axis=(0, 1))
        inputs.append(d)
    dset = np.concatenate(inputs, axis=1)

    # prepare model
    core = Core()
    if threads is not None:
        core.set_property("CPU", {inference_num_threads():threads})
    
    model=core.read_model(model=model)
    if cpu:
        compiled_model = core.compile_model(model=model, device_name="CPU")
    else:
        compiled_model = core.compile_model(model=model, device_name="GPU")

    if cropvol>0:
        orig_size = dset.shape
        orig_fuzzy_size = dset.shape
        orig_vae_size = dset.shape
        dset = dset[:, :, cropvol: orig_size[2]-cropvol*2, cropvol: orig_size[3]-cropvol*2, cropvol: orig_size[4]-cropvol*2]
    elif padvol>0:
        orig_size = dset.shape
        orig_fuzzy_size = dset.shape
        orig_vae_size = dset.shape
        print("dset:",dset.shape)
        dset = np.ascontiguousarray( np.pad(dset, pad_width=((0,0),(0,0),(padvol,padvol),(padvol,padvol),(padvol,padvol)), 
            mode='constant', constant_values = padfill))
    # apply model

    if fuzzy is not None : # or params.vae is not None or params.latent is not None
        dset_out, dset_out_fuzzy = segment_with_patches_overlap_ov(
            dset, compiled_model,  
            patch_sz=patch_sz, crop=crop, 
            bck=bck, stride=stride, 
            out_fuzzy=True) 
            # out_vae=(vae is not None),
            # out_latent_vectors=(latent is not None)                 
    else:
        dset_out = segment_with_patches_overlap_ov(dset, compiled_model, 
            patch_sz=patch_sz, crop=crop, 
            bck=bck, stride=stride, out_fuzzy=False)

    if cropvol>0:
        dset_out_ = np.full(orig_size,bck,dtype=np.int8)

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

    dset_out = np.ascontiguousarray(dset_out.squeeze(),dtype=np.int8)
    
    if fuzzy is not None:
        dset_out_fuzzy = np.ascontiguousarray(dset_out_fuzzy)

        for f in range(dset_out_fuzzy.shape[1]):
            save_output(dset_out_fuzzy[0,f,:,:,:],fuzzy+'_{}.mnc'.format(f), ref_fname=ref_file, history=_history)

    if mask is not None:
        mask = load_input(mask,as_byte=True)
        dset_out[mask<1] = bck

    save_output(dset_out, out_seg, ref_fname=ref_file, history=history, as_byte=True)
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

        segment_with_openvino(inputs, params.output, model=params.model,
                              patch_sz=params.patch_sz, crop=params.crop,
                              bck=params.bck, stride=params.stride,
                              padvol=params.padvol, cropvol=params.cropvol,
                              mask=params.mask, fuzzy=params.fuzzy, 
                              threads=params.threads, cpu=params.cpu,
                              history=_history)


    else:
      print("Run with --help")
   


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

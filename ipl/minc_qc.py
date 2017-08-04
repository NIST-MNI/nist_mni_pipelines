#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 11/21/2011
#
# Tools for creating QC images

from __future__ import print_function

import numpy as np
import numpy.ma as ma

import scipy 
import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from minc2.simple import minc2_file
from minc2_simple import minc2_file

import matplotlib.cm  as cmx
import matplotlib.colors as colors
import argparse


try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    str = str
    unicode = str
    bytes = bytes
    basestring = (str,bytes)
else:
    # 'unicode' exists, must be Python 2
    str = str
    unicode = unicode
    bytes = str
    basestring = basestring


def alpha_blend(si, so, ialpha, oalpha):
    """Perform alpha-blending
    """
    si_rgb =   si[..., :3]
    si_alpha = si[..., 3]*ialpha
    
    so_rgb =   so[..., :3]
    so_alpha = so[..., 3]*oalpha
    
    out_alpha = si_alpha + so_alpha * (1. - si_alpha)
    
    out_rgb = (si_rgb * si_alpha[..., None] +
        so_rgb * so_alpha[..., None] * (1. - si_alpha[..., None])) / out_alpha[..., None]
    
    out = np.zeros_like(si)
    out[..., :3] = out_rgb 
    out[..., 3]  = out_alpha
    
    return out


def max_blend(si,so):
    """Perform max-blending
    """
    return np.maximum(si,so)

def over_blend(si,so, ialpha, oalpha):
    """Perform max-blending
    """
    si_rgb =   si[..., :3]
    si_alpha = si[..., 3]*ialpha
    
    so_rgb =   so[..., :3]
    so_alpha = so[..., 3]*oalpha
    
    out_alpha = np.maximum(si_alpha ,  so_alpha )
    
    out_rgb = si_rgb * (si_alpha[..., None]-so_alpha[..., None]) + so_rgb * so_alpha[..., None] 
    
    out = np.zeros_like(si)
    out[..., :3] = out_rgb 
    out[..., 3]  = out_alpha
    
    return out


def qc(
    input,
    output,
    image_range=None,
    mask=None,
    mask_range=None,
    title=None,
    image_cmap='gray',
    mask_cmap='red',
    samples=5,
    mask_bg=None,
    use_max=False,
    use_over=False,
    show_image_bar=False,   # TODO:implement this?
    show_overlay_bar=False,
    dpi=100,
    ialpha=0.8,
    oalpha=0.2,
    format=None
    ):
    """QC image generation, drop-in replacement for minc_qc.pl
    Arguments:
        input -- input minc file
        output -- output QC graphics file 
        
    Keyword arguments:
        image_range -- (optional) intensity range for image
        mask  -- (optional) input mask file
        mask_range -- (optional) mask file range
        title  -- (optional) QC title
        image_cmap -- (optional) color map name for image, 
                       possibilities: red, green,blue and anything from matplotlib
        mask_cmap -- (optional) color map for mask, default red
        samples -- number of slices to show , default 5
        mask_bg  -- (optional) level for mask to treat as background
        use_max -- (optional) use 'max' colour mixing
        use_over -- (optional) use 'over' colour mixing
        show_image_bar -- show color bar for intensity range, default false
        show_overlay_bar  -- show color bar for mask intensity range, default false
        dpi -- graphics file DPI, default 100
        ialpha -- alpha channel for colour mixing of main image
        oalpha -- alpha channel for colour mixing of mask image
    """
    
    #_img=minc.Image(input)
    #_idata=_img.data
    _img=minc2_file(input)
    _img.setup_standard_order()
    _idata=_img.load_complete_volume(minc2_file.MINC2_FLOAT)
    _idims=_img.representation_dims()
    
    data_shape=_idata.shape
    spacing=[_idims[0].step,_idims[1].step,_idims[2].step]
    
    _ovl=None
    _odata=None
    omin=0
    omax=1
    
    if mask is not None:
        _ovl=minc2_file(input)
        _ovl.setup_standard_order()
        _ovl_data=_ovl.load_complete_volume(minc2_file.MINC2_FLOAT)
        if _ovl_data.shape != data_shape:
            #print("Overlay shape does not match image!\nOvl={} Image={}",repr(_ovl.data.shape),repr(data_shape))
            raise "Overlay shape does not match image!\nOvl={} Image={}",repr(_ovl_data.shape),repr(data_shape)
        if mask_range is None:
            omin=np.nanmin(_ovl_data)
            omax=np.nanmax(_ovl_data)
        else:
            omin=mask_range[0]
            omax=mask_range[1]
        _odata=_ovl_data
        
        if mask_bg is not None:
            _odata=ma.masked_less(_odata, mask_bg)
        
    slices=[]
    
    # setup ranges
    vmin=vmax=0.0
    if image_range is not None:
        vmin=image_range[0]
        vmax=image_range[1]
    else:
        vmin=np.nanmin(_idata)
        vmax=np.nanmax(_idata)

    cm = plt.get_cmap(image_cmap)
    cmo= plt.get_cmap(mask_cmap)
    cmo.set_bad('k',alpha=0.0)

    cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
    oNorm  = colors.Normalize(vmin=omin, vmax=omax)
    
    scalarMap  = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    oscalarMap = cmx.ScalarMappable(norm=oNorm, cmap=cmo)
    aspects = []
    
    # axial slices
    for j in range(0,samples):
        i=(data_shape[0]/samples)*j+(data_shape[0]%samples)/2
        si=scalarMap.to_rgba(_idata[i , : ,:])

        if _ovl is not None:
            so=oscalarMap.to_rgba(_odata[i , : ,:])
            if use_max: si=max_blend(si,so)
            elif use_over: si=over_blend(si,so, ialpha, oalpha)
            else: si=alpha_blend(si, so, ialpha, oalpha)
        slices.append( si )
        aspects.append( spacing[0]/spacing[1] )
    # coronal slices
    for j in range(0,samples):
        i=(data_shape[1]/samples)*j+(data_shape[1]%samples)/2
        si=scalarMap.to_rgba(_idata[: , i ,:])
        
        if _ovl is not None:
            so=oscalarMap.to_rgba(_odata[: , i ,:])
            if use_max: si=max_blend(si,so)
            elif use_over: si=over_blend(si,so, ialpha, oalpha)
            else: si=alpha_blend(si, so, ialpha, oalpha)
        slices.append( si )
        aspects.append( spacing[2]/spacing[0] )
        
    # sagittal slices
    for j in range(0,samples):
        i=(data_shape[2]/samples)*j+(data_shape[2]%samples)/2
        si=scalarMap.to_rgba(_idata[: , : , i])
        if _ovl is not None:
            so=oscalarMap.to_rgba(_odata[: , : , i])
            if use_max: si=max_blend(si,so)
            elif use_over: si=over_blend(si,so, ialpha, oalpha)
            else: si=alpha_blend(si, so, ialpha, oalpha)
        slices.append( si )
        aspects.append( spacing[2]/spacing[1] )
        
    w, h = plt.figaspect(3.0/samples)
    fig = plt.figure(figsize=(w,h))
    
    #outer_grid = gridspec.GridSpec((len(slices)+1)/2, 2, wspace=0.0, hspace=0.0)
    ax=None
    imgplot=None
    for i,j in enumerate(slices):
        ax =  plt.subplot2grid( (3, samples), (i/samples, i%samples) )
        imgplot = ax.imshow(j,origin='lower',cmap=cm, aspect=aspects[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_visible(False)
    # show for the last plot
    if show_image_bar:
        cbar = fig.colorbar(imgplot)
    
    
    if title is not None:
        plt.suptitle(title,fontsize=20)
        plt.subplots_adjust(wspace = 0.0 ,hspace=0.0)
    else:
        plt.subplots_adjust(top=1.0,bottom=0.0,left=0.0,right=1.0,wspace = 0.0 ,hspace=0.0)
    
    #fig.tight_layout()
    #plt.show()
    plt.savefig(output, bbox_inches='tight', dpi=dpi,format=format)
    plt.close()
    plt.close('all')

def qc_field_contour(
    input,
    output,
    image_range=None,
    title=None,
    image_cmap='gray',
    samples=5,
    show_image_bar=False, # TODO:implement this?
    dpi=100,
    format=None
    
    ):
    """show field contours
    """
    
    _img=minc2_file(input)
    _img.setup_standard_order()
    _idata=_img.load_complete_volume(minc2_file.MINC2_FLOAT)
    _idims=_img.representation_dims()
    
    data_shape=_idata.shape
    spacing=[_idims[0].step,_idims[1].step,_idims[2].step]
    
    slices=[]
    
    # setup ranges
    vmin=vmax=0.0
    if image_range is not None:
        vmin=image_range[0]
        vmax=image_range[1]
    else:
        vmin=np.nanmin(_idata)
        vmax=np.nanmax(_idata)

    cm = plt.get_cmap(image_cmap)

    cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
    
    scalarMap  = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
    for j in range(0,samples):
        i=(data_shape[0]/samples)*j+(data_shape[0]%samples)/2
        si=_idata[i , : ,:]
        slices.append( si )
        
    for j in range(0,samples):
        i=(data_shape[1]/samples)*j+(data_shape[1]%samples)/2
        si=_idata[: , i ,:]
        slices.append( si )
        
    for j in range(0,samples):
        i=(data_shape[2]/samples)*j+(data_shape[2]%samples)/2
        si=_idata[: , : , i]
        slices.append( si )
    
    w, h = plt.figaspect(3.0/samples)
    fig = plt.figure(figsize=(w,h))
    
    #outer_grid = gridspec.GridSpec((len(slices)+1)/2, 2, wspace=0.0, hspace=0.0)
    ax=None
    imgplot=None
    for i,j in enumerate(slices):
        ax =  plt.subplot2grid( (3, samples), (i/samples, i%samples) )
        imgplot = ax.contour(j,origin='lower', cmap=cm, norm=cNorm, levels=np.linspace(vmin,vmax,20))
        #plt.clabel(imgplot, inline=1, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_visible(False)
    # show for the last plot
    if show_image_bar:
        cbar = fig.colorbar(imgplot)
    
    
    if title is not None:
        plt.suptitle(title,fontsize=20)
        plt.subplots_adjust(wspace = 0.0 ,hspace=0.0)
    else:
        plt.subplots_adjust(top=1.0,bottom=0.0,left=0.0,right=1.0,wspace = 0.0 ,hspace=0.0)
    
    plt.savefig(output, bbox_inches='tight', dpi=dpi)
    plt.close('all')


# register custom maps
plt.register_cmap(cmap=colors.LinearSegmentedColormap('red',
    {'red':   ((0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0)),

      'green': ((0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0)),

      'blue':  ((0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0)),
      
      'alpha':  ((0.0, 0.0, 1.0),
                (1.0, 1.0, 1.0))         
    }))
      
plt.register_cmap(cmap=colors.LinearSegmentedColormap('green', 
    {'green': ((0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0)),

      'red':   ((0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0)),

      'blue':  ((0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0)),

      'alpha': ((0.0, 0.0, 1.0),
                (1.0, 1.0, 1.0))         
    }))

plt.register_cmap(cmap=colors.LinearSegmentedColormap('blue', 
    {'blue':  ((0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0)),

      'red':   ((0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0)),

      'green': ((0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0)),
      
      'alpha': ((0.0, 0.0, 1.0),
                (1.0, 1.0, 1.0))         
    }))

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Make QC image')

    parser.add_argument("--debug",
                    action="store_true",
                    dest="debug",
                    default=False,
                    help="Print debugging information" )
    
    parser.add_argument("--contour",
                    action="store_true",
                    dest="contour",
                    default=False,
                    help="Make contour plot" )
    
    parser.add_argument("--bar",
                    action="store_true",
                    dest="bar",
                    default=False,
                    help="Show colour-bar" )
    
    parser.add_argument("--cmap",
                    dest="cmap",
                    default=None,
                    help="Colour map" )
    
    parser.add_argument("--mask",
                    dest="mask",
                    default=None,
                    help="Add mask" )
    
    parser.add_argument("--over",
                    dest="use_over",
                    action="store_true",
                    default=False,
                    help="Overplot" )
    
    parser.add_argument("--max",
                    dest="use_max",
                    action="store_true",
                    default=False,
                    help="Use max mixing" )

    parser.add_argument("input",
                        help="Input minc file")

    parser.add_argument("output",
                        help="Output QC file")
    
    options = parser.parse_args()

    if options.debug:
        print(repr(options))

    return options    

if __name__ == '__main__':
    options = parse_options()
    if options.input is not None and options.output is not None:
        if options.contour:
            qc_field_contour(options.input,options.output,show_image_bar=options.bar,image_cmap=options.cmap)
        else:
            qc(options.input,options.output,mask=options.mask,use_max=options.use_max,use_over=options.use_over,mask_bg=0.5)
    else:
        print("Refusing to run without input data, run --help")
        exit(1)
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

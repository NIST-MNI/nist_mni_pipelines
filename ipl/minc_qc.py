# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 11/21/2011
#
# routines for creating QC images

import copy

import numpy as np
import numpy.ma as ma
import math

#import scipy 
import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from minc2_simple import minc2_file

import matplotlib.cm  as cmx
import matplotlib.colors as colors

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
    si_alpha = si[...,  3]*ialpha
    
    so_rgb =   so[..., :3]
    so_alpha = so[...,  3]*oalpha
    
    out_alpha = si_alpha + so_alpha * (1. - si_alpha)
    
    out_rgb = (si_rgb * si_alpha[..., None] +
        so_rgb * so_alpha[..., None] * (1. - si_alpha[..., None])) / out_alpha[..., None]
    
    out = np.zeros_like(si)
    out[..., :3] = out_rgb 
    out[..., 3]  = out_alpha
    
    return out


def max_blend(si,so):
    """
    Perform max-blending, just take the maximum intensity of the two images
    """
    return np.maximum(si,so)

def over_blend(si,so, ialpha, oalpha):
    """
    Perform over-blending, non-transparanet overaly takes over the image
    """
    si_rgb =   si[..., :3]
    si_alpha = si[..., 3]
    
    so_rgb =   so[..., :3]
    so_alpha = so[..., 3]
    
    #out_alpha = np.maximum(si_alpha,  so_alpha )
    
    out_rgb = si_rgb * (so_alpha[..., None]<0.5) + so_rgb * (so_alpha[..., None]>=0.5)
    
    out = np.zeros_like(si)
    out[..., :3] = out_rgb 
    out[..., 3]  = si_alpha
    
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
    samples=6,
    mask_bg=None,
    use_max=False,
    use_over=False,
    show_image_bar=False,   # TODO:implement this?
    show_overlay_bar=False,
    dpi=100,
    ialpha=0.8,
    oalpha=0.2,
    format=None,
    bg_color=None,
    fg_color=None,
    style=None,
    crop=None
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
        samples -- number of slices per dimension to show, default 6
                   should be even number, becaus it will be split across two rows 
        mask_bg  -- (optional) level for mask to treat as background
        use_max -- (optional) use 'max' colour mixing
        use_over -- (optional) use 'over' colour mixing
        show_image_bar -- show color bar for intensity range, default false
        show_overlay_bar  -- show color bar for mask intensity range, default false
        dpi -- graphics file DPI, default 100
        ialpha -- alpha channel for colour mixing of main image
        oalpha -- alpha channel for colour mixing of mask image
        format -- file format override
        bg_color -- background color
        fg_color -- foreground color
        style    -- name of the matplotlib style, see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html#sphx-glr-gallery-style-sheets-style-sheets-reference-py
        crop     -- Crop input image: [x1,x2,y1,y2,z1,z2]
    """
    
    _img=minc2_file(input)
    _img.setup_standard_order()
    _idata=_img.load_complete_volume(minc2_file.MINC2_FLOAT)
    _idims=_img.representation_dims()
    
    data_shape=_idata.shape
    # order of dimensions in representation dimension is reversed
    spacing=[_idims[2].step, _idims[1].step, _idims[0].step]
    
    _ovl=None
    _odata=None
    omin=0
    omax=1
    # setup view
    columns=samples//2
    rows=(samples//columns)*3

    if mask is not None:
        _ovl=minc2_file(mask)
        _ovl.setup_standard_order()
        _ovl_data=_ovl.load_complete_volume(minc2_file.MINC2_FLOAT)
        if _ovl_data.shape != data_shape:
            raise "Overlay shape does not match image!\nOvl={} Image={}".format(repr(_ovl_data.shape),repr(data_shape))
        if mask_range is None:
            omin=np.nanmin(_ovl_data)
            omax=np.nanmax(_ovl_data)
        else:
            omin=mask_range[0]
            omax=mask_range[1]
            ### set values below mask_range[0] to NaN
            _ovl_data[_ovl_data<omin]=math.nan
        _odata=_ovl_data
        
        if mask_bg is not None:
            _odata=ma.masked_less(_odata, mask_bg)
    

    if crop is not None:
        # apply cropping here
        _idata=_idata[crop[4]:crop[5],crop[2]:crop[3],crop[0]:crop[1]]
        if mask is not None:
            _odata=_odata[crop[4]:crop[5],crop[2]:crop[3],crop[0]:crop[1]]
        data_shape=_idata.shape

    slices=[]
    
    # setup ranges
    vmin=vmax=0.0
    if image_range is not None:
        vmin=image_range[0]
        vmax=image_range[1]
    else:
        vmin=np.nanmin(_idata)
        vmax=np.nanmax(_idata)

    cm = copy.copy(plt.get_cmap(image_cmap))
    cmo= copy.copy(plt.get_cmap(mask_cmap))

    cm.set_bad('k', alpha = 1.0)

    # make transparent background
    cmo.set_bad('k',alpha = 0.0)
    cmo.set_under('k',alpha = 0.0)

    cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
    #print("Image range:",vmin,vmax)
    oNorm  = colors.Normalize(vmin=omin, vmax=omax, clip=False)
    
    scalarMap  = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    oscalarMap = cmx.ScalarMappable(norm=oNorm, cmap=cmo)
    aspects = []
    
    # axial slices
    for j in range(0, samples):
        i=int(10+(150.0-10.0)*j/(samples-1))
        i=int(data_shape[0]*i/181.0)

        si=scalarMap.to_rgba(_idata[i , : ,:])

        if _ovl is not None:
            so = oscalarMap.to_rgba(_odata[i , : ,:])
            if    use_max: si=max_blend(si, so)
            elif use_over: si=over_blend(si, so, ialpha, oalpha)
            else:          si=alpha_blend(si, so, ialpha, oalpha)
        slices.append( si )
        aspects.append( spacing[1] / spacing[2] )

    # sagittal slices
    for j in range(0, samples//2):
        i=int(28.0+(166.0-28.0)*j/(samples-1))
        i=int(data_shape[2]*i/193.0)

        si=scalarMap.to_rgba(_idata[: , : , i])
        if _ovl is not None:
            so = oscalarMap.to_rgba(_odata[: , : , i])
            if    use_max: si=max_blend(si,so)
            elif use_over: si=over_blend(si,so, ialpha, oalpha)
            else:          si=alpha_blend(si, so, ialpha, oalpha)
        slices.append( si )
        aspects.append( spacing[0] / spacing[1] )

    for j in range(samples-1, samples//2-1,-1):
        i=int(28.0+(166.0-28.0)*j/(samples-1))
        i=int(data_shape[2]*i/193.0)

        si=scalarMap.to_rgba(_idata[: , : , i])
        if _ovl is not None:
            so = oscalarMap.to_rgba(_odata[: , : , i])
            if    use_max: si=max_blend(si,so)
            elif use_over: si=over_blend(si,so, ialpha, oalpha)
            else:          si=alpha_blend(si, so, ialpha, oalpha)
        slices.append( si )
        aspects.append( spacing[0] / spacing[1] )

    # coronal slices
    for j in range(0,samples):
        i=int(25+(195.0-25.0)*j/(samples-1))
        i=int(data_shape[1]*i/217.0)
        si=scalarMap.to_rgba(_idata[: , i ,:])
        
        if _ovl is not None:
            so = oscalarMap.to_rgba(_odata[: , i ,:])
            if    use_max: si=max_blend(si,so)
            elif use_over: si=over_blend(si,so, ialpha, oalpha)
            else:          si=alpha_blend(si, so, ialpha, oalpha)
        slices.append( si )
        aspects.append( spacing[0]/spacing[2] )

    rc={'interactive': False}
    rc['figure.frameon']=False
    #rc['aa']=True

    if bg_color is not None:
         rc['figure.edgecolor']=bg_color
         rc['figure.facecolor']=bg_color
         rc['grid.color']=bg_color
    if fg_color is not None:
         rc['text.color']=fg_color
         #axes.labelcolor
         #axes.titlecolor
    #with plt.style.context(style if style is not None else "classic"): 

    if show_image_bar or show_overlay_bar:
        columns_add=1
    else:
        columns_add=0

    with matplotlib.rc_context(rc):
        if style is not None:
            plt.style.use(style)

        w, h = plt.figaspect(rows/(columns+columns_add))
        fig = plt.figure(figsize=(w,h))

        widths=[1.0/columns]*columns
        # for color bar
        if columns_add>0:
            widths+=[widths[0]/10]
        
        grid=gridspec.GridSpec(rows, (columns+columns_add),width_ratios=widths)
        #subplotspec=gridspec.new_subplotspec(loc, rowspan, colspan)
        ax=None
        imgplot=None
        
        for i,j in enumerate(slices):
            ax =  fig.add_subplot(grid[ i//columns, i%columns])
            imgplot = ax.imshow(j, origin='lower',  cmap=cm, aspect=aspects[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.title.set_visible(False)
            ax.set_frame_on(False)

        # for color bar
        if show_image_bar:
            ax =  fig.add_subplot(grid[ 1:(rows-1), columns])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.title.set_visible(False)
            ax.set_frame_on(True)
            cbar = fig.colorbar(scalarMap,cax=ax)

        if show_overlay_bar:
            ax =  fig.add_subplot(grid[ 1:(rows-1), columns])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.title.set_visible(False)
            ax.set_frame_on(False)
            cbar = fig.colorbar(oscalarMap)
        
        if title is not None:
            plt.suptitle(title,fontsize=7, y=0.99, x=0.05, 
                         ha='left',
                         va='top',
                         color='black' if fg_color is None else fg_color,
                         backgroundcolor='none',#'white' if bg_color is None else bg_color,
                         alpha=0.8)
            #plt.text(x=0.0,y=0.97,ha='left',va='top',s=title,color='black',backgroundcolor='white')
            #plt.subplots_adjust(wspace = 0.0 ,hspace=0.0)
            plt.subplots_adjust(top=1.0,bottom=0.0,left=0.0,right=1.0,wspace = 0.0 ,hspace=0.0)
        else:
            plt.subplots_adjust(top=1.0,bottom=0.0,left=0.0,right=1.0,wspace = 0.0 ,hspace=0.0)

        # , facecolor=fig.get_facecolor(), edgecolor='none')
        plt.savefig(output, bbox_inches='tight', dpi=dpi, format=format)
        plt.close()
        plt.close('all')


def qc_field_contour(
    input,
    output,
    image_range=None,
    title=None,
    image_cmap='gray',
    samples=6,
    show_image_bar=False, # TODO:implement this?
    dpi=100,
    format=None,
    bg_color=None,
    fg_color=None
    ):
    """show field contours
    """


    _img=minc2_file(input)
    _img.setup_standard_order()
    _idata=_img.load_complete_volume(minc2_file.MINC2_FLOAT)
    _idims=_img.representation_dims()
    
    data_shape=_idata.shape
    # order of dimensions in representation dimension is reversed
    spacing=[_idims[2].step, _idims[1].step, _idims[0].step]
    columns=samples//2
    rows=(samples//columns)*3
    
    slices=[]
    aspects = []

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
        i=int(10+(150.0-10.0)*j/(samples-1))
        i=int(data_shape[0]*i/181.0)

        si=_idata[i , : ,:]
        slices.append( si )
        aspects.append( spacing[1] / spacing[2] )
        
    for j in range(0, samples//2):
        i=int(28.0+(166.0-28.0)*j/(samples-1))
        i=int(data_shape[2]*i/193.0)
        si=_idata[: , : , i]
        slices.append( si )
        aspects.append( spacing[0] / spacing[1] )

    for j in range(samples-1, samples//2-1,-1):
        i=int(28.0+(166.0-28.0)*j/(samples-1))
        i=int(data_shape[2]*i/193.0)
        si=_idata[: , : , i]
        slices.append( si )
        aspects.append( spacing[0] / spacing[1] )

    for j in range(0,samples):
        i=int(25+(195.0-25.0)*j/(samples-1))
        i=int(data_shape[1]*i/217.0)
        si=_idata[: , i ,:]
        slices.append( si )
        aspects.append( spacing[0]/spacing[2] )
        
    rc={'interactive': False}
    rc['figure.frameon']=False
    if bg_color is not None:
         rc['figure.edgecolor']=bg_color
         rc['figure.facecolor']=bg_color     
         rc['grid.color']=bg_color
    if fg_color is not None:
         rc['text.color']=fg_color
         #axes.labelcolor
         #axes.titlecolor

    with matplotlib.rc_context(rc):

        w, h = plt.figaspect(rows/columns)
        fig = plt.figure(figsize=(w,h))
        
        #outer_grid = gridspec.GridSpec((len(slices)+1)/2, 2, wspace=0.0, hspace=0.0)
        ax=None
        imgplot=None
        for i,j in enumerate(slices):
            ax =  plt.subplot2grid( (rows, columns), ( i//columns, i%columns) )
            ax.set_aspect(aspects[i])
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
        plt.close()
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

plt.register_cmap(cmap=colors.LinearSegmentedColormap('spectral',
    {
        'red': [
            (0.0, 0.0, 0.0), (0.05, 0.4667, 0.4667),
            (0.10, 0.5333, 0.5333), (0.15, 0.0, 0.0),
            (0.20, 0.0, 0.0), (0.25, 0.0, 0.0),
            (0.30, 0.0, 0.0), (0.35, 0.0, 0.0),
            (0.40, 0.0, 0.0), (0.45, 0.0, 0.0),
            (0.50, 0.0, 0.0), (0.55, 0.0, 0.0),
            (0.60, 0.0, 0.0), (0.65, 0.7333, 0.7333),
            (0.70, 0.9333, 0.9333), (0.75, 1.0, 1.0),
            (0.80, 1.0, 1.0), (0.85, 1.0, 1.0),
            (0.90, 0.8667, 0.8667), (0.95, 0.80, 0.80),
            (1.0, 0.80, 0.80),
        ],
        'green': [
            (0.0, 0.0, 0.0), (0.05, 0.0, 0.0),
            (0.10, 0.0, 0.0), (0.15, 0.0, 0.0),
            (0.20, 0.0, 0.0), (0.25, 0.4667, 0.4667),
            (0.30, 0.6000, 0.6000), (0.35, 0.6667, 0.6667),
            (0.40, 0.6667, 0.6667), (0.45, 0.6000, 0.6000),
            (0.50, 0.7333, 0.7333), (0.55, 0.8667, 0.8667),
            (0.60, 1.0, 1.0), (0.65, 1.0, 1.0),
            (0.70, 0.9333, 0.9333), (0.75, 0.8000, 0.8000),
            (0.80, 0.6000, 0.6000), (0.85, 0.0, 0.0),
            (0.90, 0.0, 0.0), (0.95, 0.0, 0.0),
            (1.0, 0.80, 0.80),
        ],
        'blue': [
            (0.0, 0.0, 0.0), (0.05, 0.5333, 0.5333),
            (0.10, 0.6000, 0.6000), (0.15, 0.6667, 0.6667),
            (0.20, 0.8667, 0.8667), (0.25, 0.8667, 0.8667),
            (0.30, 0.8667, 0.8667), (0.35, 0.6667, 0.6667),
            (0.40, 0.5333, 0.5333), (0.45, 0.0, 0.0),
            (0.5, 0.0, 0.0), (0.55, 0.0, 0.0),
            (0.60, 0.0, 0.0), (0.65, 0.0, 0.0),
            (0.70, 0.0, 0.0), (0.75, 0.0, 0.0),
            (0.80, 0.0, 0.0), (0.85, 0.0, 0.0),
            (0.90, 0.0, 0.0), (0.95, 0.0, 0.0),
            (1.0, 0.80, 0.80),
        ]
    }))


if __name__ == '__main__':
    pass
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 14/08/2015
#
# Longitudinal pipeline resampling

import shutil
import os
import sys
import csv
import traceback

# MINC stuff
from ipl.minc_tools import mincTools,mincError

def warp_scan(sample, reference, output_scan, transform=None, parameters={},corr_xfm=None):
    with mincTools() as m:  
        xfm=None
        xfms=[]

        if corr_xfm is not None:
            xfms.append(corr_xfm.xfm)
        if transform is not None:
            xfms.append(transform.xfm)

        if len(xfms)==0:
            pass
        if len(xfms)==1:
            xfm=transform.xfm
        else:
            m.xfmconcat(xfms,m.tmp('concatenated.xfm'))
            xfm=m.tmp('concatenated.xfm')

        resample_order=parameters.get('resample_order',4)
        
        m.resample_smooth(sample.scan, output_scan.scan, 
                          transform=xfm, like=reference.scan, 
                          order=resample_order)


def warp_mask(sample, reference, output_scan, transform=None, parameters={},corr_xfm=None):
    with mincTools() as m:  
        xfm=None
        xfms=[]
        
        if corr_xfm is not None:
            xfms.append(corr_xfm.xfm)
        if transform is not None:
            xfms.append(transform.xfm)
        
        if len(xfms)==0:
            pass
        if len(xfms)==1:
            xfm=transform.xfm
        else:
            m.xfmconcat(xfms,m.tmp('concatenated.xfm'))
            xfm=m.tmp('concatenated.xfm')

        resample_order=parameters.get('resample_order',4)
        m.resample_labels(sample.mask, output_scan.mask, transform=xfm, like=reference.scan, order=resample_order)


def warp_cls_back(t1w_tal, tal_cls, t1w_tal_xfm,reference, native_t1w_cls, parameters={},corr_xfm=None):
    with mincTools() as m:  
        resample_order=parameters.get('resample_order',0)
        resample_baa  =parameters.get('resample_baa',False)

        xfm=t1w_tal_xfm.xfm
        if corr_xfm is not None:
            m.xfmconcat([corr_xfm.xfm,t1w_tal_xfm.xfm],m.tmp('concatenated.xfm'))
            xfm=m.tmp('concatenated.xfm')
            
        
        m.resample_labels(tal_cls.scan, native_t1w_cls.scan, 
                          transform=xfm, 
                          like=reference.scan, 
                          order=resample_order,
                          baa=resample_baa,
                          invert_transform=True)

def warp_mask_back(t1w_tal, t1w_tal_xfm, reference, native_t1w_cls, parameters={},corr_xfm=None):
    with mincTools() as m:  
        resample_order=parameters.get('resample_order',0)
        resample_baa  =parameters.get('resample_baa',False)

        xfm=t1w_tal_xfm.xfm
        if corr_xfm is not None:
            m.xfmconcat([corr_xfm.xfm,t1w_tal_xfm.xfm],m.tmp('concatenated.xfm'))
            xfm=m.tmp('concatenated.xfm')
            
        m.resample_labels(t1w_tal.mask, native_t1w_cls.mask, 
                          transform=xfm, 
                          like=reference.scan, 
                          order=resample_order,
                          baa=resample_baa,
                          invert_transform=True)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

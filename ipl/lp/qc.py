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
from ipl.minc_qc    import qc,qc_field_contour


def draw_qc_stx(in_scan,in_outline,out_qc,options={}):
    qc(in_scan.scan,out_qc.fname,
        mask=in_outline.scan,
        mask_range=[0.0,1.0],samples=20 if options.get('big',False) else 6,
        mask_bg=0.5, use_max=True,
        bg_color=options.get('bg','black'),fg_color=options.get('fg','white'))

def draw_qc_nl(in_scan, in_outline, nl_xfm, qc_nl, options={}):
    with mincTools() as m:
        m.resample_labels(in_outline.scan,m.tmp('outline.mnc'),
            transform=nl_xfm.xfm, invert_transform=True)
        qc(in_scan.scan,qc_nl.fname,
            mask=m.tmp('outline.mnc'),
            mask_range=[0.0,1.0],samples=20 if options.get('big',False) else 6,
            mask_bg=0.5, use_max=True,
            bg_color=options.get('bg','black'),fg_color=options.get('fg','white'))


def draw_qc_mask(in_scan,out_qc,options={}):
    qc(in_scan.scan,out_qc.fname,
        mask=in_scan.mask,
        mask_range=[0.0,1.0],samples=20 if options.get('big',False) else 6,
        mask_bg=0.5, use_max=True,
        bg_color=options.get('bg','black'),fg_color=options.get('fg','white'))

def draw_qc_cls(in_scan,in_cls,out_qc,options={}):
    qc(in_scan.scan,out_qc.fname,
        mask=in_cls.scan,
        mask_range=[0.0,3.5],samples=20 if options.get('big',False) else 6,
        mask_cmap='spectral',
        mask_bg=0.5, use_max=True,
        bg_color=options.get('bg','black'),fg_color=options.get('fg','white'))


def draw_qc_lobes(in_scan,in_lobes,out_qc,options={}):
    qc(in_scan.scan,out_qc.fname,
        mask=in_lobes.scan,samples=20 if options.get('big',False) else 6,
        mask_cmap='spectral',
        mask_bg=0.5, use_max=True,
        bg_color=options.get('bg','black'),fg_color=options.get('fg','white'))


def draw_qc_add(in_scan1,in_scan2,out_qc,options={}):
    qc(in_scan1.scan,out_qc.fname,
        mask=in_scan2.scan,samples=20 if options.get('big',False) else 6,
        image_cmap='red',
        mask_cmap='green',
        mask_bg=0.5, use_max=True,
        bg_color=options.get('bg','black'),fg_color=options.get('fg','white'))

def draw_qc_nu(in_field,out_qc,options={}):
    qc_field_contour(in_field.scan,out_qc.fname,
        image_cmap='jet',samples=20 if options.get('big',False) else 6,
        bg_color=options.get('bg','black'),fg_color=options.get('fg','white'))

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

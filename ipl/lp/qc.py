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
    if options.get('big'):
        with mincTools() as m:
            m.qc(in_scan.scan,out_qc.fname,
                 big=True,mask=in_outline.scan,
                 mask_range=[0.0,1.0])
    else:
        qc(in_scan.scan,out_qc.fname,
            mask=in_outline.scan,
            mask_range=[0.0,1.0],
            mask_bg=0.5, use_max=True)


def draw_qc_mask(in_scan,out_qc,options={}):
    if options.get('big'):
        with mincTools() as m:
            m.qc(in_scan.scan,out_qc.fname,
                 big=True,mask=in_scan.mask,
                 mask_range=[0.0,1.0])
    else:
        qc(in_scan.scan,out_qc.fname,
            mask=in_scan.mask,
            mask_range=[0.0,1.0],
            mask_bg=0.5, use_max=True)

def draw_qc_cls(in_scan,in_cls,out_qc,options={}):
    if options.get('big'):
        with mincTools() as m:
            m.qc(in_scan.scan,out_qc.fname,
                 big=True,mask=in_cls.scan,
                 mask_range=[0.0,3.5],
                 spectral_mask=True)
    else:
        qc(in_scan.scan,out_qc.fname,
            mask=in_cls.scan,
            mask_range=[0.0,3.5],
            mask_cmap='spectral',
            mask_bg=0.5, use_max=True)


def draw_qc_lobes(in_scan,in_lobes,out_qc,options={}):
    if options.get('big'):
        with mincTools() as m:
            m.qc(in_scan.scan,out_qc.fname,
                 big=True,mask=in_lobes.scan,
                 spectral_mask=True)
    else:
        qc(in_scan.scan,out_qc.fname,
            mask=in_lobes.scan,
            mask_cmap='spectral',
            mask_bg=0.5, use_max=True)


def draw_qc_add(in_scan1,in_scan2,out_qc,options={}):
    if options.get('big'):
        with mincTools() as m:
            m.qc(in_scan1.scan,out_qc.fname,
                 big=True,red=True,
                 mask=in_scan2.scan,
                 green_mask=True)
    else:
        qc(in_scan1.scan,out_qc.fname,
            mask=in_scan2.scan,
            image_cmap='red',
            mask_cmap='green',
            mask_bg=0.5, use_max=True)

def draw_qc_nu(in_field,out_qc,options={}):
    qc_field_contour(in_field.scan,out_qc.fname,
        image_cmap='jet')


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

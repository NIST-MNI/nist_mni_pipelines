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
#from ipl.minc_qc    import qc,qc_field_contour



def make_aqc_nu(t1w_field,aqc_nu,options={}):
    pass

def make_aqc_stx(t1w_tal,model_outline,aqc_tal,options={}):
    with mincTools() as m:
        m.aqc(t1w_tal.scan, aqc_tal.fname,
              slices=options.get("slices",3))

def make_aqc_add(t1w_tal,add_tal,aqc,options={}):
    pass

def make_aqc_mask(t1w_tal,aqc_mask,options={}):
    pass

def make_aqc_cls(t1w_tal,tal_cls,aqc_cls,options={}):
    pass

def make_aqc_lobes( t1w_tal, tal_lob,aqc_lob,options={}):
    pass
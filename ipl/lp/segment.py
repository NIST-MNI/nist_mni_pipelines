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

from .registration import nl_registration

def extract_brain_beast(scan, parameters={},model=None):
    """extract brain using BEaST """
    with mincTools() as m:
         # TODO: come up with better default?
        beast_lib=parameters.get('beastlib','/opt/minc/share/beast-library-1.1')
        beast_res=parameters.get('resolution',2)
        beast_mask=beast_lib+os.sep+'union_mask.mnc'
        
        if m.checkfiles(inputs=[scan.scan], outputs=[scan.mask]):
            tmp_in=m.tmp('like_beast.mnc')
            m.resample_smooth(scan.scan,tmp_in,like=beast_mask)
            # run additional intensity normalizaton
            if parameters.get('normalize',True) and model is not None:
                m.volume_pol(tmp_in,model.scan,m.tmp('like_beast_norm.mnc'))
                tmp_in=m.tmp('like_beast_norm.mnc')
            
            # run beast
            beast_v10_template = beast_lib + os.sep \
                + 'intersection_mask.mnc'
            beast_v10_margin = beast_lib + os.sep + 'margin_mask.mnc'

            beast_v10_intersect = beast_lib + os.sep \
                + 'intersection_mask.mnc'

            # perform segmentation
            m.run_mincbeast(tmp_in,m.tmp('beast_mask.mnc'),
                            beast_lib=beast_lib, beast_res=beast_res)
        
            m.resample_labels(m.tmp('beast_mask.mnc'),scan.mask,like=scan.scan)

def extract_brain_nlreg(scan, parameters={},model=None):
    """extract brain using non-linear registration to the template"""
    with mincTools() as m:
        if m.checkfiles(inputs=[scan.scan], outputs=[scan.mask]):
            tmp_xfm=MriTransform(prefix=m.tempdir, name='nl_'+scan.name)
            nl_registration(scan, model, tmp_xfm, parameters=parameters)
            # warp template atlas to subject's scan
            m.resample_labels(model.mask,scan.mask, transform=tmp_xfm.xfm, invert_transform=True)


def classify_tissue(scan, cls,
                    model_name=None, 
                    model_dir=None, 
                    parameters={}, 
                    xfm=None ):
    """Tissue classification
    """
    with mincTools() as m:
        m.classify_clean([scan.scan], cls.scan, 
                         mask=scan.mask, model_dir=model_dir,
                         model_name=model_name,xfm=xfm.xfm)


def segment_lobes(tal_cls,nl_xfm, tal_lob, model=None, lobe_atlas_dir=None, 
                    parameters={}):
    """Lobe segmentation
    """
    with mincTools() as m:
        m.lobe_segment(tal_cls.scan,tal_lob.scan,
                       nl_xfm=nl_xfm.xfm,template=model.scan,
                       atlas_dir=lobe_atlas_dir)
        

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

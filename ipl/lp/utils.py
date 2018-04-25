# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 14/08/2015
#
# Longitudinal pipeline preprocessing

import shutil
import os
import sys
import csv
import traceback
import json

# MINC stuff
from ipl.minc_tools import mincTools,mincError


def create_dirs(dirs):
    for i in dirs:
        if not os.path.exists(i):
            os.makedirs(i)


def xfm_remove_scale(in_xfm,out_xfm,unscale=None):
    """remove scaling factors from linear XFM
    
    """
    _unscale=None
    if unscale is not None:
        _unscale=unscale.xfm
        
    with mincTools() as minc:
        minc.xfm_noscale(in_xfm.xfm,out_xfm.xfm,unscale=_unscale)


def xfm_concat(in_xfms,out_xfm):
    """Concatenate multiple transforms
    
    """
    with mincTools() as minc:
        minc.xfmconcat([ i.xfm for i in in_xfms],out_xfm.xfm)



def extract_volumes(in_lob, in_cls, tal_xfm, out, 
                    produce_json=False,
                    subject_id=None,
                    timepoint_id=None,
                    lobedefs=None):
    """Convert lobe segmentation to volumetric measurements
    
    """
    with mincTools() as minc:
        vol_lobes= minc.label_stats( in_lob.scan, label_defs=lobedefs )
        vol_cls  = minc.label_stats( in_cls.scan )
        params=minc.xfm2param(tal_xfm.xfm)
        vol_scale=params['scale'][0]*params['scale'][1]*params['scale'][2]
        
        volumes={ k[0]:k[1]*vol_scale for k in vol_lobes }
        _vol_cls  = { k[0]: k[1]*vol_scale for k in vol_cls }
        # TODO: figure out what to do when keys are missing, i.e something is definetely wrong
        volumes['CSF']=_vol_cls.get(1,0.0)
        volumes['GM']=_vol_cls.get(2,0.0)
        volumes['WM']=_vol_cls.get(3,0.0)
        
        volumes['ICC']=volumes['CSF']+volumes['GM']+volumes['WM']
        
        if subject_id is not None:
            volumes['id']=subject_id
        
        if timepoint_id is not None:
            volumes['timepoint']=timepoint_id
        
        volumes['scale']=vol_scale
        
        # save either as text file or json
        with open(out.fname,'w') as f:
            if produce_json:
                json.dump(volumes,f,indent=1)
            else:
                for i,j in volumes.items():
                    f.write("{} {}\n".format(i,j))

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

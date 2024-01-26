# -*- coding: utf-8 -*-

#
# @author Daniel
# @date 10/07/2011

version = '1.0'

#
# Atlas registration
#

from .general import *
from ipl.minc_tools import mincTools,mincError
from ipl import minc_qc

# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_concat(patient, tp):

    if not os.path.exists(patient[tp].nl_xfm):
        concat_v10(patient, tp)  # beast by simon fristed

    with mincTools()  as minc:
        # create QC images
        atlas_outline = patient.modeldir + os.sep + patient.modelname + '_outline.mnc'

        if not os.path.exists(patient[tp].qc_jpg['nl_t1']):
            minc.resample_smooth(patient[tp].stx2_mnc['t1'],minc.tmp('nl_stx_t1.mnc'),transform=patient[tp].nl_xfm)
            minc_qc.qc(
                minc.tmp('nl_stx_t1.mnc'),
                patient[tp].qc_jpg['nl_t1'],
                title=patient[tp].qc_title,
                image_range=[0, 120],
                mask=atlas_outline,dpi=200,use_max=True,
                samples=20,bg_color="black",fg_color="white"
            )
    return True
    


def concat_v10(patient, tp):
    
    with mincTools()  as minc:
        tmp_xfm = minc.tmp('tmp_concat.xfm')
        minc.xfmconcat([patient[tp].lng_xfm['t1'], patient.nl_xfm] , tmp_xfm)

        examplegrid = patient.nl_xfm[:-4] + '_grid_0.mnc'
        # TODO: use short for storing deformation fields
        # TODO: need to reimplement xfm_normalize in python
        
        minc.xfm_normalize(tmp_xfm, examplegrid, patient[tp].nl_xfm, exact=True)
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

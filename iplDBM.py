#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author Nicolas
# @date 21/03/2013
#
# Longitudinal deformation based morphometry
#

from iplGeneral import *
from iplMincTools import *

version = '1.0'



# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_lngDBM(patient, tp=None):

    if patient.pipeline_version == '1.0':
        pipeline_lngDBM_v10(patient, tp)
    else:
        print(' -- Chosen version not found!')

    # # Write qc images
    # ################
    with mincTools() as minc:
        minc.qc(
            patient.template['nl_template'],
            patient[tp].qc_jpg['lng_det'],
            title=patient[tp].qc_title,
            image_range=[0, 120],
            big=True,
            clamp=True,
            mask=patient[tp].lng_det['t1'],
            cyanred_mask=True )


def pipeline_lngDBM_v10(patient, tp):
    
    with mincTools() as minc:
        tmp_det = minc.tmp(tp + '_det.mnc')
        tmp_det_res = minc.tmp(tp + '_det_res.mnc')

        if not os.path.exists(patient[tp].lng_det['t1']):
            minc.grid_determinant(patient[tp].lng_igrid['t1'], tmp_det)
            
            minc.resample_smooth(tmp_det, tmp_det_res,
                                like=patient[tp].stx2_mnc['t1'],
                                resample='linear')
            
            minc.calc([tmp_det_res], '(log(1+(A[0])))',
                    patient[tp].lng_det['t1'])
        else:
            print(' -- pipeline_lngDBM is done!')


if __name__ == '__main__':
    pass

  # Using script as a stand-alone script

  # set options - the options should keep same names than in pipeline

  # use -->  runProcessing

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

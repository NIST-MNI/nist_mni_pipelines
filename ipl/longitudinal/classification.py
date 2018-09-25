# -*- coding: utf-8 -*-

#
# @author Daniel, Vladimir S. FONOV, Simon Eskildsen
# @date 10/07/2011

version = '1.0'

#
# Atlas registration
#

from .general import *
from ipl.minc_tools import mincTools,mincError


# Run preprocessing using patient info
# - Function to read info from the pipeline patient

def pipeline_classification(patient, tp):
    if os.path.exists(patient[tp].stx2_mnc['classification']) \
        and os.path.exists(patient[tp].qc_jpg['classification']):
        print(' -- Classification - Processing already done!')
    else:
        classification_v10(patient, tp)  # beast by simon fristed

    with mincTools() as minc:
        minc.qc(patient[tp].stx2_mnc['t1'],patient[tp].qc_jpg['classification'],
                title=patient[tp].qc_title, image_range=[0,120],
                mask=patient[tp].stx2_mnc['classification'],labels_mask=True,
                big=True,clamp=True   )
    return True


def classification_v10(patient, tp):
    with mincTools() as minc:  # TODO: convert to using mincTools calls
        scans= [patient[tp].stx2_mnc['t1']]
        if 't2' in patient[tp].stx2_mnc and not patient.onlyt1:
            scans.append(patient[tp].stx2_mnc['t2'])
        if 'pd' in patient[tp].stx2_mnc and not patient.onlyt1:
            scans.append(patient[tp].stx2_mnc['pd'])
            
        minc.classify_clean(scans,patient[tp].stx2_mnc['classification'],
                            mask=patient[tp].stx2_mnc['masknoles'],
                            xfm=patient[tp].nl_xfm,
                            model_name=patient.modelname,
                            model_dir=patient.modeldir)
    return 0


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

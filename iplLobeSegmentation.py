#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author Daniel, Vladimir S. FONOV, Simon Eskildsen
# @date 10/07/2011
#
# Atlas registration
#

from iplGeneral import *
from ipl.minc_tools import mincTools,mincError


# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_lobe_segmentation(patient, tp):
    if os.path.exists(patient[tp].stx2_mnc['lobes']) \
        and os.path.exists(patient[tp].vol['lobes']) \
        and os.path.exists(patient[tp].qc_jpg['lobes']):
        print(' -- Lobe Segmentation - Processing already done!')
    else:
        lobe_segmentation_v10(patient, tp)  # beast by simon fristed

    # lobes qc
    with mincTools() as minc:
        minc.qc(patient[tp].stx2_mnc['t1'],patient[tp].qc_jpg['lobes'],
                title=patient[tp].qc_title, image_range=[0,120],
                mask=patient[tp].stx2_mnc['lobes'],labels_mask=True,
                big=True,clamp=True   )

    return True


def lobe_segmentation_v10(patient, tp):

    # # doing the processing
    # ######################
    with mincTools()  as minc:

        identity = minc.tmp('identity.xfm')
        if not os.path.exists(patient[tp].stx2_mnc['lobes']):
            comm = ['param2xfm', identity]
            minc.command(comm, [], [identity])
        cls = ''

        # Do lobe segment
        if patient.dolngcls and len(list(patient.keys())) > 1:
            cls = patient[tp].stx2_mnc['lng_classification']
        else:
            cls = patient[tp].stx2_mnc['classification']

        comm = [
            'lobe_segment',
            patient[tp].nl_xfm,
            identity,
            cls,
            patient[tp].stx2_mnc['lobes'],
            '-modeldir', patient.modeldir + os.sep + patient.modelname + '_atlas/',
            '-template', patient.modeldir + os.sep + patient.modelname + '.mnc',
            ]

        minc.command(comm, [patient[tp].nl_xfm, cls],
                     [patient[tp].stx2_mnc['lobes']])

        # Compute volumes
        # Classify brain into 3 classes
        # TODO: replace with direct call to lobes_to_volumes.pl
        comm = [    
            'pipeline_volumes_nl.pl',
            patient[tp].stx2_mnc['masknoles'],
            cls,
            patient[tp].stx2_mnc['lobes'],
            patient[tp].stx2_xfm['t1'],
            patient[tp].vol['lobes'],
            '--age', str(patient[tp].age),
            '--t1',patient[tp].native['t1']
            ]
        if len(patient.sex) > 0:
            comm.extend(['--gender', patient.sex])
        
        if 't2' in patient[tp].native:
            comm.extend(['--t2', patient[tp].native['t2']])
            
        if 'pd' in patient[tp].native:
            comm.extend(['--pd', patient[tp].native['pd']])
        
        minc.command(comm, [ patient[tp].stx2_mnc['masknoles'],
                   patient[tp].stx2_mnc['classification'],
                   patient[tp].stx2_mnc['lobes'],
                   patient[tp].stx2_xfm['t1']],
                   [patient[tp].vol['lobes']])
    return 0

if __name__ == '__main__':
    pass

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

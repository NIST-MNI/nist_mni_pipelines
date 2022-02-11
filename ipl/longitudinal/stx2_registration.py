#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author Daniel
# @date 10/07/2011


#
# Atlas registration
#

from .general import *
from ipl.minc_tools import mincTools,mincError
from ipl import minc_qc

import ipl.registration
import ipl.ants_registration
import ipl.elastix_registration


version = '1.0'

# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_linearatlasregistration(patient, tp):

    if os.path.exists(patient[tp].stx2_xfm['t1']) \
        and os.path.exists(patient[tp].stx2_mnc['t1']):  # and os.path.exists(patient[tp].stx2_mnc["t2"])
        print(' -- pipeline_linearatlasregistration exists')
        return True

    if patient.pipeline_version == '1.0':
        linearatlasregistration_v10(patient, tp)  # beast by simon fristed
    else:
        print(' -- Chosen version not found!')

def linearatlasregistration_v10(patient, tp):

    # crossectional versions
    # just copy the original image as the patient linear template

    
    with mincTools() as minc:

        # assigning the templates, to the original image.

        patient.template['linear_template'] = \
            patient[tp].stx_ns_mnc['t1']
        patient.template['linear_template_mask'] = \
            patient[tp].stx_ns_mnc['masknoles']

        # atlas

        template = patient.modeldir + os.sep + patient.modelname + '.mnc'
        template_mask = patient.modeldir + os.sep + patient.modelname \
            + '_mask.mnc'

        # register ns_stx into the atlas
        ipl.registration.linear_register(patient[tp].stx_ns_mnc['t1'], template,
                             patient.template['stx2_xfm'],
                             source_mask=patient[tp].stx_ns_mnc['masknoles'],
                             target_mask=template_mask)

        # 1. concatenate all transforms
        minc.xfmconcat([patient[tp].stx_ns_xfm['t1'],
                       patient.template['stx2_xfm']],
                       patient[tp].stx2_xfm['t1'])

        # run 2nd intensity normalization and nonuniformity correction
        minc.nu_correct(patient[tp].clp['t1'], 
                        output_image=minc.tmp('clp2_t1.mnc'), 
                        mask=patient[tp].clp['mask'], 
                        mri3t=patient.mri3T )

        minc.volume_pol(
            minc.tmp('clp2_t1.mnc'),
            template,
            patient[tp].clp2['t1'],
            source_mask=patient[tp].clp['mask'],
            target_mask=template_mask,
            datatype='-short' )

        # reformat image into stx2
        minc.resample_smooth(patient[tp].clp2['t1'],
                             patient[tp].stx2_mnc['t1'],
                             transform=patient[tp].stx2_xfm['t1'],
                             like=template_mask)

        # 1bis. concatenate all transforms t2 BB
        if 't2' in patient[tp].native:

            templatet2 = template.replace('t1', 't2')
            minc.nu_correct(patient[tp].clp['t2'], 
                            output_image=minc.tmp('clp2_t2.mnc'), 
                            mask=patient[tp].clp['mask'], 
                            mri3t=patient.mri3T )

            minc.volume_pol(
                minc.tmp('clp2_t2.mnc'),
                templatet2,
                patient[tp].clp2['t2'],
                source_mask=patient[tp].clp['mask'],
                target_mask=template_mask,
                datatype='-short' )

            minc.xfmconcat([patient[tp].clp['t2t1xfm'],
                           patient[tp].stx_ns_xfm['t1'],
                           patient.template['stx2_xfm']],
                           patient[tp].stx2_xfm['t2'])

            # reformat native t2, pd and lesions images into stx2 BB
            minc.resample_smooth(patient[tp].clp2['t2'],
                                 patient[tp].stx2_mnc['t2'],
                                 transform=patient[tp].stx2_xfm['t2'],
                                 like=template_mask)

        if 'pd' in patient[tp].native:
            templatepd = template.replace('t1', 'pd')
            minc.nu_correct(patient[tp].clp['pd'], 
                            output_image=minc.tmp('clp2_pd.mnc'), 
                            mask=patient[tp].clp['mask'], 
                            mri3t=patient.mri3T )

            minc.volume_pol(
                minc.tmp('clp2_pd.mnc'),
                templatet2,
                patient[tp].clp2['pd'],
                source_mask=patient[tp].clp['mask'],
                target_mask=template_mask,
                datatype='-short' )

            minc.xfmconcat([patient[tp].clp['pdt1xfm'],
                           patient[tp].stx_ns_xfm['t1'],
                           patient.template['stx2_xfm']],
                           patient[tp].stx2_xfm['pd'])
            minc.resample_smooth(patient[tp].clp2['pd'],
                                 patient[tp].stx2_mnc['pd'],
                                 transform=patient[tp].stx2_xfm['pd'],
                                 like=template_mask)

        if 't2les' in patient[tp].native:
            minc.resample_labels(patient[tp].native['t2les'],
                                 patient[tp].stx2_mnc['t2les'],
                                 transform=patient[tp].stx2_xfm['t2'],
                                 like=template_mask)


if __name__ == '__main__':
    pass

  # Using script as a stand-alone script

  # set options - the options should keep same names than in pipeline

  # use -->  runProcessing

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

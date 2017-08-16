#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# @author Daniel
# @date 10/07/2011

version = '1.0'

#
# Atlas registration
#
from ipl.minc_tools import mincTools,mincError

from iplGeneral import *



# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline
def pipeline_atlasregistration(patient, tp=None):
    if os.path.exists(patient.nl_xfm):
        print(' -- pipeline_atlasregistration exists')
        return True
    return atlasregistration_v10(patient)


def atlasregistration_v10(patient):

    nl_level = 2

    if patient.fast:  # fast mode
        nl_level = 4

    with mincTools() as minc:
        model_t1   = patient.modeldir + os.sep + patient.modelname + '.mnc'
        model_mask = patient.modeldir + os.sep + patient.modelname + '_mask.mnc'

        minc.non_linear_register_full(
            patient.template['nl_template'],
            model_t1,
            patient.nl_xfm,
            source_mask=patient.template['nl_template_mask'],
            target_mask=model_mask,
            level=nl_level,
            )


if __name__ == '__main__':
    pass

  # Using script as a stand-alone script

  # set options - the options should keep same names than in pipeline

  # use -->  runProcessing

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

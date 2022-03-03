# -*- coding: utf-8 -*-

#
# @author Daniel
# @date 10/07/2011

version = '1.0'

#
# Atlas registration
#
from ipl.minc_tools import mincTools,mincError
from ipl import minc_qc
import ipl.registration
import ipl.ants_registration
import ipl.elastix_registration

from .general import *



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

    print("nl_method:",patient.nl_method)
    with mincTools() as minc:
        model_t1   = patient.modeldir + os.sep + patient.modelname + '.mnc'
        model_mask = patient.modeldir + os.sep + patient.modelname + '_mask.mnc'

        if patient.nl_method == 'nlfit_s' or patient.nl_method == 'minctracc':
            ipl.registration.non_linear_register_full(
                patient.template['nl_template'],
                model_t1,
                patient.nl_xfm,
                source_mask=patient.template['nl_template_mask'],
                target_mask=model_mask,
                level=nl_level,
                )
        elif patient.nl_method == 'ANTS' or patient.nl_method == 'ants': # ANTs ?
            if patient.fast:  # fast mode
                ipl.ants_registration.non_linear_register_ants2(
                        patient.template['nl_template'], model_t1,
                        patient.nl_xfm,
                        source_mask=patient.template['nl_template_mask'],
                        target_mask=model_mask,
                        start=16,
                        level=4,
                        parameters={'transformation':'SyN[.7,3,0]',
                                    'conf':  {16:50,8:50,4:20,2:20,1:20},
                                    'shrink':{16:16,8:8,4:4,2:2,1:1},
                                    'blur':  {16:12,8:6,4:3,2:1,1:0},
                                    'winsorize-image-intensities':True,
                                    'convergence':'1.e-7,10',
                                    'cost_function':'CC',
                                    'use_float': True,
                                    'cost_function_par':'1,3,Regular,1.0'}
                    )
            else:
                ipl.ants_registration.non_linear_register_ants2(
                        patient.template['nl_template'], model_t1,
                        patient.nl_xfm,
                        source_mask=patient.template['nl_template_mask'],
                        target_mask=model_mask,
                        start=16,
                        level=1,
                        parameters={'transformation':'SyN[.7,3,0]',
                                    'conf':  {16:50,8:50,4:50,2:50,1:50},
                                    'shrink':{16:16,8:8,4:4,2:2,1:1},
                                    'blur':  {16:12,8:6,4:3,2:1,1:0},
                                    'winsorize-image-intensities':True,
                                    'convergence':'1.e-7,10',
                                    'cost_function':'CC',
                                    'use_float': True,
                                    'cost_function_par':'1,3,Regular,1.0'}
                    )

        else:
            ipl.elastix_registration.register_elastix(
                    patient.template['nl_template'], model_t1,
                    patient.nl_xfm,
                    source_mask=patient.template['nl_template_mask'],
                    target_mask=model_mask,
                    nl=True )
        
        # make QC image, similar to linear ones
        if not os.path.exists(patient.qc_jpg['nl_template_nl']):
            atlas_outline = patient.modeldir + os.sep + patient.modelname + '_outline.mnc'
            minc.resample_smooth(patient.template['nl_template'],minc.tmp('nl_atlas.mnc'),transform=patient.nl_xfm)
            minc_qc.qc(
                minc.tmp('nl_atlas.mnc'),
                patient.qc_jpg['nl_template_nl'],
                title=patient.id,
                image_range=[0, 120],
                samples=20,
                dpi=200,
                mask=atlas_outline,use_max=True,
                bg_color='black',fg_color='white'
                )
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

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

import ray

@ray.remote(num_cpus=1, memory=1000 * 1024 * 1024) # uses about 10GB of RAM
def run_ants_registration(*args,**kwargs):
    n_threads=int(ray.runtime_context.get_runtime_context().get_assigned_resources()["CPU"])
    OLD_ITK_THREADS=os.environ.get('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS',None)
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']=str(n_threads)

    r=ipl.ants_registration.non_linear_register_ants2(*args,**kwargs)

    if OLD_ITK_THREADS is not None:
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']=OLD_ITK_THREADS
    return r


# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline
def pipeline_atlasregistration(patient, tp=None):
    
    if os.path.exists(patient.nl_xfm):
        return True
    return atlasregistration_v10(patient)


def atlasregistration_v10(patient):

    nl_level = patient.nl_step

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

            par={'convergence':'1.e-8,20',
                                'conf':{"32":200,"16":200,"8":100,"4":100,"2":50,"1":50},
                                'blur':{"32":24, "16":12, "8":6,  "4":2,  "2":1,"1":0.5},
                                'cost_function':'CC',
                                'cost_function_par':'1,3,Regular,1.0',
                                'transformation': 'SyN[0.1,3,0.0]',
                                'convert_grid_type': 'short'
                }
            if patient.nl_cost_fun == 'CC':
                par['cost_function']='CC'
                par['cost_function_par']='1,3,Regular,1.0'
            elif patient.nl_cost_fun == 'MI':
                par['cost_function']='MI'
                par['cost_function_par']='1,32,Regular,1.0'
            elif patient.nl_cost_fun == 'Mattes':
                par['cost_function']='Mattes'
                par['cost_function_par']='1,32,Regular,1.0'
            else:
                pass
            
            run_ants_registration_c=run_ants_registration.options(num_cpus=patient.threads)
            ray.get(run_ants_registration_c.remote(
                    patient.template['nl_template'], model_t1,
                    patient.nl_xfm,
                    source_mask=patient.template['nl_template_mask'],
                    target_mask=model_mask,
                    level=nl_level,
                    start=32,
                    parameters=par
            ))
        else:
            ipl.elastix_registration.register_elastix(
                    patient.template['nl_template'], model_t1,
                    patient.nl_xfm,
                    source_mask=patient.template['nl_template_mask'],
                    target_mask=model_mask,
                    nl=True )
        
        # make QC image, similar to linear ones
        if not os.path.exists(patient.qc_jpg['nl_template_nl']):

            modeloutline = patient.modeldir + os.sep + patient.modelname + '_brain_skull_outline.mnc'
            outline_range=[0,2]

            if not os.path.exists(modeloutline):
                modeloutline = patient.modeldir + os.sep + patient.modelname + '_outline.mnc'
                outline_range=[0,1]

            minc.resample_smooth(patient.template['nl_template'],minc.tmp('nl_atlas.mnc'),transform=patient.nl_xfm)
            minc_qc.qc(
                minc.tmp('nl_atlas.mnc'),
                patient.qc_jpg['nl_template_nl'],
                title=patient.id,
                image_range=[0, 120],
                samples=20,
                dpi=200,
                mask=modeloutline,
                mask_range=outline_range,
                use_max=True,
                bg_color='black',fg_color='white'
                )
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

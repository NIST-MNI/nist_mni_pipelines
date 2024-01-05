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
from ipl import minc_qc

from ipl  import bison
from threadpoolctl import threadpool_limits
import ray

# Run preprocessing using patient info
# - Function to read info from the pipeline patient

@ray.remote(num_cpus=4, memory=10000 * 1024 * 1024) # uses about 10GB of RAM
def run_bison(*args,**kwargs):
    #bison.infer(*args)
    with threadpool_limits(limits=4):
        bison.infer(*args,**kwargs)



def pipeline_classification(patient, tp):
    if os.path.exists(patient[tp].stx2_mnc['classification']) \
        and os.path.exists(patient[tp].qc_jpg['classification']):
        pass
    else:
        classification_v10(patient, tp)  # beast by simon fristed

    minc_qc.qc(patient[tp].stx2_mnc['t1'],patient[tp].qc_jpg['classification'],
                title=patient[tp].qc_title, image_range=[0,120],
                mask=patient[tp].stx2_mnc['classification'],dpi=200,
                samples=20,use_max=True,bg_color="black",fg_color="white" )
    return True


def classification_v10(patient, tp):

    # first run WMH classification , if possible
    if patient.wmh_bison_atlas_pfx is not None:
        #
        # generate input list in bison format
        wmh_bison_input={
            'subject':[patient.id+'_'+tp],
            't1':     [patient[tp].stx2_mnc['t1']],
            'xfm':    [patient[tp].nl_xfm],
            'mask':   [patient[tp].stx2_mnc['masknoles']],
            'output': [patient[tp].stx2_mnc['wmh']],
        }

        # FOR now, use only T1, even though it is bad
        # if 't2' in patient[tp].stx2_mnc and not patient.onlyt1:
        #     wmh_bison_input['t2']=[patient[tp].stx2_mnc['t2']]
        # if 'pd' in patient[tp].stx2_mnc and not patient.onlyt1:
        #     wmh_bison_input['pd']=[patient[tp].stx2_mnc['pd']]
        ray.get(run_bison.remote(wmh_bison_input, n_cls=1, n_jobs=4, batch=1,
                        load_pfx=patient.wmh_bison_pfx,
                        atlas_pfx=patient.wmh_bison_atlas_pfx,
                        method=patient.wmh_bison_method,
                        resample=True, inverse_xfm=True,
                        progress=True))

    if patient.bison_atlas_pfx is not None:
        # TODO: exclude WMH from the mask ?
        # generate input list in bison format
        bison_input={
            'subject': [patient.id+'_'+tp],
            't1':      [patient[tp].stx2_mnc['t1']],
            'xfm':     [patient[tp].nl_xfm],
            'mask':    [patient[tp].stx2_mnc['masknoles']],
            'output':  [patient[tp].stx2_mnc['classification']],
        }

        # FOR now, use only T1, even though it is bad
        # if 't2' in patient[tp].stx2_mnc and not patient.onlyt1:
        #     wmh_bison_input['t2']=[patient[tp].stx2_mnc['t2']]
        # if 'pd' in patient[tp].stx2_mnc and not patient.onlyt1:
        #     wmh_bison_input['pd']=[patient[tp].stx2_mnc['pd']]
        ray.get(run_bison.remote(bison_input, n_cls=3, n_jobs=4, batch=1,
                        load_pfx=patient.bison_pfx,
                        atlas_pfx=patient.bison_atlas_pfx,
                        method=patient.bison_method,
                        resample=True, inverse_xfm=True,
                        progress=True))

    else: # fall back to the old method

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


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

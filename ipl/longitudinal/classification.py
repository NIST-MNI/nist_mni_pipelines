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
def run_bison_wmh(wmh_bison_input,bison_input,wmh_out,cls_out,
                  wmh_bison_pfx=None,
                  wmh_bison_atlas_pfx=None,
                  wmh_bison_method=None,
                  bison_atlases=None,
                  bison_pfx=None,
                  bison_atlas_pfx=None,
                  bison_method=None,
                  ):
    # limit number of threads used
    n_jobs=int(ray.runtime_context.get_runtime_context().get_assigned_resources()["CPU"])
    with threadpool_limits(limits=n_jobs):
        with mincTools() as minc:
            # first segment WMH if possible
            if wmh_bison_input is not None:
                wmh_bison_input['output']=[wmh_out]
                bison_input['output']=[minc.tmp('cls.mnc')]

                bison.infer(wmh_bison_input, n_cls=1, batch=1,
                            load_pfx=wmh_bison_pfx,
                            atlas_pfx=wmh_bison_atlas_pfx,
                            method=wmh_bison_method,
                            resample=True, inverse_xfm=True,
                            n_jobs=n_jobs)
            else:
                bison_input['output']=[cls_out]
                
            # then segment everything else
            bison.infer(    bison_input, n_cls=3, batch=1,
                            load_pfx=bison_pfx,
                            atlas_pfx=bison_atlas_pfx,
                            atlases=bison_atlases,
                            method=bison_method,
                            resample=True, 
                            inverse_xfm=True,n_jobs=n_jobs)
            
            if wmh_bison_input is not None:
                # assing label WM to WMH in tissue classification results
                minc.calc([minc.tmp('cls.mnc'),wmh_out],
                          'A[1]>0.5?3:A[0]',cls_out,labels=True,datatype='-byte')  

def pipeline_classification(patient, tp):
    if os.path.exists(patient[tp].stx2_mnc['classification']) \
        and os.path.exists(patient[tp].qc_jpg['classification']):
        pass
    else:
        classification_v10(patient, tp)  # beast by simon fristed

    minc_qc.qc( patient[tp].stx2_mnc['t1'],
                patient[tp].qc_jpg['classification'],
                title=patient[tp].qc_title, 
                mask_range=[0,3],
                image_range=[0,120],
                mask=patient[tp].stx2_mnc['classification'],dpi=200,
                samples=20, 
                use_max=True,
                bg_color="black",
                fg_color="white",
                mask_cmap="jet" )
    return True


def classification_v10(patient, tp):

    # first run WMH classification , if possible
    if patient.wmh_bison_pfx is not None:
        #
        # generate input list in bison format
        wmh_bison_input={
            'subject':[patient.id+'_'+tp],
            't1':     [patient[tp].stx2_mnc['t1']],
            'xfm':    [patient[tp].nl_xfm],
            'mask':   [patient[tp].stx2_mnc['masknoles']],
            #'output': [patient[tp].stx2_mnc['wmh']],
        }
        # if patient.wmh_bison_atlas_pfx is None:
        #     # need to populate atlases
        #     wmh_bison_atlases={'t1':f"{patient.modeldir}/{patient.modelname}.mnc"}
        #     wmh_bison_atlases={'p1':f"{patient.modeldir}/{patient.modelname}.mnc"}
        # FOR now, use only T1, even though it is bad
        # if 't2' in patient[tp].stx2_mnc and not patient.onlyt1:
        #     wmh_bison_input['t2']=[patient[tp].stx2_mnc['t2']]
        # if 'pd' in patient[tp].stx2_mnc and not patient.onlyt1:
        #     wmh_bison_input['pd']=[patient[tp].stx2_mnc['pd']]
    else:
        wmh_bison_input=None

    if patient.bison_pfx is not None:
        # TODO: exclude WMH from the mask ?
        # generate input list in bison format
        bison_input={
            'subject': [patient.id+'_'+tp],
            't1':      [patient[tp].stx2_mnc['t1']],
            'xfm':     [patient[tp].nl_xfm],
            'mask':    [patient[tp].stx2_mnc['masknoles']],
            'output':  [patient[tp].stx2_mnc['classification']],
        }
        if patient.bison_atlas_pfx is None:
            # need to populate atlases
            bison_atlases={ 'av_t1':f"{patient.modeldir}/{patient.modelname}.mnc",
         #                   'av_t2':f"{patient.modeldir}/{patient.modelname.replace('_t1_','_t2_')}.mnc",
         #                   'av_pd':f"{patient.modeldir}/{patient.modelname.replace('_t1_','_pd_')}.mnc",
                            'p1':f"{patient.modeldir}/{patient.modelname.replace('_t1_','_csf_')}.mnc",
                            'p2':f"{patient.modeldir}/{patient.modelname.replace('_t1_','_gm_')}.mnc",
                            'p3':f"{patient.modeldir}/{patient.modelname.replace('_t1_','_wm_')}.mnc"
                            }
        else:
            bison_atlases=None

        # FOR now, use only T1, even though it is bad
        #if 't2' in patient[tp].stx2_mnc and not patient.onlyt1:
        #     wmh_bison_input['t2']=[patient[tp].stx2_mnc['t2']]
        #if 'pd' in patient[tp].stx2_mnc and not patient.onlyt1:
        #     wmh_bison_input['pd']=[patient[tp].stx2_mnc['pd']]
        run_bison_wmh_c=run_bison_wmh.options(num_cpus=patient.threads)
        ray.get(run_bison_wmh_c.remote(bison_input, bison_input,
                        patient[tp].stx2_mnc['wmh'],
                        patient[tp].stx2_mnc['classification'],
                        wmh_bison_pfx=patient.wmh_bison_pfx,
                        wmh_bison_atlas_pfx=patient.wmh_bison_atlas_pfx,
                        wmh_bison_method=patient.wmh_bison_method,
                        bison_atlases=bison_atlases,
                        bison_pfx=patient.bison_pfx,
                        bison_atlas_pfx=patient.bison_atlas_pfx,
                        bison_method=patient.bison_method,
                  ))

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

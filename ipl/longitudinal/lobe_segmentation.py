# -*- coding: utf-8 -*-
#
# @author Daniel, Vladimir S. FONOV, Simon Eskildsen
# @date 10/07/2011
#
# Atlas registration
#

from .general import *
from ipl.minc_tools import mincTools,mincError
from ipl import minc_qc
import json
import csv

# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_lobe_segmentation(patient, tp):
    if os.path.exists(patient[tp].stx2_mnc['lobes']) \
        and os.path.exists(patient[tp].vol['lobes']) \
        and os.path.exists(patient[tp].qc_jpg['lobes']):
        #print(' -- Lobe Segmentation - Processing already done!')
        pass
    else:
        lobe_segmentation_v10(patient, tp)  # beast by simon fristed

    # lobes qc
    minc_qc.qc(patient[tp].stx2_mnc['t1'],patient[tp].qc_jpg['lobes'],
            title=patient[tp].qc_title, image_range=[0,120],
            mask=patient[tp].stx2_mnc['lobes'],dpi=200,use_max=True,
            samples=20, bg_color="black",fg_color="white",
            mask_cmap='spectral')

    return True


def  lobes_to_json(patient, tp, lobes_txt, 
                   lobes_json=None, lobes_csv=None):
    # populate dictionary with expected values
    out={
        "SubjectID":"",
        "VisitID":"",
        "ScaleFactor":0.0,
        "Age":0.0,
        "Gender":"",
        "T1_SNR":0.0,
        "T2_SNR":0.0,
        "PD_SNR":0.0,
        "ICC_vol":0.0,
        "CSF_vol":0.0,
        "GM_vol":0.0,
        "WM_vol":0.0,
        "WMH_vol":0.0,
        "scale":0.0,
        "parietal_right_gm":0.0,
        "lateral_ventricle_left":0.0,
        "occipital_right_gm":0.0,
        "parietal_left_gm":0.0,
        "occipital_left_gm":0.0,
        "lateral_ventricle_right":0.0,
        "globus_pallidus_right":0.0,
        "globus_pallidus_left":0.0,
        "putamen_left":0.0,
        "putamen_right":0.0,
        "frontal_right_wm":0.0,
        "brainstem":0.0,
        "subthalamic_nucleus_right":0.0,
        "fornix_left":0.0,
        "frontal_left_wm":0.0,
        "subthalamic_nucleus_left":0.0,
        "caudate_left":0.0,
        "occipital_right_wm":0.0,
        "caudate_right":0.0,
        "parietal_left_wm":0.0,
        "temporal_right_wm":0.0,
        "cerebellum_left":0.0,
        "occipital_left_wm":0.0,
        "cerebellum_right":0.0,
        "temporal_left_wm":0.0,
        "thalamus_left":0.0,
        "parietal_right_wm":0.0,
        "thalamus_right":0.0,
        "frontal_left_gm":0.0,
        "frontal_right_gm":0.0,
        "temporal_left_gm":0.0,
        "temporal_right_gm":0.0,
        "3rd_ventricle":0.0,
        "4th_ventricle":0.0,
        "fornix_right":0.0,
        "extracerebral_CSF":0.0
    }
    with open(lobes_txt,'r') as f:
        for l in f:
            l=l.strip()
            if len(l)==0:
                continue
            l=l.split(' ' )
            if len(l)!=2:
                raise mincError('Invalid line in lobes file:'+str(l))
            k,v=l
            if k in out:
                if k!='Gender':
                    out[k]=float(v)
                else:
                    out[k]=v
            else:
                raise mincError('Invalid key in lobes file:'+str(k))
    out["SubjectID"]=patient.id
    out["VisitID"]=tp

    if lobes_json is not None:
        with open(lobes_json,'w') as f:
            json.dump(out,f,indent=2,sort_keys=True)

    if lobes_csv is not None:
        with open(lobes_csv,'w') as f:
            fieldnames = sorted(out.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(out)
    
    return out

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
        # TODO: reimplement in python
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
        
        ## HACK to calculate WMH volumes
        if os.path.exists(patient[tp].stx2_mnc['wmh']):
            wmh_volume=float(minc.execute_w_output(['mincstats' ,'-vol', '-binvalue' ,'1' ,'-q', patient[tp].stx2_mnc['wmh']]).strip())
            with open(patient[tp].vol['lobes'],"a") as f:
                f.write(f"WMH_vol {wmh_volume}\n")
        
        lobes_to_json(patient, tp,
                      patient[tp].vol['lobes'],
                      lobes_json=patient[tp].vol['lobes_json'],
                      lobes_csv=patient[tp].vol['lobes_csv'])
         
    return 0

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

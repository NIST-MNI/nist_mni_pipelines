#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# @author Daniel, Vladimir S. FONOV
# @date 10/07/2011
#
# Create a longitudinal model
#


import shutil
from iplGeneral import *
from iplScoopGenerateModel import generate_nonlinear_model
from ipl.minc_tools import mincTools,mincError


version = '1.1'


# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_lngtemplate(patient):

    # @ todo check if processing was already performed

    outputImages = [patient.template['nl_template'],
                    patient.template['nl_template_mask']]
                    
    for (i, tp) in patient.items():
        outputImages.append(tp.lng_xfm['t1'])

    # check if images exists

    allDone = True
    for i in outputImages:
        if not os.path.exists(i):
            print(' - Does not exists : ' + i)
            allDone = False
            break
    if allDone:
        print(' -- pipeline_linearlngtemplate is done')
    else:
        lngtemplate_v11(patient)  # VF: using 1.1 version

    return True
    # TODO add to history


def lngtemplate_v11(patient):

    with mincTools() as minc: 
        biascorr = False
        atlas = patient.modeldir + os.sep + patient.modelname + '.mnc'
        atlas_mask = patient.modeldir + os.sep + patient.modelname \
            + '_mask.mnc'

        options={'symmetric':False,
                 'protocol':[{'iter':1,'level':16},
                             {'iter':2,'level':8},
                             {'iter':2,'level':4},
                             {'iter':2,'level':2}],
                 'cleanup':True,
                 'biascorr':biascorr }

        if patient.fast:  # apply fast mode
            options['protocol']=[{'iter':1,'level':16},{'iter':2,'level':8}]

        samples= [ [tp.stx2_mnc['t1'], tp.stx2_mnc['masknoles']]
                        for (i, tp) in patient.items()]

        model = atlas
        model_mask = atlas_mask

        work_prefix = patient.workdir+os.sep+'nl'

        output=generate_nonlinear_model(samples,model=atlas,mask=atlas_mask,work_prefix=work_prefix,options=options)

        # copy output ... 
        shutil.copyfile(output['model'].scan,   patient.template['nl_template'])
        shutil.copyfile(output['model'].mask,   patient.template['nl_template_mask'])
        shutil.copyfile(output['model_sd'].scan,patient.template['nl_template_sd'])

        #TODO:
        #options.output_regu_0 = patient.template['regu_0']
        #options.output_regu_1 = patient.template['regu_1']

        minc.qc(
            patient.template['nl_template'],
            patient.qc_jpg['nl_template'],
            title=patient.id,
            image_range=[0, 120],
            big=True,
            clamp=True,
            )

        # copy each timepoint images too
        k=0
        for (i, tp) in patient.items():
            xfmfile =      output['xfm' ][k].xfm
            corrected_t1 = output['scan'][k].scan
            # creating lng images
            shutil.copyfile(corrected_t1, tp.lng_mnc['t1']) # NOT really needed?
            minc.xfm_normalize(xfmfile,atlas_mask,tp.lng_xfm['t1'], step=1.0)
            minc.xfm_normalize(xfmfile,atlas_mask,tp.lng_ixfm['t1'],step=1.0,invert=True)
            k+=1

        # cleanup temporary files
        shutil.rmtree(work_prefix)
        

if __name__ == '__main__':
    pass

  # Using script as a stand-alone script

  # We can create this as a stand alone script

  # 1. Create a patient
  # 2. Fill necessary images from the options (inputs and outputs)
  # 3. Call the function -> preprocessing_v10
  # 4. Exit

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

# -*- coding: utf-8 -*-

#
# @author Daniel, Vladimir S. FONOV, Guizz
# @date 10/07/2011

import os
import sys
import shutil
import traceback

#
# Create a longitudinal model
#

from ipl.model.generate_linear             import generate_linear_model
from ipl.minc_tools import mincTools,mincError
from ipl import minc_qc

import ipl.registration
#import ipl.ants_registration
#import ipl.elastix_registration

from .general import *

from scoop import futures, shared

version = '1.1'


# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_linearlngtemplate(patient):
    try:
        # make a vector with all otuput images

        outputImages = [patient.template['linear_template'],
                        patient.template['linear_template_mask'],
                        patient.template['stx2_xfm']]

        # tp.clp2['t1'],
        for (i, tp) in patient.items():
            outputImages.extend([tp.stx2_xfm['t1'], 
                                 tp.stx2_mnc['t1']])  # TODO add T2/PD ?

        # check if images exists

        allDone = True
        for i in outputImages:
            if not os.path.exists(i):
                print("pipeline_linearlngtemplate {} does not exist!".format(i))
                allDone = False
                break

        if allDone:
            print(' -- pipeline_linearlngtemplate is done')
        else:
            linearlngtemplate_v11(patient)  # VF. for now use this 1.1

        # # Writing QC images
        # ###################

        with mincTools() as minc:
            atlas_outline = patient.modeldir + os.sep + patient.modelname + '_outline.mnc'

            # qc linear template
            minc_qc.qc(
                patient.template['linear_template'],
                patient.qc_jpg['linear_template'],
                title=patient.id,
                image_range=[0, 120],
                samples=20,dpi=200,use_max=True,
                bg_color="black",fg_color="white",
                mask=atlas_outline
                )

            # qc stx2

            for (i, tp) in patient.items():
                minc_qc.qc(
                    tp.stx2_mnc['t1'],
                    tp.qc_jpg['stx2_t1'],
                    title=tp.qc_title,
                    image_range=[0, 120],
                    mask=atlas_outline,use_max=True,
                    samples=20,dpi=200,bg_color="black",fg_color="white"   )
    except mincError as e:
        print("Exception in average_transforms:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in average_transforms:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

    return True


class LngOptions:
    pass

# apply additional processing steps
def post_process(patient, i, tp, transform, biascorr, rigid=False):
    # bias in stx space
    modelt1   = patient.modeldir + os.sep + patient.modelname + '.mnc'
    modelmask = patient.modeldir + os.sep + patient.modelname + '_mask.mnc'
    
    with mincTools() as minc:
        xfmfile = transform.xfm
        stx_xfm_file = tp.stx_ns_xfm['t1'] if rigid else tp.stx_xfm['t1']
        clp_tp = tp.clp['t1']

        if patient.geo_corr and 't1' in patient[i].geo:
            clp_tp = tp.corr['t1']

        if biascorr is not None:
            stx_bias = biascorr.scan

            # 1. Transform the stn bias into native
            native_bias = minc.tmp('tmpbias_' + patient.id + '.mnc')
            
            minc.resample_smooth(
                stx_bias,
                native_bias,
                transform=stx_xfm_file,
                like=tp.clp['t1'],
                invert_transform=True,
                resample='linear',
                )

            # 2. Apply correction to clp image
            minc.calc([clp_tp, native_bias],
                    'A[1]>0.2?A[0]/A[1]:A[0]/0.2', tp.clp2['t1'], datatype='-short')
            clp_tp=tp.clp2['t1']
            
        else: # just run Nu correct one more time
            minc.nu_correct(clp_tp, 
                            output_image=minc.tmp('clp2_t1.mnc'), 
                            mask=tp.clp['mask'], 
                            mri3t=patient.mri3T )

            minc.volume_pol(
                minc.tmp('clp2_t1.mnc'),
                modelt1,
                tp.clp2['t1'],
                source_mask=tp.clp['mask'],
                target_mask=modelmask,
                datatype='-short' )

            clp_tp=tp.clp2['t1']

        # 3. concatenate all transforms
        if patient.skullreg:
            # skullreg
            tmpstx2xfm = minc.tmp('stx2tmp' + patient.id + '_' + i + '.xfm')
            
            minc.xfmconcat([stx_xfm_file, xfmfile,
                        patient.template['stx2_xfm']],
                        tmpstx2xfm)


            minc.skullregistration(
                clp_tp,
                patient.template['linear_template'],
                tp.clp['mask'],
                patient.template['linear_template_mask'],
                stx_xfm_file,
                tmpstx2xfm,
                patient.template['stx2_xfm'],
                )
        else:
            minc.xfmconcat([stx_xfm_file, xfmfile,
                        patient.template['stx2_xfm']],
                        tp.stx2_xfm['t1'])

        # reformat native t1, t2, pd and lesions images into stx2 BB
        minc.resample_smooth(clp_tp,
                            tp.stx2_mnc['t1'],
                            transform=tp.stx2_xfm['t1'],
                            like=modelt1)

        # 1bis. concatenate all transforms t2 BB
        # todo run N3 on T2 ?
        if 't2' in tp.native:
            clp_t2_tp = tp.clp['t2']
            
            if patient.geo_corr and 't2' in patient[i].geo:
                clp_t2_tp = tp.corr['t2']
            
            minc.xfmconcat([tp.clp['t2t1xfm'],
                        stx_xfm_file,
                        patient.template['stx2_xfm']],
                        tp.stx2_xfm['t2'])

            minc.resample_smooth(clp_t2_tp,
                    tp.stx2_mnc['t2'],
                    transform=tp.stx2_xfm['t2'], like=modelt1)

        if 'pd' in tp.native:
            # Warning: assume distortion correction for t2 and pd are the same
            clp_pd_tp = tp.clp['pd']
            
            if patient.geo_corr and 't2' in patient[i].geo:
                clp_pd_tp = tp.corr['pd']
                
            minc.xfmconcat([tp.clp['pdt1xfm'],
                        stx_xfm_file,
                        patient.template['stx2_xfm']],
                        tp.stx2_xfm['pd'])
            minc.resample_smooth(clp_pd_tp,
                    tp.stx2_mnc['pd'],
                    transform=tp.stx2_xfm['pd'], like=modelt1)

        if 't2les' in tp.native:
            stx2_t2=tp.stx2_xfm['t2']
            if 't2' in patient[i].geo and patient.geo_corr:
                tmp_t2_xfm = minc.tmp('t2_corr_xfm.xfm')
                minc.xfmconcat([patient[i].geo['t2'],tp.stx2_xfm['t2']], tmp_t2_xfm )
                stx2_t2=tmp_t2_xfm

            minc.resample_labels(tp.native['t2les'],
                    tp.stx2_mnc['t2les'],
                    transform=stx2_t2, like=modelt1)


def linearlngtemplate_v11(patient):
    with mincTools() as minc:

        atlas = patient.modeldir + os.sep + patient.modelname + '.mnc'
        atlas_mask = patient.modeldir + os.sep + patient.modelname + '_mask.mnc'
        atlas_outline = patient.modeldir + os.sep + patient.modelname + '_outline.mnc'
        atlas_mask_novent = patient.modeldir + os.sep + patient.modelname + '_mask_novent.mnc'

        # parameters for the template
        
        biasdist = 100
        # if patient.mri3T: biasdist=50
        # VF: disabling, because it seem to be unstable
        
        options={   'symmetric':False,
                    'reg_type':'-lsq12',
                    'objective':'-xcorr',
                    'iterations':4,
                    'cleanup':True,
                    'biascorr':patient.dobiascorr,
                    'biasdist':biasdist,
                    'linreg': patient.linreg }
        
        if patient.rigid:
            options['reg_type']='-lsq6' # TODO: use nsstx (?)

        if patient.symmetric:
            options['symmetric']=True 

        # Here we are relying on the time point order (1)

        samples=None
        if patient.rigid:
            samples= [ [tp.stx_ns_mnc['t1'], tp.stx_ns_mnc['masknoles']]
                        for (i, tp) in patient.items() ]
        else:
            samples= [ [tp.stx_mnc['t1'],    tp.stx_mnc['masknoles']]
                        for (i, tp) in patient.items() ]

        if patient.fast:
            options['iterations'] = 2

        work_prefix = patient.workdir+os.sep+'lin'

        output=generate_linear_model(samples, model=atlas, mask=atlas_mask, options=options, work_prefix=work_prefix)

        # copy output ... 
        shutil.copyfile(output['model'].scan,   patient.template['linear_template'])
        shutil.copyfile(output['model'].mask,   patient.template['linear_template_mask'])
        shutil.copyfile(output['model_sd'].scan,patient.template['linear_template_sd'])

        # Create the new stx space using the template
        
        # TODO: add pre-scaling in case of rigid (?)

        if patient.large_atrophy:
            ipl.registration.linear_register(patient.template['linear_template'],
                                atlas, patient.template['stx2_xfm'],
                                source_mask=atlas_mask_novent,
                                target_mask=atlas_mask_novent)
        else:
            ipl.registration.linear_register(patient.template['linear_template'],
                                atlas, patient.template['stx2_xfm'],
                                source_mask=patient.template['linear_template_mask'], 
                                target_mask=atlas_mask)

        # apply new correction to each timepoint stx file
        k=0
        jobs=[]
        print(repr(output))
        
        for (i, tp) in patient.items():
            biascorr=None
            if patient.dobiascorr:
                biascorr=output['biascorr'][k]
            # Here we are relying on the time point order (1) - see above
            jobs.append(futures.submit(post_process, patient, i, tp, output['xfm'][k], biascorr, rigid=patient.rigid))

            k+=1
        # wait for all substeps to finish
        futures.wait(jobs, return_when=futures.ALL_COMPLETED)
        
        # cleanup temporary files
        shutil.rmtree(work_prefix)
        
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

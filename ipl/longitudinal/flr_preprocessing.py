# -*- coding: utf-8 -*-
#
# @author Daniel, Vladimir S. FONOV
# @date 10/07/2011

version = '1.0'

#
# Preprocessing functions
# - mri 2 tal
# - n3 correction
# - denoising
# - vol pol

from .general import *
from .t1_preprocessing import run_nlm
from ipl.minc_tools import mincTools,mincError
from ipl import minc_qc

import ipl.registration
import ipl.ants_registration
import ipl.elastix_registration

import shutil

import ray

# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_flrpreprocessing(patient, tp):

    isDone = True
    if 'flair' in patient[tp].native:
        if os.path.exists(patient[tp].clp['flair']) \
            and os.path.exists(patient[tp].stx_xfm['flair']) \
            and os.path.exists(patient[tp].stx_mnc['flair']):
            isDone = True
        else:
            isDone = False

    if not isDone:
        flrpreprocessing_v10(patient, tp)

    if not 't2' in patient[tp].native:
        return True

    # 7. Qc image: t1/flr registration #TODO: convert to call to minctoools

    if 'flair' in patient[tp].native and \
        os.path.exists(patient[tp].stx_mnc['t1']) and \
        os.path.exists(patient[tp].stx_mnc['flair']):

        with mincTools( ) as minc:
            minc_qc.qc(
                patient[tp].stx_mnc['t1'],
                patient[tp].qc_jpg['t1flair'],
                title=patient[tp].qc_title,
                image_range=[0, 100],
                mask=patient[tp].stx_mnc['flair'],
                samples=20,dpi=200,use_max=True,
                image_cmap='red',
                mask_cmap='green',
                )
    return True


def pipeline_flrpreprocessing_s0(patient, tp):
    if 'flair' in patient[tp].native:
        if patient.denoise:
            run_nlm_c = run_nlm.options(num_cpus=patient.threads)
            ray.get(run_nlm_c.remote(patient[tp].native['flair'],  patient[tp].den['flair']))
    return True

def flrpreprocessing_v10(patient, tp):

    with  mincTools() as minc:

        model = patient.modeldir + os.sep + patient.modelname + '.mnc'
        
        modelflr = model.replace('t1', 'flair')
        if not os.path.exists(modelflr): # use T2 model
            modelflr = model.replace('t1', 't2')
        if not os.path.exists(modelflr): # use t1
            modelflr = model
        
        modelmask = patient.modeldir + os.sep + patient.modelname + '_mask.mnc'

        # TODO combine the two sequences in a for s in ["t2","pd","flair"] kind of...
        # # FLR Preprocessing
        # ##################

        if not 'flair' in patient[tp].native:
            print(' -- No FLAIR image!')
        elif os.path.exists(patient[tp].clp['flair']) \
            and os.path.exists(patient[tp].stx_xfm['flair']) \
            and os.path.exists(patient[tp].stx_mnc['flair']):
            pass
        else:

            tmpflr =   minc.tmp('float_flr.mnc')
            tmpmask = minc.tmp('mask_flr.mnc')
            tmpnlm =  minc.tmp('nlm_flr.mnc')
            tmpn3 =   minc.tmp('n3_flr.mnc')
            tmp_flr_t1_xfm = minc.tmp('flr_t1_0.xfm')
            tmp_flr_stx_xfm = minc.tmp('flr_stx_0.xfm')
            tmpstats = minc.tmp('volpol_flr.stats')

            # minc.convert(patient[tp].native['flair'], tmpflr)

            # for s in ['xspace', 'yspace', 'zspace']:
            #     spacing = minc.query_attribute(tmpflr, s + ':spacing')

            #     if spacing.count( 'irregular' ):
            #         minc.set_attribute( tmpflr, s + ':spacing', 'regular__' )
            
            # 1. Do nlm
            if patient.denoise:
                tmpnlm = patient[tp].den['flair']
            else:
                minc.convert_and_fix(patient[tp].native['flair'], tmpflr)
                tmpnlm = tmpflr

            # # manual initialization

            init_xfm = None
            # VF: this is probably incorrect!
            if 'stx_flr' in patient[tp].manual \
                and os.path.exists(patient[tp].manual['stx_flr']):
                init_xfm = patient[tp].manual['stx_flr']

            ipl.registration.linear_register_to_self(  # patient[tp].clp["flrt1xfm"],
                tmpnlm,
                patient[tp].native['t1'],
                tmp_flr_t1_xfm,
                # changes by SFE  TODO: use onlyt1 flag here?
                #mask='target',
                #model=patient.modelname,
                #modeldir=patient.modeldir,
                #target_talxfm=patient[tp].stx_xfm['t1'],
                init_xfm=init_xfm,
                nocrop=True,
                noautothreshold=True,
                )

            minc.xfmconcat([tmp_flr_t1_xfm, patient[tp].stx_xfm['t1']],
                           tmp_flr_stx_xfm)  # patient[tp].stx_xfm['flair']

            if patient.n4:
                minc.winsorize_intensity(tmpnlm,minc.tmp('trunc_flr.mnc'))
                minc.binary_morphology(minc.tmp('trunc_flr.mnc'),'',minc.tmp('otsu_flr.mnc'),binarize_bimodal=True)
                minc.defrag(minc.tmp('otsu_flr.mnc'),minc.tmp('otsu_defrag_flr.mnc'))
                minc.autocrop(minc.tmp('otsu_defrag_flr.mnc'),minc.tmp('otsu_defrag_expanded_flr.mnc'),isoexpand='50mm')
                minc.binary_morphology(minc.tmp('otsu_defrag_expanded_flr.mnc'),'D[25] E[25]',minc.tmp('otsu_expanded_closed_flr.mnc'))
                minc.resample_labels(minc.tmp('otsu_expanded_closed_flr.mnc'),minc.tmp('otsu_closed_flr.mnc'),like=minc.tmp('trunc_flr.mnc'))
                
                minc.calc([minc.tmp('trunc_flr.mnc'),minc.tmp('otsu_closed_flr.mnc')], 'A[0]*A[1]',  minc.tmp('trunc_masked_flr.mnc'))
                minc.calc([tmpnlm,minc.tmp('otsu_closed_flr.mnc')],'A[0]*A[1]' ,minc.tmp('masked_flr.mnc'))
                
                minc.resample_labels( modelmask, minc.tmp('brainmask_flr.mnc'),
                        transform=tmp_flr_stx_xfm, invert_transform=True,
                        like=minc.tmp('otsu_defrag_flr.mnc') )
                
                minc.calc([minc.tmp('otsu_defrag_flr.mnc'),minc.tmp('brainmask_flr.mnc')],'A[0]*A[1]',minc.tmp('weightmask_flr.mnc'))
                
                dist=200
                if patient.mri3T: dist=50 # ??
                
                minc.n4(minc.tmp('masked_flr.mnc'),
                        output_field=patient[tp].nuc['flair'],
                        output_corr=tmpn3,
                        iter='200x200x200x200',
                        weight_mask=minc.tmp('weightmask_flr.mnc'),
                        mask=minc.tmp('otsu_closed_flr.mnc'),
                        distance=dist,
                        downsample_field=4,
                        datatype='short'
                        )
                # shrink?
                minc.volume_pol(
                    tmpn3,
                    modelflr,
                    patient[tp].clp['flair'],
                    source_mask=minc.tmp('weightmask_flr.mnc'),
                    target_mask=modelmask,
                    datatype='-short',
                    )
                
            elif patient.mask_n3:

                # 3. Reformat t1 mask
                minc.resample_labels(patient[tp].stx_mnc['mask'],
                        tmpmask, like=tmpnlm, transform=tmp_flr_stx_xfm,
                        invert_transform=True)

                # 4. Apply n3
                minc.nu_correct(tmpnlm, output_image=tmpn3,
                                mask=tmpmask, mri3t=patient.mri3T,
                                output_field=patient[tp].nuc['flair'],
                                downsample_field=4,
                                datatype='short')

                # 5. vol pol
                minc.volume_pol(
                    tmpn3,
                    modelflr,
                    patient[tp].clp['flair'],
                    target_mask=modelmask,
                    source_mask=tmpmask,
                    datatype='-short',
                    )
            else:

                # 4. Apply n3
                minc.nu_correct(tmpnlm, output_image=tmpn3,
                                mri3t=patient.mri3T,
                                output_field=patient[tp].nuc['flair'],
                                downsample_field=4,
                                datatype='short')

                # 5. vol pol
                minc.volume_pol(tmpn3, modelflr, patient[tp].clp['flair'],
                                datatype='-short')

            # register to the stx space
            flr_corr = patient[tp].clp['flair']
            t1_corr = patient[tp].clp['t1']
            
            if 't1' in patient[tp].geo and patient.geo_corr:
                t1_corr = patient[tp].corr['t1']

            if 'flair' in patient[tp].geo and patient.geo_corr:
                flr_corr = patient[tp].corr['flair']
                
                minc.resample_smooth( patient[tp].clp['flair'],
                                    flr_corr,
                                    transform=patient[tp].geo['flair'] )


            # 6. second round of co-registration
            ipl.registration.linear_register_to_self(
                flr_corr,
                t1_corr,
                patient[tp].clp['flrt1xfm'],
                init_xfm=tmp_flr_t1_xfm,
                nocrop=True,
                noautothreshold=True,
                close=True,
                )

            # 7. create final flr stx transform
            minc.xfmconcat([patient[tp].clp['flrt1xfm'],
                           patient[tp].stx_xfm['t1']],
                           patient[tp].stx_xfm['flair'])

            # 7. Resample n3 image to stx
            minc.resample_smooth(flr_corr,
                                 patient[tp].stx_mnc['flair'], 
                                 like=model,
                                 transform=patient[tp].stx_xfm['flair'])

            # # 8. concat flrnat->t1nat and t1nat->stx native BB
            minc.xfmconcat([patient[tp].clp['flrt1xfm'],
                            patient[tp].stx_ns_xfm['t1']],
                            patient[tp].stx_ns_xfm['flair'])


if __name__ == '__main__':
    pass

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

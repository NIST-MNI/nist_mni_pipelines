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
from ipl.minc_tools import mincTools,mincError
from ipl import minc_qc

import ipl.registration
import ipl.ants_registration
import ipl.elastix_registration

import shutil

# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_t2pdpreprocessing(patient, tp):

    isDone = True
    for s in list(patient[tp].native.keys()):
        if s == 't1' or s == 't2les':
            continue  # t1 has its own processing

        if os.path.exists(patient[tp].clp[s]) \
            and os.path.exists(patient[tp].stx_xfm[s]) \
            and os.path.exists(patient[tp].stx_mnc[s]):
            isDone = True
        else:
            isDone = False

    if 't2les' in patient[tp].native:
        if isDone and os.path.exists(patient[tp].stx_mnc['masknoles']) \
            and os.path.exists(patient[tp].stx_mnc['t2les']) \
            and os.path.exists(patient[tp].stx_ns_mnc['masknoles']) \
            and os.path.exists(patient[tp].qc_jpg['t1t2']):
            return True
        else:
            isDone = False

    if not isDone:
        t2pdpreprocessing_v10(patient, tp)

    if not 't2' in patient[tp].native:
        return True

  # 7. Qc image: t1/t2 registration #TODO: convert to call to minctoools

    if os.path.exists(patient[tp].stx_mnc['t1']) \
        and os.path.exists(patient[tp].stx_mnc['t2']):

        with mincTools( ) as minc:
            minc_qc.qc(
                patient[tp].stx_mnc['t1'],
                patient[tp].qc_jpg['t1t2'],
                title=patient[tp].qc_title,
                image_range=[0, 100],
                mask=patient[tp].stx_mnc['t2'],
                samples=20,dpi=200,use_max=True,
                image_cmap='red',
                mask_cmap='green',
                )
    return True

def t2pdpreprocessing_v10(patient, tp):

    with  mincTools() as minc:

        model = patient.modeldir + os.sep + patient.modelname + '.mnc'
        
        modelt2 = model.replace('t1', 't2')
        modelpd = model.replace('t1', 'pd')
        modelmask = patient.modeldir + os.sep + patient.modelname \
            + '_mask.mnc'

        # TODO combine the two sequences in a for s in ["t2","pd","flair"] kind of...
        # # T2 Preprocessing
        # ##################

        if not 't2' in patient[tp].native:
            print(' -- No T2 image!')
        elif os.path.exists(patient[tp].clp['t2']) \
            and os.path.exists(patient[tp].stx_xfm['t2']) \
            and os.path.exists(patient[tp].stx_mnc['t2']):
            pass
        else:

            tmpt2 =   minc.tmp('float_t2.mnc')
            tmpmask = minc.tmp('mask_t2.mnc')
            tmpnlm =  minc.tmp('nlm_t2.mnc')
            tmpn3 =   minc.tmp('n3_t2.mnc')
            tmp_t2_t1_xfm = minc.tmp('t2_t1_0.xfm')
            tmp_t2_stx_xfm = minc.tmp('t2_stx_0.xfm')
            tmpstats = minc.tmp('volpol_t2.stats')

            minc.convert(patient[tp].native['t2'], tmpt2)

            for s in ['xspace', 'yspace', 'zspace']:
                spacing = minc.query_attribute(tmpt2, s + ':spacing')

                if spacing.count( 'irregular' ):
                    minc.set_attribute( tmpt2, s + ':spacing', 'regular__' )
            
            # 1. Do nlm
            if patient.denoise:
                minc.nlm(tmpt2, tmpnlm, beta=0.7)
            else:
                tmpnlm = tmpt2

            # # manual initialization

            init_xfm = None
            # VF: this is probably incorrect!
            if 'stx_t2' in patient[tp].manual \
                and os.path.exists(patient[tp].manual['stx_t2']):
                init_xfm = patient[tp].manual['stx_t2']

            ipl.registration.linear_register_to_self(  # patient[tp].clp["t2t1xfm"],
                tmpnlm,
                patient[tp].native['t1'],
                tmp_t2_t1_xfm,
                # changes by SFE  TODO: use onlyt1 flag here?
                #mask='target',
                #model=patient.modelname,
                #modeldir=patient.modeldir,
                #target_talxfm=patient[tp].stx_xfm['t1'],
                init_xfm=init_xfm,
                nocrop=True,
                noautothreshold=True,
                )

            minc.xfmconcat([tmp_t2_t1_xfm, patient[tp].stx_xfm['t1']],
                           tmp_t2_stx_xfm)  # patient[tp].stx_xfm["t2"]

            if patient.n4:
                minc.winsorize_intensity(tmpnlm,minc.tmp('trunc_t2.mnc'))
                minc.binary_morphology(minc.tmp('trunc_t2.mnc'),'',minc.tmp('otsu_t2.mnc'),binarize_bimodal=True)
                minc.defrag(minc.tmp('otsu_t2.mnc'),minc.tmp('otsu_defrag_t2.mnc'))
                minc.autocrop(minc.tmp('otsu_defrag_t2.mnc'),minc.tmp('otsu_defrag_expanded_t2.mnc'),isoexpand='50mm')
                minc.binary_morphology(minc.tmp('otsu_defrag_expanded_t2.mnc'),'D[25] E[25]',minc.tmp('otsu_expanded_closed_t2.mnc'))
                minc.resample_labels(minc.tmp('otsu_expanded_closed_t2.mnc'),minc.tmp('otsu_closed_t2.mnc'),like=minc.tmp('trunc_t2.mnc'))
                
                minc.calc([minc.tmp('trunc_t2.mnc'),minc.tmp('otsu_closed_t2.mnc')], 'A[0]*A[1]',  minc.tmp('trunc_masked_t2.mnc'))
                minc.calc([tmpnlm,minc.tmp('otsu_closed_t2.mnc')],'A[0]*A[1]' ,minc.tmp('masked_t2.mnc'))
                
                minc.resample_labels( modelmask, minc.tmp('brainmask_t2.mnc'),
                        transform=tmp_t2_stx_xfm, invert_transform=True,
                        like=minc.tmp('otsu_defrag_t2.mnc') )
                
                minc.calc([minc.tmp('otsu_defrag_t2.mnc'),minc.tmp('brainmask_t2.mnc')],'A[0]*A[1]',minc.tmp('weightmask_t2.mnc'))
                
                dist=200
                if patient.mri3T: dist=50 # ??
                
                minc.n4(minc.tmp('masked_t2.mnc'),
                        output_field=patient[tp].nuc['t2'],
                        output_corr=tmpn3,
                        iter='200x200x200x200',
                        weight_mask=minc.tmp('weightmask_t2.mnc'),
                        mask=minc.tmp('otsu_closed_t2.mnc'),
                        distance=dist,
                        downsample_field=4,
                        datatype='short'
                        )
                # shrink?
                minc.volume_pol(
                    tmpn3,
                    modelt2,
                    patient[tp].clp['t2'],
                    source_mask=minc.tmp('weightmask_t2.mnc'),
                    target_mask=modelmask,
                    datatype='-short',
                    )
                
            elif patient.mask_n3:

                # 3. Reformat t1 mask
                minc.resample_labels(patient[tp].stx_mnc['mask'],
                        tmpmask, like=tmpnlm, transform=tmp_t2_stx_xfm,
                        invert_transform=True)

                # 4. Apply n3
                minc.nu_correct(tmpnlm, output_image=tmpn3,
                                mask=tmpmask, mri3t=patient.mri3T,
                                output_field=patient[tp].nuc['t2'],
                                downsample_field=4,
                                datatype='short')

                # 5. vol pol
                minc.volume_pol(
                    tmpn3,
                    modelt2,
                    patient[tp].clp['t2'],
                    target_mask=modelmask,
                    source_mask=tmpmask,
                    datatype='-short',
                    )
            else:

                # 4. Apply n3
                minc.nu_correct(tmpnlm, output_image=tmpn3,
                                mri3t=patient.mri3T,
                                output_field=patient[tp].nuc['t2'],
                                downsample_field=4,
                                datatype='short')

                # 5. vol pol
                minc.volume_pol(tmpn3, modelt2, patient[tp].clp['t2'],
                                datatype='-short')

            # register to the stx space
            t2_corr = patient[tp].clp['t2']
            t1_corr = patient[tp].clp['t1']
            
            if 't1' in patient[tp].geo and patient.geo_corr:
                t1_corr = patient[tp].corr['t1']

            if 't2' in patient[tp].geo and patient.geo_corr:
                t2_corr = patient[tp].corr['t2']
                
                minc.resample_smooth( patient[tp].clp['t2'],
                                    t2_corr,
                                    transform=patient[tp].geo['t2'] )


            # 6. second round of co-registration
            ipl.registration.linear_register_to_self(
                t2_corr,
                t1_corr,
                patient[tp].clp['t2t1xfm'],
                init_xfm=tmp_t2_t1_xfm,
                nocrop=True,
                noautothreshold=True,
                close=True,
                )

            # 7. create final T2 stx transform
            minc.xfmconcat([patient[tp].clp['t2t1xfm'],
                           patient[tp].stx_xfm['t1']],
                           patient[tp].stx_xfm['t2'])

            # 7. Resample n3 image to stx
            minc.resample_smooth(t2_corr,
                                 patient[tp].stx_mnc['t2'], 
                                 like=model,
                                 transform=patient[tp].stx_xfm['t2'])

            # # 8. concat t2nat->t1nat and t1nat->stx native BB
            minc.xfmconcat([patient[tp].clp['t2t1xfm'],
                            patient[tp].stx_ns_xfm['t1']],
                            patient[tp].stx_ns_xfm['t2'])

        # # PD Preprocessing
        if not 'pd' in patient[tp].native:
            print(' -- No PD image!')
        elif os.path.exists(patient[tp].clp['pd']) \
            and os.path.exists(patient[tp].stx_mnc['pd']):
            print(' -- PD preprocessing exists!')
        else:

            # create tmpdir
            tmppd =    minc.tmp('float_pd.mnc')
            tmpmask =  minc.tmp('mask_pd.mnc')
            tmpnlm =   minc.tmp('nlm_pd.mnc')
            tmpn3 =    minc.tmp('n3_pd.mnc')
            tmpstats = minc.tmp('volpol_pd.stats')
            tmp_pd_t1_xfm = minc.tmp('pd_t1_0.xfm')
            tmp_pd_stx_xfm = minc.tmp('pd_stx_0.xfm')

            minc.convert(patient[tp].native['pd'], tmppd)

            for s in ['xspace', 'yspace', 'zspace']:
                spacing = minc.query_attribute(tmppd, s + ':spacing')
                
                if spacing.count( 'irregular' ):
                    minc.set_attribute( tmppd, s + ':spacing', 'regular__' )
                        
            # 1. Do nlm
          
            if patient.denoise:
                minc.nlm(tmppd, tmpnlm, beta=0.7)
            else:
                tmpnlm=tmppd


            # 2. Best lin reg T2 to T1
            # if there is no T2, we register the data
            if not 't2' in patient[tp].native:

                # tmp_xfm=tmpdir+'tmp_t1pd.xfm"'
                print(' -- Using PD to register to T1')  # TODO: convert to minctools
                
                init_xfm = None
                # VF: this is probably incorrect!
                if 'stx_pd' in patient[tp].manual \
                    and os.path.exists(patient[tp].manual['stx_pd']):
                    init_xfm = patient[tp].manual['stx_pd']

                ipl.registration.linear_register_to_self(
                    tmpnlm,
                    patient[tp].native['t1'],
                    tmp_pd_t1_xfm,
                    mask='target',
                    model=patient.modelname,
                    modeldir=patient.modeldir,
                    target_talxfm=patient[tp].stx_xfm['t1'],
                    init_xfm=init_xfm,
                    nocrop=True,
                    noautothreshold=True,
                    )
                    
                minc.xfmconcat([tmp_pd_t1_xfm,
                                patient[tp].stx_xfm['t1']],
                                tmp_pd_stx_xfm)
            else:
                # assume dual echo ?
                tmp_pd_stx_xfm = patient[tp].clp['t2t1xfm']
                        


            if patient.n4:
                # using advanced N4 recipe from Gabriel A. Devenyi: 
                #mincmath -clamp -const2 \$(mincstats -quiet -pctT 1 $input.mnc) \$(mincstats -quiet -pctT 95 $input.mnc) $input.mnc \$tmpdir/trunc.mnc &&

                #headmask \$tmpdir/trunc.mnc \$tmpdir/otsu.mnc &&
                #mincdefrag \$tmpdir/otsu.mnc \$tmpdir/otsu_defrag.mnc 1 6 &&
                #autocrop -isoexpand 50mm \$tmpdir/otsu_defrag.mnc \$tmpdir/otsu_expanded.mnc &&
                #itk_morph --exp 'D[25] E[25]' \$tmpdir/otsu_expanded.mnc \$tmpdir/otsu_expanded_closed.mnc &&
                #mincresample -keep -near -like \$tmpdir/otsu_defrag.mnc \$tmpdir/otsu_expanded_closed.mnc \$tmpdir/otsu_closed.mnc &&

                #minccalc -expression 'A[0]*A[1]' \$tmpdir/trunc.mnc \$tmpdir/otsu_closed.mnc \$tmpdir/masked.mnc &&
                #minccalc -expression 'A[0]*A[1]'  $input.mnc \$tmpdir/otsu_closed.mnc \$tmpdir/trunc-final_masked.mnc &&

                #bestlinreg_g -nmi -lsq12 -sec_target ${REGISTRATIONMODEL2} \$tmpdir/masked.mnc ${REGISTRATIONMODEL} \$tmpdir/0.xfm &&

                #itk_resample --clobber --invert_transform --labels --like \$tmpdir/trunc-final_masked.mnc --transform \$tmpdir/0.xfm ${REGISTRATIONBRAINMASK} \$tmpdir/brainmask.mnc &&

                #mincmath -mult \$tmpdir/otsu_defrag.mnc \$tmpdir/brainmask.mnc \$tmpdir/weightmask.mnc &&

                #N4BiasFieldCorrection -d 3 --verbose -r 1 -b [200] -c [200x200x200x200,0.0] -w \$tmpdir/weightmask.mnc -x \$tmpdir/otsu_closed.mnc -i \$tmpdir/trunc-final_masked.mnc -o $output.mnc &&
                #rm -r \$tmpdir                    
                minc.winsorize_intensity(tmpnlm,minc.tmp('trunc_pd.mnc'))
                minc.binary_morphology(minc.tmp('trunc_pd.mnc'),'',minc.tmp('otsu_pd.mnc'),binarize_bimodal=True)
                minc.defrag(minc.tmp('otsu_pd.mnc'),minc.tmp('otsu_defrag_pd.mnc'))
                minc.autocrop(minc.tmp('otsu_defrag_pd.mnc'),minc.tmp('otsu_defrag_expanded_pd.mnc'),isoexpand='50mm')
                minc.binary_morphology(minc.tmp('otsu_defrag_expanded_pd.mnc'),'D[25] E[25]',minc.tmp('otsu_expanded_closed_pd.mnc'))
                minc.resample_labels(minc.tmp('otsu_expanded_closed_pd.mnc'),minc.tmp('otsu_closed_pd.mnc'),like=minc.tmp('trunc_pd.mnc'))
                
                minc.calc([minc.tmp('trunc_pd.mnc'),minc.tmp('otsu_closed_pd.mnc')], 'A[0]*A[1]',  minc.tmp('trunc_masked_pd.mnc'))
                minc.calc([tmpnlm,minc.tmp('otsu_closed_pd.mnc')],'A[0]*A[1]' ,minc.tmp('masked_pd.mnc'))
                
                minc.resample_labels( modelmask, minc.tmp('brainmask_pd.mnc'),
                        transform=tmp_pd_stx_xfm, invert_transform=True,
                        like=minc.tmp('otsu_defrag_pd.mnc') )
                
                minc.calc([minc.tmp('otsu_defrag_pd.mnc'),minc.tmp('brainmask_pd.mnc')],'A[0]*A[1]',minc.tmp('weightmask_pd.mnc'))
                
                #N4BiasFieldCorrection -d 3 --verbose -r 1 -b [200] -c [200x200x200x200,0.0] -w \$tmpdir/weightmask.mnc -x \$tmpdir/otsu_closed.mnc -i \$tmpdir/trunc-final_masked.mnc -o $output.mnc &&
                dist=200
                if patient.mri3T: dist=50 # ??
                
                minc.n4(minc.tmp('masked_pd.mnc'),
                        output_field=patient[tp].nuc['pd'],
                        output_corr=tmpn3,
                        iter='200x200x200x200',
                        weight_mask=minc.tmp('weightmask_pd.mnc'),
                        mask=minc.tmp('otsu_closed_pd.mnc'),
                        distance=dist
                        )
                # shrink?
                minc.volume_pol(
                    tmpn3,
                    modelpd,
                    patient[tp].clp['pd'],
                    source_mask=minc.tmp('weightmask_pd.mnc'),
                    target_mask=modelmask,
                    datatype='-short',
                    )
                
            elif patient.mask_n3:
                # # 3. Reformat t1 mask
                minc.resample_labels(patient[tp].stx_mnc['mask'],
                        tmpmask, transform=patient[tp].stx_xfm['pd'],
                        invert_transform=True, like=tmpnlm)
                # 4. Apply n3
                minc.nu_correct(tmpnlm, output_image=tmpn3,
                                mask=tmpmask, mri3t=patient.mri3T,
                                output_field=patient[tp].nuc['pd'])
                # 5. vol pol
                minc.volume_pol(
                    tmpn3,
                    modelpd,
                    patient[tp].clp['pd'],
                    target_mask=modelmask,
                    source_mask=tmpmask,
                    datatype='-short',
                    )
            else:

                # 4. Apply n3
                minc.nu_correct(tmpnlm, output_image=tmpn3,
                                mri3t=patient.mri3T,
                                output_field=patient[tp].nuc['pd'])

                # 5. vol pol
                minc.volume_pol(tmpn3, modelpd, patient[tp].clp['pd'],
                                datatype='-short')


            # register to the stx space
            t1_corr = patient[tp].clp['t1']
            pd_corr = patient[tp].clp['pd']
            
            # assume dual echo
            if 't1' in patient[tp].geo and patient.geo_corr:
                t1_corr = patient[tp].corr['t1']

            if 'pd' in patient[tp].geo and patient.geo_corr:
                pd_corr = patient[tp].corr['pd']
                
                minc.resample_smooth( patient[tp].clp['pd'],
                                    pd_corr,
                                    transform=patient[tp].geo['t2'] )

            if not 't2' in patient[tp].native:
                # 6. second round of co-registration
                ipl.registration.linear_register_to_self(
                    pd_corr,
                    t1_corr,
                    patient[tp].clp['pdt1xfm'],
                    init_xfm=init_xfm,
                    nocrop=True,
                    noautothreshold=True,
                    close=True,
                    )
            else:
                shutil.copyfile(patient[tp].clp['t2t1xfm'],patient[tp].clp['pdt1xfm'])

            # 7. create final T2 stx transform
            minc.xfmconcat([patient[tp].clp['pdt1xfm'],
                           patient[tp].stx_xfm['t1']],
                           patient[tp].stx_xfm['pd'])

            # 7. Resample n3 image to stx
            minc.resample_smooth(pd_corr,
                                 patient[tp].stx_mnc['pd'], 
                                 like=model,
                                 transform=patient[tp].stx_xfm['pd'])

            # # 8. concat t2nat->t1nat and t1nat->stx native BB
            minc.xfmconcat([patient[tp].clp['pdt1xfm'],
                            patient[tp].stx_ns_xfm['t1']],
                            patient[tp].stx_ns_xfm['pd'])

        # # T2les Preprocessing
        # #####################

        if not 't2les' in patient[tp].native:
            pass
        elif os.path.exists(patient[tp].stx_mnc['t2les']) \
            and os.path.exists(patient[tp].stx_mnc['masknoles']) \
            and os.path.exists(patient[tp].stx_ns_mnc['masknoles']):
            pass
        else:

            tmpdilated = minc.tmp('dilated.mnc')
            tmp_t2_xfm = patient[tp].stx_xfm['t2']
            
            
            if 't2' in patient[tp].geo and patient.geo_corr:
                tmp_t2_xfm = minc.tmp('t2_corr_xfm.xfm')
                minc.xfmconcat([patient[tp].geo['t2'],patient[tp].stx_xfm['t2']], tmp_t2_xfm )
            
                
            # 6. Resample lesions tostx
            minc.resample_labels(patient[tp].native['t2les'],
                    patient[tp].stx_mnc['t2les'], 
                    transform=tmp_t2_xfm,
                    like=patient[tp].stx_mnc['t1'])

            # 7. Dilate lesions
            minc.binary_morphology(patient[tp].stx_mnc['t2les'], 'D[1]' ,tmpdilated)
            
            # 8. Remove lesions from mask
            minc.calc([patient[tp].stx_mnc['mask'],tmpdilated],
                      'A[0]>0.5&&A[1]<0.5?1:0',
                      patient[tp].stx_mnc['masknoles'], labels=True)

            # 9. Invert of stx_t1
            ixfm = minc.tmp( 'stx_invert.xfm')
            minc.xfminvert(patient[tp].stx_xfm['t1'], ixfm)

            combinedxfm = minc.tmp('combined.xfm')
            minc.xfmconcat([patient[tp].stx_ns_xfm['t1'], ixfm],combinedxfm)
            
            # 12. reformat stxmask noles
            minc.resample_labels(patient[tp].stx_mnc['masknoles'],
                                 patient[tp].stx_ns_mnc['masknoles'],
                                 like=patient[tp].stx_mnc['masknoles'],
                                 transform=combinedxfm)

if __name__ == '__main__':
    pass

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

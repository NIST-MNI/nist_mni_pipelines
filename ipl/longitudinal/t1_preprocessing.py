# -*- coding: utf-8 -*-

#
# @author Daniel, Vladimir S. FONOV
# @date 10/07/2011
version = '1.0'

from ipl.minc_tools import mincTools,mincError
from ipl import minc_qc

import ipl.registration
import ipl.ants_registration
import ipl.elastix_registration

from .general import *
from .patient import *

import shutil

import ray
from threadpoolctl import threadpool_limits
import traceback

# try:
#     from ipl.apply_multi_model_ov import segment_with_openvino
#     _have_segmentation_ov=True
# except:
#     _have_segmentation_ov=False


try:
    from ipl.apply_multi_model_onnx import segment_with_onnx
    _have_segmentation_onnx=True
except:
    _have_segmentation_onnx=False
    traceback.print_exc(file=sys.stdout)
    print("Missing onnxruntime, will not be able to run onnx models")


def pipeline_t1preprocessing_s0(patient, tp):

    # if redskull is available, use it to create initial mask
    if patient.redskull_native and patient.redskull_onnx is not None:
        run_redskull_onnx_c = run_redskull_onnx.options(num_cpus=patient.threads)
        ray.get(run_redskull_onnx_c.remote(
                    patient[tp].native['t1'], patient[tp].clp['brain_skull'], 
                    out_brain_mask=patient[tp].clp['mask'],
                    out_qc=patient[tp].qc_jpg['synthstrip'],
                    normalize_1x1x1=True,
                    redskull_model=patient.redskull_onnx))
    elif patient.synthstrip_onnx is not None:
        if not os.path.exists(patient[tp].clp['mask']):
            # apply synthstrip in the native space to ease everything else
            # need to resample to 1x1x1mm^2
            run_synthstrip_onnx_c = run_synthstrip_onnx.options(num_cpus=patient.threads)
            ray.get(run_synthstrip_onnx_c.remote(
                        patient[tp].native['t1'], patient[tp].clp['mask'], 
                        out_qc=patient[tp].qc_jpg['synthstrip'],
                        normalize_1x1x1=True,
                        synthstrip_model=patient.synthstrip_onnx))
            
    # 3. denoise
    if patient.denoise:
        if not os.path.exists( patient[tp].den['t1'] ):
            run_nlm_c = run_nlm.options(num_cpus=patient.threads)
            ray.get(run_nlm_c.remote(patient[tp].native['t1'],  patient[tp].den['t1']))

    return True


# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_t1preprocessing(patient, tp):
    # checking if processing was performed
    if os.path.exists(patient[tp].qc_jpg['stx_t1']) \
        and os.path.exists(patient[tp].clp['t1']) \
        and (os.path.exists(patient[tp].clp['mask']) or (patient.synthstrip_onnx is None and patient.redskull_onnx is None) ) \
        and os.path.exists(patient[tp].stx_xfm['t1']) \
        and os.path.exists(patient[tp].stx_mnc['t1']) \
        and os.path.exists(patient[tp].stx_ns_xfm['t1']) \
        and   os.path.exists(patient[tp].stx_ns_mnc['t1']) \
        and ( os.path.exists(patient[tp].stx_ns_mnc['brain_skull']) \
              or patient.redskull_onnx is None ):
        pass
    else:
        # # Run the appropiate version
        t1preprocessing_v10(patient, tp)

    # Writing QC images
    # #####################
    # qc stx registration

    modeloutline = patient.modeldir + os.sep + patient.modelname + '_brain_skull_outline.mnc'
    outline_range=[1,2]
    mask_cmap='autumn'

    if not os.path.exists(modeloutline):
        modeloutline = patient.modeldir + os.sep + patient.modelname + '_outline.mnc'
        outline_range=[0.5,1]
        mask_cmap='red'

    if not os.path.exists(patient[tp].qc_jpg['stx_t1']):
        minc_qc.qc(
            patient[tp].stx_mnc['t1'],
            patient[tp].qc_jpg['stx_t1'],
            title=patient[tp].qc_title,
            image_range=[0, 150],
            mask=modeloutline,
            dpi=200,    use_over=True, 
            ialpha=1.0, oalpha=1.0,
            samples=20,
            mask_range=outline_range,
            bg_color="black",fg_color="white",
            mask_cmap=mask_cmap
            )
    
    return True



@ray.remote(num_cpus=4, memory=10000 * 1024 * 1024) # 
def run_redskull_onnx(in_t1w, out_redskull, 
        unscale_xfm=None, out_ns_skull=None, out_ns_redskull=None, 
        out_qc=None,qc_title=None,reference=None,
        redskull_model=None,normalize_1x1x1=False,
        qc_image_range=[5,95],
        out_brain_mask=None,
        redskull_var='seg' ):
    assert _have_segmentation_onnx, "Failed to import segment_with_onnx"
    n_threads=int(ray.runtime_context.get_runtime_context().get_assigned_resources()["CPU"])

    with mincTools() as minc:
        # run redskull segmentation to create skull mask
            
        if not os.path.exists(out_redskull):
            if normalize_1x1x1:
                minc.resample_smooth(in_t1w, minc.tmp('t1_1x1x1.mnc'), unistep=1.0)
                in_t1w_=minc.tmp('t1_1x1x1.mnc')
                out_redskull_=minc.tmp('brain_1x1x1.mnc')
            else:
                in_t1w_=in_t1w
                out_redskull_=out_redskull

            if redskull_var=='seg':
                segment_with_onnx([in_t1w_], out_redskull_,
                                    threads=n_threads,
                                    settings=dict(
                                        whole=False, freesurfer=False, 
                                        normalize=True, 
                                        dist=False,
                                        use_gaussian_weights=True,
                                        padvol=16,
                                        patch_sz=[160, 160, 160],
                                        stride=80,
                                        n_classes=3,
                                        models=[redskull_model])
                                    ) # 
            # elif redskull_var=='synth': # experimental
            #     segment_with_onnx([in_t1w_], out_redskull_,
            #                         model=redskull_model,
            #                         whole=True, freesurfer=True, 
            #                         normalize=True, 
            #                         dist=True, largest=True,
            #                         threads=n_threads 
            #                         ) # 
            if normalize_1x1x1:
                minc.resample_labels(out_redskull_, out_redskull, 
                                     like=in_t1w, datatype='byte')

        if out_qc is not None:
            minc_qc.qc(
                in_t1w,
                out_qc,
                title=qc_title,
                image_range=qc_image_range,mask_cmap='jet',
                mask=out_redskull ,dpi=200,use_max=True,
                samples=20,bg_color="black",fg_color="white",
                percentile=True,mask_range=[0.5,2]
                )
            
        if out_brain_mask is not None:
            ### extract largest connected component

            minc.calc([out_redskull], 'abs(A[0]-1)<0.5?1:0', 
                minc.tmp("brain.mnc"), labels=True)
            minc.fill_holes(minc.tmp("brain.mnc"), out_brain_mask)
            # minc.crop(minc.tmp("brain_filled.mnc"),minc.tmp("brain1.mnc"),crop=1)
            # minc.resample_labels(minc.tmp("brain1.mnc"),out_brain_mask,like=in_t1w,order=0)
            #minc.zero_border(minc.tmp("brain1.mnc"),out_brain_mask)

        # generate unscaling transform
        if unscale_xfm is not None:
            minc.calc([out_redskull],'abs(A[0]-2)<0.5?1:0', 
                minc.tmp("skull.mnc"), labels=True)
            
            if out_ns_skull is None:
                minc.resample_labels(minc.tmp("skull.mnc"), out_ns_skull, transform=unscale_xfm,like=reference)
            if out_ns_redskull is None:
                minc.resample_labels(out_redskull, out_ns_redskull, transform=unscale_xfm,like=reference)

@ray.remote(num_cpus=4) 
def run_nlm(in_t1w, out_den):
    n_threads=int(ray.runtime_context.get_runtime_context().get_assigned_resources()["CPU"])
    #os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']=os.environ['OMP_NUM_THREADS']
    _omp_num_threads=os.environ.get('OMP_NUM_THREADS',None)
    _itk_num_threads=os.environ.get('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS',None)

    with mincTools() as minc:
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']=str(n_threads)
        os.environ['OMP_NUM_THREADS']=str(n_threads)
        minc.convert_and_fix(in_t1w, minc.tmp('fixed.mnc'))
        minc.nlm( minc.tmp('fixed.mnc'), out_den, beta=0.7 )

    if _omp_num_threads is not None:
        os.environ['OMP_NUM_THREADS']=_omp_num_threads
    if _itk_num_threads is not None:
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']=_itk_num_threads



@ray.remote(num_cpus=4, memory=10000 * 1024 * 1024) # uses about 10GB of RAM
def run_synthstrip_onnx(in_t1w, out_synthstrip, 
        out_qc=None, qc_title=None, normalize_1x1x1=False,
        synthstrip_model=None ):
    
    assert _have_segmentation_onnx, "Failed to import segment_with_onnx"

    n_threads=int(ray.runtime_context.get_runtime_context().get_assigned_resources()["CPU"])

    with mincTools() as minc:
        # run redskull segmentation to create skull mask
        if not os.path.exists(out_synthstrip):
            if normalize_1x1x1:
                minc.resample_smooth(in_t1w, minc.tmp('t1_1x1x1.mnc'), unistep=1.0)
                segment_with_onnx([minc.tmp('t1_1x1x1.mnc')], minc.tmp('brain_1x1x1.mnc'),
                                    
                                    settings=dict(whole=True,freesurfer=True,normalize=True,
                                                 threads=n_threads, dist=True,largest=True,
                                                 models=[synthstrip_model],
                                                 )
                                    ) # 
                minc.resample_labels(minc.tmp('brain_1x1x1.mnc'),out_synthstrip,like=in_t1w,datatype='byte')
            else:
                segment_with_onnx([in_t1w], out_synthstrip,
                                    settings=dict(whole=True,freesurfer=True,normalize=True,
                                                  threads=n_threads, dist=True,largest=True,
                                                  models=[synthstrip_model],
                                                  )
                                    ) # 

        if out_qc is not None:
            minc_qc.qc(
                in_t1w,
                out_qc,
                title=qc_title,
                image_range=[0, 120],
                mask=out_synthstrip,dpi=200,use_max=True,
                samples=20,bg_color="black",fg_color="white"
                )



def t1preprocessing_v10(patient, tp):

    # # processing data
    # ##################
    with mincTools() as minc:
        tmpt1 =    minc.tmp('float_t1.mnc')
        tmpmask =  minc.tmp('mask_t1.mnc')
        tmpn3 =    minc.tmp('n3_t1.mnc')
        tmpstats = minc.tmp('volpol_t1.stats')
        tmpxfm =   minc.tmp('stx_t1.xfm')
        tmpnlm =   minc.tmp('nlm_t1.mnc')

        modelt1   = patient.modeldir + os.sep + patient.modelname + '.mnc'
        modelmask = patient.modeldir + os.sep + patient.modelname + '_mask.mnc'

        init_xfm = None
        if 'stx_t1' in patient[tp].manual \
            and os.path.exists(patient[tp].manual['stx_t1']):
            init_xfm = patient[tp].manual['stx_t1']

        # Manual clp t1
        if 'clp_t1' in patient[tp].manual \
            and os.path.exists(patient[tp].manual['clp_t1']):
                
            shutil.copyfile(patient[tp].manual['clp_t1'],  patient[tp].clp['t1'])
            tmpt1 = patient[tp].clp['t1']  # In order to make the registration if needed

        # if we have a native mask
        have_native_mask=os.path.exists(patient[tp].clp['mask'])
        if have_native_mask:
            tmpmask = patient[tp].clp['mask']

        if not os.path.exists( patient[tp].clp['t1'] ):
            # 3. denoise
            if patient.denoise:
                tmpnlm = patient[tp].den['t1']
            else:
                minc.convert_and_fix(patient[tp].native['t1'], tmpt1)
                tmpnlm = tmpt1

            if     not os.path.exists( patient[tp].clp['t1'] ) \
                or not os.path.exists( patient[tp].nuc['t1']):

                if patient.n4:
                    if have_native_mask: # using synthstrip for N4 mask
                        dist=200
                        if patient.mri3T: dist=50 # ??
                        
                        minc.n4(tmpt1,
                                output_field=patient[tp].nuc['t1'],
                                output_corr=tmpn3,
                                iter='200x200x200x200',
                                weight_mask=tmpmask,
                                mask=tmpmask,
                                distance=dist,
                                downsample_field=4,
                                datatype='short'
                                )
                    else:
                        minc.winsorize_intensity(tmpt1,minc.tmp('trunc_t1.mnc'))
                        minc.binary_morphology(minc.tmp('trunc_t1.mnc'),'',minc.tmp('otsu_t1.mnc'),binarize_bimodal=True)
                        minc.defrag(minc.tmp('otsu_t1.mnc'),minc.tmp('otsu_defrag_t1.mnc'))
                        minc.autocrop(minc.tmp('otsu_defrag_t1.mnc'),minc.tmp('otsu_defrag_expanded_t1.mnc'),isoexpand='50mm')
                        minc.binary_morphology(minc.tmp('otsu_defrag_expanded_t1.mnc'),'D[25] E[25]',minc.tmp('otsu_expanded_closed_t1.mnc'))
                        minc.resample_labels(minc.tmp('otsu_expanded_closed_t1.mnc'),minc.tmp('otsu_closed_t1.mnc'),like=minc.tmp('trunc_t1.mnc'))
                        
                        minc.calc([minc.tmp('trunc_t1.mnc'),minc.tmp('otsu_closed_t1.mnc')], 'A[0]*A[1]',  minc.tmp('trunc_masked_t1.mnc'))
                        minc.calc([tmpt1,minc.tmp('otsu_closed_t1.mnc')],'A[0]*A[1]' ,minc.tmp('masked_t1.mnc'))
                        
                        ipl.registration.linear_register( minc.tmp('trunc_masked_t1.mnc'), modelt1, tmpxfm,
                                init_xfm=init_xfm, 
                                objective='-nmi', conf=patient.linreg )
                        
                        minc.resample_labels( modelmask, minc.tmp('brainmask_t1.mnc'),
                                transform=tmpxfm, invert_transform=True,
                                like=minc.tmp('otsu_defrag_t1.mnc') )
                        
                        minc.calc([minc.tmp('otsu_defrag_t1.mnc'),minc.tmp('brainmask_t1.mnc')],'A[0]*A[1]',minc.tmp('weightmask_t1.mnc'))
                    
                        dist=200
                        if patient.mri3T: dist=50 # ??
                        
                        minc.n4(minc.tmp('masked_t1.mnc'),
                                output_field=patient[tp].nuc['t1'],
                                output_corr=tmpn3,
                                iter='200x200x200x200',
                                weight_mask=minc.tmp('weightmask_t1.mnc'),
                                mask=minc.tmp('otsu_closed_t1.mnc'),
                                distance=dist,
                                downsample_field=4,
                                datatype='short'
                                )
                    # shrink?
                    minc.volume_pol(
                        tmpn3,
                        modelt1,
                        patient[tp].clp['t1'],
                        source_mask=minc.tmp('weightmask_t1.mnc'),
                        target_mask=modelmask,
                        datatype='-short',
                        )
                elif patient.mask_n3:
                    # 2. Reformat mask
                    if patient.synthstrip_onnx is None:
                        ipl.registration.linear_register( tmpt1, modelt1, tmpxfm,
                                init_xfm=init_xfm, 
                                objective='-nmi', 
                                conf=patient.linreg )

                        minc.resample_labels( modelmask, tmpmask,
                                transform=tmpxfm, invert_transform=True,
                                like=tmpnlm )

                    minc.nu_correct( tmpnlm, output_image=tmpn3,
                                    mask=tmpmask, 
                                    mri3t=patient.mri3T,
                                    output_field=patient[tp].nuc['t1'],
                                    downsample_field=4,
                                    datatype='short')

                    minc.volume_pol(
                        tmpn3,
                        modelt1,
                        patient[tp].clp['t1'],
                        source_mask=tmpmask,
                        target_mask=modelmask,
                        datatype='-short',
                        )
                else:
                    minc.nu_correct( tmpnlm,
                                     mask=(tmpmask if have_native_mask else None),
                                     output_image=tmpn3,
                                     mri3t=patient.mri3T,
                                     output_field=patient[tp].nuc['t1'],
                                     downsample_field=4,
                                     datatype='short')

                    minc.volume_pol( tmpn3, modelt1, patient[tp].clp['t1'], 
                                     datatype='-short' )
        # register to the stx space
        t1_corr = patient[tp].clp['t1']
        
        if 't1' in patient[tp].geo and patient.geo_corr:
            t1_corr = patient[tp].corr['t1'] #TODO: avoid double resampling for the output!
            minc.resample_smooth( patient[tp].clp['t1'],
                                  t1_corr,
                                  transform=patient[tp].geo['t1'] )

        # TODO: implement skull-based scaling here?
        if not os.path.exists( patient[tp].stx_xfm['t1']):
            if have_native_mask:
                # HACK: using masks for initial registration
                ipl.registration.linear_register( tmpmask, modelmask,
                                    minc.tmp('mask_init.xfm'),
                                    init_xfm=init_xfm,
                                    objective='-xcorr',  # should use -zscore or -ssc ??
                                    conf=patient.linreg)

                ipl.registration.linear_register( t1_corr, modelt1,
                                    patient[tp].stx_xfm['t1'],
                                    init_xfm=minc.tmp('mask_init.xfm'),
                                    objective='-nmi', 
                                    conf=patient.linreg,
                                    source_mask=tmpmask,
                                    target_mask=modelmask)
            else:
                ipl.registration.linear_register( t1_corr, modelt1,
                                    patient[tp].stx_xfm['t1'],
                                    init_xfm=init_xfm,
                                    objective='-nmi', 
                                    conf=patient.linreg)
                                    # target_mask=modelmask
            

        minc.resample_smooth( t1_corr,
                              patient[tp].stx_mnc['t1'], like=modelt1,
                              transform=patient[tp].stx_xfm['t1'] )

        # stx no scale
        minc.xfm_noscale( patient[tp].stx_xfm['t1'], patient[tp].stx_ns_xfm['t1'],
                          unscale=patient[tp].stx_ns_xfm['unscale_t1'])

        minc.resample_smooth(t1_corr,
                             patient[tp].stx_ns_mnc['t1'],
                             like=modelt1,
                             transform=patient[tp].stx_ns_xfm['t1'])

        if patient.redskull_onnx is not None:
            run_redskull_onnx_c = run_redskull_onnx.options(num_cpus=patient.threads)

            ray.get(run_redskull_onnx_c.remote(
                patient[tp].stx_mnc['t1'], 
                patient[tp].stx_mnc['brain_skull'],
                unscale_xfm=patient[tp].stx_ns_xfm['unscale_t1'],
                out_ns_skull=patient[tp].stx_ns_mnc["skull"], 
                out_ns_redskull=patient[tp].stx_ns_mnc["brain_skull"],
                out_qc=patient[tp].qc_jpg['stx_skull'],
                qc_title=patient[tp].qc_title, 
                reference=modelmask,
                redskull_model=patient.redskull_onnx,
                redskull_var=patient.redskull_var ))
            
            # adjust scaling factor based on the skull here? 

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

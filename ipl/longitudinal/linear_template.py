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
from ipl.model.registration             import xfmavg
from ipl.minc_tools import mincTools,mincError
from ipl import minc_qc

import ipl.registration
#import ipl.ants_registration
#import ipl.elastix_registration

from .general import *

import ray

version = '1.1'


# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_linearlngtemplate(patient):
    print("pipeline_linearlngtemplate")
    try:
        # make a vector with all output images
        outputImages = [patient.template['linear_template'],
                        patient.template['linear_template_mask'],
                        patient.template['stx2_xfm']]

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
                    samples=20,dpi=200  )
    except mincError as e:
        print("Exception in pipeline_linearlngtemplate:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in pipeline_linearlngtemplate:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

    return True


class LngOptions:
    pass


# SKULL registraion code starts here
# TODO: move to a separate file ?
s1_template="""
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
(UseDirectionCosines "true")

(AutomaticTransformInitialization "{automatic_transform_init}")
(AutomaticTransformInitializationMethod "{automatic_transform_init_method}")
(AutomaticScalesEstimation "true")
(AutomaticParameterEstimation "true")
(MaximumStepLength {max_step})

(Registration         "MultiResolutionRegistration")
(Interpolator         "BSplineInterpolator" )
(ResampleInterpolator "FinalBSplineInterpolator" )
(Resampler            "DefaultResampler" )
(ShowExactMetricValue {exact_metric})

(FixedImagePyramid  "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

(Optimizer "{optimizer}")
(Transform "{transform}")
(Metric    "{metric}")

(HowToCombineTransforms "Compose")

(ErodeMask "false")

(NumberOfResolutions {resolutions})

(ImagePyramidSchedule {pyramid} )

(MaximumNumberOfIterations {iterations} )
(RequiredRatioOfValidSamples 0.01)
(MaximumNumberOfSamplingAttempts 10)

//
// restrict parameters search to scale only\
// https://elastix.lumc.nl/doxygen/classitk_1_1AffineDTI3DTransform.html
// T(x) = R G S (x-c) + t + c
//   R = Rx Ry Rz (rotation matrices)
//   G = Gx Gy Gz (shear matrices)
//   S = diag( [sx sy sz] ) (scaling matrix)
//    c = center of rotation
//    t = translation
//    see: APPENDIX A in https://doi.org/10.1002/mrm.21890 
//        "The B-matrix must be rotated when correcting for subject motion in DTI data"
//
//       Rx Ry Rz Gx Gy Gz Sx Sy Sz tx ty tz
(Scales 100000000 100000000 100000000  100000000 100000000 100000000 -1 -1 -1     100000000 100000000 100000000)

(NumberOfSpatialSamples  {samples} )

(NewSamplesEveryIteration "{new_samples}")
(ImageSampler             "{sampler}" )

(BSplineInterpolationOrder 1)

(FinalBSplineInterpolationOrder 1)

(DefaultPixelValue    0)

(WriteResultImage     "false")

// The pixel type and format of the resulting deformed moving image
(ResultImagePixelType  "float")
(ResultImageFormat     "mnc")
"""

def generate_skull_mask_dist_map(in_brain_skull, outertable, out_mask, out_dist, width1=4, width2=3, top=None):
    with mincTools() as m:
        if top is not None:
            m.resample_labels(top,m.tmp("brain_top.mnc"), like=in_brain_skull)
            m.calc([in_brain_skull, m.tmp("brain_top.mnc")],"A[0]>0.5&&A[1]>0.5",m.tmp("brain_skull.mnc"),datatype="byte")
            in_brain_skull=m.tmp("brain_skull.mnc")
        # HACK?
        m.binary_morphology(in_brain_skull,f"D[3] E[3]",m.tmp("brain_skull_c3.mnc"),binarize_threshold=0.5)

        m.binary_morphology(m.tmp("brain_skull_c3.mnc"),f"D[{width1}]",m.tmp("brain_skull_d.mnc"))
        m.calc([in_brain_skull, m.tmp("brain_skull_d.mnc")], "A[1]>0.5&&A[0]<0.5?1:0", outertable, datatype="byte")

        m.binary_morphology(outertable,f"D[{width2}]", out_mask)
        
        m.command(['itk_distance', '--signed', '--label','1', 
                    m.tmp("brain_skull_c3.mnc"), m.tmp("brain_skull_dist.mnc")])
        
        if top is not None:
            m.calc([m.tmp("brain_skull_dis@ext:ms-python.black-formattert.mnc"), m.tmp("brain_top.mnc")] ,"A[1]>0.5?abs(A[0]):0",out_dist)
        else:
            m.calc([m.tmp("brain_skull_dist.mnc")] ,"abs(A[0])",out_dist)



def register_using_skull(in_t1w, in_skull,     in_brain, 
                         in_ref, in_ref_skull, in_ref_brain,
                         out_lin_xfm,    
                         top=None, _id=""):
    # 
    # using 1 Core?
    # os.environ['OMP_NUM_THREADS']='1'
    # os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'

    #out_par=f"{pfx}_skull_lin.par"
    if not os.path.exists(out_lin_xfm):

        with mincTools() as m:
            
            # 0th part, standard mode - register using the brain mask 
            # TODO: figure out, maybe better to use
            # the brain mask from the redskull?
            ipl.registration.linear_register(in_t1w,
                                    in_ref, m.tmp("s0.xfm"),
                                    source_mask=in_brain, 
                                    target_mask=in_ref_brain)
            
            m.resample_labels(in_skull, m.tmp("in_skull.mnc"), like=in_ref_skull, transform=m.tmp("s0.xfm"))
            m.resample_smooth(in_t1w, m.tmp("warp_t1w_s0.mnc"),like=in_ref_skull, transform=m.tmp("s0.xfm"))

            generate_skull_mask_dist_map(m.tmp("in_skull.mnc"),  m.tmp("in_mask.mnc"),  m.tmp("in_mask2.mnc"), m.tmp("in_dist.mnc"), top=top)
            generate_skull_mask_dist_map(in_ref_skull, m.tmp("ref_mask.mnc"),m.tmp("ref_mask2.mnc"), m.tmp("ref_dist.mnc"),top=top)

            parameters_s1 = {"optimizer":   'AdaptiveStochasticGradientDescent',
                        "transform":   'AffineDTITransform',#'AffineTransform',
                        "metric":      'AdvancedMeanSquares',
                        "resolutions": 2,
                        "pyramid": "2 2 2 1 1 1",
                        "iterations": 4000,
                        "samples": 4096,
                        "sampler": "Random",
                        "max_step": 0.2,
                        "measure": 'false',
                        "automatic_transform_init": 'false',
                        "automatic_transform_init_method": "CenterOfGravity",
                        "exact_metric": 'false',
                        "new_samples": 'true'
                    }
            
            parameters_s2 = {"optimizer":   'AdaptiveStochasticGradientDescent',
                        "transform":   'AffineDTITransform',#'AffineTransform',
                        "metric":      'AdvancedNormalizedCorrelation',
                        "resolutions": 2,
                        "pyramid": "2 2 2 1 1 1",
                        "iterations": 4000,
                        "samples": 4096,
                        "sampler": "RandomSparseMask",
                        "max_step": 0.1,
                        "measure": 'false',
                        "automatic_transform_init": 'false',
                        "automatic_transform_init_method": "CenterOfGravity",
                        "exact_metric": 'false',
                        "new_samples": 'true'
                    }

            # run 1st iteration linear registration using distance maps , adjusting only scales?
            par_s1=s1_template.format(**parameters_s1)
            par_s2=s1_template.format(**parameters_s2)

            # 
            # 1st iteration, using distance maps only
            ipl.elastix_registration.register_elastix(m.tmp("in_dist.mnc"),
                                                    m.tmp("ref_dist.mnc"),
                                                    source_mask=m.tmp("in_mask2.mnc"),
                                                    target_mask=m.tmp("ref_mask2.mnc"),
                                                    output_xfm=m.tmp("s1.xfm"),
                                                    parameters=par_s1,
                                                    output_par=m.tmp("s1.par"),
                                                    nl=False)
            
            m.resample_labels(m.tmp("in_mask.mnc"), m.tmp("warp_mask_s1.mnc"),  like=in_ref_skull,  transform=m.tmp("s1.xfm"))
            m.resample_smooth(m.tmp("warp_t1w_s0.mnc"), m.tmp("warp_t1w_s1.mnc"),like=in_ref_skull, transform=m.tmp("s1.xfm"))

            # run 2nd iteration linear registration using T1w + outer skull mask , adjusting only scales?
            ipl.elastix_registration.register_elastix(m.tmp("warp_t1w_s1.mnc"),
                                                    in_ref,
                                                    source_mask=m.tmp("warp_mask_s1.mnc"),
                                                    target_mask=m.tmp("ref_mask.mnc"),
                                                    output_xfm=m.tmp("s2.xfm"),
                                                    parameters=par_s2,
                                                    output_par=m.tmp("s2.par"),
                                                    nl=False)
            
            # final linar transform
            m.xfmconcat([m.tmp("s0.xfm"),m.tmp("s1.xfm"), m.tmp("s2.xfm")], out_lin_xfm)

        return out_lin_xfm



# apply additional processing steps
@ray.remote
def post_process(patient, i, tp, transform, biascorr, rigid=False, transform2=None, scale_xfm=None):
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
            native_log_bias = minc.tmp('tmpbias_' + patient.id + '.mnc')
            minc.calc([stx_bias],"A[0]>0.1?log(A[0]):0.0", minc.tmp('logbias_' + patient.id + '.mnc'))

            # TODO: maybe better to resample with different fill value here?
            minc.resample_smooth(
                minc.tmp('logbias_' + patient.id + '.mnc'),
                native_log_bias,
                transform=stx_xfm_file,
                like=tp.clp['t1'],
                invert_transform=True,
                resample='linear',
                order=1
                )

            # 2. Apply correction to clp image
            minc.calc([clp_tp, native_log_bias],
                    'A[0]/exp(A[1])', minc.tmp('clp2_t1.mnc'), datatype='-short')
            # apply normalization once again
            minc.volume_pol(
                minc.tmp('clp2_t1.mnc'),
                modelt1,
                tp.clp2['t1'],
                source_mask=tp.clp['mask'],
                target_mask=modelmask,
                datatype='-short' )

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
            minc.xfmconcat([stx_xfm_file, 
                            xfmfile, 
                            #transform2.xfm,  # DISABLED for now
                            scale_xfm, 
                            patient.template['stx2_xfm'] ],
                        tp.stx2_xfm['t1'] )
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
    print("linearlngtemplate_v11")

    with mincTools() as minc:

        atlas = patient.modeldir + os.sep + patient.modelname + '.mnc'
        atlas_mask = patient.modeldir + os.sep + patient.modelname + '_mask.mnc'
        # REDSKULL version 
        # atlas_skull_mask = patient.modeldir + os.sep + patient.modelname + '_skull.mnc'
        # REDSKULL version
        atlas_brain_skull = patient.modeldir + os.sep + patient.modelname + '_brain_skull.mnc' 

        atlas_outline = patient.modeldir + os.sep + patient.modelname + '_outline.mnc'
        atlas_mask_novent = patient.modeldir + os.sep + patient.modelname + '_mask_novent.mnc'

        # parameters for the template
        biasdist = 100
        # if patient.mri3T: biasdist=50
        # VF: disabling, because it seem to be unstable
        # using skull mask if patients.skullreg
        options={   'symmetric':False,
                    'reg_type':'-lsq12',
                    'objective':'-xcorr',
                    'iterations':4,
                    'cleanup':True,
                    'biascorr':patient.dobiascorr,
                    'biasdist':biasdist,
                    'linreg': patient.linreg }
        
        options_2={   
                    'symmetric':False,
                    'reg_type':'-lsq9', # TODO: ?
                    'objective':'-xcorr',
                    'iterations':8,
                    'cleanup':True,
                    'biascorr':False,
                    'biasdist':100,
                    'linreg': 'bestlinreg_20180117_scaling',
                    'norot':   True,
                    'noshift': True,
                    'noshear': True,
                    'close':   True }
        
        if patient.rigid or patient.skullreg:
            options['reg_type']='-lsq6'

        if patient.symmetric:
            options['symmetric']=True 
        tps=[ i for (i, _) in patient.items() ]

        # Here we are relying on the time point order (1)
        if patient.skullreg or patient.rigid:
            print("linearlngtemplate_v11: rigid or skullreg")
            samples= [ [tp.stx_ns_mnc['t1'], tp.stx_ns_mnc['masknoles']]
                        for (i, tp) in patient.items() ]
            # replicate skullreg (?)
            unscale_xfms = [ tp.stx_ns_xfm['unscale_t1'] 
                        for (i, tp) in patient.items() ]
            xfmavg(unscale_xfms,minc.tmp('avg_unscale.xfm'))
            minc.xfminvert(minc.tmp('avg_unscale.xfm'), patient.template['scale_xfm'])
        else:
            print("linearlngtemplate_v11: default")
            samples= [ [tp.stx_mnc['t1'],    tp.stx_mnc['masknoles']]
                        for (i, tp) in patient.items() ]
            # generate identity
            minc.param2xfm(patient.template['scale_xfm'])

        if patient.fast:
            options['iterations'] = 2

        work_prefix = patient.workdir + os.sep + 'lin'

        print("linearlngtemplate_v11: generate_linear_model")
        if not patient.skullreg and not patient.rigid:
            output = generate_linear_model(samples, model=atlas, mask=atlas_mask, 
                        options=options, work_prefix=work_prefix)
        else:
            output = generate_linear_model(samples, options=options, 
                        work_prefix=work_prefix)

        # if patient.skullreg: 
        #     # run 2nd stage here
        #     # THIS IS UNSTABLE too :(
        #     samples_2 = []
        #     # generate new samples
        #     for i,_ in enumerate(output['xfm']):
        #         corr_xfm=output['xfm'][i].xfm
        #         scan=output['scan'][i].scan

        #         outertable=output['scan'][i].scan + '_outer.mnc'
        #         outertable_mask=output['scan'][i].scan + '_outer_mask.mnc'
        #         outertable_dist=output['scan'][i].scan + '_outer_dist.mnc'

        #         minc.resample_labels(patient[tps[i]].stx_ns_mnc['skull'],
        #              minc.tmp('{}_skull.mnc'.format(i)),
        #              transform=corr_xfm, like=scan)

        #         #minc.binary_morphology(minc.tmp('{}_skull.mnc'.format(i)),'D[3]',skull_mask)
        #         generate_skull_mask_dist_map(minc.tmp('{}_skull.mnc'.format(i)), outertable, outertable_mask, outertable_dist  )

        #         samples_2.append([outertable_dist, outertable_mask])
        #     #
        #     work_prefix_2 = patient.workdir+os.sep+'lin2'
        #     output_2 = generate_linear_model(samples_2, options=options_2, work_prefix=work_prefix_2)
        #     #minc.resample_smooth(output_2['model'].scan,   patient.template['linear_template'],     transform=patient.template['scale_xfm'])

        #     # need to apply full transformation to all files
        #     # to generate final version of the template
        #     _masks=[]
        #     _masks_redskull=[]
        #     _t1s=[]
        #     for i,_ in enumerate(output_2['xfm']):
        #         minc.xfmconcat([output['xfm'][i].xfm, output_2['xfm'][i].xfm, patient.template['scale_xfm']], minc.tmp("full_{}.xfm".format(i)))
        #         # generate samples
        #         minc.resample_smooth(patient[tps[i]].stx_ns_mnc['t1'], minc.tmp("t1_{}.mnc".format(i)), transform=minc.tmp("full_{}.xfm".format(i)))
        #         minc.resample_labels(patient[tps[i]].stx_ns_mnc['mask'], minc.tmp("mask_{}.mnc".format(i)), transform=minc.tmp("full_{}.xfm".format(i)))
        #         minc.resample_labels(patient[tps[i]].stx_ns_mnc['redskull'], minc.tmp("redskull_{}.mnc".format(i)), transform=minc.tmp("full_{}.xfm".format(i)))

        #         _t1s+=[minc.tmp("t1_{}.mnc".format(i))]
        #         _masks+=[minc.tmp("mask_{}.mnc".format(i))]
        #         _masks_redskull+=[minc.tmp("redskull_{}.mnc".format(i))]

        #     minc.average(_masks,minc.tmp("all_masks.mnc"))
        #     minc.multiple_volume_similarity(_masks_redskull, maj=patient.template['linear_template_redskull'], bg=True)
        #     minc.average(_t1s, patient.template['linear_template'], sdfile=patient.template['linear_template_sd'])
        #     minc.calc([minc.tmp("all_masks.mnc")],      "A[0]>0.5?1:0",patient.template['linear_template_mask'],labels=True)

        if patient.skullreg:
            #minc.resample_smooth(output['model'].scan,   patient.template['linear_template'],     transform=patient.template['scale_xfm'])
            #minc.resample_labels(output['model'].mask,   patient.template['linear_template_mask'],transform=patient.template['scale_xfm'])
            #minc.resample_smooth(output['model_sd'].scan,patient.template['linear_template_sd'],  transform=patient.template['scale_xfm'])
            # resample skull segs
            _masks_redskull=[]
            _scans=[]
            _masks=[]

            for i,_ in enumerate(output['xfm']):
                corr_xfm=output['xfm'][i].xfm
                scaled_xfm=minc.tmp('{}_scaled.xfm'.format(i))
                
                minc.xfmconcat([corr_xfm, patient.template['scale_xfm']], scaled_xfm)

                minc.resample_smooth(patient[tps[i]].stx_ns_mnc['t1'],      
                                    minc.tmp("t1_{}.mnc".format(i)), 
                                    transform=scaled_xfm)
                
                minc.resample_labels(patient[tps[i]].stx_ns_mnc['mask'],     
                                     minc.tmp("mask_{}.mnc".format(i)), 
                                     transform=scaled_xfm)
                
                minc.resample_labels(patient[tps[i]].stx_ns_mnc['redskull'], 
                                     minc.tmp("redskull_{}.mnc".format(i)), 
                                     transform=scaled_xfm)


                _masks_redskull.append(minc.tmp("redskull_{}.mnc".format(i)))
                _scans.append(minc.tmp("t1_{}.mnc".format(i)))
                _masks.append(minc.tmp("mask_{}.mnc".format(i)))

            minc.multiple_volume_similarity(_masks,maj=patient.template['linear_template_mask'],bg=True)
            minc.multiple_volume_similarity(_masks_redskull, maj=patient.template['linear_template_redskull'], bg=True)
            minc.average(_scans, patient.template['linear_template'], sdfile=patient.template['linear_template_sd'])

        elif patient.rigid:
            minc.resample_smooth(output['model'].scan,   patient.template['linear_template'],     transform=patient.template['scale_xfm'])
            minc.resample_labels(output['model'].mask,   patient.template['linear_template_mask'],transform=patient.template['scale_xfm'])
            minc.resample_smooth(output['model_sd'].scan,patient.template['linear_template_sd'],  transform=patient.template['scale_xfm'])
        else:
            shutil.copyfile(output['model'].scan,   patient.template['linear_template'])
            shutil.copyfile(output['model'].mask,   patient.template['linear_template_mask'])
            shutil.copyfile(output['model_sd'].scan,patient.template['linear_template_sd'])

        # Create the new stx space using the template
        if patient.skullreg:
            # registering using skull is extremely unstable :(
            # ipl.registration.linear_register(patient.template['linear_template'],
            #                     atlas, patient.template['stx2_xfm'],
            #                     source_mask=patient.template['linear_template_mask'], 
            #                     target_mask=atlas_mask_novent)
            
            # 
            register_using_skull(patient.template['linear_template'],
                                 patient.template['linear_template_redskull'],
                                 patient.template['linear_template_mask'],
                                 atlas,
                                 atlas_brain_skull,
                                 atlas_mask,
                                 patient.template['stx2_xfm'])
            

        elif patient.large_atrophy:
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
        
        for (i, tp) in patient.items():
            biascorr=None
            if patient.dobiascorr:
                biascorr=output['biascorr'][k]
            # Here we are relying on the time point order (1) - see above
            if patient.skullreg:
                # transform2 disabled for now
                jobs.append(post_process.remote( patient, i, tp, output['xfm'][k], biascorr, rigid=True, 
                            transform2=None, scale_xfm=patient.template['scale_xfm']))
                
            else:
                jobs.append(post_process.remote( patient, i, tp, output['xfm'][k], biascorr, rigid=patient.rigid ))
            k+=1
        
        # wait for all substeps to finish
        ray.wait(jobs, num_returns=len(jobs))
        
        # cleanup temporary files
        shutil.rmtree(work_prefix)
        
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

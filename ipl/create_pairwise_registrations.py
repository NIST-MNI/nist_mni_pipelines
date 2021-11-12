#! /usr/bin/env python
# TODO: move to ..

import shutil
import os
import json

from iplMincTools import mincTools,mincError
from scoop import futures, shared

def generate_xfm_model(i , j, xfm1, xfm2, mri1, mri2, mask1, mask2, seg1, seg2, output_base,step=2,baa=False):
    with mincTools(verbose=2) as minc:
        # all xfms are mapping subject to common space, so to map one subject to another it will be xfm1 * xfm2^1
        minc.xfminvert(xfm2,minc.tmp('xfm1.xfm'))
        # concatenate xfms
        minc.xfmconcat([xfm1,minc.tmp('xfm1.xfm')],minc.tmp('xfm1_dot_xfm2_inv.xfm'))
        # normalize xfms
        minc.xfm_normalize(minc.tmp('xfm1_dot_xfm2_inv.xfm'),mri1,output_base+'_map.xfm',step=step)

        # resample mris
        minc.resample_smooth(mri2,minc.tmp('mri2.mnc'),transform=output_base+'_map.xfm',invert_transform=True)
        # resample segs
        minc.resample_labels(seg2,minc.tmp('seg2.mnc'),transform=output_base+'_map.xfm',invert_transform=True,baa=baa)
        minc.resample_labels(mask2,minc.tmp('mask2.mnc'),transform=output_base+'_map.xfm',invert_transform=True,baa=baa)
        # calculate CC, MI
        cc = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='cc')
        nmi = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='nmi')
        ncc = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='ncc')
        msq = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='msq')
        # calculate label overlap
        gtc = minc.label_similarity(seg1,minc.tmp('seg2.mnc'),method='gtc')

        # write out result
        with open(output_base+'_similarity.txt','w') as f:
            f.write("{},{},{},{},{},{},{}\n".format(i,j,cc,ncc,nmi,msq,gtc))

def generate_xfm_direct_minctracc(i , j, mri1, mri2, mask1, mask2, seg1, seg2, output_base,step=2,baa=False):
    with mincTools(verbose=2) as minc:
        # normalize xfms
        minc.non_linear_register_full(mri1,mri2,output_base+'_map.xfm',level=step,source_mask=mask1,target_mask=mask2)

        # resample mris
        minc.resample_smooth(mri2,minc.tmp('mri2.mnc'),transform=output_base+'_map.xfm',invert_transform=True)
        # resample segs
        minc.resample_labels(seg2,minc.tmp('seg2.mnc'),transform=output_base+'_map.xfm',invert_transform=True,baa=baa)
        minc.resample_labels(mask2,minc.tmp('mask2.mnc'),transform=output_base+'_map.xfm',invert_transform=True,baa=baa)
        # calculate CC, MI
        cc = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='cc')
        nmi = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='nmi')
        ncc = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='ncc')
        msq = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='msq')
        # calculate label overlap
        gtc = minc.label_similarity(seg1,minc.tmp('seg2.mnc'),method='gtc')

        # write out result
        with open(output_base+'_similarity.txt','w') as f:
            f.write("{},{},{},{},{},{},{}\n".format(i,j,cc,ncc,nmi,msq,gtc))


def generate_xfm_direct_ANTS_CC(i , j, mri1, mri2, mask1, mask2, seg1, seg2, output_base,baa=False,step=2):
    with mincTools(verbose=2) as minc:
        # normalize xfms
        param_cc={'cost_function':'CC','iter':'40x40x40x00'}
        
        minc.non_linear_register_ants(mri1,mri2,minc.tmp('transform.xfm'),target_mask=mask2,parameters=param_cc)
        minc.xfm_normalize(minc.tmp('transform.xfm'),mri1,output_base+'_map.xfm',step=step)
        
        # resample mris
        minc.resample_smooth(mri2,minc.tmp('mri2.mnc'),transform=output_base+'_map.xfm',invert_transform=True)
        # resample segs
        minc.resample_labels(seg2,minc.tmp('seg2.mnc'),transform=output_base+'_map.xfm',invert_transform=True,baa=baa)
        minc.resample_labels(mask2,minc.tmp('mask2.mnc'),transform=output_base+'_map.xfm',invert_transform=True,baa=baa)
        # calculate CC, MI
        cc = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='cc')
        nmi = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='nmi')
        ncc = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='ncc')
        msq = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='msq')
        # calculate label overlap
        gtc = minc.label_similarity(seg1,minc.tmp('seg2.mnc'),method='gtc')

        # write out result
        with open(output_base+'_similarity.txt','w') as f:
            f.write("{},{},{},{},{},{},{}\n".format(i,j,cc,ncc,nmi,msq,gtc))


def generate_xfm_direct_ANTS_MI(i , j, mri1, mri2, mask1, mask2, seg1, seg2, output_base,baa=False,step=2):
    with mincTools(verbose=2) as minc:
        # normalize xfms
        param_mi={'cost_function':'MI','iter':'40x40x40x00','cost_function_par':'1,32'}

        minc.non_linear_register_ants(mri1,mri2,minc.tmp('transform.xfm'),target_mask=mask2,parameters=param_mi)
        minc.xfm_normalize(minc.tmp('transform.xfm'),mri1,output_base+'_map.xfm',step=step)
        
        # resample mris
        minc.resample_smooth(mri2,minc.tmp('mri2.mnc'),transform=output_base+'_map.xfm',invert_transform=True)
        # resample segs
        minc.resample_labels(seg2,minc.tmp('seg2.mnc'),transform=output_base+'_map.xfm',invert_transform=True,baa=baa)
        minc.resample_labels(mask2,minc.tmp('mask2.mnc'),transform=output_base+'_map.xfm',invert_transform=True,baa=baa)
        # calculate CC, MI
        cc = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='cc')
        nmi = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='nmi')
        ncc = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='ncc')
        msq = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='msq')
        # calculate label overlap
        gtc = minc.label_similarity(seg1,minc.tmp('seg2.mnc'),method='gtc')

        # write out result
        with open(output_base+'_similarity.txt','w') as f:
            f.write("{},{},{},{},{},{},{}\n".format(i,j,cc,ncc,nmi,msq,gtc))
            
def generate_xfm_direct_elastix_cc(i , j, mri1, mri2, mask1, mask2, seg1, seg2, output_base,baa=False,step=2):
    with mincTools(verbose=2) as minc:

        param_cc="""
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
(UseDirectionCosines "true")

(ShowExactMetricValue "false")

(Registration "MultiResolutionRegistration")
(Interpolator "BSplineInterpolator" )
(ResampleInterpolator "FinalBSplineInterpolator" )
(Resampler "DefaultResampler" )

(FixedImagePyramid  "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

(Optimizer "AdaptiveStochasticGradientDescent")
(Transform "BSplineTransform")
(Metric "AdvancedNormalizedCorrelation")

(FinalGridSpacingInPhysicalUnits 4)

(HowToCombineTransforms "Compose")

(ErodeMask "false")

(NumberOfResolutions 3)

(ImagePyramidSchedule 8 8 8  4 4 4 2 2 2)

(MaximumNumberOfIterations 2000 2000 2000 )
(MaximumNumberOfSamplingAttempts 3)

(NumberOfSpatialSamples 1024 1024 4096 )

(NewSamplesEveryIteration "true")
(ImageSampler "Random" )

(BSplineInterpolationOrder 1)

(FinalBSplineInterpolationOrder 3)

(DefaultPixelValue 0)

(WriteResultImage "false")

// The pixel type and format of the resulting deformed moving image
(ResultImagePixelType "float")
(ResultImageFormat "mnc")
"""        
        # normalize xfms
        minc.register_elastix(mri1,mri2,output_xfm=minc.tmp('transform.xfm'),source_mask=mask1,target_mask=mask2,parameters=param_cc)
        minc.xfm_normalize(minc.tmp('transform.xfm'),mri1,output_base+'_map.xfm',step=step)

	# resample mris
        minc.resample_smooth(mri2,minc.tmp('mri2.mnc'),transform=output_base+'_map.xfm',invert_transform=True)  
        # resample segs
        minc.resample_labels(seg2,minc.tmp('seg2.mnc'),transform=output_base+'_map.xfm',invert_transform=True,baa=baa)
        minc.resample_labels(mask2,minc.tmp('mask2.mnc'),transform=output_base+'_map.xfm',invert_transform=True,baa=baa)
        # calculate CC, MI
        cc = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='cc')
        nmi = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='nmi')
        ncc = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='ncc')
        msq = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='msq')
        # calculate label overlap
        gtc = minc.label_similarity(seg1,minc.tmp('seg2.mnc'),method='gtc')

        # write out result
        with open(output_base+'_similarity.txt','w') as f:
            f.write("{},{},{},{},{},{},{}\n".format(i,j,cc,ncc,nmi,msq,gtc))


def generate_xfm_direct_elastix_mi(i , j, mri1, mri2, mask1, mask2, seg1, seg2, output_base,baa=False,step=2):
    with mincTools(verbose=2) as minc:

        param_mi="""
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
(UseDirectionCosines "true")

(ShowExactMetricValue "false")

(Registration "MultiResolutionRegistration")
(Interpolator "BSplineInterpolator" )
(ResampleInterpolator "FinalBSplineInterpolator" )
(Resampler "DefaultResampler" )

(FixedImagePyramid  "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

(Optimizer "AdaptiveStochasticGradientDescent")
(Transform "BSplineTransform")
(Metric "AdvancedMattesMutualInformation")

(FinalGridSpacingInPhysicalUnits 4)

(HowToCombineTransforms "Compose")

(ErodeMask "false")

(NumberOfResolutions 3)

(ImagePyramidSchedule 8 8 8  4 4 4 2 2 2)

(MaximumNumberOfIterations 2000 2000 2000 )
(MaximumNumberOfSamplingAttempts 3)

(NumberOfSpatialSamples 1024 1024 4096 )

(NewSamplesEveryIteration "true")
(ImageSampler "Random" )

(BSplineInterpolationOrder 1)

(FinalBSplineInterpolationOrder 3)

(DefaultPixelValue 0)

(WriteResultImage "false")

// The pixel type and format of the resulting deformed moving image
(ResultImagePixelType "float")
(ResultImageFormat "mnc")
"""
        # normalize xfms
        minc.register_elastix(mri1,mri2,output_xfm=minc.tmp('transform.xfm'),source_mask=mask1,target_mask=mask2,parameters=param_mi)
        minc.xfm_normalize(minc.tmp('transform.xfm'),mri1,output_base+'_map.xfm',step=step)
        
        # resample mris
        minc.resample_smooth(mri2,minc.tmp('mri2.mnc'),transform=output_base+'_map.xfm',invert_transform=True)
        # resample segs
        minc.resample_labels(seg2,minc.tmp('seg2.mnc'),transform=output_base+'_map.xfm',invert_transform=True,baa=baa)
        minc.resample_labels(mask2,minc.tmp('mask2.mnc'),transform=output_base+'_map.xfm',invert_transform=True,baa=baa)
        # calculate CC, MI
        cc = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='cc')
        nmi = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='nmi')
        ncc = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='ncc')
        msq = minc.similarity(mri1,minc.tmp('mri2.mnc'),ref_mask=mask1,sample_mask=minc.tmp('mask2.mnc'),method='msq')
        # calculate label overlap
        gtc = minc.label_similarity(seg1,minc.tmp('seg2.mnc'),method='gtc')

        # write out result
        with open(output_base+'_similarity.txt','w') as f:
            f.write("{},{},{},{},{},{},{}\n".format(i,j,cc,ncc,nmi,msq,gtc))


if __name__ == '__main__':
    model='model_nl'
    output='pairwise'
    input_prefix='minc_prep_bbox/'
    step_size=2
    
    model_results={}

    with open(model+os.sep+'results.json','r') as f:
        model_results=json.load(f)

    if not os.path.exists(output):
        os.makedirs(output)
    # generate fake seg and mri names
    #TODO replace with CSV file input 
    mri= [input_prefix+k['name'] for k in model_results['scan']]
    mask=[input_prefix+k['name'].rstrip('.mnc')+'_mask.mnc' for k in model_results['scan']]
    seg= [input_prefix+k['name'].rstrip('.mnc')+'_glm.mnc' for k in model_results['scan']]

    print(repr(mri))
    print(repr(mask))
    print(repr(seg))
    rr=[]
    
    # generate uniform file names!
    for (i,j) in enumerate(model_results['xfm']):
        for (k,t) in enumerate(model_results['xfm']):
            if i!=k:
                if not os.path.exists(output+os.sep+'A_{:02d}_{:02d}_map.xfm'.format(i,k) ) :
                    rr.append( generate_xfm_model.remote(i,k,
                                j['xfm'],t['xfm'],
                                mri[i],mri[k],
                                mask[i],mask[k],
                                seg[i],seg[k],
                                output+os.sep+'A_{:02d}_{:02d}'.format(i,k) ) )
                            
                if not os.path.exists(output+os.sep+'B_{:02d}_{:02d}_map.xfm'.format(i,k) ) :
                    rr.append( generate_xfm_direct_minctracc.remote(i,k,
                                mri[i],mri[k],
                                mask[i],mask[k],
                                seg[i],seg[k],
                                output+os.sep+'B_{:02d}_{:02d}'.format(i,k),
                                step=2) )

                if not os.path.exists(output+os.sep+'C_{:02d}_{:02d}_map.xfm'.format(i,k) ) :
                    rr.append( generate_xfm_direct_ANTS_CC.remote(i,k,
                                mri[i],mri[k],
                                mask[i],mask[k],
                                seg[i],seg[k],
                                output+os.sep+'C_{:02d}_{:02d}'.format(i,k) ) )

                if not os.path.exists( output+os.sep+'D_{:02d}_{:02d}_map.xfm'.format(i,k) ) :
                    rr.append( generate_xfm_direct_ANTS_MI.remote(i,k,
                                mri[i],mri[k],
                                mask[i],mask[k],
                                seg[i],seg[k],
                                output+os.sep+'D_{:02d}_{:02d}'.format(i,k) ) )
                            
                if not os.path.exists( output+os.sep+'E_{:02d}_{:02d}_map.xfm'.format(i,k) ) :
                    rr.append( generate_xfm_direct_elastix_cc.remote(i,k,
                                mri[i],mri[k],
                                mask[i],mask[k],
                                seg[i],seg[k],
                                output+os.sep+'E_{:02d}_{:02d}'.format(i,k) ) )

                if not os.path.exists( output+os.sep+'F_{:02d}_{:02d}_map.xfm'.format(i,k) ) :
                    rr.append( generate_xfm_direct_elastix_mi.remote(i,k,
                                mri[i],mri[k],
                                mask[i],mask[k],
                                seg[i],seg[k],
                                output+os.sep+'F_{:02d}_{:02d}'.format(i,k) ) )
                
    ray.wait(rr,num_returns=len(rr))
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

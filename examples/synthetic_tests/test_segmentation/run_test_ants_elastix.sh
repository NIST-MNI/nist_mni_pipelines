#! /bin/sh


PREFIX=$(pwd)/../../python
PARALLEL=1

export PYTHONPATH=$PREFIX:$PYTHONPATH


cat - > library_description_ants.json <<END
{
  "reference_model": "data/ellipse_0_blur.mnc",
  "reference_mask":  "data/ellipse_0_mask.mnc",
  "reference_local_model" : null,
  "reference_local_mask" :  null,
  "library":"seg_subjects.lst",
  "build_remap":         [ [1, 1], [2,2], [3,3], [4,4], [5,5], [6,6],  [7,7], [8,8] ],
  "build_flip_remap":    null,
  "parts": 0,
  "classes": 9,
  "build_symmetric": false,
  "build_symmetric_flip": false,
  "symmetric_lut": null,
  "denoise": false,
  "denoise_beta": null,
  "linear_register": false,
  "local_linear_register": true,
  "non_linear_register": true,
  "resample_order": 2,
  "resample_baa": true,
  "non_linear_register_ants": true
}
END


if [ ! -e test_lib_ants/library.json ];then
# create the library
python -m scoop -n $PARALLEL $PREFIX/iplScoopFusionSegmentation.py  \
  --create library_description_ants.json --output test_lib_ants --debug
fi

# run cross-validation


cat - > cv_ants2.json <<END
{
  "validation_library":"seg_subjects.lst",
  "iterations":-1,
  "cv":1,
  "fuse_variant":"fuse",
  "cv_variant": "cv2", 
  "regularize_variant":"gc"
}
END

cat - > elastix_lin.txt <<END
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
(UseDirectionCosines "true")

(ShowExactMetricValue "false")

(Registration "MultiResolutionRegistration")
(Interpolator "BSplineInterpolator" )
(ResampleInterpolator "FinalBSplineInterpolator" )
(Resampler    "DefaultResampler" )

(FixedImagePyramid  "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

(Optimizer "AdaptiveStochasticGradientDescent")
(Transform "EulerTransform")
(Metric    "AdvancedNormalizedCorrelation")

(AutomaticTransformInitialization "true")
(AutomaticScalesEstimation "true")
(MaximumStepLength 1.0)

(HowToCombineTransforms "Compose")

(ErodeMask "false")

(NumberOfResolutions 1)

(ImagePyramidSchedule  4 4 4 )

(MaximumNumberOfIterations 200 )
(MaximumNumberOfSamplingAttempts 3)

(NumberOfSpatialSamples 1024 )
(CheckNumberOfSamples false)

(NewSamplesEveryIteration "true")
(ImageSampler "Random" )

(BSplineInterpolationOrder 1)

(FinalBSplineInterpolationOrder 1)

(DefaultPixelValue 0)

(WriteResultImage "false")

// The pixel type and format of the resulting deformed moving image
(ResultImagePixelType "float")
(ResultImageFormat "mnc")
END

cat - > segment_ants2.json <<END
{
  "initial_register": false,
  "initial_local_register": true,
  
  "local_register_type": "elastix",
  "local_register_options": "@elastix_lin.txt",
  
  "non_linear_pairwise": true,
  "non_linear_register": false,
  "non_linear_register_ants": true,
  "non_linear_pairwise_ants": true,

  "simple_fusion": false,
  "non_linear_register_level": 4,
  "pairwise_level": 4,

  "resample_order": 2,
  "resample_baa": true,
  "library_preselect": 3,
  "segment_symmetric": false,

  "fuse_options":
  {
      "patch": 1,
      "search": 1,
      "threshold": 0.0,
      "gco_diagonal": false,
      "gco_wlabel": 0.0001,
      "gco":true
  }

}
END

cat - > ec_train_ants2.json <<END
{
  "method"    : "AdaBoost",
  "method_n"  : 200,
  "border_mask": true,
  "border_mask_width": 2,
  "use_coord": true,
  "use_joint": true,
  "patch_size": 1,
  "use_raw":    true,
  "normalize_input": false,
  "primary_features": 1,
  "max_samples": -1,
  "sample_pick":"first",

  "antialias_labels": false,
  "blur_labels": 2,
  "expit_labels": 2,
  "normalize_labels": true
}
END

python -m scoop -n $PARALLEL \
  $PREFIX/iplScoopFusionSegmentation.py \
   --output test_cv_ants \
   --debug  \
   --segment test_lib_ants \
   --cv cv_ants2.json \
   --options segment_ants2.json 
#   --train-ec ec_train_ants2.json

#! /bin/sh

PREFIX=$(pwd)/../../..

if [ -z $PARALLEL ];then
PARALLEL=4
fi

export PYTHONPATH=$PREFIX:$PYTHONPATH
xport OMP_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

PYTHON=python3

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
  
  "initial_register": null,
  "initial_local_register": {
    "type": "ants",
    "bbox": true
   },
  
  "non_linear_register": true,
  "resample_order": 2,
  "resample_baa": true,
  "non_linear_register_ants": true
}
END


if [ ! -e test_lib_ants/library.json ];then
# create the library
${PYTHON} -m scoop -n $PARALLEL $PREFIX/iplScoopFusionSegmentation.py  \
  --create library_description_ants.json --output test_lib_ants --debug
fi

# run cross-validation

cat - > cv_ants.json <<END
{
  "validation_library":"seg_subjects.lst",
  "iterations":-1,
  "cv":1
}
END


cat - > segment_ants.json <<END
{

  "initial_register": null,
  "initial_local_register": {
    "type": "ants",
    "bbox": true
   },
  
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
  },

  "qc_options":
  {
    "clamp": false,
    "image_range": [0,200],
    "spectral_mask": true,
    "big": true,
    "contours": true
  }
}
END

${PYTHON} -m scoop -n ${PARALLEL} \
  $PREFIX/iplScoopFusionSegmentation.py \
   --output test_cv_ants_no_ec \
   --debug  \
   --segment test_lib_ants \
   --cv cv_ants.json \
   --options segment_ants.json 


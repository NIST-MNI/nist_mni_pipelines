#! /bin/sh
set -e 

PREFIX=$(pwd)/../../python

export PYTHONPATH=$PREFIX:$PYTHONPATH


cat - > library_description.json <<END
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
  "non_linear_register": false,
  "resample_order": 2,
  "resample_baa": true
}
END


if [ ! -e test_lib/library.json ];then
# create the library
python -m scoop -n 4 $PREFIX/iplScoopFusionSegmentation.py  \
  --create library_description.json --output test_lib --debug \
  --cleanup
fi

# run cross-validation

cat - > cv.json <<END
{
  "validation_library":"seg_subjects.lst",
  "iterations":10,
  "cv":2,
  "fuse_variant":"fuse_1",
  "ec_variant":"ec_2",
  "cv_variant":"cv_2",
  "regularize_variant":"reg_1"
}
END


cat - > segment.json <<END
{
  "initial_local_register": true,
  "non_linear_pairwise": false,
  "non_linear_register": false,

  "simple_fusion": false,
  "non_linear_register_level": 4,
  "pairwise_level": 4,

  "resample_order": 2,
  "resample_baa": true,
  "library_preselect": 3,
  "segment_symmetric": false,

  "fuse_options":
  {
      "patch": 0,
      "search": 0,
      "threshold": 0.0,
      "gco_diagonal": false,
      "gco_wlabel": 0.001,
      "gco":false
  }

}
END

cat - > ec_train.json <<END
{
  "method" : "AdaBoost",
  "method_n" : 100,
  "border_mask": true,
  "border_mask_width": 3,
  "use_coord": false,
  "use_joint": false,

  "train_rounds": 3,
  "train_cv": 2
}
END

mkdir -p test_cv
python -m scoop -n 4 -vvv \
  $PREFIX/iplScoopFusionSegmentation.py \
   --output test_cv \
   --debug  \
   --segment test_lib \
   --cv cv.json \
   --options segment.json \
   --train-ec ec_train.json \
   --cleanup

#! /bin/sh


PREFIX=$(pwd)/../../python

export PYTHONPATH=$PREFIX:$PYTHONPATH


cat - > library_description_triple.json <<END
{
  "reference_model": "data/ellipse_0_blur.mnc",
  "reference_mask":  "data/ellipse_0_mask.mnc",
  "reference_add": [ "data/ellipse_0_blur.mnc", "data/ellipse_0_blur.mnc"],
  "reference_local_model" : null,
  "reference_local_mask" :  null,
  "library":"seg_subjects_triple.lst",
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
  "resample_baa": true,
  "modalities" : 3 
}
END


if [ ! -e test_lib_triple/library.json ];then
# create the library
python -m scoop -n 1 $PREFIX/iplScoopFusionSegmentation.py  \
  --create library_description_triple.json --output test_lib_triple --debug
fi

# run cross-validation
cat - > cv_triple.json <<END
{
  "validation_library":"seg_subjects_triple.lst",
  "iterations":10,
  "cv":2,
  "fuse_variant":"fuse_1",
  "ec_variant":"ec_1",
  "cv_variant":"cv_1",
  "regularize_variant":"reg_1"
}
END


cat - > segment_triple.json <<END
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
      "patch": 1,
      "search": 1,
      "threshold": 0.0,
      "gco_diagonal": false,
      "gco_wlabel": 0.001,
      "gco":false
  }

}
END

cat - > ec_train_triple.json <<END
{
  "method" : "AdaBoost",
  "method_n" : 100,
  "border_mask": true,
  "border_mask_width": 3,
  "use_coord": true,
  "use_joint": true,

  "train_rounds": 3,
  "train_cv": 2
}
END

mkdir -p test_cv_triple

python -m scoop -n 1 -vvv \
  $PREFIX/iplScoopFusionSegmentation.py \
   --output test_cv_triple \
   --debug  \
   --segment test_lib_triple \
   --cv cv_triple.json \
   --options segment_triple.json \
   --train-ec ec_train_triple.json

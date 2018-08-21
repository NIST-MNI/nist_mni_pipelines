#! /bin/sh


PREFIX=$(pwd)/../../../

if [ -z $PARALLEL ];then
PARALLEL=4
fi

export PYTHONPATH=$PREFIX:$PYTHONPATH


cat - > library_description.yaml <<END
---
reference_model: data/ellipse_0_blur.mnc
reference_mask: data/ellipse_0_mask.mnc
reference_local_model:
reference_local_mask:
library: seg_subjects.lst
build_remap:
- [1, 1]
- [2, 2]
- [3, 3]
- [4, 4]
- [5, 5]
- [6, 6]
- [7, 7]
- [8, 8]
build_flip_remap:
parts: 0
classes: 9
build_symmetric: false
build_symmetric_flip: false
symmetric_lut:
denoise: false
denoise_beta:
linear_register: false
local_linear_register: true
non_linear_register: false
resample_order: 2
resample_baa: true
END


if [ ! -e test_lib/library.yaml ] && [ ! -e test_lib/library.json ] ;then
# create the library
python -m scoop -n 4 $PREFIX/ipl_segmentation_library_prepare.py  \
  --create library_description.yaml --output test_lib --debug
fi


# run cross-validation

cat - > cv.yaml <<END
---
validation_library: seg_subjects.lst
iterations: 10
cv: 2
fuse_variant: fuse_1
ec_variant: ec_1
cv_variant: cv_1
regularize_variant: reg_1
END


cat - > segment.yaml <<END
---
initial_local_register: true
non_linear_pairwise: false
non_linear_register: false
simple_fusion: false
non_linear_register_level: 4
pairwise_level: 4
resample_order: 2
resample_baa: true
library_preselect: 3
segment_symmetric: false
fuse_options:
  patch: 0
  search: 0
  threshold: 0
  gco_diagonal: false
  gco_wlabel: 0.001
  gco: false
END

cat - > ec_train.yaml <<END
---
method: AdaBoost
method_n: 100
border_mask: true
border_mask_width: 3
use_coord: true
use_joint: true
train_rounds: 3
train_cv: 2
END

mkdir -p test_cv
python -m scoop -n $PARALLEL -vvv \
  $PREFIX/ipl_segmentation_library_prepare.py \
   --output test_cv \
   --debug  \
   --library test_lib \
   --cv cv.yaml \
   --options segment.yaml \
   --train-ec ec_train.yaml

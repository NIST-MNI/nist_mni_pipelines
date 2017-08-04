#! /bin/sh


PREFIX=$(pwd)/../../python

export PYTHONPATH=$PREFIX:$PYTHONPATH

PARALLEL=3
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

cat - > library_description.json <<END
{
  "reference_model": "adni_ec_bbox/model_t1w.mnc",
  "reference_mask":  "adni_ec_bbox/model_t1w.mnc",
  
  "reference_local_model" : null,
  "reference_local_mask" :  null,
  "library":"ec_library_bbox.csv",

  "build_remap":         [  [18 ,1],  
                            [1  ,2],   
                            [240,3], 
                            [16 ,4],  
                            [21 ,5],  
                            [19 ,6],  
                            [6  ,7],   
                            [47 ,8],  
                            [74 ,9],  
                            [87 ,10] ],
                            
  "build_flip_remap":    [  [3,  1],
                            [199,2],
                            [245,3],
                            [20, 4],
                            [4,  5],
                            [241,6],
                            [248,7],
                            [90, 8],
                            [5,  9],
                            [91, 10]  
                            ],
  "classes": 11,
  "build_symmetric": true,
  "build_symmetric_flip": true,
  "symmetric_lut": null,
  "denoise": false,
  "denoise_beta": null,
  
  "initial_register": false,
  "initial_local_register": false,
  
  "non_linear_register": true,
  "non_linear_register_ants": true,
  "non_linear_register_level": 1,
  
  
  "resample_order": 2,
  "resample_baa": true,
  "extend_boundary": 4,
  "op_mask": "E[2] D[4]",
  
  "create_patch_norm_lib": true,
  "patch_norm_lib_pct": 0.1,
  "patch_norm_lib_patch": 2
}
END


if [ ! -e test_lib_nl2/library.json ];then
# create the library

python -m scoop -n $PARALLEL $PREFIX/iplScoopFusionSegmentation.py  \
  --create library_description.json --output test_lib_nl2 --debug
fi
# run cross-validation

cat - > cv.json <<END
{
  "validation_library":"ec_library_bbox.csv",
  "iterations":-1,
  "cv":1,
  "fuse_variant":"fuse_1",
  "cv_variant":"cv_2",
  "regularize_variant":"reg_1"
}
END


cat - > segment.json <<END
{
  "initial_local_register": false,
  "non_linear_pairwise": false,
  
  "non_linear_register": true,
  "non_linear_register_ants": true,
  
  "simple_fusion": false,
  "non_linear_register_level": 1,
  "pairwise_level": 4,

  "resample_order": 2,
  "resample_baa": true,
  "library_preselect": 3,
  "segment_symmetric": true,

  "fuse_options":
  {
      "patch": 2,
      "search": 1,
      "threshold": 0.0,
      "gco_diagonal": false,
      "gco_wlabel": 0.001,
      "gco":false
  }

}
END

cat - > ec_train.json <<END
{
  "method" :   "AdaBoost",
  "method_n" : 500,
  "method2" :   "AdaBoost",
  "method2_n" : 500,
  
  "border_mask": true,
  "border_mask_width": 3,
  "use_coord": true,
  "use_joint": true,

  "train_rounds": -1,
  "train_cv": 1
}
END

mkdir -p test_cv_nl2

if [ ! -e test_cv_nl2/cv_2_stats.json ];then
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
python -m scoop -n $PARALLEL -vvv \
  $PREFIX/iplScoopFusionSegmentation.py \
   --output test_cv_nl2 \
   --debug  \
   --segment test_lib_nl2 \
   --cv cv.json \
   --options segment.json  \
   --cleanup 
fi

mkdir -p test_cv_nl2_re

cat - > cv_re.json <<END
{
  "validation_library":"ec_library_bbox_re.csv",
  "iterations":-1,
  "cv":1,
  "fuse_variant":"fuse_1",
  "cv_variant":"cv_2",
  "regularize_variant":"reg_1"
}
END


rm -f ec_library_bbox_re.csv

for i in $(cat ec_library_bbox.csv);do
t1w=$(echo $i|cut -d , -f 1)
seg=$(echo $i|cut -d , -f 2)
n=$(basename $t1w .mnc)
a=test_cv_nl2/${n}_cv_2_seg.mnc
echo $t1w,$seg,$a>>ec_library_bbox_re.csv
done


if [ ! -e test_cv_nl2_re/cv_2_stats.json ];then

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
python -m scoop -n $PARALLEL -vvv \
  $PREFIX/iplScoopFusionSegmentation.py \
   --output test_cv_nl2_re \
   --debug  \
   --segment test_lib_nl2 \
   --cv cv_re.json \
   --options segment.json  \
   --cleanup --ext
fi   


if [ ! -e test_cv_nl2_re_ec/cv_2_stats.json ];then
PARALLEL=3
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
python -m scoop -n $PARALLEL -vvv \
  $PREFIX/iplScoopFusionSegmentation.py \
   --output test_cv_nl2_re_ec \
   --debug  \
   --segment test_lib_nl2 \
   --cv cv_re.json \
   --options segment.json  \
   --cleanup --ext \
   --train-ec ec_train.json
fi   

cat - > cv_re_2.json <<END
{
  "validation_library":"ec_library_bbox_re.csv",
  "iterations":-1,
  "cv":1,
  "fuse_variant":"fuse_1",
  "cv_variant":"cv_3",
  "regularize_variant":"reg_1",
  "ec_variant":"ec2"
}
END

cat - > ec_train2.json <<END
{
  "method" :   "AdaBoostPP",
  "method_n" : 500,
  
  "method2" :   "AdaBoostPP",
  "method2_n" : 500,
  
  "border_mask":       true,
  "border_mask_width": 4,
  
  "use_coord": true,
  "use_joint": true,

  "train_rounds": -1,
  "train_cv": 1
}
END


if [ ! -e test_cv_nl2_re_ec/cv_3_stats.json ];then
PARALLEL=3
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
python -m scoop -n $PARALLEL -vvv \
  $PREFIX/iplScoopFusionSegmentation.py \
   --output test_cv_nl2_re_ec \
   --debug  \
   --segment test_lib_nl2 \
   --cv      cv_re_2.json \
   --options segment.json  \
   --cleanup --ext \
   --train-ec ec_train2.json
fi   

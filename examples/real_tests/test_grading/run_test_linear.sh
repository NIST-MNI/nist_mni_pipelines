#! /bin/sh


PREFIX=$(pwd)/../../python

export PYTHONPATH=$PREFIX:$PYTHONPATH


cat - > library_description.json <<END
{
  "reference_model": "snipe_library/NC/T1/ADNI.stx_011_S_0002_m00_bbox_snipe.mnc",
  "reference_mask":  "snipe_library/whole.mnc",
  
  "reference_local_model" : null,
  "reference_local_mask" :  null,
  "library":"snipe_library.lst",

  "build_remap":         [ [2,1],
                           [4,2],
                           [19,3],
                           [21,4]],
                            
  "build_flip_remap":    null,
  "parts": 0,
  "classes": 5,
  "groups": 2,
  "build_symmetric": false,
  "build_symmetric_flip": false,
  "symmetric_lut": null,
  "denoise": false,
  "denoise_beta": null,
  
  "initial_register": false,
  "initial_local_register": false,
  
  "non_linear_register": true,
  "non_linear_register_ants": true,
  
  "non_linear_register_level": 2,
  "non_linear_register_start": 8,
  
  "non_linear_register_options": {
        "conf":   {"8":100,"4":40,"2":40,"1": 20  },
        "blur":   {"8":8  ,"4":4 ,"2":2, "1": 1   },
        "shrink": {"8":8  ,"4":4 ,"2":2, "1": 1   },

        "transformation": "SyN[ .25, 1.0 , 1.0 ]",
        "use_histogram_matching": true,
        "cost_function":"CC",
        "cost_function_par":"1,3,Regular,1.0"
  },
  
  "resample_order": 1,
  "resample_baa": true,
  "extend_boundary": 4,
  "op_mask": "E[2] D[4]",
  
  "create_patch_norm_lib": false
}
END


if [ ! -e test_lib/library.json ];then
# create the library
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

python -m scoop -n 4 $PREFIX/iplScoopFusionGrading.py  \
  --create library_description.json --output test_lib --debug
fi
# run cross-validation

cat - > cv.json <<END
{
  "validation_library":"snipe_library.lst",
  "iterations":-1,
  "cv":1,
  "fuse_variant":"fuse",
  "cv_variant":"cv",
  "regularize_variant":"reg_1"
}
END


cat - > grade_lin.json <<END
{
  "initial_local_register": true,
  "non_linear_pairwise": false,
  "non_linear_register": false,
  "non_linear_register_ants": true,
  
  "non_linear_register_level": 2,
  "non_linear_register_start": 8,

  "non_linear_register_options": {
        "conf":   {"8":100,"4":40,"2":40,"1": 20  },
        "blur":   {"8":8  ,"4":4 ,"2":2, "1": 1   },
        "shrink": {"8":8  ,"4":4 ,"2":2, "1": 1   },

        "transformation": "SyN[ .25, 1.0 , 1.0 ]",
        "use_histogram_matching": true,
        "cost_function":"CC",
        "cost_function_par":"1,3,Regular,1.0"
  },  

  "simple_fusion": false,
  "resample_order": 1,
  "resample_baa": true,
  "library_preselect": -1,
  "segment_symmetric": false,

  "fuse_options":
  {
      "patch": 1,
      "search": 1,
      "threshold": 0.0,
      "top": 4
  }

}
END

mkdir -p test_cv_lin

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
python -m scoop -n 4 -vvv \
  $PREFIX/iplScoopFusionGrading.py \
   --output test_cv_lin \
   --debug  \
   --grade test_lib \
   --cv cv.json \
   --options grade_lin.json 

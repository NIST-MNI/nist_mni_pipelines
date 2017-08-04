#! /usr/bin/env python

# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 12/10/2014
#
# Run fusion segmentation test

import os

from ipl.segment import *

if __name__=='__main__':
    
    prefix='test_python'
    debug=True
    library_dir=prefix+'_lib'
    cv_dir=prefix+'_cv'
    
    library_description={
      "reference_model": "data/ellipse_0_blur.mnc",
      "reference_mask":  "data/ellipse_0_mask.mnc",
      "reference_local_model" : None,
      "reference_local_mask" :  None,
      "library":"seg_subjects.lst",
      "build_remap":         [ [1, 1], [2,2], [3,3], [4,4], [5,5], [6,6],  [7,7], [8,8] ],
      "build_flip_remap":    None,
      "parts": 0,
      "classes": 9,
      "build_symmetric": False,
      "build_symmetric_flip": False,
      "symmetric_lut": None,
      "denoise": False,
      "denoise_beta": None,
      "linear_register": False,
      "local_linear_register": True,
      "non_linear_register": False,
      "resample_order": 2,
      "resample_baa": True
    }
    
    
    # create library if needed
    if not os.path.exists(library_dir+os.sep+'library.json'):
        generate_library(library_description, library_dir, debug=debug)
    
    # run cross-validation segmenttion test
    cv = {
      "validation_library":"seg_subjects.lst",
      "iterations":2,
      "cv":2,
      "fuse_variant":"fuse",
      "ec_variant":"ec_p",
      "cv_variant":"cv_p"
    }
    
    segmentation_options = {
      "local_linear_register": True,
      "non_linear_pairwise": False,
      "non_linear_register": False,

      "simple_fusion": False,
      "non_linear_register_level": 4,
      "pairwise_level": 4,

      "resample_order": 2,
      "resample_baa": True,
      "library_preselect": 3,
      "segment_symmetric": False,

      "fuse_options":
      {
          "patch": 1,
          "search": 1,
          "threshold": 0.0,
          "gco_diagonal": False,
          "gco_wlabel": 0.001,
          "gco":True
      }
    }
    library=load_library_info( library_dir )
    
    cv_fusion_segment(cv,
                      library,
                      cv_dir,
                      segmentation_options,
                      debug=debug)
    


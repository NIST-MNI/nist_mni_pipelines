#! /usr/bin/env python

# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 12/10/2014
#
# Run fusion segmentation test

import os
import copy

from sklearn.grid_search import ParameterGrid

# scoop parallel execution
from scoop import futures, shared


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
    cv_base = {
      "validation_library": "seg_subjects.lst",
      "iterations":-1,
      "cv": 1,
      "fuse_variant":"fuse",
      "ec_variant":"adaboost",
      "cv_variant":"cv"
    }
    
    
    segmentation_base = {
      "local_linear_register": True,
      "non_linear_pairwise": False,
      "non_linear_register": True,
      "non_linear_register_ants": True, 
      "simple_fusion": False,
      "non_linear_register_level": 2,
      "resample_order": 2,
      "resample_baa": True,
      "library_preselect":2,
      "segment_symmetric":True,
      
      "fuse_options":
        {
          "patch": 3,
          "search": 1,
          "threshold": 0.0,
          "gco_diagonal": True,
          "gco_wlabel": 0.0001,
          "gco":True
        },

      "filters":
        {
        "denoise": True,
        "denoise_beta": 0.7,
        "denoise_patch": 2,
        "denoise_search": 1,
        "normalize": True
        }
    }
        
    ec_base= {
      "method" : "AdaBoost",
      "method_n" : 300,
      "method_n_jobs": 1,
      "border_mask" : True,
      "border_mask_width" : 2,
      "use_coord": True ,
      "use_joint": True, 
      "patch_size": 1,
      "use_raw" : False,
      "primary_features" : -1,
      "split" : None,
    }
    
    library=load_library_info( library_dir )
    
    results=[]

    parameters=[]
    for preselect in [2, 3, 4]:
      ec_name='ec'
      for use_ants in [False,True]:
        for nl_level in [2,4]:
          for pairwise in [False,True]:
            fusion_name='fuse_nl{}_a{}_p{}_pre{}'.format(nl_level,use_ants,pairwise,preselect)

            cv_name=fusion_name+'_'+ec_name

            # create a job-specific options
            segmentation_options=copy.deepcopy(segmentation_base)
            cv=copy.deepcopy(cv_base)
            ec=copy.deepcopy(ec_base)

            # set parameters
            segmentation_options['non_linear_pairwise']=pairwise
            segmentation_options['non_linear_register']=not pairwise
            segmentation_options['library_preselect']=preselect

            if pairwise:
              segmentation_options['pairwise_level']=nl_level
              segmentation_options['non_linear_pairwise_ants']=use_ants
            else:
              segmentation_options['non_linear_register_level']=nl_level
              segmentation_options['non_linear_register_ants']=use_ants

            cv['fuse_variant']=fusion_name
            cv['ec_variant']=ec_name
            cv['cv_variant']=cv_name
            # set output names
            parameters.append(segmentation_options)
            results.append( futures.submit(
                              cv_fusion_segment, cv, library,
                              cv_dir, segmentation_options, debug=debug) )
    
    futures.wait(results, return_when=futures.ALL_COMPLETED)
    # TODO: gather stats?
    print("Done!")


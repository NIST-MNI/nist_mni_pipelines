#! /usr/bin/env python


import shutil
import os
import sys
import csv
import traceback
import argparse
import json
import tempfile
import re
import copy
import random

# MINC stuff
from iplMincTools import mincTools,mincError

if __name__=='__main__':
  with mincTools() as minc:


    for s in range(0,8):
      for g in range(0,8):
        par={'conf':{},
            'smooth_update':s,
            'smooth_field':g,
            'update_rule':1,
            'grad_type':0,
            'max_step':2.0 }
        
        xfm="test_{}_{}_ldd.xfm".format(s,g)
        grid="test_{}_{}_ldd_grid_0.mnc".format(s,g)
        grid_m="test_{}_{}_ldd_grid_m.mnc".format(s,g)
        test_out="test_{}_{}_ldd_test.mnc".format(s,g)
        test_qc="test_{}_{}_ldd_test.jpg".format(s,g)
        
        minc.non_linear_register_ldd(
          "data/ellipse_0_blur.mnc","data/ellipse_1_blur.mnc","test_{}_{}_vel.mnc".format(s,g),
          output_xfm=xfm, start=8,level=2,parameters=par)
        
        minc.grid_magnitude(grid,grid_m)

        minc.resample_smooth("data/ellipse_0_blur.mnc",test_out,transform=xfm)
        
        minc.qc("data/ellipse_1_blur.mnc",test_qc,mask=test_out,image_range=[0,100],mask_range=[0,100])
        
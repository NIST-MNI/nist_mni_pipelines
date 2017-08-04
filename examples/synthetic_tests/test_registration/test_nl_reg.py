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

elx_par1="""
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
(UseDirectionCosines "true")

(ShowExactMetricValue "false")

(Registration "MultiResolutionRegistration")
(Interpolator "BSplineInterpolator" )
(ResampleInterpolator "FinalBSplineInterpolator" )
(Resampler "DefaultResampler" )

(FixedImagePyramid  "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

(Optimizer "AdaptiveStochasticGradientDescent")
(Transform "BSplineTransform")
(Metric "AdvancedNormalizedCorrelation")

(FinalGridSpacingInPhysicalUnits 32)

(HowToCombineTransforms "Compose")

(ErodeMask "false")

(NumberOfResolutions 2)

(ImagePyramidSchedule 8 8 8  4 4 4 )

(MaximumNumberOfIterations 200 200 200 )
(MaximumNumberOfSamplingAttempts 3)

(NumberOfSpatialSamples 1024 )

(NewSamplesEveryIteration "true")
(ImageSampler "Random" )

(BSplineInterpolationOrder 1)

(FinalBSplineInterpolationOrder 1)

(DefaultPixelValue 0)

(WriteResultImage "false")

// The pixel type and format of the resulting deformed moving image
(ResultImagePixelType "float")
(ResultImageFormat "mnc")
"""


elx_par2="""
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
(UseDirectionCosines "true")

(ShowExactMetricValue "false")

(Registration "MultiResolutionRegistration")
(Interpolator "BSplineInterpolator" )
(ResampleInterpolator "FinalBSplineInterpolator" )
(Resampler "DefaultResampler" )

(FixedImagePyramid  "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

(Optimizer "AdaptiveStochasticGradientDescent")
(Transform "BSplineTransform")
(Metric "AdvancedNormalizedCorrelation")

(FinalGridSpacingInPhysicalUnits 2)

(HowToCombineTransforms "Compose")

(ErodeMask "false")

(NumberOfResolutions 2)

(ImagePyramidSchedule 4 4 4  2 2 2 )

(MaximumNumberOfIterations 200 200 200 )
(MaximumNumberOfSamplingAttempts 3)

(NumberOfSpatialSamples 1024 )

(NewSamplesEveryIteration "true")
(ImageSampler "Random" )

(BSplineInterpolationOrder 1)

(FinalBSplineInterpolationOrder 1)

(DefaultPixelValue 0)

(WriteResultImage "false")

// The pixel type and format of the resulting deformed moving image
(ResultImagePixelType "float")
(ResultImageFormat "mnc")
"""


if __name__=='__main__':
  with mincTools() as minc:
    minc.register_elastix("data/ellipse_0_blur.mnc","data/ellipse_1_blur.mnc",
        output_par="test_4mm_0_1_par.txt",output_xfm="test_4mm_0_1.xfm",parameters=elx_par1)
    
    minc.grid_magnitude("test_4mm_0_1_grid_0.mnc","test_4mm_0_1_grid_m.mnc")
    
    minc.register_elastix("data/ellipse_0_blur.mnc","data/ellipse_1_blur.mnc",output_par="test_4mm_0_1_2_par.txt",
        output_xfm="test_4mm_0_1_2.xfm",parameters=elx_par1,init_xfm="test_4mm_0_1.xfm")

    minc.grid_magnitude("test_4mm_0_1_2_grid_0.mnc","test_4mm_0_1_2_grid_m.mnc")
    
    minc.register_elastix("data/ellipse_0_blur.mnc","data/ellipse_1_blur.mnc",output_par="test_4mm_0_1_3_par.txt",
        output_xfm="test_4mm_0_1_3.xfm",parameters=elx_par1,init_par="test_4mm_0_1_par.txt")
        
    minc.grid_magnitude("test_4mm_0_1_3_grid_0.mnc","test_4mm_0_1_3_grid_m.mnc")
    
    minc.register_elastix("data/ellipse_0_blur_.mnc","data/ellipse_1_blur_.mnc",output_par="test_1mm_0_1_par.txt",output_xfm="test_1mm_0_1.xfm",parameters=elx_par1)
    minc.grid_magnitude("test_1mm_0_1_grid_0.mnc","test_1mm_0_1_grid_m.mnc")
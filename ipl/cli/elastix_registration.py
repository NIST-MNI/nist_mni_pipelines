#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 29/06/2015
#
# registration tools


from __future__ import print_function

import os
import sys
import shutil
import tempfile
import subprocess
import re
import fcntl
import traceback
import collections
import math

# command-line interface
import argparse

# local stuff
from   ipl.minc_tools    import mincTools,mincError
import ipl.elastix_registration


def parse_options():
    parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    description="Run elastix registration")
    
    parser.add_argument("--verbose",
                    action="store_true",
                    default=False,
                    help="Be verbose",
                    dest="verbose")
                    
    parser.add_argument("source",
                    help="Source file")
    
    parser.add_argument("target",
                    help="Target file")
    
    parser.add_argument("--output_par",
                    help="Output transformation file, elastix format",
                    default=None)
    
    parser.add_argument("--output_xfm",
                    help="Output transformation file, MINC xfm format",
                    default=None)
    
    parser.add_argument("--source_mask",
                        default= None,
                        help="Source mask")
    parser.add_argument("--target_mask",
                        default= None,
                        help="Target mask")
    parser.add_argument("--init_xfm",
                        default   = None,
                        help="Initial transformation, minc format")
    parser.add_argument("--init_par",
                        default   = None,
                        help="Initial transformation elastix format")

    parser.add_argument("--optimizer",
                        default="AdaptiveStochasticGradientDescent",
                        help="Elastix optimizer",
                        choices=["AdaptiveStochasticGradientDescent",
                                      "CMAEvolutionStrategy" ,
                                      "ConjugateGradient",
                                      "ConjugateGradientFRPR",
                                      "FiniteDifferenceGradientDescent",
                                      "QuasiNewtonLBFGS",
                                      "RegularStepGradientDescent",
                                      "RSGDEachParameterApart"]
                        )
    
    parser.add_argument("--transform",
                        default="BSplineTransform",
                        help="Elastix transform",
                        choices=[    "BSplineTransform", 
                                     "SimilarityTransform",
                                     "AffineTransform",
                                     "AffineDTITransform",
                                     "EulerTransform",
                                     "MultiBSplineTransformWithNormal",
                                     "TranslationTransform"]
                        )
    
    parser.add_argument("--metric",
                        default="AdvancedNormalizedCorrelation",
                        help="Elastix metric",
                        choices=[    "AdvancedNormalizedCorrelation",
                                     "AdvancedMattesMutualInformation",
                                     "NormalizedMutualInformation",
                                     "AdvancedKappaStatistic",
                                     "KNNGraphAlphaMutualInformation",
                                     "AdvancedMeanSquares"])

    parser.add_argument("--resolutions",
                        default=3,
                        type=int,
                        help="Number of resolutions")
    
    parser.add_argument("--pyramid",
                        default="8 8 8 4 4 4 2 2 2",
                        help="Downsampling program")
    
    parser.add_argument("--iterations",
                        default=4000,
                        help="Number of iterations per level")
    
    parser.add_argument("--elx_iterations",
                        default=1,
                        help="Number of times elastix will run")
    
    parser.add_argument("--samples",
                        default=4096,
                        help="Number of samples")
    
    parser.add_argument("--sampler",
                        default="Random",
                        help="Elastix sampler")
    
    parser.add_argument("--grid_spacing",
                        default=10,
                        type=float,
                        help="Final node-distance for B-Splines")
    
    parser.add_argument("--max_step",
                        default="1.0",
                        help="Elastix maximum optimizer step")
    
    parser.add_argument("--work_dir",
                        default   = None,
                        help="Work directory")
    
    parser.add_argument("--downsample",
                        default = None,
                        help="Downsample to given voxel size ",
                        type=float)
    
    parser.add_argument("--downsample_grid",
                        default=None,
                        help="Downsample output grid by factor",
                        type=int)
    
    parser.add_argument("--tags",
                        default=None,
                        help="tags")
    
    parser.add_argument("--nl",
                    dest="nl",
                    action="store_true",
                    help="Use nonlinear mode",
                    default=False)
    
    parser.add_argument("--lin",
                    dest="nl",
                    action="store_false",
                    help="Use linear mode",
                    default=False)
    
    parser.add_argument("--output_log",
                        default   = None,
                        help="Output log file")
    
    parser.add_argument("-M","--measure",
                        default = False,
                        action = "store_true",
                        help = "Measure mode",
                        dest="measure")
                    
    parser.add_argument("--close",
                    dest="close",
                    action="store_true",
                    help="Do not initialize transform",
                    default=False)
    
    options = parser.parse_args()
    return options


def main():
    options = parse_options()

    if options.source is None or options.target is None:
         print("Error in arguments, run with --help")
         print(repr(options))
    else:
        if not options.nl and options.transform=="BSplineTransform":
            options.transform="SimilarityTransform"
        
        parameters= {
            "optimizer":   options.optimizer,
            "transform":   options.transform,
            "metric":      options.metric,
            "resolutions": options.resolutions,
            "pyramid":     options.pyramid,
            "iterations":  options.iterations,
            "samples":     options.samples,
            "sampler":     options.sampler,
            "grid_spacing":options.grid_spacing,
            "max_step":    options.max_step,
            "measure":     options.measure,
            "automatic_transform_init": not options.close
            }
        
        out=ipl.elastix_registration.register_elastix( 
                    options.source, options.target, 
                    output_par = options.output_par,
                    output_xfm = options.output_xfm,
                    source_mask= options.source_mask,
                    target_mask= options.target_mask,
                    init_xfm   = options.init_xfm,
                    init_par   = options.init_par,
                    parameters = parameters,
                    work_dir   = options.work_dir,
                    downsample = options.downsample,
                    downsample_grid=options.downsample_grid,
                    nl         = options.nl,
                    output_log = options.output_log,
                    tags       = options.tags,
                    verbose    = 2,
                    iterations = options.elx_iterations)
        if options.measure:
            print(out)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

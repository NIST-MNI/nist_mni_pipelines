#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# @author Renzo Phellan
# @date 18/02/2021
#
# Generate average model
from __future__ import print_function

import shutil
import os
import sys
import csv
import traceback
import argparse
import json

from ipl.minc_tools    import mincTools,mincError

#
from ipl.lp.pipeline        import standard_pipeline, default_pipeline_options
from ipl.lp.structures      import MriScan,MriTransform,MriQCImage,MriAux
from ipl.lp.structures      import save_pipeline_output,load_pipeline_output
from ipl.lp.utils           import *


def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Run seeg pipeline')

    parser.add_argument("input",
                        help="Input directory, it contains preimplantation MR and CT, "\
                         "and postimplantation CT DICOM directories")

    parser.add_argument("pre_imp_mr",
                        help="Name of the preimplantation MR DICOM directory")

    parser.add_argument("post_imp_ct",
                        help="Name of the postimplantation CT DICOM directory")

    parser.add_argument("subject",
                        help="Subject id")

    parser.add_argument("visit",
                        help="Subject visit")

    parser.add_argument("output",
                        help="Output directory, required for application of method")

    options = parser.parse_args()

    return options


def main():
    options = parse_options()
    try:
        #First, transform the input dicom files to minc and store them in a minc directory
        minc_dir = os.path.join(options.output, 'minc')
        minc_pre_imp_mr = 'minc_pre_imp_mr'
        minc_post_imp_ct = 'minc_post_imp_ct'

        create_dirs([minc_dir])

        pre_imp_mr_files = os.path.join(options.input, options.pre_imp_mr, 'IM*')
        os.system('dcm2mnc ' + pre_imp_mr_files + ' ' + minc_dir + ' -anon' + ' -fname ' +
                  minc_pre_imp_mr + ' -dname' + " ''")

        post_imp_ct_files = os.path.join(options.input, options.post_imp_ct, 'IM*')
        os.system('dcm2mnc ' + post_imp_ct_files + ' ' + minc_dir + ' -anon' + ' -fname ' +
                  minc_post_imp_ct + ' -dname' + " ''")

        #Add here anonymization by removing headers of the file (TBD)

        #Second, call the pipeline with the dicom files
        minc_pre_imp_mr_path = os.path.join(minc_dir, minc_pre_imp_mr + '.mnc')
        pipeline_command = ("python -m scoop -n 4 ipl_preprocess_pipeline.py " +
                            options.subject + " " + options.visit + " " +
                            minc_pre_imp_mr_path +
                            " --output " + options.output +
                            " --options test/pipeline_options.json")
        os.system(pipeline_command)

    except :
        print("Exception :{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 
#

import shutil
import os
import sys
import csv
import traceback


# MINC stuff
from ipl.minc_tools import mincTools,mincError

def split_labels_seg(sample):
    ''' split-up one multi-label segmentation into a set of files'''
    try:
        with mincTools() as m:
            if sample.seg is not None:
                base=sample.seg.rsplit('.mnc',1)[0]+'_%03d.mnc'
                sample.seg_split=m.split_labels(sample.seg,base)
            if sample.seg_f is not None:
                base=sample.seg_f.rsplit('.mnc',1)[0]+'_%03d.mnc'
                sample.seg_f_split=m.split_labels(sample.seg,base)
    except mincError as e:
        print("Exception in split_labels_seg:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in split_labels_seg:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

def merge_labels_seg(sample):
    ''' merge multiple segmentation into a single files'''
    try:
        with mincTools() as m:
            if any(sample.seg_split):
                if sample.seg is None:
                    sample.seg=sample.seg_split[0].rsplit('_000.mnc',1)[0]+'.mnc'
                m.merge_labels(sample.seg_split,sample.seg)
            if any(sample.seg_f_split):
                if sample.seg_f is None:
                    sample.seg_f=sample.seg_f_split[0].rsplit('_000.mnc',1)[0]+'.mnc'
                m.merge_labels(sample.seg_f_split,sample.seg_f)
    except mincError as e:
        print("Exception in merge_labels_seg:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in merge_labels_seg:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

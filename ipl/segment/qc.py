# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 
#

import shutil
import os
import sys
import csv
import copy
import re
import json


import argparse

# MINC stuff
from ipl.minc_tools import mincTools,mincError
import traceback

# local things
from ipl.segment.structures import *


def make_contours(input, output, width=1):
  """Convert multi-label image into another multilabel images with borders only 
    Arguments:
    input -- input minc file
    output -- output file

    Keyword arguments:
    width -- width of the border to leave behind, default 1 (voxels)
  """
  with mincTools() as m:
    m.command(['c3d', input,'-split',
              '-foreach', 
                '-dup', '-erode', '1' ,'{}x{}x{}'.format(width,width,width), '-scale', '-1', 
                 '-add',
              '-endfor',
              '-merge',
              '-type', 'short','-o',output],
            inputs=[input],outputs=[output],
            verbose=True)

def generate_qc_image(sample_seg,
                      sample, 
                      sample_qc, 
                      options={},
                      model=None,
                      symmetric=False,
                      labels=2,
                      title=None):
  """Gnerate QC image for multilabel segmentation
    Arguments:
    sample_seg -- input segmentation
    sample -- input file
    sample_qc -- output QC file
    
    Keyword arguments:
    options -- options as dictionary with following keys:
             lut_file -- LUT file for minclookup, default None
             spectral_mask -- boolean , if spectral mask should be used, default False
             discrete_mask -- boolean , if discrete mask should be used, default False
             image_range -- list of two real values
             clamp -- boolean, if range clamp should be used
             big
             contours
             contour_width
             crop 
    model -- reference model, default None
    symmetric -- boolean, if symmetric QC is needed 
    width -- width of the border to leave behind, default 1 (voxels)
    labels -- integer, number of labels present, default 2
    title -- QC image title
  """
  try:
    
    #TODO: implement advanced features
    qc_lut=options.get('lut_file',None)
    spectral_mask=options.get('spectral_mask',False)
    discrete_mask=options.get('discrete_mask',False)
    image_range=options.get('image_range',None)
    clamp=options.get('clamp',False)
    big=options.get('big',False)
    contours=options.get('contours',False)
    contour_width=options.get('contour_width',1)
    crop=options.get('crop',None)
    
    if qc_lut is not None:
        spectral_mask=False
        discrete_mask=True
    
    with mincTools() as m:
      seg=sample_seg.seg
      seg_f=sample_seg.seg_f
      scan=sample.scan
      scan_f=sample.scan_f
      
      if crop is not None:
          # remove voxels from the edge
          m.autocrop(scan,m.tmp('scan.mnc'),isoexpand=-crop)
          scan=m.tmp('scan.mnc')
          m.resample_labels(seg,m.tmp('seg.mnc'),like=scan)
          seg=m.tmp('seg.mnc')
          
          if symmetric:
            m.autocrop(scan_f,m.tmp('scan_f.mnc'),isoexpand=-crop)
            scan_f=m.tmp('scan_f.mnc')
            m.resample_labels(seg_f,m.tmp('seg_f.mnc'),like=scan)
            seg_f=m.tmp('seg_f.mnc')
      
      if contours:
        make_contours(seg,m.tmp('seg_contours.mnc'),width=contour_width)
        seg=m.tmp('seg_contours.mnc')
        if symmetric:
          make_contours(seg_f,m.tmp('seg_f_contours.mnc'),width=contour_width)
          seg_f=m.tmp('seg_f_contours.mnc')
          
      if symmetric:
        
        m.qc( scan,
            m.tmp('qc.png'),
            mask=seg,
            mask_range=[0,labels-1],
            big=False,
            clamp=clamp,
            image_range=image_range,
            spectral_mask=spectral_mask,
            discrete_mask=discrete_mask,
            mask_lut=qc_lut)
        
        m.qc( scan_f,
            m.tmp('qc_f.png'),
            mask=seg_f,
            mask_range=[0,labels-1],
            image_range=image_range,
            big=False,
            clamp=clamp,
            spectral_mask=spectral_mask,
            discrete_mask=discrete_mask,
            title=title,
            mask_lut=qc_lut)
        
        m.command(['montage','-tile','2x1','-geometry','+1+1',
                   m.tmp('qc.png'),m.tmp('qc_f.png'),sample_qc],
                  inputs=[m.tmp('qc.png'),m.tmp('qc_f.png')], 
                  outputs=[sample_qc])
      else:
        m.qc( scan,
            sample_qc,
            mask=seg,
            mask_range=[0,labels-1],
            image_range=image_range,
            big=True, 
            mask_lut=qc_lut,
            spectral_mask=spectral_mask,
            discrete_mask=discrete_mask,
            clamp=clamp,
            title=title)
        
    return [sample_qc]
  except mincError as e:
      print("Exception in generate_qc_image:{}".format(str(e)))
      traceback.print_exc(file=sys.stdout)
      raise
  except :
      print("Exception in generate_qc_image:{}".format(sys.exc_info()[0]))
      traceback.print_exc(file=sys.stdout)
      raise




def parse_options():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Run QC step manually')
    
    parser.add_argument('--scan',
                    help="Underlying scan")

    parser.add_argument('--scan_f',
                    help="flipped scan")
    
    parser.add_argument('--seg',
                    help="Segmentation")

    parser.add_argument('--seg_f',
                    help="flipped segmentation")
    
    parser.add_argument('--spectral_mask', 
                action="store_true",
                default=False )
    
    parser.add_argument('--discrete_mask', 
                action="store_true",
                default=False )

    parser.add_argument('--clamp', 
                action="store_true",
                default=False )
    
    parser.add_argument('--big', 
                action="store_true",
                default=False )
    
    parser.add_argument('--contours', 
                action="store_true",
                default=False )
    
    parser.add_argument('--contour_width',
                    default=1,
                    type=int,
                    help="contour_width")
    
    parser.add_argument('--image_range',
                    nargs=2,
                    help="Range")
    
    parser.add_argument('--lut_file',
                    help="LUT")
    
    parser.add_argument('--crop',
                    type=int,
                    default=None,
                    help="Crop files")
    
    parser.add_argument('--labels',
                    type=int,
                    default=2,
                    help="Number of labels")
    
    parser.add_argument('output')
    
    return parser.parse_args()

    
#crop=options.get('crop',None)
    
if __name__ == '__main__':
    options = parse_options()
    
    if options.output is None or options.scan is None:
        print("Provide some input")
        exit(1)
        
    segment_symmetric=False
    if options.scan_f is not None:
        segment_symmetric=True
    
    sample_scan=MriDataset(name='scan', scan=options.scan,scan_f=options.scan_f )
    sample_seg=MriDataset(name='seg', seg=options.seg,seg_f=options.seg_f )
    class_number=1
    
    qc_options={
    'lut_file':options.lut_file,
    'spectral_mask':options.spectral_mask,
    'dicrete_mask':options.discrete_mask,
    'image_range':options.image_range,
    'clamp':options.clamp,
    'big':options.big,
    'contours':options.contours,
    'contour_width':options.contour_width,
    'crop':options.crop
        }
    
    generate_qc_image(sample_seg,
                     sample_scan, 
                     options.output, 
                     options=qc_options,
                     symmetric=segment_symmetric,
                     labels=options.labels)    

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

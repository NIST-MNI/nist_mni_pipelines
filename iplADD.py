#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# @author Vladimir Fonov
# @date 09/03/2017
import shutil
import os
import sys
import csv
import traceback

import json
import six


# MINC stuff
from ipl.minc_tools import mincTools,mincError

# internal funcions
import ipl.segment 
import ipl.grading 

# scoop parallel execution
from scoop import futures, shared

version = '1.0'


# run additional segmentation or grading 
# specifyin in .add option
# this part runs subject-specific part

def pipeline_run_add(patient):
    for i,j in enumerate( patient.add ):
        # apply to the template if 'apply_on_template' is on
        output_name=j.get('name','seg_{}'.format(i))
        if j.get('apply_on_template',False) and 'segment_options' in j and 'segment_library' in j:
            
            # let's run segmentation
            library=j['segment_library']
            options=j['segment_options']
            
            output_prefix=patient.template['nl_template_prefix']+'_'+output_name
            
            if isinstance(options, six.string_types):
                with open(options,'r') as f:
                    options=json.load(f)
            
            library=ipl.segment.load_library_info( library )
            
            print(json.dumps(library,indent=2))
            
            ipl.segment.fusion_segment(patient.template['nl_template'], 
                            library,
                            output_prefix,
                            input_mask=patient.template['nl_template_mask'],
                            parameters=options,
                            work_dir=None,
                            fuse_variant='seg',
                            regularize_variant='',
                            cleanup=True)
    

# this part runs timepoint-specific part
def pipeline_run_add_tp(patient, tp):
    for i,j in enumerate( patient.add ):
        if 'segment_options' in j and 'segment_library' in j:
            output_name=j.get('name','seg_{}'.format(i))
            
            if j.get('apply_on_template',False):
                template_prefix=patient.template['nl_template_prefix']+'_'+output_name
                output_prefix=patient[tp].stx2_mnc['add_prefix']+'_'+output_name
                
                nl_xfm=patient[tp].lng_xfm['t1']
                template_seg=template_prefix+'_seg.mnc'
                output_seg=output_prefix+'_seg.mnc'
                
                with mincTools(resample=patient.resample) as minc: 
                    minc.resample_labels(template_seg, output_seg,
                            transform=nl_xfm,
                            invert_transform=True,
                            like=patient[tp].stx2_mnc['t1'], 
                            baa=True,
                            order=1) # TODO: make it a parameter?
                
                # TODO: produce volume measurements
                # TODO: add option to use jacobian integration
                patient[tp].add[output_name]={'seg':output_seg,'vol':None} 
            else:
                # TODO: use partial volume mode here?
                # let's run segmentation
                library=j['segment_library']
                options=j['segment_options']
                modality=j.get('modality','t1')
                
                if isinstance(options, six.string_types):
                    with open(options,'r') as f:
                        options=json.load(f)
                
                library=ipl.segment.load_library_info( library )
                output_prefix=patient[tp].stx2_mnc['add_prefix']+'_'+output_name
                
                ipl.segment.fusion_segment(patient[tp].stx2_mnc[modality], 
                                library,
                                output_prefix,
                                input_mask=patient[tp].stx2_mnc["mask"],
                                parameters=options,
                                work_dir=None,
                                fuse_variant='seg',
                                regularize_variant='',
                                cleanup=True)
                
                output_seg=output_prefix+'_seg.mnc'
                output_vol=output_prefix+'_vol.json'
                patient[tp].add[output_name]={'seg':output_seg,'vol':output_vol}
        # grading 
        elif 'grading_options' in j and 'grading_library' in j:
            library=j['grading_library']
            options=j['grading_options']
            use_nl= j.get('use_nl',False)
            modality=j.get('modality','t1')
            output_name=j.get('name','grad_{}'.format(i))
            
            if isinstance(options, six.string_types):
                with open(options,'r') as f:
                    options=json.load(f)

            library=ipl.grading.load_library_info( library )
            output_prefix=patient[tp].stx2_mnc['add_prefix']+'_'+output_name
            
            ipl.grading.fusion_grading(patient[tp].stx2_mnc[modality], 
                            library,
                            output_prefix,
                            input_mask=patient[tp].stx2_mnc["mask"],
                            parameters=options,
                            work_dir=None,
                            fuse_variant='grad',
                            regularize_variant='',
                            cleanup=True)
            
            output_grad=output_prefix+'_grad.mnc'
            output_grad_vol=output_prefix+'_grad.json'
            patient[tp].add[output_name]={'grad':output_grad,'vol':output_grad_vol}

if __name__ == '__main__':
    pass

  # Using script as a stand-alone script
  # do nothing

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

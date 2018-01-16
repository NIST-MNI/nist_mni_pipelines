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
        
        if j.get('apply_on_template',False):
            if 'segment_options' in j and j.get('ANIMAL',False):
                # use ANIMAL style
                # TODO:
                pass
            elif 'segment_options' in j and j.get('WARP',False):
                # use just nonlinear warping
                # TODO:
                pass
            elif 'segment_options' in j and 'segment_library' in j:
                # use label fusion 
                # let's run segmentation
                library=j['segment_library']
                options=j['segment_options']
                
                output_prefix=patient.template['nl_template_prefix']+'_'+output_name
                
                if isinstance(options, six.string_types):
                    with open(options,'r') as f:
                        options=json.load(f)
                
                library=ipl.segment.load_library_info( library )
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
        if 'segment_options' in j:
            output_name=j.get('name','seg_{}'.format(i))
            library=None
            if 'segment_library' in j:
                library=ipl.segment.load_library_info( j['segment_library'] )
                
            options=j['segment_options']
            
            if isinstance(options, six.string_types):
                with open(options,'r') as f:
                    options=json.load(f)
            
            if j.get('apply_on_template',False):
                template_prefix=patient.template['nl_template_prefix']+'_'+output_name
                output_prefix=patient[tp].stx2_mnc['add_prefix']+'_'+output_name
                
                nl_xfm=patient[tp].lng_xfm['t1']
                nl_igrid=patient[tp].lng_igrid['t1']
                nl_idet=patient[tp].lng_det['t1']
                
                template_seg=template_prefix+'_seg.mnc'
                output_seg=output_prefix+'_seg.mnc'
                output_vol=output_prefix+'_vol.json'
                label_map=options.get('label_map',None)
                
                if label_map is None and library is not None:
                    label_map=library.get('label_map',None)
                
                with mincTools() as minc: 
                    if options.get('warp',True):
                        minc.resample_labels(template_seg, output_seg,
                                transform=nl_xfm,
                                invert_transform=True,
                                like=patient[tp].stx2_mnc['t1'], 
                                baa=options.get("resample_baa",True),
                                order=options.get("resample_order",1)) # TODO: make it a parameter?
                        ipl.segment.seg_to_volumes(output_seg,output_vol,label_defs=label_map)

                    if options.get('jacobian',False):
                        # perform jacobian integration within each ROI
                        minc.grid_determinant(nl_igrid,minc.tmp("det.mnc"))
                        minc.resample_smooth(minc.tmp("det.mnc"), nl_idet, like=template_seg)
                        ipl.segment.seg_to_volumes(output_seg, output_vol, label_defs=label_map, volume=nl_idet)

                patient[tp].add[output_name]={'seg':output_seg, 'vol':output_vol} 
            else:
                # TODO: use partial volume mode here?
                # let's run segmentation
                modality=j.get('modality','t1')
                
                output_prefix=patient[tp].stx2_mnc['add_prefix']+'_'+output_name
                if library is not None:
                    ipl.segment.fusion_segment(patient[tp].stx2_mnc[modality], 
                                    library,
                                    output_prefix,
                                    input_mask=patient[tp].stx2_mnc["mask"],
                                    parameters=options,
                                    work_dir=None,
                                    fuse_variant='seg',
                                    regularize_variant='',
                                    cleanup=True)
                elif j.get('ANIMAL',False):
                    # TODO: implement ANIMAL
                    pass
                elif j.get('WARP', False):
                    # TODO: implement atlas warping
                    pass
                
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

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
        output_prefix=patient.template['nl_template_prefix']+'_'+output_name
        print("ADD:{}".format(output_name))
        if j.get('apply_on_template',False):
            if j.get('ANIMAL',False):
                # HACK: run tissue classification on template, followed by lobe-segment
                with mincTools() as minc: 
                    minc.classify_clean([patient.template['nl_template']],output_prefix+'_cls.mnc',
                                        mask=patient.template['nl_template_mask'],
                                        xfm=patient.nl_xfm,
                                        model_name=patient.modelname,
                                        model_dir=patient.modeldir)
                    identity = minc.tmp('identity.xfm')
                    minc.command(['param2xfm', identity], [], [identity])
                    comm = [
                        'lobe_segment',
                        patient.nl_xfm,
                        identity,
                        output_prefix+'_cls.mnc',
                        output_prefix+'_seg.mnc',
                        '-modeldir', patient.modeldir + os.sep + patient.modelname + '_atlas/',
                        '-template', patient.modeldir + os.sep + patient.modelname + '.mnc',
                        ]
                    minc.command(comm, [patient.nl_xfm, output_prefix+'_cls.mnc'], [output_prefix+'_seg.mnc'])
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
                
                if isinstance(options, six.string_types):
                    with open(options,'r') as f:
                        options=json.load(f)
                
                library=ipl.segment.SegLibrary( library )
                print(repr(library))
                if os.path.exists(output_prefix+'_seg.mnc'):
                    print('ADD:{} already done!'.format(output_name))
                else:
                    ipl.segment.fusion_segment(patient.template['nl_template'], 
                                library,
                                output_prefix,
                                input_mask=patient.template['nl_template_mask'],
                                parameters=options,
                                work_dir=patient.workdir+os.sep+'template_'+output_name,
                                fuse_variant='seg',
                                regularize_variant='',
                                cleanup=True)
            

# this part runs timepoint-specific part
def pipeline_run_add_tp(patient, tp):
    
    for i,j in enumerate( patient.add ):
        output_name=j.get('name','seg_{}'.format(i))
        print("ADD TP:{}".format(output_name))
        
        library=None
        if 'segment_library' in j:
            library=ipl.segment.load_library_info( j['segment_library'] )
            
        options=j.get('segment_options',{})
        
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
            elif j.get('ANIMAL',False): # HACK: preset label map for ANIMAL 
                label_map=[ [30,  'frontal_left_wm'],
                            [210, 'frontal_left_gm'],
                            [17,  'frontal_right_wm'],
                            [211, 'frontal_right_gm'],
                            [83,  'temporal_left_wm'],
                            [218, 'temporal_left_gm'],
                            [59,  'temporal_right_wm'],
                            [219, 'temporal_right_gm'],
                            [57,  'parietal_left_wm'],
                            [6,   'parietal_left_gm'],
                            [105, 'parietal_right_wm'],
                            [2,   'parietal_right_gm'],
                            [73,  'occipital_left_wm'],
                            [8,   'occipital_left_gm'],
                            [45,  'occipital_right_wm'],
                            [4,   'occipital_right_gm'],
                            [67,  'cerebellum_left'],
                            [76,  'cerebellum_right'],
                            [20,  'brainstem'],
                            [3,   'lateral_ventricle_left'],
                            [9,   'lateral_ventricle_right'],
                            [232, '3rd_ventricle'],
                            [233, '4th_ventricle'],
                            [255, 'extracerebral_CSF'],
                            [39,  'caudate_left'],
                            [53,  'caudate_right'],
                            [14,  'putamen_left'],
                            [16,  'putamen_right'],
                            [102, 'thalamus_left'],
                            [203, 'thalamus_right'],
                            [33,  'subthalamic_nucleus_left'],
                            [23,  'subthalamic_nucleus_right'],
                            [12,  'globus_pallidus_left'],
                            [11,  'globus_pallidus_right'],
                            [29,  'fornix_left'],
                            [254, 'fornix_right'],
                            [28,  'skull'] ]
                
            with mincTools() as minc: 
                if j.get('warp',False):
                    minc.resample_labels(template_seg, output_seg,
                            transform=nl_xfm,
                            invert_transform=True,
                            like=patient[tp].stx2_mnc['t1'], 
                            baa=options.get("resample_baa",True),
                            order=options.get("resample_order",1)) # TODO: make it a parameter?
                    ipl.segment.seg_to_volumes(output_seg, output_vol,label_map=label_map)

                if j.get('jacobian',False):
                    # perform jacobian integration within each ROI
                    minc.grid_determinant(nl_igrid,minc.tmp("det.mnc"))
                    minc.resample_smooth(minc.tmp("det.mnc"), nl_idet, like=template_seg)
                    ipl.segment.seg_to_volumes(template_seg, output_vol, label_map=label_map, volume=nl_idet)

            patient[tp].add[output_name]={'seg':output_seg, 'vol':output_vol} 
        elif 'segment_options' in j: #HACK: figure out how to distinguish between grading and segmentation
            # TODO: use partial volume mode here?
            # let's run segmentation
            # options=j.get('segment_options',{})
            modality=j.get('modality','t1')
            
            output_prefix=patient[tp].stx2_mnc['add_prefix']+'_'+output_name
            if library is not None:
                ipl.segment.fusion_segment(patient[tp].stx2_mnc[modality], 
                                library,
                                output_prefix,
                                input_mask=patient[tp].stx2_mnc["mask"],
                                parameters=options,
                                work_dir=patient.workdir+os.sep+tp+'_'+output_name,
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

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

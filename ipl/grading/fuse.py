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

# MINC stuff
from ipl.minc_tools import mincTools,mincError

import ray

from .filter           import *
from .structures       import *
from .registration     import *
from .resample         import *
from .preselect        import *
from .qc               import *
from .fuse_grading     import *

import traceback

def seg_to_volumes_grad(seg, output_json, label_map=None,grad=None,median=False):
    with mincTools( verbose=2 ) as m:
        _out=m.label_stats(seg,label_defs=label_map,volume=grad,median=median)
        # convert to a dictionary
        # label_id, volume, mx, my, mz,[mean/median]
        out={i[0]: { 'volume':i[1], 'x':i[2], 'y':i[3], 'z': i[4], 'grad':i[5] } for i in _out }

        with open(output_json,'w') as f:
            json.dump(out,f,indent=1)
        return out

def fusion_grading( input_scan,
                    library_description,
                    output_segment,
                    input_mask=None,
                    parameters={},
                    exclude=[],
                    work_dir=None,
                    debug=False,
                    ec_variant=None,
                    fuse_variant=None,
                    regularize_variant=None,
                    add=[],
                    cleanup=False,
                    cleanup_xfm=False,
                    exclude_re=None):
    """Apply fusion segmentation"""
    
    if debug: 
        print( "Segmentation parameters:")
        print( repr(parameters) )
        
    out_variant=''
    if fuse_variant is not None:
        out_variant+=fuse_variant
        
    if regularize_variant is not None:
        out_variant+='_'+regularize_variant
        
    if ec_variant is not None:
        out_variant+='_'+ec_variant
        
    if work_dir is None:
        work_dir=output_segment+os.sep+'work_segment'

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    work_lib_dir=  work_dir+os.sep+'library'
    work_lib_dir_f=work_dir+os.sep+'library_f'

    if not os.path.exists(work_lib_dir):
        os.makedirs(work_lib_dir)

    if not os.path.exists(work_lib_dir_f):
        os.makedirs(work_lib_dir_f)

    library_nl_samples_avail=library_description['nl_samples_avail']
    library_modalities=library_description.get('modalities',1)-1
    
    # perform symmetric segmentation
    segment_symmetric=    parameters.get('segment_symmetric', False )

    # read filter paramters
    pre_filters=          parameters.get('pre_filters', None )
    post_filters=         parameters.get('post_filters', parameters.get( 'filters', None ))
    

    # perform local linear registration
    do_initial_register       = parameters.get( 'initial_register',
                                parameters.get( 'linear_register', {}))
    
    if do_initial_register is not None and isinstance(do_initial_register,dict):
        initial_register          = do_initial_register
        do_initial_register = True
    else:
        initial_register={}
    
    inital_reg_type           = parameters.get( 'initial_register_type',
                                parameters.get( 'linear_register_type',
                                initial_register.get('type','-lsq12')))
    
    inital_reg_ants           = parameters.get( 'initial_register_ants',
                                parameters.get( 'linear_register_ants', False))
    
    inital_reg_options        = parameters.get( 'initial_register_options', 
                                initial_register.get('options',None) )
    
    inital_reg_downsample     = parameters.get( 'initial_register_downsample', 
                                initial_register.get('downsample',None))

    inital_reg_use_mask       = parameters.get( 'initial_register_use_mask', 
                                initial_register.get('use_mask',False))
    
    initial_reg_objective     = initial_register.get('objective','-xcorr')

    # perform local linear registration
    do_initial_local_register = parameters.get( 'initial_local_register', 
                                parameters.get( 'local_linear_register', {}) )
    if do_initial_local_register is not None and isinstance(do_initial_local_register,dict):
        initial_local_register=do_initial_local_register
        do_initial_local_register=True
    else:
        initial_local_register={}

    local_reg_type            =  parameters.get( 'local_register_type',
                                    initial_local_register.get('type','-lsq12'))
                                    
    local_reg_ants            =  parameters.get( 'local_register_ants', False)
    
    local_reg_opts            =  parameters.get( 'local_register_options', 
                                    initial_local_register.get('options',None))
    
    local_reg_bbox            =  parameters.get( 'local_register_bbox', 
                                    initial_local_register.get('bbox',False ))
    
    local_reg_downsample      =  parameters.get( 'local_register_downsample', 
                                    initial_local_register.get('downsample',None))
    
    local_reg_use_mask        =  parameters.get( 'local_register_use_mask', 
                                    initial_local_register.get('use_mask',True))
    
    local_reg_objective       =  initial_local_register.get('objective','-xcorr')
    # if non-linear registraiton should be performed for library creation
    do_nonlinear_register=parameters.get('non_linear_register', False )

    # if non-linear registraiton should be performed with ANTS
    do_nonlinear_register_ants=parameters.get('non_linear_register_ants',False )
    nonlinear_register_type   = parameters.get( 'non_linear_register_type',None)
    if nonlinear_register_type is None:
        if do_nonlinear_register_ants: 
            nonlinear_register_type='ants'
    
    # if non-linear registraiton should be performed pairwise
    do_pairwise          =parameters.get('non_linear_pairwise', False )
    
    # if pairwise registration should be performed using ANTS
    do_pairwise_ants     =parameters.get('non_linear_pairwise_ants', True )
    pairwise_register_type   = parameters.get( 'non_linear_pairwise_type',None)
    if pairwise_register_type is None:
        if do_pairwise_ants: 
            pairwise_register_type='ants'

    # should we use ANTs
    library_preselect=        parameters.get('library_preselect', 10)
    library_preselect_step=   parameters.get('library_preselect_step', None)
    library_preselect_method= parameters.get('library_preselect_method', 'MI')
    

    nlreg_level        =  parameters.get('non_linear_register_level', 2)
    nlreg_start        =  parameters.get('non_linear_register_start', 16)
    nlreg_options      =  parameters.get('non_linear_register_options', None)
    nlreg_downsample   =  parameters.get('non_linear_register_downsample', None)
    

    pairwise_level     =  parameters.get('pairwise_level', 2)
    pairwise_start     =  parameters.get('pairwise_start', 16)
    pairwise_options   =  parameters.get('pairwise_options', None)

    fuse_options       =  parameters.get('fuse_options', None)

    resample_order      = parameters.get('resample_order', 2)
    label_resample_order= parameters.get( 'label_resample_order',resample_order)
    
    resample_baa        = parameters.get('resample_baa', True)

    use_median          = parameters.get('use_median', False)
    # QC image paramters
    qc_options          = parameters.get('qc_options', None)

    # special case for training error correction, assume input scan is already pre-processed
    run_in_bbox         = parameters.get('run_in_bbox', False)

    classes_number      = library_description['classes_number']
    groups              = library_description['groups']
    seg_datatype        = 'byte'

    output_info         = {}

    sample= MriDataset(scan=input_scan, seg=None, 
                       mask=input_mask, protect=True,
                       add=add)
    # get parameters
    model = MriDataset(scan=library_description['model'], 
                       mask=library_description['model_mask'],
                       add= library_description.get('model_add',[]) )
    
    local_model = MriDataset(scan=library_description['local_model'],
                             mask=library_description['local_model_mask'],
                             scan_f=library_description.get('local_model_flip',None),
                             mask_f=library_description.get('local_model_mask_flip',None),
                             seg=   library_description.get('local_model_seg',None),
                             seg_f= library_description.get('local_model_seg_flip',None),
                             add=   library_description.get('local_model_add',[]),
                             add_f= library_description.get('local_model_add_flip',[]),
                             )

    library = library_description['library']
    
    sample_modalities=len(add)
    
    print("\n\n")
    print("Sample modalities:{}".format(sample_modalities))
    print("\n\n")
    # apply the same steps as used in library creation to perform segmentation:

    # global
    initial_xfm=None
    nonlinear_xfm=None
    bbox_sample=None
    nl_sample=None
    bbox_linear_xfm=None

    sample_filtered=MriDataset(prefix=work_dir, name='flt_'+sample.name, add_n=sample_modalities )

    # QC file
    # TODO: allow for alternative location, extension
    sample_qc=work_dir+os.sep+'qc_'+sample.name+'_'+out_variant+'.jpg'

    if run_in_bbox:
        segment_symmetric=False
        do_initial_register=False
        do_initial_local_register=False
        # assume filter already applied!
        pre_filters=None
        post_filters=None

    if segment_symmetric:
        # need to flip the inputs
        flipdir=work_dir+os.sep+'flip'
        if not os.path.exists(flipdir):
            os.makedirs(flipdir)

        sample.scan_f=flipdir+os.sep+os.path.basename(sample.scan)
        sample.add_f=['' for (i,j) in enumerate(sample.add)]

        for (i,j) in enumerate(sample.add):
            sample.add_f[i]=flipdir+os.sep+os.path.basename(sample.add[i])

        if sample.mask is not None:
            sample.mask_f=flipdir+os.sep+'mask_'+os.path.basename(sample.scan)
        generate_flip_sample( sample )
    
    if pre_filters is not None:
        apply_filter( sample.scan, 
                    sample_filtered.scan,
                    pre_filters,
                    model=model.scan,
                    model_mask=model.mask)
        
        if sample.mask is not None:
            shutil.copyfile(sample.mask,sample_filtered.mask)
        
        for i,j in enumerate(sample.add):
            shutil.copyfile(sample.add[i],sample_filtered.add[i])
            
        sample=sample_filtered
    else:
        sample_filtered=None

    output_info['sample_filtered']=sample_filtered
    
    if do_initial_register:
        initial_xfm=MriTransform(prefix=work_dir, name='init_'+sample.name )
        
        if inital_reg_type=='elx' or inital_reg_type=='elastix' :
            elastix_registration( sample, 
                model, initial_xfm,
                symmetric=segment_symmetric,
                parameters=inital_reg_options,
                nl=False,
                downsample=inital_reg_downsample
                )
        elif inital_reg_type=='ants' or inital_reg_ants:
            linear_registration( sample, 
                model, initial_xfm,
                symmetric=segment_symmetric, 
                reg_type=inital_reg_type,
                linreg=inital_reg_options,
                ants=True,
                downsample=inital_reg_downsample
                )
        else:
            linear_registration( sample, 
                model, initial_xfm,
                symmetric=segment_symmetric, 
                reg_type=inital_reg_type,
                linreg=inital_reg_options,
                downsample=inital_reg_downsample,
                objective=initial_reg_objective
                )
        
        output_info['initial_xfm']=initial_xfm
        

    # local 
    bbox_sample = MriDataset(prefix=work_dir, name='bbox_init_'+sample.name, 
                             add_n=sample_modalities )
    
    
    if do_initial_local_register:
        bbox_linear_xfm=MriTransform(prefix=work_dir, name='bbox_init_'+sample.name )
        
        if local_reg_type=='elx' or local_reg_type=='elastix' :
            elastix_registration( sample, 
                local_model, 
                bbox_linear_xfm,
                symmetric=segment_symmetric,
                init_xfm=initial_xfm,
                resample_order=resample_order,
                parameters=local_reg_opts,
                bbox=local_reg_bbox,
                downsample=local_reg_downsample
                )
        elif local_reg_type=='ants' or local_reg_ants:
            linear_registration( sample, 
                local_model, 
                bbox_linear_xfm,
                init_xfm=initial_xfm,
                symmetric=segment_symmetric,
                reg_type=local_reg_type,
                linreg=local_reg_opts,
                resample_order=resample_order,
                ants=True,
                close=True,
                bbox=local_reg_bbox,
                downsample=local_reg_downsample
                )
        else:
            linear_registration( sample, 
                local_model, 
                bbox_linear_xfm,
                init_xfm=initial_xfm,
                symmetric=segment_symmetric,
                reg_type=local_reg_type,
                linreg=local_reg_opts,
                resample_order=resample_order,
                close=True,
                bbox=local_reg_bbox,
                downsample=local_reg_downsample,
                objective=local_reg_objective
                )

    else: 
        bbox_linear_xfm=initial_xfm

    output_info['bbox_initial_xfm']=bbox_linear_xfm
    bbox_sample.mask=None
    bbox_sample.seg=None
    bbox_sample.seg_f=None
    
    warp_sample(sample, local_model, bbox_sample,
                transform=bbox_linear_xfm,
                symmetric=segment_symmetric,
                symmetric_flip=segment_symmetric,# need to flip symmetric dataset
                resample_order=resample_order,
                filters=post_filters,
                )
    
    output_info['bbox_sample']=bbox_sample

    # TODO: run local intensity normalization

    # 3. run non-linear registration if needed
    if do_nonlinear_register:
        nl_sample=MriDataset(prefix=work_dir, name='nl_'+sample.name, add_n=sample_modalities )
        nonlinear_xfm=MriTransform(prefix=work_dir, name='nl_'+sample.name )


        if nonlinear_register_type=='elx' or nonlinear_register_type=='elastix' :
            elastix_registration( bbox_sample, local_model,
                nonlinear_xfm,
                symmetric=segment_symmetric,
                level=nlreg_level,
                start_level=nlreg_start,
                parameters=nlreg_options,
                nl=True,
                downsample=nlreg_downsample )
        elif nonlinear_register_type=='ants' or do_nonlinear_register_ants:
            non_linear_registration( bbox_sample, local_model,
                nonlinear_xfm,
                symmetric=segment_symmetric,
                level=nlreg_level,
                start_level=nlreg_start,
                parameters=nlreg_options,
                ants=True,
                downsample=nlreg_downsample )
        else:
            non_linear_registration( bbox_sample, local_model,
                nonlinear_xfm,
                symmetric=segment_symmetric,
                level=nlreg_level,
                start_level=nlreg_start,
                parameters=nlreg_options,
                ants=False,
                downsample=nlreg_downsample )

        print("\n\n\nWarping the sample!:{}\n\n\n".format(bbox_sample))
        nl_sample.seg=None
        nl_sample.seg_f=None
        nl_sample.mask=None

        warp_sample(bbox_sample, local_model, nl_sample,
                    transform=nonlinear_xfm,
                    symmetric=segment_symmetric,
                    resample_order=resample_order)

        output_info['nl_sample']=nl_sample
    else:
        nl_sample=bbox_sample

    output_info['nonlinear_xfm']=nonlinear_xfm

    if exclude_re is not None:
        _exclude_re=re.compile(exclude_re)
        selected_library=[i for i in library if not _exclude_re.match(i[2]) and i[2] not in exclude]
    else:
        selected_library=[i for i in library if i[2] not in exclude]

    selected_library_f=[]

    if segment_symmetric: # fill up with all entries
        selected_library_f=selected_library

    # library pre-selection if needed
    # we need balanced number of samples for each group
    if library_preselect>0 and library_preselect < len(selected_library):
        loaded=False
        loaded_f=False

        if os.path.exists(work_lib_dir+os.sep+'sel_library.json'):
            with open(work_lib_dir+os.sep+'sel_library.json','r') as f:
                selected_library=json.load(f)
            loaded=True

        if segment_symmetric and os.path.exists(work_lib_dir_f+os.sep+'sel_library.json'):
            with open(work_lib_dir_f+os.sep+'sel_library.json','r') as f:
                selected_library_f=json.load(f)
            loaded_f=True

        if do_nonlinear_register:
            if not loaded:
                selected_library=preselect(nl_sample,
                                       selected_library, 
                                       method=library_preselect_method, 
                                       number=library_preselect,
                                       use_nl=library_nl_samples_avail,
                                       step=library_preselect_step,
                                       lib_add_n=library_modalities,
                                       groups=groups) 
            if segment_symmetric:
                if not loaded_f:
                    selected_library_f=preselect(nl_sample,
                                             selected_library, 
                                             method=library_preselect_method,
                                             number=library_preselect,
                                             use_nl=library_nl_samples_avail,
                                             flip=True,
                                             step=library_preselect_step,
                                             lib_add_n=library_modalities,
                                             groups=groups)
        else:
            if not loaded:
                selected_library=preselect(bbox_sample,
                                       selected_library,
                                       method=library_preselect_method,
                                       number=library_preselect,
                                       use_nl=False,
                                       step=library_preselect_step,
                                       lib_add_n=library_modalities,
                                       groups=groups)
            if segment_symmetric:
                if not loaded_f:
                    selected_library_f=preselect(bbox_sample, selected_library,
                                             method=library_preselect_method,
                                             number=library_preselect,
                                             use_nl=False,flip=True,
                                             step=library_preselect_step,
                                             lib_add_n=library_modalities,
                                             groups=groups)

        if not loaded:
            with open(work_lib_dir+os.sep+'sel_library.json','w') as f:
                json.dump(selected_library,f)

        if not loaded_f:
            if segment_symmetric:
                with open(work_lib_dir_f+os.sep+'sel_library.json','w') as f:
                    json.dump(selected_library_f,f)
                    
        output_info['selected_library']=selected_library
        if segment_symmetric:
            output_info['selected_library_f']=selected_library_f
    
    selected_library_scan=[]
    selected_library_xfm=[]
    selected_library_warped2=[]
    selected_library_xfm2=[]

    selected_library_scan_f=[]
    selected_library_xfm_f=[]
    selected_library_warped_f=[]
    selected_library_warped2_f=[]
    selected_library_xfm2_f=[]
    
    for (i,j) in enumerate(selected_library):
        d=MriDataset(scan=j[2],seg=j[3], add=j[4:4+library_modalities],group=int(j[0]), grading=float(j[1]) )
        
        selected_library_scan.append(d)
        
        selected_library_warped2.append(    MriDataset(name=d.name, prefix=work_lib_dir, add_n=sample_modalities,group=int(j[0]), grading=float(j[1]) ))
        selected_library_xfm2.append(       MriTransform(name=d.name,prefix=work_lib_dir ))

        if library_nl_samples_avail:
            selected_library_xfm.append(    MriTransform(xfm=j[4+library_modalities], xfm_inv=j[5+library_modalities] ) )
            
    output_info['selected_library_warped2']=selected_library_warped2
    output_info['selected_library_xfm2']=selected_library_xfm2
    if library_nl_samples_avail:
        output_info['selected_library_xfm']=selected_library_xfm

    if segment_symmetric:
        for (i,j) in enumerate(selected_library_f):
            d=MriDataset(scan=j[2],seg=j[3], add=j[4:4+library_modalities], group=int(j[0]), grading=float(j[1]) )
            selected_library_scan_f.append(d)
            selected_library_warped2_f.append(MriDataset(name=d.name, prefix=work_lib_dir_f, add_n=sample_modalities ))
            selected_library_xfm2_f.append(MriTransform( name=d.name, prefix=work_lib_dir_f ))
            
            if library_nl_samples_avail:
                selected_library_xfm_f.append(   MriTransform(xfm=j[4+library_modalities], xfm_inv=j[5+library_modalities] ))

        output_info['selected_library_warped2_f']=selected_library_warped2_f
        output_info['selected_library_xfm2_f']=selected_library_xfm2_f
        if library_nl_samples_avail:
            output_info['selected_library_xfm_f']=selected_library_xfm_f
                
    # nonlinear registration to template or individual
    
    if do_pairwise: # Right now ignore precomputed transformations
        results=[]
        if debug:
            print("Performing pairwise registration")
        
        for (i,j) in enumerate(selected_library):
            # TODO: make clever usage of precomputed transform if available
            if pairwise_register_type=='elx' or pairwise_register_type=='elastix' :
                results.append( 
                    elastix_registration.remote(
                    bbox_sample,
                    selected_library_scan[i],
                    selected_library_xfm2[i],
                    level=pairwise_level,
                    start_level=pairwise_start,
                    parameters=pairwise_options,  
                    nl=True,
                    output_inv_target=selected_library_warped2[i],
                    warp_seg=True,
                    resample_order=resample_order,
                    resample_baa=resample_baa
                    ) )
            elif pairwise_register_type=='ants' or do_pairwise_ants:
                results.append( 
                    non_linear_registration.remote(
                    bbox_sample,
                    selected_library_scan[i],
                    selected_library_xfm2[i],
                    level=pairwise_level,
                    start_level=pairwise_start,
                    parameters=pairwise_options,  
                    ants=True,
                    output_inv_target=selected_library_warped2[i],
                    warp_seg=True,
                    resample_order=resample_order,
                    resample_baa=resample_baa
                    ) )
            else:
                results.append( 
                    non_linear_registration.remote(
                    bbox_sample,
                    selected_library_scan[i],
                    selected_library_xfm2[i],
                    level=pairwise_level,
                    start_level=pairwise_start,
                    parameters=pairwise_options,  
                    ants=False,
                    output_inv_target=selected_library_warped2[i],
                    warp_seg=True,
                    resample_order=resample_order,
                    resample_baa=resample_baa
                    ) )
            
        if segment_symmetric:
            for (i,j) in enumerate(selected_library_f):
                # TODO: make clever usage of precomputed transform if available
                if pairwise_register_type=='elx' or pairwise_register_type=='elastix' :
                    results.append( 
                        elastix_registration.remote(
                        bbox_sample,
                        selected_library_scan_f[i],
                        selected_library_xfm2_f[i],
                        level=pairwise_level,
                        start_level=pairwise_start,
                        parameters=pairwise_options,  
                        nl=True,
                        output_inv_target=selected_library_warped2_f[i],
                        warp_seg=True,
                        flip=True,
                        resample_order=resample_order,
                        resample_baa=resample_baa
                        ) )
                elif pairwise_register_type=='ants' or do_pairwise_ants:
                    results.append( 
                        non_linear_registration.remote(
                        bbox_sample,
                        selected_library_scan_f[i],
                        selected_library_xfm2_f[i],
                        level=pairwise_level,
                        start_level=pairwise_start,
                        parameters=pairwise_options,  
                        ants=True,
                        output_inv_target=selected_library_warped2_f[i],
                        warp_seg=True,
                        flip=True,
                        resample_order=resample_order,
                        resample_baa=resample_baa
                        ) )
                else:
                    results.append( 
                        non_linear_registration.remote(
                        bbox_sample,
                        selected_library_scan_f[i],
                        selected_library_xfm2_f[i],
                        level=pairwise_level,
                        start_level=pairwise_start,
                        parameters=pairwise_options,  
                        ants=False,
                        output_inv_target=selected_library_warped2_f[i],
                        warp_seg=True,
                        flip=True,
                        resample_order=resample_order,
                        resample_baa=resample_baa
                        ) )
        # TODO: do we really need to wait for result here?
        ray.wait(results, num_returns=len(results))
    else:
            
        results=[]
        
        for (i, j) in enumerate(selected_library):
            
            lib_xfm=None
            if library_nl_samples_avail:
                lib_xfm=selected_library_xfm[i]
                
            results.append( 
                concat_resample.remote(
                 selected_library_scan[i],
                 lib_xfm ,
                 nonlinear_xfm,
                 selected_library_warped2[i],
                 resample_order=resample_order,
                 label_resample_order=label_resample_order,
                 resample_baa=resample_baa
                ) )
                
        if segment_symmetric:
            for (i, j) in enumerate(selected_library_f):
                lib_xfm=None
                if library_nl_samples_avail:
                    lib_xfm=selected_library_xfm_f[i]

                results.append( 
                    concat_resample.remote(
                    selected_library_scan_f[i],
                    lib_xfm,
                    nonlinear_xfm,
                    selected_library_warped2_f[i],
                    resample_order=resample_order,
                    label_resample_order=label_resample_order,
                    resample_baa=resample_baa,
                    flip=True
                    ) )
        # TODO: do we really need to wait for result here?
        ray.wait(results,num_retuns=len(results))

    results=[]

    sample_seg=MriDataset(name='bbox_seg_' + sample.name+out_variant, prefix=work_dir )
    sample_grad=MriDataset(name='bbox_grad_' + sample.name+out_variant, prefix=work_dir )
    
    results.append( 
        fuse_grading.remote(
        bbox_sample,
        sample_seg,
        selected_library_warped2, 
        flip=False,
        classes_number=classes_number,
        fuse_options=fuse_options,
        model=local_model,
        debug=debug,
        fuse_variant=fuse_variant,
        groups=groups
        ))

    if segment_symmetric:
        results.append( 
            fuse_grading.remote(
            bbox_sample,
            sample_seg,
            selected_library_warped2_f, 
            flip=True,
            classes_number=classes_number,
            fuse_options=fuse_options,
            model=local_model,
            debug=debug,
            fuse_variant=fuse_variant,
            groups=groups
            ))

    ray.wait(results, num_retuns=len(results))

    output_info['fuse']=results[0].result()
    if segment_symmetric:
        output_info['fuse_f']=results[1].result()

    if qc_options:
        # generate QC images
        output_info['qc'] = generate_qc_image(sample_seg,
                                              bbox_sample, 
                                              sample_qc, 
                                              options=qc_options,
                                              model=local_model,
                                              symmetric=segment_symmetric,
                                              labels=library_description['classes_number'])
    # cleanup if need
    if cleanup:
        shutil.rmtree(work_lib_dir)
        shutil.rmtree(work_lib_dir_f)
        if nl_sample is not None:
            nl_sample.cleanup()
    
    if cleanup_xfm:
        if nonlinear_xfm is not None:
            nonlinear_xfm.cleanup()
    
    if not run_in_bbox:
        # TODO: apply error correction here
        # rename labels to final results
        sample_seg_native=MriDataset(name='seg_' + sample.name+out_variant, prefix=work_dir )
        
        warp_rename_seg( sample_seg, sample, sample_seg_native, 
                        transform=bbox_linear_xfm, invert_transform=True, 
                        lut=library_description['map'] , 
                        symmetric=segment_symmetric,
                        symmetric_flip=segment_symmetric,
                        use_flipped=segment_symmetric,  # needed to flip .seg_f back to right orientation
                        flip_lut=library_description['flip_map'],
                        resample_baa=resample_baa, 
                        resample_order=label_resample_order,
                        datatype=seg_datatype )
        
        warp_sample(sample_seg, sample, sample_seg_native,
                    transform=bbox_linear_xfm, invert_transform=True,
                    symmetric=segment_symmetric,
                    symmetric_flip=segment_symmetric,# need to flip symmetric dataset
                    resample_order=resample_order)
        
        output_info['sample_seg_native']=sample_seg_native
        
        if segment_symmetric:
            join_left_right(sample_seg_native, output_segment+'_seg.mnc', output_segment+'_grad.mnc', datatype=seg_datatype)
        else:
            shutil.copyfile(sample_seg_native.seg, output_segment+'_seg.mnc')
            shutil.copyfile(sample_seg_native.scan, output_segment+'_grad.mnc')
        
        output_info['output_segment']=output_segment+'_seg.mnc'
        output_info['output_grading']=output_segment+'_grad.mnc'
        
        volumes=seg_to_volumes_grad( output_segment+'_seg.mnc',
                                output_segment+'_vol.json',
                                label_map=library_description.get('label_map',None),
                                grad=output_segment+'_grad.mnc',
                                median=use_median)
        
        output_info['output_volumes']=volumes
        output_info['output_volumes_json']=output_segment+'_vol.json'
        
        # TODO: cleanup more here (?)
        
        return (output_segment+'_seg.mnc', output_segment+'_grad.mnc', volumes, output_info)
    else: # special case, needed to train error correction TODO: remove?
        volumes=seg_to_volumes_grad(sample_seg.seg, 
                               output_segment+'_vol.json', 
                               grad=sample_seg.scan,
                               median=use_median)
        return (sample_seg.seg, sample_seg.scan, volumes, output_info)


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

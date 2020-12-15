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
import traceback

import yaml
# MINC stuff
from ipl.minc_tools import mincTools,mincError

# scoop parallel execution
from scoop import futures, shared

from .filter           import *
from .structures       import *
from .registration     import *
from .resample         import *
from .error_correction import *
from .preselect        import *
from .qc               import *
from .fuse_segmentations import *
from .library          import *
from .analysis         import *


def fusion_segment( input_scan,
                    library_description,
                    output_segment,
                    input_mask = None,
                    parameters = {},
                    exclude    =[],
                    work_dir = None,
                    debug = False,
                    ec_variant = None,
                    fuse_variant = None,
                    regularize_variant = None,
                    add=[],
                    cleanup = False,
                    cleanup_xfm = False,
                    presegment = None,
                    preprocess_only = False):
    """Apply fusion segmentation"""
    try:
        if debug: 
            print( "Segmentation parameters:")
            print( repr(parameters) )
            print( "presegment={}".format(repr(presegment)))
            
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

        work_lib_dir =   work_dir+os.sep+'library'
        work_lib_dir_f = work_dir+os.sep+'library_f'

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
        
        # if linear registration should be performed
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
        do_nonlinear_register     =  parameters.get('non_linear_register', False )
        
        # generate segmentation library (needed for label fusion, not needed for single atlas based or external tool)
        generate_library          =  parameters.get('generate_library', True )
        
        # if non-linear registraiton should be performed pairwise
        do_pairwise               =  parameters.get('non_linear_pairwise', False )
        # if pairwise registration should be performed using ANTS
        do_pairwise_ants          =  parameters.get('non_linear_pairwise_ants', True )
        pairwise_register_type    =  parameters.get( 'non_linear_pairwise_type',None)
        if pairwise_register_type is None:
            if do_pairwise_ants: 
                pairwise_register_type='ants'
        
        library_preselect=        parameters.get('library_preselect', 10)
        library_preselect_step=   parameters.get('library_preselect_step', None)
        library_preselect_method= parameters.get('library_preselect_method', 'MI')
        

        # if non-linear registraiton should be performed with ANTS
        do_nonlinear_register_ants=parameters.get('non_linear_register_ants',False )
        nlreg_level        =  parameters.get('non_linear_register_level', 2)
        nlreg_start        =  parameters.get('non_linear_register_start', 16)
        nlreg_options      =  parameters.get('non_linear_register_options', None)
        nlreg_downsample   =  parameters.get('non_linear_register_downsample', None)
        
        nonlinear_register_type   = parameters.get( 'non_linear_register_type',None)
        if nonlinear_register_type is None:
            if do_nonlinear_register_ants: 
                nonlinear_register_type='ants'

        pairwise_level     =  parameters.get('pairwise_level', 2)
        pairwise_start     =  parameters.get('pairwise_start', 16)
        pairwise_options   =  parameters.get('pairwise_options', None)

        fuse_options       =  parameters.get('fuse_options', None)
        
        resample_order      = parameters.get('resample_order', 2)
        resample_baa        = parameters.get('resample_baa', True)

        # error correction parametrs
        ec_options          = parameters.get('ec_options', None)
        
        # QC image paramters
        qc_options          = parameters.get('qc_options', None)

        
        # special case for training error correction, assume input scan is already pre-processed
        run_in_bbox         = parameters.get('run_in_bbox', False)
        
        # mask output
        mask_output         = parameters.get('mask_output', True)
        
        classes_number      = library_description["classes_number"]
        seg_datatype        = library_description["seg_datatype"]
        gco_energy          = library_description["gco_energy"]
        
        
        output_info         = {}
        
        input_sample        = MriDataset(scan=input_scan,  seg=presegment, 
                                         mask=input_mask, protect=True,
                                         add=add)
        
        sample              = input_sample
        
        # get parameters
        model = MriDataset(scan=library_description["model"],
                           mask=library_description["model_mask"],
                           add= library_description["model_add"])
        
        local_model = MriDataset(scan=  library_description["local_model"],
                                 mask=  library_description["local_model_mask"],
                                 scan_f=library_description["local_model_flip"],
                                 mask_f=library_description["local_model_mask_flip"],
                                 seg=   library_description["local_model_seg"],
                                 seg_f= library_description["local_model_seg_flip"],
                                 add=   library_description["local_model_add"],
                                 add_f= library_description["local_model_add_flip"],
                                 )

        library = library_description["library"]
        
        sample_modalities = len(add)
        
        print("\n\n")
        print("Additional sample modalities:{}".format(sample_modalities))
        print("\n\n")
        # apply the same steps as used in library creation to perform segmentation:

        # global
        initial_xfm=None
        nonlinear_xfm=None
        bbox_sample=None
        nl_sample=None
        bbox_linear_xfm=None
        flipdir = work_dir+os.sep+'flip'

        sample_filtered = MriDataset(prefix=work_dir, name='flt_'+sample.name, add_n=sample_modalities )

        # QC file
        # TODO: allow for alternative location, extension
        #sample_qc=work_dir+os.sep+'qc_'+sample.name+'_'+out_variant+'.jpg'
        sample_qc = output_segment+'_qc.jpg'

        if run_in_bbox:
            segment_symmetric = False # that would depend ?
            do_initial_register = False
            do_initial_local_register = False
            # assume filter already applied!
            pre_filters =None
            post_filters = None
            print("Running in the box")

        if pre_filters is not None:
            apply_filter(sample.scan,
                         sample_filtered.scan,
                         pre_filters,
                         model=model.scan,
                         model_mask=model.mask)
            
            if sample.mask is not None:
                shutil.copyfile(sample.mask, sample_filtered.mask)
            else:
                sample_filtered.mask=None
            
            for i,j in enumerate(sample.add):
                shutil.copyfile(sample.add[i], sample_filtered.add[i])
                
            sample = sample_filtered
        else:
            sample_filtered=None
        
        output_info['sample_filtered']=sample_filtered

        if segment_symmetric:
            # need to flip the inputs
            if not os.path.exists(flipdir):
                os.makedirs(flipdir)

            sample.scan_f=flipdir+os.sep+os.path.basename(sample.scan)
            sample.add_f=['' for (i,j) in enumerate(sample.add)]

            for (i,j) in enumerate(sample.add):
                sample.add_f[i]=flipdir+os.sep+os.path.basename(sample.add[i])

            if sample.mask is not None:
                sample.mask_f=flipdir+os.sep+'mask_'+os.path.basename(sample.scan)
            else:
                sample.mask_f=None
            
            generate_flip_sample(sample)

        if presegment is None:
            sample.seg = None
            sample.seg_f = None

        
        if do_initial_register is not None:
            initial_xfm=MriTransform(prefix=work_dir, name='init_'+sample.name )
            
            if inital_reg_type=='elx' or inital_reg_type=='elastix' :
                elastix_registration(sample,
                                    model, initial_xfm,
                                    symmetric=segment_symmetric,
                                    parameters=inital_reg_options,
                                    nl=False,
                                    use_mask=inital_reg_use_mask,
                                    downsample=inital_reg_downsample
                                    )
            elif inital_reg_type=='ants' or inital_reg_ants:
                linear_registration(sample,
                                    model, initial_xfm,
                                    symmetric=segment_symmetric,
                                    reg_type=inital_reg_type,
                                    linreg=inital_reg_options,
                                    ants=True,
                                    use_mask=inital_reg_use_mask,
                                    downsample=inital_reg_downsample
                                    )
            else:
                linear_registration(sample,
                                    model, initial_xfm,
                                    symmetric=segment_symmetric,
                                    reg_type=inital_reg_type,
                                    linreg=inital_reg_options,
                                    downsample=inital_reg_downsample,
                                    use_mask=inital_reg_use_mask,
                                    objective=initial_reg_objective
                                    )
            
            output_info['initial_xfm']=initial_xfm
            

        # local 
        bbox_sample = MriDataset(prefix=work_dir, name='bbox_init_'+sample.name,
                                 add_n=sample_modalities )
        # a hack to have sample mask
        bbox_sample_mask = MriDataset(prefix=work_dir, name='bbox_init_'+sample.name )


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
                    use_mask=local_reg_use_mask,
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
                    use_mask=local_reg_use_mask,
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
                    use_mask=local_reg_use_mask,
                    objective=local_reg_objective,
                    downsample=local_reg_downsample )

        else: 
            bbox_linear_xfm=initial_xfm

        output_info['bbox_initial_xfm']=bbox_linear_xfm
        
        bbox_sample.mask=None
        bbox_sample.mask_f=None
        
        if sample.seg is None:
            bbox_sample.seg=None
            bbox_sample.seg_f=None
        
        warp_sample(sample, local_model, bbox_sample,
                    transform=bbox_linear_xfm,
                    symmetric=segment_symmetric,
                    symmetric_flip=segment_symmetric,# need to flip symmetric dataset
                    resample_order=resample_order,
                    filters=post_filters,
                    )
        
        if sample.seg is not None:
            _lut = None
            _flip_lut = None
            if not run_in_bbox: # assume that labels are already renamed
                _lut=invert_lut(library_description.get("map",None))
                _flip_lut=invert_lut(library_description.get("flip_map",None))

            warp_rename_seg( sample, local_model, bbox_sample,
                transform=bbox_linear_xfm,
                symmetric=segment_symmetric,
                symmetric_flip=segment_symmetric,
                lut      = _lut,
                flip_lut = _flip_lut,
                resample_order=resample_order,
                resample_baa=resample_baa)

        output_info['bbox_sample']=bbox_sample
        
        if preprocess_only:
            if cleanup:
                shutil.rmtree(work_lib_dir)
                shutil.rmtree(work_lib_dir_f)
                if os.path.exists(flipdir):
                    shutil.rmtree(flipdir)
                if pre_filters is not None:
                    sample_filtered.cleanup()
            return (None,output_info)

        # 3. run non-linear registration if needed
        # TODO: skip if sample presegmented
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
            nl_sample.mask_f=None
            
            warp_sample(bbox_sample, local_model, nl_sample,
                        transform=nonlinear_xfm,
                        symmetric=segment_symmetric,
                        resample_order=resample_order,
                        filters=post_filters,
                        )
            
            warp_model_mask(local_model, bbox_sample_mask,
                        transform=nonlinear_xfm,
                        symmetric=segment_symmetric,
                        resample_order=resample_order)
            
            bbox_sample.mask = bbox_sample_mask.mask
            bbox_sample.mask_f = bbox_sample_mask.mask_f
            
            output_info['bbox_sample'] = bbox_sample
            output_info['nl_sample'] = nl_sample
        else:
            nl_sample=bbox_sample
            # use mask from the model directly?
            bbox_sample.mask = local_model.mask
            bbox_sample.mask_f = local_model.mask
            
        output_info['nonlinear_xfm'] = nonlinear_xfm

        if generate_library:
            # remove excluded samples TODO: use regular expressions for matching?
            selected_library = [ i for i in library if i[0] not in exclude]
            selected_library_f = []
            
            if segment_symmetric: # fill up with all entries
                selected_library_f = copy.deepcopy(selected_library)

            # library pre-selection if needed
            # TODO: skip if sample presegmented
            if library_preselect>0 and library_preselect < len(selected_library):
                loaded = False
                loaded_f = False
                
                if os.path.exists(work_lib_dir + os.sep + 'sel_library.yaml'):
                    with open(work_lib_dir + os.sep + 'sel_library.yaml', 'r') as f:
                        selected_library = yaml.safe_load(f)
                    for i in selected_library:
                        i.prefix = work_lib_dir
                    loaded = True

                if segment_symmetric and os.path.exists(work_lib_dir_f + os.sep + 'sel_library_f.yaml'):
                    with open(work_lib_dir + os.sep + 'sel_library_f.yaml', 'r') as f:
                        selected_library_f= yaml.safe_load(f)
                    for i in selected_library_f:
                        i.prefix = work_lib_dir_f
                    loaded_f = True
                
                if do_nonlinear_register:
                    if not loaded:
                        selected_library = preselect(nl_sample,
                                            selected_library, 
                                            method=library_preselect_method, 
                                            number=library_preselect,
                                            use_nl=library_nl_samples_avail,
                                            step=library_preselect_step,
                                            lib_add_n=library_modalities) 
                    if segment_symmetric:
                        if not loaded_f:
                            selected_library_f = preselect(nl_sample,
                                                    selected_library_f, 
                                                    method=library_preselect_method,
                                                    number=library_preselect,
                                                    use_nl=library_nl_samples_avail,
                                                    flip=True,
                                                    step=library_preselect_step,
                                                    lib_add_n=library_modalities)
                else:
                    if not loaded:
                        selected_library = preselect(bbox_sample,
                                            selected_library,
                                            method=library_preselect_method,
                                            number=library_preselect,
                                            use_nl=False,
                                            step=library_preselect_step,
                                            lib_add_n=library_modalities)
                    if segment_symmetric:
                        if not loaded_f:
                            selected_library_f = preselect(bbox_sample,
                                                    selected_library_f,
                                                    method=library_preselect_method,
                                                    number=library_preselect,
                                                    use_nl=False,flip=True,
                                                    step=library_preselect_step,
                                                    lib_add_n=library_modalities)

                if not loaded:
                    with open(work_lib_dir + os.sep + 'sel_library.yaml', 'w') as f:
                        f.write( yaml.dump( selected_library ) )

                if not loaded_f and segment_symmetric:
                    with open(work_lib_dir + os.sep + 'sel_library_f.yaml', 'w') as f:
                        f.write( yaml.dump( selected_library_f ) )
                            
                output_info['selected_library'] = selected_library
                if segment_symmetric:
                    output_info['selected_library_f'] = selected_library_f
            
            selected_library_scan=[]
            selected_library_xfm=[]
            selected_library_warped2=[]
            selected_library_xfm2=[]

            selected_library_scan_f=[]
            selected_library_xfm_f=[]
            selected_library_warped_f=[]
            selected_library_warped2_f=[]
            selected_library_xfm2_f=[]
            
            for (i, j) in enumerate(selected_library):
                d = MriDataset(scan=j[0], seg=j[1], add=list(j[2:2+library_modalities]))
                
                selected_library_scan.append(d)
                
                selected_library_warped2.append(    MriDataset(name=d.name, prefix=work_lib_dir, add_n=sample_modalities ))
                selected_library_xfm2.append(       MriTransform(name=d.name,prefix=work_lib_dir ))

                if library_nl_samples_avail:
                    selected_library_xfm.append(    MriTransform(xfm=j[2+library_modalities], xfm_inv=j[3+library_modalities] ) )
                    
            output_info['selected_library_warped2']=selected_library_warped2
            output_info['selected_library_xfm2']=selected_library_xfm2
            
            if library_nl_samples_avail:
                output_info['selected_library_xfm']=selected_library_xfm

            if segment_symmetric:
                for (i,j) in enumerate(selected_library_f):
                    d=MriDataset(scan=j[0],seg=j[1], add=j[2:2+library_modalities] )
                    selected_library_scan_f.append(d)
                    selected_library_warped2_f.append(MriDataset(name=d.name, prefix=work_lib_dir_f, add_n=sample_modalities ))
                    selected_library_xfm2_f.append(MriTransform( name=d.name, prefix=work_lib_dir_f ))
                    
                    if library_nl_samples_avail:
                        selected_library_xfm_f.append(   MriTransform(xfm=j[2+library_modalities], xfm_inv=j[3+library_modalities] ))

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
                        results.append( futures.submit(
                            elastix_registration, 
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
                        results.append( futures.submit(
                            non_linear_registration, 
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
                        results.append( futures.submit(
                            non_linear_registration, 
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
                        
                        if pairwise_register_type == 'elx' or pairwise_register_type == 'elastix':
                            results.append( futures.submit(
                                elastix_registration, 
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
                            results.append( futures.submit(
                                non_linear_registration, 
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
                            results.append( futures.submit(
                                non_linear_registration, 
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
                futures.wait(results, return_when=futures.ALL_COMPLETED)
            else: # use precomputer transformations
                
                results=[]
                
                for (i, j) in enumerate(selected_library):
                    
                    lib_xfm=None
                    if library_nl_samples_avail:
                        lib_xfm=selected_library_xfm[i]
                        
                    results.append( futures.submit( 
                        concat_resample,
                        selected_library_scan[i],
                        lib_xfm ,
                        nonlinear_xfm,
                        selected_library_warped2[i],
                        resample_order=resample_order,
                        resample_baa=resample_baa
                        ) )
                        
                if segment_symmetric:
                    for (i, j) in enumerate(selected_library_f):
                        lib_xfm=None
                        if library_nl_samples_avail:
                            lib_xfm=selected_library_xfm_f[i]

                        results.append( futures.submit(
                            concat_resample,
                            selected_library_scan_f[i],
                            lib_xfm,
                            nonlinear_xfm,
                            selected_library_warped2_f[i],
                            resample_order=resample_order,
                            resample_baa=resample_baa,
                            flip=True
                            ) )
                # TODO: do we really need to wait for result here?
                futures.wait(results, return_when=futures.ALL_COMPLETED)
        else: # no library generated
            selected_library=[]
            selected_library_f=[]
            selected_library_warped2=[]
            selected_library_warped2_f=[]
        
        results=[]

        sample_seg = MriDataset(name='bbox_seg_' + sample.name+out_variant, prefix=work_dir )
        sample_seg.mask = None
        sample_seg.mask_f = None

        print(local_model        )

        results.append(futures.submit(
            fuse_segmentations,
            bbox_sample,
            sample_seg,
            selected_library_warped2, 
            flip=False,
            classes_number=classes_number,
            fuse_options=fuse_options,
            gco_energy=gco_energy,
            ec_options=ec_options,
            model=local_model,
            debug=debug,
            ec_variant=ec_variant,
            fuse_variant=fuse_variant,
            regularize_variant=regularize_variant
            ))

        if segment_symmetric:
            results.append( futures.submit(
                fuse_segmentations,
                bbox_sample,
                sample_seg,
                selected_library_warped2_f, 
                flip=True,
                classes_number=classes_number,
                fuse_options=fuse_options,
                gco_energy=gco_energy,
                ec_options=ec_options,
                model=local_model,
                debug=debug,
                ec_variant=ec_variant,
                fuse_variant=fuse_variant,
                regularize_variant=regularize_variant
                ))

        futures.wait(results, return_when=futures.ALL_COMPLETED)

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
                                    labels=library_description["classes_number"])
        # cleanup if need
        if cleanup:
            shutil.rmtree(work_lib_dir)
            shutil.rmtree(work_lib_dir_f)
            if os.path.exists(flipdir):
                shutil.rmtree(flipdir)
                
            if nl_sample is not None:
                nl_sample.cleanup()
            
            if pre_filters is not None:
                sample_filtered.cleanup()
            
        if cleanup_xfm:
            # TODO: remove more xfms(?)
            if nonlinear_xfm is not None:
                nonlinear_xfm.cleanup()
        
        if not run_in_bbox:
            # TODO: apply error correction here
            # rename labels to final results
            sample_seg_native = MriDataset(name='seg_' + sample.name+out_variant, prefix=work_dir )
            sample_seg_native2 = MriDataset(name='seg2_' + sample.name+out_variant, prefix=work_dir )
            
            warp_rename_seg(sample_seg, input_sample, sample_seg_native, 
                            transform=bbox_linear_xfm, invert_transform=True, 
                            lut=library_description["map"],
                            symmetric=segment_symmetric,
                            symmetric_flip=segment_symmetric,
                            use_flipped=segment_symmetric,  # needed to flip .seg_f back to right orientation
                            flip_lut=library_description["flip_map"],
                            resample_baa=resample_baa, 
                            resample_order=resample_order,
                            datatype=seg_datatype )
            
            output_info['sample_seg_native'] = sample_seg_native
            output_info['used_labels']       = make_segmented_label_list(library_description,symmetric=segment_symmetric)
            
            _output_segment=output_segment+'_seg.mnc'
            
            if segment_symmetric:
                join_left_right(sample_seg_native, sample_seg_native2.seg, datatype=seg_datatype)
            else:
                sample_seg_native2=sample_seg_native
                #shutil.copyfile(sample_seg_native.seg, output_segment+'_seg.mnc')
            
            if mask_output and input_mask is not None:
                #
                with mincTools() as minc:
                    minc.calc([sample_seg_native2.seg, input_mask],'A[1]>0.5?A[0]:0',_output_segment,labels=True)
            else:
                shutil.copyfile(sample_seg_native2.seg, _output_segment)
                
            output_info['output_segment'] = _output_segment
            output_info['output_volumes'] = seg_to_volumes(_output_segment, 
                                            output_segment+'_vol.json', 
                                            label_map=library_description["label_map"])
            
            output_info['output_volumes_json'] = output_segment+'_vol.json'

            # TODO: cleanup more here (?)
            
            return (_output_segment,output_info)
        else: # special case, needed to train error correction
            return (sample_seg.seg,output_info)
        
    except mincError as e:
        print("Exception in fusion_segment:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout )
        raise
    except :
        print("Exception in fusion_segment:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

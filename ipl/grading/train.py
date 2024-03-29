# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 
#

from __future__ import print_function

import shutil
import os
import sys
import csv
import copy
import traceback

# MINC stuff
# from ipl.minc_tools import mincTools,mincError

import ray

from .filter           import *
from .structures       import *
from .registration     import *
from .resample         import *
from .model            import *
from .library          import *


def inv_dict(d):
    return { v:k for (k,v) in d.items() }

def generate_library(parameters, output, debug=False,cleanup=False):
    '''Actual generation of the segmentation library'''
    try:
        if debug: print(repr(parameters))

        # read parameters
        reference_model           = parameters[ 'reference_model']
        reference_mask            = parameters.get( 'reference_mask', None)
        reference_model_add       = parameters.get( 'reference_model_add', [])
        
        reference_local_model     = parameters.get( 'reference_local_model', None)
        reference_local_mask      = parameters.get( 'reference_local_mask', None)
        
        reference_local_model_flip= parameters.get( 'reference_local_model_flip', None)
        reference_local_mask_flip = parameters.get( 'reference_local_mask_flip', None)
        
        library                   = parameters[ 'library' ]
        
        work_dir                  = parameters.get( 'workdir',output+os.sep+'work')
        
        train_groups              = parameters[ 'groups']
        
        # should we build symmetric model
        build_symmetric           = parameters.get( 'build_symmetric' ,False)
        
        # should we build symmetric flipped model
        build_symmetric_flip      = parameters.get( 'build_symmetric_flip' ,False)
        
        # lookup table for renaming labels for more compact representation
        build_remap               = parameters.get( 'build_remap' ,{})
        
        # lookup table for renaming labels for more compact representation, 
        # when building symmetrized library 
        build_flip_remap          = parameters.get( 'build_flip_remap' ,{})
        
        # lookup table for renaming labels for more compact representation, 
        # when building symmetrized library
        build_unflip_remap        = parameters.get( 'build_unflip_remap' ,{})
        
        if not build_unflip_remap and build_flip_remap and build_remap:
            build_unflip_remap = create_unflip_remap(build_remap,build_flip_remap)
        
        # label map
        label_map                 = parameters.get( 'label_map' ,None)
        
        # perform filtering as final stage of the library creation
        pre_filters       =         parameters.get( 'pre_filters' , None )
        post_filters      =         parameters.get( 'post_filters' , parameters.get( 'filters', None ))
        
        resample_order            = parameters.get( 'resample_order',2)
        label_resample_order      = parameters.get( 'label_resample_order',resample_order)
        
        # use boundary anti-aliasing filter when resampling labels
        resample_baa              = parameters.get( 'resample_baa',True)
        
        # perform label warping to create final library
        do_warp_labels            = parameters.get( 'warp_labels',False)

        # extent bounding box to reduce boundary effects
        extend_boundary           = parameters.get( 'extend_boundary',4)

        # extend maks 
        #dilate_mask               = parameters.get( 'dilate_mask',3)
        op_mask                    = parameters.get( 'op_mask','E[2] D[4]')

        # if linear registration should be performed
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
        do_nonlinear_register     = parameters.get( 'non_linear_register',False)
        
        # if non-linear registraiton should be performed with ANTS
        do_nonlinear_register_ants= parameters.get( 'non_linear_register_ants',False)
        nlreg_level               =  parameters.get('non_linear_register_level', 2)
        nlreg_start               =  parameters.get('non_linear_register_start', 16)
        nlreg_options             =  parameters.get('non_linear_register_options', None)
        nlreg_downsample          =  parameters.get('non_linear_register_downsample', None)
        
        nonlinear_register_type   = parameters.get( 'non_linear_register_type',None)
        if nonlinear_register_type is None:
            if do_nonlinear_register_ants: 
                nonlinear_register_type='ants'
        
        
        
        modalities                = parameters.get( 'modalities',1 ) - 1

        create_patch_norm_lib     = parameters.get( 'create_patch_norm_lib',False)
        patch_norm_lib_pct        = parameters.get( 'patch_norm_lib_pct', 0.1 )
        patch_norm_lib_sub        = parameters.get( 'patch_norm_lib_sub', 1 )
        patch_norm_lib_patch      = parameters.get( 'patch_norm_lib_patch', 2 ) # 5x5x5 patches
        
        # prepare directories
        if not os.path.exists(output):
            os.makedirs(output)
        
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        
        # 0. go over input samples, prepare variables
        input_samples=[]
        filtered_samples=[]
        lin_xfm=[]
        lin_samples=[]
        tmp_lin_samples=[]
        bbox_lin_xfm=[]
        #nl_xfm=[]
        #bbox_samples=[]
        
        final_samples=[]
        warped_samples=[]
        final_transforms=[]
        tmp_log_samples=[]
        
        patch_norm_db = output + os.sep  + 'patch_norm.db'
        patch_norm_idx = output + os.sep + 'patch_norm.idx'
        
        # identity xfm
        identity_xfm=MriTransform(prefix=work_dir,   name='identity' )
        with mincTools() as m:
            m.param2xfm(identity_xfm.xfm)
            m.param2xfm(identity_xfm.xfm_f)
        
        # check if library is list, if it is not, assume it's a reference to a csv file
        if library is not list:
            with open(library,'r') as f:
                library=list(csv.reader(f))

        # setup files
        model = MriDataset(scan=reference_model, mask=reference_mask, add=reference_model_add)

        for (j,i) in enumerate(library):
            scan=i[0]
            seg=i[1]
            add=i[2:modalities+2] # additional modalties
            group=None
            grading=None
            
            mask = work_dir + os.sep + 'fake_mask_' + os.path.basename(scan)
            create_fake_mask(seg, mask)

            if len(i)>modalities+2: # assume that the extra columns is group and grading
                group=    int(i[modalities+2])
                grading=float(i[modalities+3])

            sample=                 MriDataset(scan=scan, seg=seg, mask=mask,protect=True, add=add, group=group, grading=grading)
            input_samples.append(   sample )
            filtered_samples.append( MriDataset(  prefix=work_dir, name='flt_'+sample.name,      add_n=modalities, group=group, grading=grading ) )
            
            lin_xfm.append(          MriTransform(prefix=work_dir, name='lin_'+sample.name      ) )
            bbox_lin_xfm.append(     MriTransform(prefix=work_dir, name='lin_bbox_'+sample.name ) )
            lin_samples.append(      MriDataset(  prefix=work_dir, name='lin_'+sample.name,      add_n=modalities, group=group, grading=grading ) )
            tmp_lin_samples.append(  MriDataset(  prefix=work_dir, name='tmp_lin_'+ sample.name, add_n=modalities, group=group, grading=grading ) )
            tmp_log_samples.append(  MriDataset(  prefix=work_dir, name='tmp_log_'+ sample.name, group=group, grading=grading  ) )
            final_samples.append(    MriDataset(  prefix=output,   name=sample.name,        add_n=modalities, group=group, grading=grading ) )
            warped_samples.append(   MriDataset(  prefix=output,   name='nl_'+sample.name,  add_n=modalities, group=group, grading=grading ) )
            final_transforms.append( MriTransform(prefix=output,   name='nl_'+sample.name ) )

        # temp array
        results=[]
        
        if pre_filters is not None:
            # apply pre-filtering before other stages
            filter_all=[]
            
            for (j,i) in enumerate(input_samples):
                # a HACK?
                filtered_samples[j].seg      = input_samples[j].seg
                filtered_samples[j].group    = input_samples[j].group
                filtered_samples[j].grading  = input_samples[j].grading
                filtered_samples[j].mask     = input_samples[j].mask
                
                filter_all.append( 
                    filter_sample.remote( input_samples[j], filtered_samples[j], pre_filters, model=model
                    ))
            
            ray.wait(filter_all, num_returns=len(filter_all))
        else:
            filtered_samples=input_samples
            
        if build_symmetric:
            # need to flip the inputs
            flipdir=work_dir+os.sep+'flip'
            if not os.path.exists(flipdir):
                os.makedirs(flipdir)
            flip_all=[]

            labels_datatype='short'# TODO: determine optimal here
            #if largest_label>255:labels_datatype='short'

            for (j,i) in enumerate(filtered_samples):
                i.scan_f=flipdir+os.sep+os.path.basename(i.scan)
                i.add_f=[]
                for (k,j) in enumerate(i.add):
                    i.add_f.append(flipdir+os.sep+os.path.basename(i.add[k]))

                if i.mask is not None:
                    i.mask_f=flipdir+os.sep+'mask_'+os.path.basename(i.scan)

                flip_all.append( generate_flip_sample.remote( i, labels_datatype=labels_datatype  )  )

            ray.wait(flip_all, num_returns=len(flip_all))
        
        # 1. run global linear registration if nedded
        if do_initial_register:
            for (j,i) in enumerate(filtered_samples):
                if inital_reg_type=='elx' or inital_reg_type=='elastix' :
                    results.append( 
                        elastix_registration.remote( i, model, lin_xfm[j], 
                        symmetric=build_symmetric, 
                        parameters=inital_reg_options,
                        ) )
                elif inital_reg_type=='ants' or inital_reg_ants:
                    results.append( 
                        linear_registration.remote( i, model, lin_xfm[j], 
                        symmetric=build_symmetric, 
                        linreg=inital_reg_options,
                        ants=True
                        ) )
                else:
                    results.append( 
                        linear_registration.remote( i, model, lin_xfm[j], 
                        symmetric=build_symmetric, 
                        reg_type=inital_reg_type,
                        linreg=inital_reg_options,
                        objective=initial_reg_objective
                        ) )
            # TODO: do we really need to wait for result here?
            ray.wait( results, num_returns=len(results) )
            # TODO: determine if we need to resample input files here
            lin_samples=input_samples
        else:
            lin_samples=input_samples

        # 2. for each part run linear registration, apply flip and do symmetric too
        # 3. perform local linear registrtion and local intensity normalization if needed
        # create a local reference model
        local_model=None
        local_model_ovl=None
        local_model_avg=None
        local_model_sd=None
        
        if reference_local_model is None  :
            local_model    =MriDataset( prefix=output, name='local_model', add_n=modalities    )
            local_model_ovl=MriDataset( prefix=output, name='local_model_ovl' )
            local_model_avg=MriDataset( prefix=output, name='local_model_avg', add_n=modalities )
            local_model_sd =MriDataset( prefix=output, name='local_model_sd',  add_n=modalities  )
            
            if not os.path.exists( local_model.scan ):
                for (j,i) in enumerate( filtered_samples ):
                    xfm=None
                    if do_initial_register:
                        xfm=lin_xfm[j]
                    
                    results.append( 
                        warp_rename_seg.remote( i, model, tmp_lin_samples[j],
                            transform=xfm,
                            symmetric=build_symmetric,
                            symmetric_flip=build_symmetric,
                            lut=build_remap, 
                            flip_lut=build_flip_remap,
                            resample_order=0,
                            resample_baa=False # This is quick and dirty part
                        ) )
            
                ray.wait(results, num_returns=len(results))
                create_local_model(tmp_lin_samples, model, local_model, extend_boundary=extend_boundary, op=op_mask)
                
            if not os.path.exists(local_model.scan_f) and build_symmetric and build_symmetric_flip:
                create_local_model_flip(local_model, model, remap=build_unflip_remap, op=op_mask)
        else:
            local_model=MriDataset(scan=reference_local_model, mask=reference_local_mask)

            local_model.scan_f=reference_local_model_flip
            local_model.mask_f=reference_local_mask_flip
        
        if do_initial_local_register:
            for (j,i) in enumerate(lin_samples):
                init_xfm=None
                if do_initial_register:
                    init_xfm=lin_xfm[j]
                
                if local_reg_type=='elx' or local_reg_type=='elastix' :
                    results.append( 
                        elastix_registration.remote( i, local_model, bbox_lin_xfm[j], 
                        init_xfm=init_xfm,
                        symmetric=build_symmetric,
                        parameters=local_reg_opts,
                        bbox=local_reg_bbox
                        ) )
                elif local_reg_type=='ants' or local_reg_ants:
                    results.append( 
                        linear_registration.remote( i, local_model, bbox_lin_xfm[j], 
                        init_xfm=init_xfm,
                        symmetric=build_symmetric,
                        reg_type=local_reg_type,
                        linreg=local_reg_opts,
                        close=True,
                        ants=True,
                        bbox=local_reg_bbox
                        ) )
                else:
                    if not do_initial_register:
                        init_xfm=identity_xfm # to avoid strange initialization errors 
                        
                    results.append( 
                        linear_registration.remote( i, local_model, bbox_lin_xfm[j], 
                        init_xfm=init_xfm,
                        symmetric=build_symmetric,
                        reg_type=local_reg_type,
                        linreg=local_reg_opts,
                        close=True,
                        bbox=local_reg_bbox,
                        objective=local_reg_objective
                        ) )

            # TODO: do we really need to wait for result here?
            ray.wait(results, num_returns=len(results))
        else:
            bbox_lin_xfm=lin_xfm
        

        # create bbox samples
        results=[]
        for (j, i) in enumerate(input_samples):
            xfm=None

            if i.mask is None:
                final_samples[j].mask=None
                
            if i.mask_f is None:
                final_samples[j].mask_f=None

            if do_initial_local_register or do_initial_register:
                xfm=bbox_lin_xfm[j]
            #
            results.append( 
                warp_rename_seg.remote( i, local_model, final_samples[j],
                    transform=xfm,
                    symmetric=build_symmetric,
                    symmetric_flip=build_symmetric,
                    lut=build_remap,
                    flip_lut=build_flip_remap,
                    resample_order=label_resample_order,
                    resample_baa=resample_baa
                ) )
                    
        ray.wait(results, num_returns=len(results))
    
        results=[]
        for (j, i) in enumerate(input_samples):
            xfm=None
            if do_initial_local_register or do_initial_register:
                xfm=bbox_lin_xfm[j]
            
            results.append( 
                warp_sample.remote( i, local_model, final_samples[j],
                    transform=xfm,
                    symmetric=build_symmetric,
                    symmetric_flip=build_symmetric,
                    resample_order=resample_order,
                    filters=post_filters,
                    ) )
                    
        ray.wait(results, num_returns=len(results))

        if create_patch_norm_lib:
            #for (j, i) in enumerate(final_samples):
            #    results.append( futures.submit(
            #        log_transform_sample, i , tmp_log_samples[j] ) )
            # 
            # ray.wait(results,num_returns=len(results))
        
            create_patch_norm_db( final_samples, patch_norm_db, 
                                  patch_norm_idx,
                                  pct=patch_norm_lib_pct, 
                                  sub=patch_norm_lib_sub,
                                  patch=patch_norm_lib_patch)
        results=[]
        if do_nonlinear_register:
            for (j, i) in enumerate(final_samples):
                # TODO: decide what to do with mask
                i.mask=None
                
                if nonlinear_register_type=='elx' or nonlinear_register_type=='elastix' :
                    results.append( 
                        elastix_registration.remote(
                            i,
                            local_model, 
                            final_transforms[j],
                            symmetric=build_symmetric,
                            level=nlreg_level,
                            parameters=nlreg_options,
                            output_sample=warped_samples[j],
                            warp_seg=True,
                            resample_order=resample_order,
                            resample_baa=resample_baa,
                            nl=True,
                            downsample=nlreg_downsample
                        ) )
                elif nonlinear_register_type=='ants' or do_nonlinear_register_ants:
                    results.append( 
                        non_linear_registration.remote(
                            i,
                            local_model, 
                            final_transforms[j],
                            symmetric=build_symmetric,
                            level=nlreg_level,
                            parameters=nlreg_options,
                            output_sample=warped_samples[j],
                            warp_seg=True,
                            resample_order=resample_order,
                            resample_baa=resample_baa,
                            ants=True,
                            downsample=nlreg_downsample
                        ) )
                else:
                    results.append( 
                        non_linear_registration.remote(
                            i,
                            local_model, 
                            final_transforms[j],
                            symmetric=build_symmetric,
                            level=nlreg_level,
                            parameters=nlreg_options,
                            output_sample=warped_samples[j],
                            warp_seg=True,
                            resample_order=resample_order,
                            resample_baa=resample_baa,
                            ants=False,
                            downsample=nlreg_downsample
                        ) )

                final_samples[j].mask=None
            # TODO: do we really need to wait for result here?
            ray.wait(results, num_returns=len(results))

            with mincTools() as m:
                # a hack, to replace a rough model with a new one
                if os.path.exists(local_model.seg):
                    os.unlink(local_model.seg)

                # create majority voted model segmentation, for ANIMAL segmentation if needed
                segs=['multiple_volume_similarity']

                segs.extend([ i.seg for i in warped_samples ])
                if build_symmetric: segs.extend([ i.seg_f for i in warped_samples ])

                segs.extend(['--majority', local_model.seg, '--bg', '--overlap', local_model_ovl.scan ] )
                m.command(segs,inputs=[],outputs=[local_model.seg,local_model_ovl.scan])

                avg=['mincaverage','-float']
                avg.extend([ i.scan for i in warped_samples ])
                if build_symmetric: avg.extend([ i.scan_f for i in warped_samples ])
                avg.extend([local_model_avg.scan, '-sdfile', local_model_sd.scan ] )
                m.command(avg,inputs=[],outputs=[local_model_avg.scan,local_model_sd.scan])

                for i in range(modalities):
                    avg=['mincaverage','-float']
                    avg.extend([ j.add[i] for j in warped_samples ])
                    if build_symmetric: avg.extend([ j.add_f[i] for j in warped_samples ])

                    avg.extend([local_model_avg.add[i], '-sdfile', local_model_sd.add[i] ] )
                    m.command(avg,inputs=[],outputs=[local_model_avg.add[i],local_model_sd.add[i]])
        else:
            with mincTools() as m:
                # a hack, to replace a rough model with a new one
                if os.path.exists(local_model.seg):
                    os.unlink(local_model.seg)

                # create majority voted model segmentation, for ANIMAL segmentation if needed
                segs=['multiple_volume_similarity']
                segs.extend([ i.seg for i in final_samples ])

                if build_symmetric: segs.extend([ i.seg_f for i in final_samples ])

                segs.extend(['--majority', local_model.seg, '--bg','--overlap', local_model_ovl.scan] )
                m.command(segs,inputs=[],outputs=[local_model.seg,local_model_ovl.scan])
                
                avg=['mincaverage','-float']
                avg.extend([ i.scan for i in final_samples ])
                if build_symmetric: avg.extend([ i.scan_f for i in final_samples ])
                avg.extend([local_model_avg.scan, '-sdfile', local_model_sd.scan ] )
                m.command(avg,inputs=[],outputs=[local_model_avg.scan,local_model_sd.scan])

                for i in range(modalities):
                    avg=['mincaverage','-float']
                    avg.extend([ j.add[i] for j in final_samples ])
                    if build_symmetric: avg.extend([ j.add_f[i] for j in final_samples ])
                    avg.extend([local_model_avg.add[i], '-sdfile', local_model_sd.add[i] ] )
                    m.command(avg,inputs=[],outputs=[local_model_avg.add[i],local_model_sd.add[i]])
        
        # number of classes including bg
        classes_number=2
        # 6. create training library description
        with mincTools() as m:
            classes_number=int(m.execute_w_output(['mincstats', '-q', '-max',local_model.seg ]).rstrip("\n"))+1

        library_description={}
        # library models
        library_description['model']          = model.scan
        library_description['model_mask']     = model.mask
        library_description['model_add']      = model.add

        library_description['local_model']    = local_model.scan
        library_description['local_model_add']= local_model.add
        library_description['local_model_mask']=local_model.mask
        library_description['local_model_seg']= local_model.seg

        # library parameters
        library_description['map']=inv_dict(dict(build_remap))
        library_description['classes_number']=  classes_number
        library_description['nl_samples_avail']=do_nonlinear_register
        library_description['modalities']=modalities+1
        library_description['groups']=train_groups
        library_description['label_map'] = label_map

        if build_symmetric and build_symmetric_flip:
            library_description['local_model_flip']     =local_model.scan_f
            library_description['local_model_add_flip'] =local_model.add_f
            library_description['local_model_mask_flip']=local_model.mask_f
            library_description['local_model_seg_flip'] =local_model.seg_f
            library_description['flip_map']=inv_dict(dict(build_flip_remap))
        else:
            library_description['local_model_flip']=None
            library_description['local_model_add_flip']=[]
            library_description['local_model_mask_flip']=None
            library_description['flip_map']={}
        
        library_description['library']=[]
        
        for (j, i) in enumerate(final_samples):
            ss=[i.group,i.grading]
            ss.extend([i.scan, i.seg ])
            ss.extend(i.add)
            
            if do_nonlinear_register:
                ss.extend( [ final_transforms[j].xfm, final_transforms[j].xfm_inv, warped_samples[j].scan, warped_samples[j].seg  ])

            library_description['library'].append(ss)
            
            if build_symmetric:
                ss=[i.group,i.grading]
                ss.extend([i.scan_f, i.seg_f ])
                ss.extend(i.add_f)
                
                if do_nonlinear_register:
                    ss.extend( [ final_transforms[j].xfm_f, final_transforms[j].xfm_f_inv, warped_samples[j].scan_f, warped_samples[j].seg_f ])

                library_description['library'].append(ss)
        
        save_library_info( library_description, output)
        # cleanup
        if cleanup:
            shutil.rmtree(work_dir)
        
    except mincError as e:
        print("Exception in generate_library:{}".format(str(e)),file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise
    except :
        print("Exception in generate_library:{}".format(sys.exc_info()[0]),file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise    

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 14/08/2015
#
# Longitudinal pipeline preprocessing

import shutil
import os
import sys
import csv
import traceback

# MINC stuff
from ipl.minc_tools import mincTools,mincError

# distributed
import ray


# local stuff
from .structures   import *
from .preprocess   import *
from .utils        import *
from .registration import *
from .resample     import *
from .segment      import *
from .qc           import *

@ray.remote
def iter_step(  t1w_scan, 
                iteration,
                output_dir,
                prev_iter={},
                options =None,
                t2w_scan=None,
                pdw_scan=None,
                work_dir=None,
                subject_id=None,
                timepoint_id=None,
                corr_t1w=None,
                corr_t2w=None,
                corr_pdw=None ):
    """
    drop-in replacement for the standard pipeline

    Argumets: t1w_scan   -- `MriScan` for T1w scan
              iteration  -- iteration number 
              output_dir -- string pointing to output directory
              
    Kyword arguments:
            prev_iter -- information from previous iteration
            options   -- pipeline optins (dict)
            t2w_scan  -- T2w scan
            pdw_scan  -- PDw scan
            work_dir  -- string pointing to work directory , default None - use output_dir
            subject_id -- ID of subject
            timepoint_id -- ID of timepoint
    """
    try:
        print("running iter_step options={}".format(repr(options)))

        if options is None:
            # TODO: load defaults from a settings file?
            if iteration >0 :
                options= {
                'model':     'mni_icbm152_t1_tal_nlin_sym_09c',
                'model_dir': '/opt/minc/share/icbm152_model_09c',
                't1w_nuc':   {},
                't2w_nuc':   {},
                'pdw_nuc':   {},
                't1w_stx':   {
                    'type':'ants',
                    'resample':False,
                    #'options': {
                        #'levels': 3,
                        #'conf':  {'3':1000,'2':1000,'1':1000}, 
                        #'blur':  {'3':8,   '2':4,   '1': 2 }, 
                        #'shrink':{'3':8,   '2':4,   '1': 2 },
                        #'convergence':'1.e-8,20',
                        #'cost_function':'MI',
                        #'cost_function_par':'1,32,random,0.3',
                        #'transformation':'similarity[ 0.3 ]',
                        #}
                    },
                'stx':       {
                        'noscale':False,
                    },
                'beast':     { 'beastlib':  '/opt/minc/share/beast-library-1.1' },
                'tissue_classify': {},
                'lobe_segment': {},
                'nl':        True,
                'lobes':     True,
                'cls'  :     True,
                'qc':        True,
                'denoise':   {},
                
                }
            else:
                options= {
                'model':     'mni_icbm152_t1_tal_nlin_sym_09c',
                'model_dir': '/opt/minc/share/icbm152_model_09c',
                't1w_nuc':   {},
                't2w_nuc':   {},
                'pdw_nuc':   {},
                't1w_stx':   {
                    'type':'ants',
                    'resample':False,
                    #'options': {
                        #'levels': 2,
                        #'conf':  {'2':1000,'1':1000}, 
                        #'blur':  {'2':4, '1': 2 }, 
                        #'shrink':{'2':4, '1': 2 },
                        #'convergence':'1.e-8,20',
                        #'cost_function':'MI',
                        #'cost_function_par':'1,32,random,0.3',
                        #'transformation':'similarity[ 0.3 ]',
                        #}
                    },
                'stx':       {
                        'noscale':False,
                    },
                'beast':     { 'beastlib':  '/opt/minc/share/beast-library-1.1' },
                'tissue_classify': {},
                'lobe_segment': {},
                'nl':        True,
                'lobes':     True,
                'cls'  :     True,
                'qc':        True,
                'denoise':   {},
                
                }
        
        dataset_id=subject_id
        
        if dataset_id is None:
            dataset_id=t1w_scan.name
            
        if timepoint_id is not None:
            dataset_id=dataset_id+'_'+timepoint_id
        
        # generate model reference
        model_dir =options['model_dir']
        model_name=options['model']

        model_t1w=MriScan(scan=model_dir+os.sep+options['model']+'.mnc',
                        mask=model_dir+os.sep+options['model']+'_mask.mnc')

        model_outline=MriScan(scan=model_dir+os.sep+options['model']+'_outline.mnc',
                            mask=None)

        lobe_atlas_dir=options.get('lobe_atlas_dir',None)
        lobe_atlas_defs=options.get('lobe_atlas_defs',None)

        if lobe_atlas_dir is None:
            lobe_atlas_dir=model_dir + os.sep + model_name + '_atlas'+os.sep

        if lobe_atlas_defs is None:
            lobe_atlas_defs=model_dir + os.sep + model_name + '_atlas'+os.sep+'lobe_defs.csv'
            if not os.path.exists(lobe_atlas_defs):
                lobe_atlas_defs=None

        if work_dir is None:
            work_dir=output_dir+os.sep+'work_'+dataset_id
        
        run_qc=options.get('qc',True)
        run_nl=options.get('nl',True)
        run_cls=options.get('cls',True)
        run_lobes=options.get('lobes',True)
        denoise_parameters=options.get('denoise',None)
        create_unscaled=options.get('stx',{}).get('noscale',False)
        
        clp_dir=work_dir+os.sep+'clp'
        tal_dir=work_dir+os.sep+'tal'
        nl_dir =work_dir+os.sep+'nl'
        cls_dir=work_dir+os.sep+'tal_cls'
        qc_dir =work_dir+os.sep+'qc'
        lob_dif=work_dir+os.sep+'lob'
        vol_dir=work_dir+os.sep+'vol'
        
        # create all
        create_dirs([clp_dir,tal_dir,nl_dir,cls_dir,qc_dir,lob_dif,vol_dir])
        
        # files produced by pipeline
        # native space
        t1w_den=MriScan(prefix=clp_dir,  name='den_'+dataset_id,   modality='t1w', mask=None, iter=iteration)
        t1w_field=MriScan(prefix=clp_dir,name='fld_'+dataset_id,   modality='t1w', mask=None, iter=iteration)
        t1w_nuc=MriScan(prefix=clp_dir,  name='n4_' +dataset_id,   modality='t1w', mask=None, iter=iteration)
        t1w_clp=MriScan(prefix=clp_dir,  name='clamp_'+dataset_id, modality='t1w', mask=None, iter=iteration)
        # warp cls and mask back into native space
        native_t1w_cls=MriScan(prefix=clp_dir,  name='cls_'+dataset_id, modality='t1w', iter=iteration)
        # stereotaxic space
        t1w_tal_xfm=MriTransform(prefix=tal_dir,name='tal_xfm_'+dataset_id, iter=iteration)
        t1w_tal_noscale_xfm=MriTransform(prefix=tal_dir,name='tal_noscale_xfm_'+dataset_id, iter=iteration)
        unscale_xfm=MriTransform(prefix=tal_dir,name='unscale_xfm_'+dataset_id, iter=iteration)
        
        t1w_tal=MriScan(prefix=tal_dir, name='tal_'+dataset_id, modality='t1w', iter=iteration)
        prev_t1w_xfm=None
        t1w_tal_noscale=MriScan(prefix=tal_dir, name='tal_noscale_'+dataset_id,modality='t1w', iter=iteration)

        # tissue classification results
        tal_cls=MriScan(prefix=cls_dir, name='cls_'+dataset_id, iter=iteration)
        # lobe segmentation results
        tal_lob=MriScan(prefix=lob_dif, name='lob_'+dataset_id, iter=iteration)

        # nl space
        nl_xfm= MriTransform(prefix=nl_dir, name='nl_'+dataset_id, iter=iteration)

        # QC files
        qc_tal= MriQCImage(prefix=qc_dir,name='tal_t1w_'+dataset_id, iter=iteration)
        qc_mask=MriQCImage(prefix=qc_dir,name='tal_mask_'+dataset_id,iter=iteration)
        qc_cls= MriQCImage(prefix=qc_dir,name='tal_cls_'+dataset_id, iter=iteration)
        qc_lob= MriQCImage(prefix=qc_dir,name='tal_lob_'+dataset_id, iter=iteration)
        qc_nu=  MriQCImage(prefix=qc_dir,name='nu_'+dataset_id,      iter=iteration)
        
        # AUX files
        lob_volumes=MriAux(prefix=vol_dir,name='vol_'+dataset_id, iter=iteration)
        lob_volumes_json=MriAux(prefix=vol_dir,name='vol_'+dataset_id,suffix='.json', iter=iteration)
        summary_file=MriAux(prefix=work_dir,name='summary_'+dataset_id,suffix='.json', iter=iteration)
        
        print("Iteration step dataset:{} iteration:{}".format(dataset_id,iteration))
        
        # actual processing steps
        # 1. preprocessing
        if prev_iter is not None:
            t1w_scan.mask=prev_iter['native_t1w_cls'].mask
            t1w_den.mask =prev_iter['native_t1w_cls'].mask
            t1w_nuc.mask =prev_iter['native_t1w_cls'].mask
            t1w_clp.mask =prev_iter['native_t1w_cls'].mask
            prev_t1w_xfm =prev_iter['t1w_tal_xfm']
            print("Previous iteration:")
            print(repr(prev_iter))

        iter_summary={
                    'iter':         iteration,
                    'input_t1w':    t1w_scan,
                    'output_dir':   output_dir,
                    'dataset_id':   dataset_id,
                    "t1w_field":    t1w_field,
                    "t1w_nuc":      t1w_nuc,
                    "t1w_clp":      t1w_clp,
                    "t1w_tal_xfm":  t1w_tal_xfm,
                    "t1w_tal_noscale_xfm":t1w_tal_noscale_xfm,
                    "t1w_tal":      t1w_tal,
                    "t1w_tal_noscale":t1w_tal_noscale,
                    
                    "corr_t1w": corr_t1w,
                    "corr_t2w": corr_t2w,
                    "corr_pdw": corr_pdw,
                    }


        if denoise_parameters is not None:
            # reuse old denoising
            if prev_iter is not None :
                t1w_den=prev_iter.get('t1w_den',None)
                t1w_den.mask=prev_iter['native_t1w_cls'].mask
            else:
                denoise(t1w_scan, t1w_den, parameters=denoise_parameters)
            
            iter_summary["t1w_den"]=t1w_den

            # non-uniformity correction
            estimate_nu(t1w_den, t1w_field,
                        parameters=options.get('t1w_nuc',{}))
            if run_qc: 
                draw_qc_nu(t1w_field,qc_nu)
                iter_summary["qc_nu"]=qc_nu

            # apply field
            apply_nu(t1w_den, t1w_field, t1w_nuc,
                    parameters=options.get('t1w_nuc',{}))
        else:
            # non-uniformity correction
            estimate_nu(t1w_scan, t1w_field,
                        parameters=options.get('t1w_nuc',{}))

            if run_qc: 
                draw_qc_nu(t1w_field,qc_nu)
                iter_summary["qc_nu"]=qc_nu

            # apply field
            apply_nu(t1w_scan, t1w_field, t1w_nuc,
                    parameters=options.get('t1w_nuc',{}))
        
        # normalize intensity
        normalize_intensity(t1w_nuc, t1w_clp,
                            parameters=options.get('t1w_clp',{}),
                            model=model_t1w)
        # TODO coregister other modalities here?
        
        # register to STX space
        lin_registration(t1w_clp, model_t1w, t1w_tal_xfm, 
                        parameters=options.get('t1w_stx',{}),
                        init_xfm=prev_t1w_xfm,
                        corr_xfm=corr_t1w)

        warp_scan(t1w_clp,model_t1w,t1w_tal,transform=t1w_tal_xfm,corr_xfm=corr_t1w)

        if run_qc: 
            draw_qc_stx(t1w_tal,model_outline,qc_tal)
            iter_summary["qc_tal"]=qc_tal
        
        # run beast to create brain mask
        extract_brain_beast(t1w_tal,parameters=options.get('beast'),model=model_t1w)
        if run_qc: 
            draw_qc_mask(t1w_tal,qc_mask)
            iter_summary["qc_mask"]=qc_mask
        
        # create unscaled version
        if create_unscaled:
            xfm_remove_scale(t1w_tal_xfm, t1w_tal_noscale_xfm, unscale=unscale_xfm)
            iter_summary["t1w_tal_noscale_xfm"]=t1w_tal_noscale_xfm
            #warp scan to create unscaled version
            warp_scan(t1w_clp,model_t1w,t1w_tal_noscale,transform=t1w_tal_noscale_xfm,corr_xfm=corr_t1w)
            # warping mask from tal space to unscaled tal space
            warp_mask(t1w_tal, model_t1w, t1w_tal_noscale, transform=unscale_xfm)
            iter_summary["t1w_tal_noscale"]=t1w_tal_noscale
        
        # perform non-linear registration
        if run_nl: 
            nl_registration(t1w_tal, model_t1w, nl_xfm, 
                        parameters=options.get('nl_reg',{}))
            iter_summary["nl_xfm"]=nl_xfm

        # run tissue classification
        if run_nl and run_cls: 
            classify_tissue(t1w_tal, tal_cls, model_name=model_name, 
                        model_dir=model_dir, xfm=nl_xfm,
                        parameters=options.get('tissue_classify',{}))
            iter_summary["tal_cls"]=tal_cls
            if run_qc: 
                draw_qc_cls(t1w_tal,tal_cls,qc_cls)
                iter_summary["qc_cls"]=qc_cls
            
            warp_cls_back(t1w_tal, tal_cls, t1w_tal_xfm, t1w_nuc, native_t1w_cls,corr_xfm=corr_t1w)
            iter_summary["native_t1w_cls"]=native_t1w_cls
            
        # run lobe segmentation
        if run_nl and run_cls and run_lobes: 
            segment_lobes( tal_cls, nl_xfm, tal_lob, 
                    model=model_t1w, 
                    lobe_atlas_dir=lobe_atlas_dir, 
                    parameters=options.get('lobe_segment',{}))
            
            iter_summary["tal_lob"]=tal_lob
            if run_qc: 
                draw_qc_lobes( t1w_tal, tal_lob,qc_lob)
                iter_summary["qc_lob"]=qc_lob

            # calculate volumes
            extract_volumes(tal_lob, tal_cls, t1w_tal_xfm, lob_volumes, 
                            subject_id=subject_id, timepoint_id=timepoint_id , lobedefs=lobe_atlas_defs)
        
            extract_volumes(tal_lob, tal_cls, t1w_tal_xfm, lob_volumes_json, 
                            produce_json=True,subject_id=subject_id, timepoint_id=timepoint_id,lobedefs=lobe_atlas_defs)
            
            iter_summary["lob_volumes"]=     lob_volumes
            iter_summary["lob_volumes_json"]=lob_volumes_json
        
        save_summary(iter_summary,summary_file.fname)
        return iter_summary
    
    except mincError as e:
        print("Exception in iter_step:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in iter_step:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

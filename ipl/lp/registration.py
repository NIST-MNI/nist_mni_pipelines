# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 14/08/2015
#
# Longitudinal pipeline registration

import shutil
import os
import sys
import csv
import traceback

# MINC stuff
from ipl.minc_tools import mincTools,mincError

import ipl.registration
import ipl.ants_registration
import ipl.elastix_registration


def lin_registration(scan, model, out_xfm, init_xfm=None, parameters={}, corr_xfm=None, par=None, log=None, ref_model=None):
    """Perform linear registration
    
    """
    with mincTools() as m:
        if ref_model is None:
            ref_model = model
            
        if not m.checkfiles(inputs=[scan.scan, model.scan],outputs=[out_xfm.xfm]):
            return
        
        use_inverse    = parameters.get('inverse',  False)
        lin_mode       = parameters.get('type',     'ants')
        options        = parameters.get('options',   None)
        downsample     = parameters.get('downsample',None)
        close          = parameters.get('close',     False)
        resample       = parameters.get('resample',  False)
        objective      = parameters.get('objective','-xcorr')
        use_model_mask = parameters.get('use_model_mask',False)
        
        print("Running lin_registration with parameters:{}".format(repr(parameters)))
        
        _init_xfm=None
        _in_scan       = scan.scan
        _in_mask       = scan.mask
        
        _in_model      = model.scan
        _in_model_mask = model.mask
        _out_xfm       = out_xfm.xfm
        
        if init_xfm is not None:
            _init_xfm=init_xfm.xfm

        if corr_xfm is not None:
            # apply distortion correction before linear registration, 
            # but then don't include it it into linear XFM
            _in_scan=m.tmp('corr_scan.mnc')
            m.resample_smooth(scan.scan,_in_scan, transform=corr_xfm.xfm)
            if scan.mask is not None:
                _in_mask=m.tmp('corr_scan_mask.mnc')
                m.resample_labels(scan.mask,_in_mask, transform=corr_xfm.xfm, like=_in_scan)

        if init_xfm is not None and resample:
            #_init_xfm=init_xfm.xfm
            _init_xfm=None
            _out_xfm=m.tmp('out.xfm')
            m.resample_smooth(_in_scan, m.tmp('scan_scan.mnc'), transform=init_xfm.xfm, like=ref_model.scan)
            _in_scan=m.tmp('scan_scan.mnc')
            if scan.mask is not None:
                m.resample_labels(scan.mask, _in_mask, transform=init_xfm.xfm, like=ref_model.scan)
                _in_mask=m.tmp('scan_mask.mnc')

        print("lin_registration: mode={} init_xfm={} scan_mask={} use_inverse={}".format(lin_mode,_init_xfm,scan.mask,use_inverse))
        
        _model_mask=None
        
        # use model mask even if scan mask is unspecified!
        # to run experminets mostly
        if use_model_mask or _in_mask is not None:
            _model_mask=model.mask
        
        _save_out_xfm=_out_xfm
        if use_inverse:
            _save_out_xfm=_out_xfm
            _out_xfm=m.tmp('inverted_out.xfm')
            
            save_in_scan=_in_scan
            save_in_mask=_in_mask
            
            _in_scan=_in_model
            _in_mask=_in_model_mask
            
            _in_model=save_in_scan
            _in_model_mask=save_in_mask
            
        
        if lin_mode=='ants':
            ipl.ants_registration.linear_register_ants2(
                _in_scan,
                _in_model,
                _out_xfm,
                source_mask=_in_mask,
                target_mask=_model_mask,
                init_xfm=_init_xfm,
                parameters=options,
                close=close,
                downsample=downsample,
                )
        elif lin_mode=='elx':
            output_par=None
            output_log=None
            
            if par is not None:
                output_par=par.fname
                
            if log is not None:
                output_log=log.fname
                
            ipl.elastix_registration.register_elastix(
                _in_scan,
                _in_model,
                output_xfm=_out_xfm,
                source_mask=_in_mask,
                target_mask=_model_mask,
                init_xfm=_init_xfm,
                downsample=downsample,
                parameters=options,
                nl=False,
                output_log=output_log,
                output_par=output_par
                )
        elif lin_mode=='mritotal':
            # going to use mritotal directly
            #m.command()
            model_name = os.path.basename(model.scan).rsplit('.mnc',1)[0]
            model_dir  = os.path.dirname(model.scan)
            # TODO: add more options?
            cmd=['mritotal','-model', model_name,'-modeldir',model_dir, _in_scan, _out_xfm]
            if options is not None:
                cmd.extend(options)
                
            m.command(cmd, 
                      inputs=[_in_scan],
                      outputs=[_out_xfm])
        else:
            ipl.registration.linear_register(
                _in_scan,
                _in_model,
                _out_xfm,
                source_mask=_in_mask,
                target_mask=_model_mask,
                init_xfm=_init_xfm,
                objective=objective,
                downsample=downsample,
                conf=options,
                parameters=lin_mode
                )
        
        if use_inverse: # need to invert transform 
            m.xfminvert(_out_xfm,_save_out_xfm)
            _out_xfm=_save_out_xfm
        
        if init_xfm is not None and resample:
            m.xfmconcat([init_xfm.xfm,_out_xfm],out_xfm.xfm)
            
            
def intermodality_co_registration(scan, ref, out_xfm, 
                                  init_xfm=None, 
                                  parameters={}, 
                                  corr_xfm=None, 
                                  corr_ref=None,
                                  par=None, 
                                  log=None):
    with mincTools() as m:
        
        if not m.checkfiles(inputs=[scan.scan, ref.scan],outputs=[out_xfm.xfm]):
            return
        
        lin_mode  =parameters.get('type',      'ants')
        options   =parameters.get('options',    None)
        downsample=parameters.get('downsample', None)
        close     =parameters.get('close',      True)
        resample  =parameters.get('resample',   False)
        objective =parameters.get('objective',  '-nmi')
        nl        =parameters.get('nl',         False)
        
        print("Running intermodality_co_registration with parameters:{}".format(repr(parameters)))
        
        _init_xfm=None
        _in_scan=scan.scan
        _in_mask=scan.mask
        _out_xfm=out_xfm.xfm
        
        if init_xfm is not None:
            _init_xfm=init_xfm.xfm

        if corr_xfm is not None:
            # apply distortion correction before linear registration, 
            # but then don't include it it into linear XFM
            _in_scan=m.tmp('corr_scan.mnc')
            m.resample_smooth(scan.scan,_in_scan, transform=corr_xfm.xfm)
            if scan.mask is not None:
                _in_mask=m.tmp('corr_scan_mask.mnc')
                m.resample_labels(scan.mask,_in_mask, transform=corr_xfm.xfm, like=_in_scan)

        if init_xfm is not None and resample:
            #_init_xfm=init_xfm.xfm
            _init_xfm=None
            _out_xfm=m.tmp('out.xfm')
            m.resample_smooth(_in_scan,m.tmp('scan_scan.mnc'), transform=init_xfm.xfm, like=ref.scan)
            _in_scan=m.tmp('scan_scan.mnc')
            if scan.mask is not None:
                m.resample_labels(scan.mask,_in_mask, transform=init_xfm.xfm, like=ref.scan)
                _in_mask=m.tmp('scan_mask.mnc')

        print("intermodality_co_registration: mode={} init_xfm={} scan_mask={}".format(lin_mode, _init_xfm, scan.mask))
        
        if lin_mode=='ants':
            ipl.ants_registration.linear_register_ants2(
                _in_scan,
                ref.scan,
                _out_xfm,
                source_mask=_in_mask,
                target_mask=ref.mask,
                init_xfm=_init_xfm,
                parameters=options,
                close=close,
                downsample=downsample,
                )
        elif lin_mode=='elx':
            output_par=None
            output_log=None
            
            if par is not None:
                output_par=par.fname
                
            if log is not None:
                output_log=log.fname
                
            ipl.elastix_registration.register_elastix(
                _in_scan,
                ref.scan,
                output_xfm=_out_xfm,
                source_mask=_in_mask,
                target_mask=ref.mask,
                init_xfm=_init_xfm,
                downsample=downsample,
                parameters=options,
                nl=nl,
                output_log=output_log,
                output_par=output_par
                )
        else:
            ipl.registration.linear_register(
                _in_scan,
                ref.scan,
                _out_xfm,
                source_mask=_in_mask,
                target_mask=ref.mask,
                init_xfm=_init_xfm,
                objective=objective,
                downsample=downsample,
                conf=options,
                parameters=lin_mode,
                close=close
                )
        
        if init_xfm is not None and resample:
            m.xfmconcat([init_xfm.xfm,_out_xfm],out_xfm.xfm)
    
    
def nl_registration(scan, model, out_xfm, init_xfm=None, parameters={}):
    """Perform non-linear registration
    
    """
    nl_mode     =parameters.get('type','ants')
    options     =parameters.get('options',None)
    downsample  =parameters.get('downsample',None)
    level       =parameters.get('level',2)
    start       =parameters.get('start_level',32)

    with mincTools() as m:
        
        if not m.checkfiles(inputs=[scan.scan,model.scan],outputs=[out_xfm.xfm]):
            return
    
        _init_xfm=None
        if init_xfm is not None:
            _init_xfm=init_xfm.xfm
        
        if nl_mode=='ants':
            ipl.ants_registration.non_linear_register_ants2(
                scan.scan,
                model.scan,
                out_xfm.xfm,
                source_mask=scan.mask,
                target_mask=model.mask,
                init_xfm=_init_xfm,
                parameters=options,
                downsample=downsample,
                level=level,
                start=start,
                )
        elif nl_mode=='elx':
            ipl.elastix_registration.register_elastix(
                scan.scan,
                model.scan,
                output_xfm=out_xfm.xfm,
                source_mask=scan.mask,
                target_mask=model.mask,
                init_xfm=_init_xfm,
                downsample=downsample,
                parameters=options,
                nl=True
                )
        else:
            objective='-xcorr'
            if options is not None:
                objective=options.get('objective')
                
            ipl.registration.non_linear_register_full(
                scan.scan,
                model.scan,
                out_xfm.xfm,
                source_mask=scan.mask,
                target_mask=model.mask,
                init_xfm=_init_xfm,
                downsample=downsample,
                parameters=options,
                level=level,
                start=start,
                )

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

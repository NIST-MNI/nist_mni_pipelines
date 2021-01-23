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
import ipl.registration
import ipl.ants_registration
import ipl.elastix_registration

def linear_registration(
    sample,
    model,
    output_xfm,
    output_sample=None,
    output_invert_xfm=None,
    init_xfm=None,
    symmetric=False,
    ants=False,
    reg_type ='-lsq12',
    objective='-xcorr',
    linreg=None,
    work_dir=None,
    close=False,
    warp_seg=False,
    resample_order=2,
    resample_aa=None,
    resample_baa=False,
    downsample=None,
    bbox=False,
    use_mask=True
    ):
    """perform linear registration to the model, and calculate inverse"""
    try:
        _init_xfm=None
        _init_xfm_f=None

        if init_xfm is not None:
            _init_xfm=init_xfm.xfm
            if symmetric:
                _init_xfm_f=init_xfm.xfm_f
        print("Linear registration:{} obj:{} ants:{} symmetric:{}".format(reg_type,objective,ants,symmetric))
        print("\toptions: {}".format(repr(linreg)))
        print("\tsample  :{}".format(repr(sample)))
        print("\tmodel   :{}".format(repr(model)))

        with mincTools() as m:
            
            #TODO: check more files?
            if not m.checkfiles(inputs=[sample.scan], outputs=[output_xfm.xfm]):  return
            
            #if _init_xfm is None:
            #    _init_xfm=_init_xfm_f=m.tmp('identity.xfm')
            #    m.param2xfm(m.tmp('identity.xfm'))

            scan=sample.scan
            scan_f=sample.scan_f
            
            mask=sample.mask
            mask_f=sample.mask_f
            
            model_mask=model.mask
            model_mask_f=model.mask
            
            if mask   is None:  model_mask=None
            if mask_f is None:  model_mask_f=None

            if not use_mask:
                mask=None
                model_mask=None
                mask_f=None
                model_mask_f=None
            
            _output_xfm  =output_xfm.xfm
            _output_xfm_f=output_xfm.xfm_f

            if bbox:
                print("Running in bbox! _init_xfm={} _init_xfm_f={}\n\n\n".format(_init_xfm,_init_xfm_f))
                scan=m.tmp('scan.mnc')
                m.resample_smooth(sample.scan, scan, like=model.scan, transform=_init_xfm)
                if sample.mask is not None and use_mask:
                    mask=m.tmp('mask.mnc')
                    m.resample_labels(sample.mask, mask, like=model.scan, transform=_init_xfm)
                _init_xfm=None
                close=True
                _output_xfm=m.tmp('output.xfm')

                if symmetric:
                    scan_f=m.tmp('scan_f.mnc')
                    m.resample_smooth(sample.scan_f, scan_f, like=model.scan, transform=_init_xfm_f)
                    if sample.mask_f is not None and  use_mask:
                        mask_f=m.tmp('mask_f.mnc')
                        m.resample_labels(sample.mask_f, mask_f, like=model.scan, transform=_init_xfm_f)
                    _init_xfm_f=None
                    _output_xfm_f=m.tmp('output_f.xfm')

                #os.system('cp -v {} {} {} {} ./'.format(scan,mask,scan_f,mask_f))

            if symmetric:
                if ants:
                    ipl.ants_registration.linear_register_ants2(
                        scan,
                        model.scan,
                        _output_xfm,
                        source_mask=mask,
                        target_mask=model_mask,
                        init_xfm=_init_xfm,
                        parameters=linreg,
                        close=close,
                        downsample=downsample,
                        verbose=0
                        )
                    ipl.ants_registration.linear_register_ants2(
                        scan_f,
                        model.scan,
                        _output_xfm_f,
                        source_mask=mask_f,
                        target_mask=model_mask_f,
                        init_xfm=_init_xfm_f,
                        parameters=linreg,
                        close=close,
                        downsample=downsample,
                        verbose=0
                        )
                else:
                    ipl.registration.linear_register(
                        scan,
                        model.scan,
                        _output_xfm,
                        source_mask=mask,
                        target_mask=model_mask,
                        init_xfm=_init_xfm,
                        objective=objective,
                        parameters=reg_type,
                        conf=linreg,
                        close=close,
                        downsample=downsample,
                        )
                        
                    ipl.registration.linear_register(
                        scan_f,
                        model.scan,
                        _output_xfm_f,
                        source_mask=mask_f,
                        target_mask=model_mask_f,
                        init_xfm=_init_xfm_f,
                        objective=objective,
                        parameters=reg_type,
                        conf=linreg,
                        close=close,
                        downsample=downsample,
                        )
            else:
                if ants:
                    ipl.ants_registration.linear_register_ants2(
                        scan,
                        model.scan,
                        _output_xfm,
                        source_mask=mask,
                        target_mask=model_mask,
                        init_xfm=_init_xfm,
                        parameters=linreg,
                        close=close,
                        downsample=downsample,
                        )
                else:
                    ipl.registration.linear_register(
                        scan,
                        model.scan,
                        _output_xfm,
                        source_mask=mask,
                        target_mask=model_mask,
                        init_xfm=_init_xfm,
                        parameters=reg_type,
                        conf=linreg,
                        close=close,
                        downsample=downsample,
                        )
            if bbox :
                if init_xfm is not None:
                    m.xfmconcat([init_xfm.xfm,_output_xfm],output_xfm.xfm)
                    if symmetric:
                        m.xfmconcat([init_xfm.xfm_f,_output_xfm_f],output_xfm.xfm_f)
                else:
                    shutil.copyfile(_output_xfm,output_xfm.xfm)
                    if symmetric:
                        shutil.copyfile(_output_xfm_f,output_xfm.xfm_f)

            if output_invert_xfm is not None:
                m.xfminvert(output_xfm.xfm, output_invert_xfm.xfm)
                if symmetric:
                    m.xfminvert(output_xfm.xfm_f, output_invert_xfm.xfm_f)

            if output_sample is not None: 
                m.resample_smooth(sample.scan, output_sample.scan, 
                                  transform=output_xfm.xfm, 
                                  like=model.scan, 
                                  order=resample_order)
                if warp_seg:
                    m.resample_labels(sample.seg,  output_sample.seg,  
                                      transform=output_xfm.xfm, 
                                      aa=resample_aa, 
                                      order=resample_order, 
                                      like=model.scan, 
                                      baa=resample_baa)

                if symmetric:
                    m.resample_smooth(sample.scan_f, output_sample.scan_f, 
                                      transform=output_xfm.xfm_f, 
                                      like=model.scan, 
                                      order=resample_order)
                    if warp_seg:
                        m.resample_labels(sample.seg_f,  output_sample.seg_f,  
                                          transform=output_xfm.xfm_f, 
                                          aa=resample_aa, 
                                          order=resample_order, 
                                          like=model.scan, 
                                          baa=resample_baa)

        return True
    except mincError as e:
        print("Exception in linear_registration:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in linear_registration:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise


def elastix_registration(
    sample,
    model,
    output_xfm,
    output_sample=None,
    output_invert=True,
    init_xfm=None,
    symmetric=False,
    work_dir=None,
    warp_seg=False,
    resample_order=2,
    resample_aa=None,
    resample_baa=False,
    downsample=None,
    parameters=None,
    bbox=False,
    nl=False,
    level=2,
    start_level=None, # not really used
    use_mask=True
    ):
    """perform elastix registration to the model, and calculate inverse"""
    try:

        with mincTools() as m:
            
            #TODO: check more files?
            if not m.checkfiles(inputs=[sample.scan], outputs=[output_xfm.xfm]):  return 
        
            _init_xfm=None
            _init_xfm_f=None
            
            if init_xfm is not None:
                _init_xfm=init_xfm.xfm
                if symmetric:
                    _init_xfm_f=init_xfm.xfm_f
                    
            mask=sample.mask
            mask_f=sample.mask_f
            model_mask=model.mask
            
            if mask is None:
                model_mask=None

            if not use_mask:
                mask=None
                model_mask=None
                mask_f=None
                model_mask_f=None

            scan=sample.scan
            scan_f=sample.scan_f
            
            _output_xfm=output_xfm.xfm
            _output_xfm_f=output_xfm.xfm_f
            
            if bbox:
                scan=m.tmp('scan.mnc')
                m.resample_smooth(sample.scan, scan, like=model.scan, transform=_init_xfm)
                if sample.mask is not None and use_mask:
                    mask=m.tmp('mask.mnc')
                    m.resample_labels(sample.mask, mask, like=model.scan, transform=_init_xfm)
                _init_xfm=None
                close=True
                _output_xfm=m.tmp('output.xfm')
                
                if symmetric:
                    scan_f=m.tmp('scan_f.mnc')
                    m.resample_smooth(sample.scan_f, scan_f, like=model.scan, transform=_init_xfm_f)
                    if sample.mask_f is not None and use_mask:
                        mask_f=m.tmp('mask_f.mnc')
                        m.resample_labels(sample.mask_f, mask_f, like=model.scan, transform=_init_xfm_f)
                    _init_xfm_f=None
                    _output_xfm_f=m.tmp('output_f.xfm')
            
            #TODO: update elastix registration to downsample xfm?
            if symmetric:
                ipl.elastix_registration.register_elastix(
                    scan,
                    model.scan,
                    output_xfm=_output_xfm,
                    source_mask=mask,
                    target_mask=model_mask,
                    init_xfm=_init_xfm,
                    downsample=downsample,
                    downsample_grid=level,
                    parameters=parameters,
                    nl=nl
                    )
                ipl.elastix_registration.register_elastix(
                    scan_f,
                    model.scan,
                    output_xfm=_output_xfm_f,
                    source_mask=mask_f,
                    target_mask=model_mask,
                    init_xfm=_init_xfm_f,
                    downsample=downsample,
                    downsample_grid=level,
                    parameters=parameters,
                    nl=nl
                    )
            else:
                ipl.elastix_registration.register_elastix(
                    scan,
                    model.scan,
                    output_xfm=_output_xfm,
                    source_mask=mask,
                    target_mask=model_mask,
                    init_xfm=_init_xfm,
                    downsample=downsample,
                    downsample_grid=level,
                    parameters=parameters,
                    nl=nl
                    )
            
            if bbox :
                if init_xfm is not None:
                    m.xfmconcat([init_xfm.xfm,_output_xfm],output_xfm.xfm)
                    if symmetric:
                        m.xfmconcat([init_xfm.xfm_f,_output_xfm_f],output_xfm.xfm_f)
                else:
                    shutil.copyfile(_output_xfm,output_xfm.xfm)
                    if symmetric:
                        shutil.copyfile(_output_xfm_f,output_xfm.xfm_f)
                
            if output_invert:
                if nl:
                    m.xfm_normalize(output_xfm.xfm, model.scan, output_xfm.xfm_inv, step=level, invert=True)
                else:
                    m.xfminvert(output_xfm.xfm, output_xfm.xfm_inv)
                    
                if symmetric:
                    if nl:
                        m.xfm_normalize(output_xfm.xfm_f, model.scan, output_xfm.xfm_f_inv, step=level, invert=True)
                    else:
                        m.xfminvert(output_xfm.xfm_f, output_xfm.xfm_f_inv)

            if output_sample is not None: 
                m.resample_smooth(sample.scan, output_sample.scan, 
                                  transform=output_xfm.xfm, 
                                  like=model.scan, order=resample_order)
                if warp_seg:
                    m.resample_labels(sample.seg,  output_sample.seg,  
                                      transform=output_xfm.xfm, 
                                      aa=resample_aa, order=resample_order, 
                                      like=model.scan, baa=resample_baa)

                if symmetric:
                    m.resample_smooth(sample.scan_f, output_sample.scan_f, 
                                      transform=output_xfm.xfm_f, 
                                      like=model.scan, order=resample_order)
                    if warp_seg:
                        m.resample_labels(sample.seg_f,  output_sample.seg_f,  
                                          transform=output_xfm.xfm_f, 
                                          aa=resample_aa, order=resample_order, 
                                          like=model.scan, baa=resample_baa)

        return True
    except mincError as e:
        print("Exception in elastix_registration:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in elastix_registration:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise



def non_linear_registration(
    sample,
    model,
    output,
    output_sample=None,
    output_invert=True,
    init_xfm=None,
    level=2,
    start_level=8,
    symmetric=False,
    parameters=None,
    work_dir=None,
    ants=False,
    warp_seg=False,
    resample_order=2,
    resample_aa=None,
    resample_baa=False,
    output_inv_target=None,
    flip=False,
    downsample=None,
    ):
    """perform linear registration to the model, and calculate inverse"""

    try:
        _init_xfm=None
        _init_xfm_f=None
        
        if init_xfm is not None:
            _init_xfm=init_xfm.xfm
            if symmetric:
                _init_xfm_f=init_xfm.xfm_f


        with mincTools() as m:

            #TODO: check more files?
            if not m.checkfiles(inputs=[sample.scan], outputs=[output.xfm]):  return 
        
            
            if symmetric:
                # TODO: split up into two jobs?
                if not os.path.exists( output.xfm ) or \
                   not os.path.exists( output.xfm_f ) :
                    if ants:
                        ipl.ants_registration.non_linear_register_ants2(
                            sample.scan,
                            model.scan,
                            m.tmp('forward')+'.xfm',
                            target_mask=model.mask,
                            parameters=parameters,
                            downsample=downsample,
                            level=level,
                            start=start_level,
                            #work_dir=work_dir
                            )
                        ipl.ants_registration.non_linear_register_ants2(
                            sample.scan_f,
                            model.scan,
                            m.tmp('forward_f')+'.xfm',
                            target_mask=model.mask,
                            parameters=parameters,
                            downsample=downsample,
                            level=level,
                            start=start_level,
                            #work_dir=work_dir
                            )
                    else:
                        ipl.registration.non_linear_register_full(
                            sample.scan,
                            model.scan,
                            m.tmp('forward')+'.xfm',
                            #source_mask=sample.mask,
                            target_mask=model.mask,
                            init_xfm=_init_xfm,
                            parameters=parameters,
                            level=level,
                            start=start_level,
                            downsample=downsample,
                            #work_dir=work_dir
                            )
                        ipl.registration.non_linear_register_full(
                            sample.scan_f,
                            model.scan,
                            m.tmp('forward_f')+'.xfm',
                            #source_mask=sample.mask_f,
                            target_mask=model.mask,
                            init_xfm=_init_xfm_f,
                            parameters=parameters,
                            level=level,
                            start=start_level,
                            downsample=downsample,
                            #work_dir=work_dir
                            )
                    m.xfm_normalize(m.tmp('forward')+'.xfm',model.scan,output.xfm,step=level)
                    #TODO: regularize here
                    m.xfm_normalize(m.tmp('forward_f')+'.xfm',model.scan,output.xfm_f,step=level)
                    
                    if output_invert:
                        if ants:
                            m.xfm_normalize(m.tmp('forward')+'_inverse.xfm', model.scan, output.xfm_inv, step=level )
                            m.xfm_normalize(m.tmp('forward_f')+'_inverse.xfm',model.scan, output.xfm_f_inv, step=level )
                        else:
                            m.xfm_normalize(m.tmp('forward')+'.xfm', model.scan, output.xfm_inv, step=level, invert=True)
                            m.xfm_normalize(m.tmp('forward_f')+'.xfm',model.scan, output.xfm_f_inv, step=level, invert=True)
            else:
                if not os.path.exists( output.xfm ) :
                    if flip:
                        if ants:
                            ipl.ants_registration.non_linear_register_ants2(
                                sample.scan_f,
                                model.scan,
                                m.tmp('forward')+'.xfm',
                                target_mask=model.mask,
                                parameters=parameters,
                                downsample=downsample,
                                level=level,
                                start=start_level,
                                #work_dir=work_dir
                                )
                        else:
                            ipl.registration.non_linear_register_full(
                                sample.scan_f,
                                model.scan,
                                m.tmp('forward')+'.xfm',
                                #source_mask=sample.mask_f,
                                target_mask=model.mask,
                                init_xfm=_init_xfm,
                                parameters=parameters,
                                level=level,
                                start=start_level,
                                downsample=downsample,
                                #work_dir=work_dir
                                )
                    else:
                        if ants:
                            ipl.ants_registration.non_linear_register_ants2(
                                sample.scan,
                                model.scan,
                                m.tmp('forward')+'.xfm',
                                target_mask=model.mask,
                                parameters=parameters,
                                downsample=downsample,
                                level=level,
                                start=start_level,
                                #work_dir=work_dir
                                )
                        else:
                            ipl.registration.non_linear_register_full(
                                sample.scan,
                                model.scan,
                                m.tmp('forward')+'.xfm',
                                #source_mask=sample.mask,
                                target_mask=model.mask,
                                init_xfm=_init_xfm,
                                parameters=parameters,
                                level=level,
                                start=start_level,
                                downsample=downsample,
                                #work_dir=work_dir
                                )
                    m.xfm_normalize(m.tmp('forward')+'.xfm', model.scan, output.xfm, step=level)
                    
                    if output_invert:
                        if ants: # ANTS produces forward and invrese 
                            m.xfm_normalize(m.tmp('forward')+'_inverse.xfm', model.scan, output.xfm_inv, step=level )
                        else:
                            m.xfm_normalize(m.tmp('forward')+'.xfm', model.scan, output.xfm_inv, step=level, invert=True)
           
            if output_sample is not None: 
                m.resample_smooth(sample.scan, output_sample.scan, 
                                  transform=output.xfm_inv, 
                                  like=model.scan, 
                                  order=resample_order, 
                                  invert_transform=True)
                
                for (i,j) in enumerate(sample.add):
                    m.resample_smooth(sample.add[i], output_sample.add[i], 
                                  transform=output.xfm_inv, 
                                  like=model.scan, 
                                  order=resample_order, 
                                  invert_transform=True)
                if warp_seg:
                    m.resample_labels(sample.seg,  output_sample.seg,  
                                      transform=output.xfm_inv, 
                                      aa=resample_aa, 
                                      order=resample_order, 
                                      like=model.scan, 
                                      invert_transform=True, 
                                      baa=resample_baa)

                if symmetric:
                    m.resample_smooth(sample.scan_f, output_sample.scan_f, 
                                      transform=output.xfm_f_inv, 
                                      like=model.scan, 
                                      invert_transform=True, 
                                      order=resample_order)
                    
                    for (i,j) in enumerate(sample.add_f):
                        m.resample_smooth(sample.add_f[i], output_sample.add_f[i], 
                                  transform=output.xfm_f_inv, 
                                  like=model.scan, 
                                  order=resample_order, 
                                  invert_transform=True)
                    
                    if warp_seg:
                        m.resample_labels(sample.seg_f,  output_sample.seg_f,  
                                          transform=output.xfm_f_inv, 
                                          aa=resample_aa, 
                                          order=resample_order, 
                                          like=model.scan, 
                                          invert_transform=True, 
                                          baa=resample_baa )

            if output_inv_target is not None:
                m.resample_smooth(model.scan, output_inv_target.scan, 
                                  transform=output.xfm, 
                                  like=sample.scan, 
                                  order=resample_order, 
                                  invert_transform=True)

                for (i,j) in enumerate(output_inv_target.add):
                    m.resample_smooth(model.add[i], output_inv_target.add[i], 
                                  transform=output.xfm_inv, 
                                  like=model.scan, 
                                  order=resample_order, 
                                  invert_transform=True)

                if warp_seg:
                    m.resample_labels(model.seg,  output_inv_target.seg,  
                                      transform=output.xfm, 
                                      aa=resample_aa, 
                                      order=resample_order, 
                                      like=sample.scan, 
                                      invert_transform=True, 
                                      baa=resample_baa)

                if symmetric:
                    m.resample_smooth(model.scan, output_inv_target.scan_f, 
                                      transform=output.xfm_f, 
                                      like=sample.scan, 
                                      invert_transform=True, 
                                      order=resample_order)

                    for (i,j) in enumerate(output_inv_target.add):
                        m.resample_smooth(model.add_f[i], output_inv_target.add_f[i], 
                                      transform=output.xfm_f, 
                                      like=sample.scan, 
                                      invert_transform=True, 
                                      order=resample_order)

                    if warp_seg:
                        m.resample_labels(model.seg,  output_inv_target.seg_f,  
                                          transform=output.xfm_f, 
                                          aa=resample_aa, 
                                          order=resample_order, 
                                          like=sample.scan, 
                                          invert_transform=True, 
                                          baa=resample_baa )
        
    except mincError as e:
        print("Exception in non_linear_registration:{}".format(repr(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in non_linear_registration:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

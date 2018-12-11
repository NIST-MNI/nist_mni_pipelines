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
import ipl.minc_hl as hl

def fix_spacing(scan):
    """make sure all spacing in 3D volume are regular
    
    Arguments: `scan` scan to be fixed
    """
    with mincTools() as minc:
        for s in ['xspace', 'yspace', 'zspace']:
            spacing = minc.query_attribute( scan, s + ':spacing' )

            if spacing.count( 'irregular' ):
                minc.set_attribute( scan, s + ':spacing', 'regular__' )
    return scan

def denoise(in_scan, out_scan, parameters={}):
    """Apply patch-based denoising
    
    Arguments: in `MriScan` input
               out `MriScan` output
               parameters `dict` of parameters
    """
    use_anlm=parameters.get('anlm', False )
    denoise_beta=parameters.get('beta', 0.7 )
    patch=parameters.get('patch', 2 )
    search=parameters.get('search', 2 )
    regularize=parameters.get('regularize', None )
    with mincTools() as minc:
        if use_anlm:
            minc.anlm( in_scan.scan, out_scan.scan, beta=denoise_beta, patch=patch, search=search, regularize=regularize ) 
        else:
            minc.nlm( in_scan.scan, out_scan.scan, beta=denoise_beta, patch=patch, search=search ) 
        # TODO: maybe USE anlm sometimes?
    

def estimate_nu(in_scan, out_field, parameters={},model=None):
    """Estimate non-uniformity correction field
    
    Arguments: in `MriScan` input
               out_field `MriScan` output
               parameters `dict` of parameters
    """
    with mincTools() as minc:
        #
        #print("Running N4, parameters={}".format(repr(parameters)))
        #traceback.print_stack()
        weight_mask=None
        init_xfm=None # TODO: maybe add as a parameter, in case manual registration was done?
        if in_scan.mask is not None and os.path.exists(in_scan.mask):
            weight_mask=in_scan.mask
        else:
            #TODO: maybe use some kind of threshold here instead of built-in?
            pass
       
        if not minc.checkfiles(inputs=[in_scan.scan], outputs=[out_field.scan]):
            return
        
        if parameters.get('disable',False):
            minc.calc([in_scan.scan],'1.0',out_field.scan,datatype='-float')
        else:
            if parameters.get('use_stx_mask',False) and model is not None:
                # method from Gabriel
                minc.winsorize_intensity(in_scan.scan,minc.tmp('trunc_t1.mnc'))
                minc.binary_morphology(minc.tmp('trunc_t1.mnc'),'',minc.tmp('otsu_t1.mnc'),binarize_bimodal=True)
                minc.defrag(minc.tmp('otsu_t1.mnc'),minc.tmp('otsu_defrag_t1.mnc'))
                minc.autocrop(minc.tmp('otsu_defrag_t1.mnc'),minc.tmp('otsu_defrag_expanded_t1.mnc'),isoexpand='50mm')
                minc.binary_morphology(minc.tmp('otsu_defrag_expanded_t1.mnc'),'D[25] E[25]',minc.tmp('otsu_expanded_closed_t1.mnc'))
                minc.resample_labels(minc.tmp('otsu_expanded_closed_t1.mnc'),minc.tmp('otsu_closed_t1.mnc'),like=minc.tmp('trunc_t1.mnc'))
                
                minc.calc([minc.tmp('trunc_t1.mnc'),minc.tmp('otsu_closed_t1.mnc')], 'A[0]*A[1]',  minc.tmp('trunc_masked_t1.mnc'))
                minc.calc([in_scan.scan,minc.tmp('otsu_closed_t1.mnc')],'A[0]*A[1]' ,minc.tmp('masked_t1.mnc'))
                
                ipl.registration.linear_register( minc.tmp('trunc_masked_t1.mnc'), model.scan, minc.tmp('stx.xfm'),
                        init_xfm=init_xfm, objective='-nmi',conf='bestlinreg_new')
                
                minc.resample_labels( model.mask, minc.tmp('brainmask_t1.mnc'),
                        transform=minc.tmp('stx.xfm'), invert_transform=True,
                        like=minc.tmp('otsu_defrag_t1.mnc') )
                
                minc.calc([minc.tmp('otsu_defrag_t1.mnc'),minc.tmp('brainmask_t1.mnc')],'A[0]*A[1]',minc.tmp('weightmask_t1.mnc'))
                
                minc.n4(minc.tmp('masked_t1.mnc'),
                        output_field=out_field.scan,
                        shrink=parameters.get('shrink',4),
                        iter=parameters.get('iter','200x200x200x200'),
                        weight_mask=minc.tmp('weightmask_t1.mnc'),
                        mask=minc.tmp('otsu_closed_t1.mnc'),
                        distance=parameters.get('distance',200),
                        datatype=parameters.get('datatype',None)
                        )
            else:
                minc.n4(in_scan.scan, 
                    output_field=out_field.scan,
                    weight_mask=weight_mask,
                    shrink=parameters.get('shrink',4),
                    datatype=parameters.get('datatype',None),
                    iter=parameters.get('iter','200x200x200'),
                    distance=parameters.get('distance',200),
                    downsample_field=parameters.get('downsample_field',None))

def apply_nu(in_scan, field, out_scan, parameters={}):
    """ Apply non-uniformity correction 
    """
    with mincTools() as minc:
        if not minc.checkfiles(inputs=[field.scan],outputs=[out_scan.scan]):
            return
        minc.resample_smooth(field.scan,minc.tmp('fld.mnc'),like=in_scan.scan,order=1)
        minc.calc([in_scan.scan,minc.tmp('fld.mnc')],
                  'A[0]/A[1]', out_scan.scan)


def normalize_intensity(in_scan, out_scan,
                        parameters={}, 
                        model=None):
    """ Perform global intensity scale normalization
    """
    # TODO: make output exp file
    with mincTools() as minc:
        
        if not minc.checkfiles(inputs=[in_scan.scan],outputs=[out_scan.scan]):
            return
        
        if parameters is not None and not parameters.get('disable',False):
            order = parameters.get('order',1)
            _model=None
            
            # 
            if model is None:
                _model = parameters.get('model',None)
            else:
                _model = model.scan
            
            if _model is None:
                raise mincError('Need model ')
            
            scan_mask  = None
            model_mask = None
            
            if in_scan.mask is not None and model is not None:
                scan_mask  = in_scan.mask
                model_mask = model.mask
            elif parameters.get('nuyl',False):
                minc.nuyl_normalize(in_scan.scan,_model,out_scan.scan,
                                    source_mask=scan_mask,
                                    target_mask=model_mask)
            elif parameters.get('nuyl2',False):
                hl.nuyl_normalize2( in_scan.scan,_model,out_scan.scan,
                                    #source_mask=input_mask,target_mask=model_mask,
                                    fwhm=parameters.get('nuyl2_fwhm',2.0),
                                    iterations=parameters.get('nuyl2_iter',4),
                                    )
            else:
                minc.volume_pol(in_scan.scan, _model, out_scan.scan,
                                order=order, 
                                source_mask=scan_mask, 
                                target_mask=model_mask)
        else: # HACK just by-pass processing if parameters are empty or disabled
            shutil.copyfile(in_scan.scan,out_scan.scan)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

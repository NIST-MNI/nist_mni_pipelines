# -*- coding: utf-8 -*-

#
# @author Daniel
# @date 10/07/2011

version = '1.0'

#
# Do the VBM masks of GM WM and CSF
#
from .general import *

from ipl.minc_tools import mincTools,mincError
import ipl.registration
import ipl.ants_registration
import ipl.elastix_registration

# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_vbm(patient, tp, options):
    if      os.path.exists(patient[tp].vbm['wm'])   \
        and os.path.exists(patient[tp].vbm['gm'])   \
        and os.path.exists(patient[tp].vbm['csf'])  \
        and os.path.exists(patient[tp].vbm['idet'])  \
        and os.path.exists(patient[tp].vbm['grid']) \
        and os.path.exists(patient[tp].vbm['xfm']) :
        print(' -- Processing already done')
        return True

    return VBM_v10(patient, tp, options)

def resample_modulate( inp, label, xfm, jacobian, out, ref_hires, resolution, fwhm):
    with mincTools() as minc:
        if minc.checkfiles(inputs=[inp,xfm,jacobian], outputs=[out]):
            mod=minc.tmp('modulate_{}.mnc'.format(label))
            mod_r=minc.tmp('modulate_{}_r.mnc'.format(label))
            mod_rb=minc.tmp('modulate_{}_rb.mnc'.format(label))
            minc.calc([inp,jacobian], "(abs(A[0]-{})<0.5&&A[1]>-1)?1.0/(1.0+A[1]):0".format(label), mod )
            minc.resample_smooth(mod, mod_r,transform=xfm,like=ref_hires)
            minc.blur(mod_r,mod_rb,fwhm=fwhm)
            # downsample, convert to short here ?
            minc.resample_smooth(mod_rb, out, uniformize=resolution, datatype='short')
    
    
def VBM_v10(patient, tp, options):
    vbm_fwhm       = options.get('vbm_fwhm'      ,8)
    vbm_resolution = options.get('vbm_resolution',2)
    vbm_nl_level   = options.get('vbm_nl_level'  ,None)
    vbm_nl_method  = options.get('vbm_nl_method','minctracc')
    
    
    modelt1   = patient.modeldir + os.sep + patient.modelname + '.mnc'
    modelmask = patient.modeldir + os.sep + patient.modelname + '_mask.mnc'

    with mincTools() as minc:
        # create reference mask...
        #minc.resample_labels(modelmask,minc.tmp('ref.mnc'),uniformize=vbm_resolution)
        # create deformation field at required step size
        
        # TODO: create (regularize?) dbm-specific XFM to calculate jacobians only 
        if vbm_nl_method is not None and vbm_nl_level is not None:
            if vbm_nl_method=='minctracc':
                ipl.registration.non_linear_register_full(
                    patient[tp].stx2_mnc['t1'], modelt1,
                    patient[tp].vbm['xfm'],
                    source_mask=patient[tp].stx2_mnc['mask'], 
                    target_mask=modelmask,
                    level=vbm_nl_level )
            elif vbm_nl_method=='ANTS':
                ipl.ants_registration.non_linear_register_ants(
                    patient[tp].stx2_mnc['t1'], modelt1,
                    patient[tp].vbm['xfm'],
                    source_mask=patient[tp].stx2_mnc['mask'],
                    target_mask=modelmask )
            else:
                ipl.elastix_registration.register_elastix(
                    patient[tp].stx2_mnc['t1'], modelt1,
                    patient[tp].vbm['xfm'],
                    source_mask=patient[tp].stx2_mnc['mask'], 
                    target_mask=modelmask )
        else : # reuse existing xfm files
            minc.xfm_normalize(patient[tp].nl_xfm, 
                               modelmask, 
                               patient[tp].vbm['xfm'], 
                               step=vbm_resolution)
        
        xfm=patient[tp].vbm['xfm']
        cls=patient[tp].stx2_mnc['classification']
        
        minc.xfm_normalize(patient[tp].vbm['xfm'], cls, minc.tmp('nl')+'.xfm', exact=True)
        
        det=minc.tmp('det.mnc')
        minc.grid_determinant(minc.tmp('nl')+'_grid_0.mnc',det)

        resample_modulate(cls, 1, xfm, det, patient[tp].vbm['csf'],modelmask,  vbm_resolution, vbm_fwhm)
        resample_modulate(cls, 2, xfm, det, patient[tp].vbm['gm'], modelmask,  vbm_resolution, vbm_fwhm)
        resample_modulate(cls, 3, xfm, det, patient[tp].vbm['wm'], modelmask,  vbm_resolution, vbm_fwhm)
        
        # create determinant of inverse transform for DBM analysis
        minc.xfm_normalize(patient[tp].vbm['xfm'], patient[tp].vbm['csf'], minc.tmp('inl')+'.xfm', exact=True, invert=True)
        minc.grid_determinant(minc.tmp('inl')+'_grid_0.mnc',patient[tp].vbm['idet'])
        
        
    return 0


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

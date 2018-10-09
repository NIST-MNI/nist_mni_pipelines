# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 4/01/2016
#
# high level tools

from __future__ import print_function

import os
import sys
import shutil
import tempfile
import traceback

#from sigtools.modifiers import kwoargs,autokwoargs

# local stuff
from .minc_tools import mincTools
from .minc_tools import logger

# import xfmavg from model/registration
from .model.registration import xfmavg

import numpy as np
from sklearn import linear_model

def label_normalize(sample, sample_labels, ref, ref_labels, out=None,sample_mask=None, ref_mask=None,median=False,order=3,debug=False):
    '''Use label-based intensity normalization'''
    with mincTools() as minc:
        if not mincTools.checkfiles(outputs=[out]):     return
        
        ref_stats   = {i[0]:i[5] for i in minc.label_stats(ref_labels,      volume=ref, mask=ref_mask,median=median)}
        sample_stats= {i[0]:i[5] for i in minc.label_stats(sample_labels,volume=sample,mask=sample_mask,median=median)}
        x=[]
        y=[]
        
        for i in ref_stats:
            # use 0-intercept
            if i in sample_stats:
                #x.append( [1.0, sample_stats[i], sample_stats[i]*sample_stats[i] ] )
                x.append( sample_stats[i] )
                y.append( ref_stats[i] )
                #print('{} -> {}'.format(sample_stats[i],ref_stats[i]))
        # FIX origin? (HACK)
        x.append(0.0)
        y.append(0.0)
        # run linear regression
        clf = linear_model.LinearRegression()
        __x=np.array(x)
        
        _x=np.column_stack( ( np.power(__x,i) for i in range(1,order+1) ) )
        _y=np.array( y )
        #print(_x)
        #print(_y)
        clf.fit(_x, _y)
        
        if debug:
            import matplotlib.pyplot as plt
            logger.warn('Coefficients: \n', clf.coef_)
            #print('[0.0 100.0] -> {}'.format(clf.predict([[1.0,0.0,0.0], [1.0,100.0,100.0*100.0]] )))
            
            plt.scatter(_x[:,0], _y,  color='black')
            #plt.plot(_x[:,0], clf.predict(_x), color='blue', linewidth=3)
            prx=np.linspace(0,100,20)
            prxp=np.column_stack( ( np.power(prx,i) for i in range(1,order+1) ) )
            plt.plot( prx , clf.predict( prxp ), color='red', linewidth=3)

            plt.xticks(np.arange(0,100,5))
            plt.yticks(np.arange(0,100,5))

            plt.show()
        # create command-line for minccalc 
        cmd=''
        for i in range(order):
            if i==0:
                cmd+='A[0]*{}'.format(clf.coef_[i])
            else:
                cmd+='+'+'*'.join(['A[0]']*(i+1))+'*{}'.format(clf.coef_[i])
        if out is not None:
            minc.calc([sample],cmd,out)
        return cmd

def nuyl_normalize2(
    source,target,
    output,
    source_mask=None,
    target_mask=None,
    linear=False,
    iterations=4,
    filter_gradients=True,
    fwhm=2.0,
    verbose=0,
    remove_bg=False,
    ):
    """normalize intensities, using areas with uniform intensity """
    with mincTools(verbose=verbose) as minc:
        if not mincTools.checkfiles(outputs=[output]):     return
        # create gradient maps
        
        if filter_gradients:
            minc.blur(source,minc.tmp('source_grad.mnc'),fwhm,gmag=True,output_float=True)
            minc.blur(target,minc.tmp('target_grad.mnc'),fwhm,gmag=True,output_float=True)
            # create masks of areas with low gradient
            minc.binary_morphology(minc.tmp('source_grad.mnc'),'D[1] I[0]',minc.tmp('source_grad_mask.mnc'),binarize_bimodal=True)
            source_mask=minc.tmp('source_grad_mask.mnc')
            
            minc.binary_morphology(minc.tmp('target_grad.mnc'),'D[1] I[0]',minc.tmp('target_grad_mask.mnc'),binarize_bimodal=True)
            target_mask=minc.tmp('target_grad_mask.mnc')
            
            if remove_bg:
                minc.binary_morphology(source,'D[8]',minc.tmp('source_mask.mnc'),binarize_bimodal=True)
                minc.binary_morphology(target,'D[8]',minc.tmp('target_mask.mnc'),binarize_bimodal=True)
                minc.calc([source_mask,minc.tmp('source_mask.mnc')],'A[0]>0.5&&A[1]>0.5?1:0',minc.tmp('source_grad_mask2.mnc'))
                minc.calc([target_mask,minc.tmp('target_mask.mnc')],'A[0]>0.5&&A[1]>0.5?1:0',minc.tmp('target_grad_mask2.mnc'))
                source_mask=minc.tmp('source_grad_mask2.mnc')
                target_mask=minc.tmp('target_grad_mask2.mnc')
                
            if source_mask is not None:
                minc.resample_labels(source_mask,minc.tmp('source_mask.mnc'),like=minc.tmp('source_grad_mask.mnc'))
                minc.calc([minc.tmp('source_grad_mask.mnc'),minc.tmp('source_mask.mnc')],'A[0]>0.5&&A[1]>0.5?1:0',minc.tmp('source_mask2.mnc'))
                source_mask=minc.tmp('source_mask2.mnc')
                
            if target_mask is not None:
                minc.resample_labels(target_mask,minc.tmp('target_mask.mnc'),like=minc.tmp('target_grad_mask.mnc'))
                minc.calc([minc.tmp('target_grad_mask.mnc'),minc.tmp('target_mask.mnc')],'A[0]>0.5&&A[1]>0.5?1:0',minc.tmp('target_mask2.mnc'))
                target_mask=minc.tmp('target_mask2.mnc')

        # now run iterative normalization
        for i in range(iterations):
            if (i+1)==iterations: out=output
            else: out=minc.tmp('{}.mnc'.format(i))
            
            minc.nuyl_normalize(source,target,out,source_mask=source_mask,target_mask=target_mask,linear=linear)
            source=out

        # done here?

def patch_normalize(sample, sample_labels, ref, ref_labels, out=None,sample_mask=None, ref_mask=None,median=False,order=3,debug=False):
    '''Use label-based intensity normalization'''
    with mincTools() as minc:
        if not mincTools.checkfiles(outputs=[out]):     return
        
        ref_stats   = {i[0]:i[5] for i in minc.label_stats(ref_labels,      volume=ref, mask=ref_mask,median=median)}
        sample_stats= {i[0]:i[5] for i in minc.label_stats(sample_labels,volume=sample,mask=sample_mask,median=median)}
        x=[]
        y=[]
        
        for i in ref_stats:
            # use 0-intercept
            if i in sample_stats:
                #x.append( [1.0, sample_stats[i], sample_stats[i]*sample_stats[i] ] )
                x.append( sample_stats[i] )
                y.append( ref_stats[i] )
                #print('{} -> {}'.format(sample_stats[i],ref_stats[i]))
        # FIX origin? (HACK)
        x.append(0.0)
        y.append(0.0)
        # run linear regression
        clf = linear_model.LinearRegression()
        __x=np.array(x)
        
        _x=np.column_stack( ( np.power(__x,i) for i in range(1,order+1) ) )
        _y=np.array( y )
        #print(_x)
        #print(_y)
        clf.fit(_x, _y)
        
        if debug:
            import matplotlib.pyplot as plt
            print('Coefficients: \n', clf.coef_)
            #print('[0.0 100.0] -> {}'.format(clf.predict([[1.0,0.0,0.0], [1.0,100.0,100.0*100.0]] )))
            
            plt.scatter(_x[:,0], _y,  color='black')
            #plt.plot(_x[:,0], clf.predict(_x), color='blue', linewidth=3)
            prx=np.linspace(0,100,20)
            prxp=np.column_stack( ( np.power(prx,i) for i in range(1,order+1) ) )
            plt.plot( prx , clf.predict( prxp ), color='red', linewidth=3)

            plt.xticks(np.arange(0,100,5))
            plt.yticks(np.arange(0,100,5))

            plt.show()
        # create command-line for minccalc 
        cmd=''
        for i in range(order):
            if i==0:
                cmd+='A[0]*{}'.format(clf.coef_[i])
            else:
                cmd+='+'+'*'.join(['A[0]']*(i+1))+'*{}'.format(clf.coef_[i])
        if out is not None:
            minc.calc([sample],cmd,out)
        return cmd



if __name__ == '__main__':
    pass
    
  
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on


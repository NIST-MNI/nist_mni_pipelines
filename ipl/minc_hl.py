#!/usr/bin/env python
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
from ipl.minc_tools import mincTools,mincError
#from ipl.optfunc import optfunc
#from clize import run

# numpy & scipy
#from scipy import stats
import numpy as np
from sklearn import linear_model

try:
    # needed to read and write XFM files
    import pyezminc
except:
    pass

try:
    # needed for matrix log and exp
    import scipy.linalg
except:
    pass

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


def xfmavg(inputs,output):
    # TODO: handle inversion flag correctly
    all_linear=True
    all_nonlinear=True
    input_xfms=[]
    if not mincTools.checkfiles(inputs=inputs,
                        outputs=[output ]):
        return
    for j in inputs:
        x=pyezminc.read_transform(j)
        if x[0][0] and len(x)==1 and (not x[0][1]):
            # this is a linear matrix
            input_xfms.append(x[0])
        else:
            all_linear&=False
            # strip identity matrixes
            nl=[]
            _identity=np.asmatrix(np.identity(4))
            _eps=1e-6
            for i in x:
                if i[0]:
                     if scipy.linalg.norm(_identity-i[2])>_eps: # this is non-identity matrix
                        all_nonlinear&=False
                else:
                    nl.append(i)
            if len(nl)!=1: 
                all_nonlinear&=False
            else:
                input_xfms.append(nl[0])
    if all_linear:
        acc=np.asmatrix(np.zeros([4,4],dtype=np.complex))
        for i in input_xfms:
            acc+=scipy.linalg.logm(i[2])
        acc/=len(input_xfms)
        out_xfm=[(True,False,scipy.linalg.expm(acc).real)]
        pyezminc.write_transform(output,out_xfm)
    elif all_nonlinear:
        input_grids=[]
        for i in input_xfms:
            input_grids.append(i[2])
        output_grid=output.rsplit('.xfm',1)[0]+'_grid_0.mnc'
        with mincTools(verbose=2) as m:
            m.average(input_grids,output_grid)
        out_xfm=[(False,False,output_grid)]
        print("xfmavg output:{}".format(repr(out_xfm)))
        pyezminc.write_transform(output,out_xfm)
    else:
        raise Exception("Mixed XFM files provided as input")



if __name__ == '__main__':
    #optfunc.run([nuyl_normalize2,label_normalize]) 
    # TODO: re-implement using optparse
    pass
    
    
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on


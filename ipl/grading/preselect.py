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

import traceback


def preselect(sample, 
              library,
              method='MI',
              number=10,
              mask=None,
              use_nl=False,
              flip=False,
              step=None,
              lib_add_n=0,
              groups=None):
    '''calculate requested similarity function and return top number of elements from the library'''
    results=[]
    column=2 # skip over grading and group
    
    # TODO: use multiple modalities for preselection?
    if use_nl:
        column=6+lib_add_n
        
    for (i,j) in enumerate(library):
        results.append( 
            calculate_similarity.remote( sample, MriDataset(scan=j[column]), method=method, mask=mask, flip=flip, step=step
            ) )
    ray.wait(results,num_returns=len(results))

    val=[ (ray.get(j), int(library[i][0]), library[i] ) for (i,j) in enumerate(results)]
    
    if groups is None:
      val_sorted=sorted(val, key=lambda s: s[0] )
      return [ i[2] for i in val_sorted[ 0:number] ]
    else:
      s_number=number/groups
      res=[]
      
      for i in range(groups):
        val_sorted=sorted( [v for v in val if v[1]==i]  , key=lambda s: s[0] )
        res.extend( val_sorted[0:s_number] )
      
      return [ i[2] for i in res ]


def calculate_similarity(sample1, sample2,
                         mask=None, method='MI',
                         flip=False, step=None):
    try:
        with mincTools() as m:
            scan=sample1.scan
            
            if flip:
                scan=sample1.scan_f
                
            # figure out step size, minctracc works extremely slow when step size is smaller then file step size
            info_sample1=m.mincinfo( sample1.scan )
            
            cmds=[ 'minctracc', scan, sample2.scan, '-identity' ]
            
            if method=='MI':
                cmds.extend( ['-nmi', '-blur_pdf', '9'] )
            else:
                cmds.append( '-xcorr' )

            if step is None:
                step= max( abs( info_sample1['xspace'].step ) ,
                           abs( info_sample1['yspace'].step ) ,
                           abs( info_sample1['zspace'].step ) )
            
            cmds.extend([
                '-step', str(step), str(step), str(step),
                '-simplex', '1',
                '-tol', '0.01',
                '-lsq6',
                '-est_center',
                '-clob',
                m.tmp('similarity.xfm')
                ])

            if mask is not None:
                cmds.extend( ['-source_mask', mask])

            output=re.search( '^Final objective function value = (\S+)' , m.execute_w_output(cmds, verbose=0), flags=re.MULTILINE).group(1)

            return float(output)
            
    except mincError as e:
        print("Exception in calculate_similarity:{}".format( str(e)) )
        traceback.print_exc( file=sys.stdout )
        raise

    except :
        print("Exception in calculate_similarity:{}".format( sys.exc_info()[0]) )
        traceback.print_exc( file=sys.stdout )
        raise

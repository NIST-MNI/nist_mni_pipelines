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


def create_local_model(tmp_lin_samples, model, local_model,
                       extend_boundary=4, 
                       op=None, 
                       symmetric=False ):
    '''create an average segmentation and use it to create local model'''
    try:
        with mincTools() as m:
            segs=['multiple_volume_similarity']
            segs.extend([ i.seg for i in tmp_lin_samples ])
            
            if symmetric: segs.extend([ i.seg_f for i in tmp_lin_samples ])
            
            segs.extend(['--majority', m.tmp('majority.mnc')] )
            m.execute(segs)
            maj=m.tmp('majority.mnc')
            
            if op is not None:
                m.binary_morphology(maj, op, m.tmp('majority_op.mnc'),binarize_threshold=0.5)
                maj=m.tmp('majority_op.mnc')
                
            # TODO: replace mincreshape/mincbbox with something more sensible
            out=m.execute_w_output(['mincbbox', '-threshold', '0.5', '-mincreshape', maj ]).rstrip("\n").split(' ')
            
            s=[ int(i) for i in out[1].split(',') ]
            c=[ int(i) for i in out[3].split(',') ]
            
            start=[s[0]-extend_boundary,   s[1]-extend_boundary   ,s[2]-extend_boundary  ]
            ext=  [c[0]+extend_boundary*2, c[1]+extend_boundary*2 ,c[2]+extend_boundary*2]
            
            # reshape the mask
            m.execute(['mincreshape',
                       '-start','{},{},{}'.format(start[0], start[1], start[2]),
                       '-count','{},{},{}'.format(ext[0],   ext[1],   ext[2]  ),
                       maj , local_model.mask , '-byte' ] )
                
            m.resample_smooth(model.scan, local_model.scan, like=local_model.mask, order=0)
            m.resample_labels(m.tmp('majority.mnc'),local_model.seg, like=local_model.mask, order=0)
            
            for (i,j) in enumerate(model.add):
                m.resample_smooth(model.add[i], local_model.add[i], like=local_model.mask, order=0)
            
    except mincError as e:
        print("Exception in create_local_model:{}".format(repr(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in create_local_model:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise
        
def create_local_model_flip(local_model, model, remap={},
                            extend_boundary=4, op=None ):
    try:
        with mincTools() as m:
            m.param2xfm(m.tmp('flip_x.xfm'), scales=[-1.0, 1.0, 1.0])
            m.resample_labels(local_model.seg, m.tmp('flip_seg.mnc'), 
                               transform=m.tmp('flip_x.xfm'),  
                               order=0, remap=remap, like=model.scan)
            
            seg=m.tmp('flip_seg.mnc')
            
            if op is not None:
                m.binary_morphology(seg, op, m.tmp('flip_seg_op.mnc'),binarize_threshold=0.5)
                seg=m.tmp('flip_seg_op.mnc')
            
            # TODO: replace mincreshape/mincbbox with something more sensible
            out=m.execute_w_output(['mincbbox', '-threshold', '0.5', '-mincreshape', seg ]).rstrip("\n").split(' ')
            
            s=[ int(i) for i in out[1].split(',') ]
            c=[ int(i) for i in out[3].split(',') ]
            
            start=[s[0]-extend_boundary,   s[1]-extend_boundary   ,s[2]-extend_boundary  ]
            ext=  [c[0]+extend_boundary*2, c[1]+extend_boundary*2 ,c[2]+extend_boundary*2]
            # reshape the mask
            m.execute(['mincreshape',
                       '-start','{},{},{}'.format(start[0], start[1], start[2]),
                       '-count','{},{},{}'.format(ext[0],   ext[1],   ext[2]  ),
                       seg,
                       local_model.mask_f,
                       '-byte' ] )
            
            m.resample_smooth(local_model.scan, local_model.scan_f, 
                              like=local_model.mask_f, order=0, transform=m.tmp('flip_x.xfm'))
            
            for (i,j) in enumerate(model.add_f):
                m.resample_smooth(model.add[i], local_model.add_f[i], 
                              like=local_model.mask_f, order=0, transform=m.tmp('flip_x.xfm'))
    
    except mincError as e:
        print("Exception in create_local_model_flip:{}".format(str(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in create_local_model_flip:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

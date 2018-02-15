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
import ipl.minc_hl as hl


def filter_sample(input, output, filters, model=None):
    
    apply_filter(input.scan, output.scan, filters,
                 model=model.scan, input_mask=input.mask, 
                 model_mask=model.mask)
    # TODO: parallelalize?
    for (i,j) in enumerate( input.add ):
        apply_filter(input.add[i], output.add[i], filters, 
                     model=model.add[i], input_mask=i.mask, 
                     model_mask=model.mask)
    

def apply_filter(input, output, filters, model=None, input_mask=None, model_mask=None, input_labels=None, model_labels=None):
    output_scan=input
    try:
        if filters is not None :
            
            with mincTools() as m:
                if filters.get('denoise',False):
                    # TODO: choose between ANLM and NLM here?
                    m.anlm(output_scan,m.tmp('denoised.mnc'),
                        beta   =filters.get('beta',0.5),
                        patch  =filters.get('patch',1),
                        search =filters.get('search',1),
                        regularize=filters.get('regularize',None))
                        
                    output_scan  =m.tmp('denoised.mnc')
                    
                if filters.get('nuc',False) : # RUN N4
                    m.n4(output_scan,output_corr=m.tmp('n4.mnc'),
                        weight_mask=input_mask,
                        shrink=parameters.get('nuc_shrink',4),
                        iter=parameters.get('nuc_iter','200x200x200'),
                        distance=parameters.get('nuc_distance',200))
                    output_scan  =m.tmp('n4.mnc')
                
                if filters.get('normalize',False) and model is not None:
                    
                    if filters.get('nuyl',False):
                        m.nuyl_normalize(output_scan,model,m.tmp('normalized.mnc'),
                                         source_mask=input_mask,target_mask=model_mask)
                    elif filters.get('nuyl2',False):
                        hl.nuyl_normalize2(output_scan,model,m.tmp('normalized.mnc'),
                                           #source_mask=input_mask,target_mask=model_mask,
                                           fwhm=filters.get('nuyl2_fwhm',2.0),
                                           iterations=filters.get('nuyl2_iter',4))
                    else:
                        m.volume_pol(output_scan,model,    m.tmp('normalized.mnc'),
                                     source_mask=input_mask,target_mask=model_mask)
                    output_scan = m.tmp('normalized.mnc')
                
                # TODO: implement more filters
                patch_norm = filters.get('patch_norm',None)
                
                if patch_norm is not None:
                    print("Running patch normalization")
                    db  = patch_norm.get('db',None)
                    idx = patch_norm.get('idx',None)
                    thr = patch_norm.get('threshold',None)
                    spl = patch_norm.get('spline',None)
                    med = patch_norm.get('median',None)
                    it  = patch_norm.get('iterations',None)
                    if db is not None and idx and not None:
                        # have all the pieces
                        m.patch_norm(output_scan, m.tmp('patch_norm.mnc'), 
                                    index=idx, db=db, threshold=thr, spline=spl,
                                    median=med, field = m.tmp('patch_norm_field.mnc'),
                                    iterations=it)
                        output_scan = m.tmp('patch_norm.mnc')
                
                label_norm = filters.get('label_norm',None)
                
                if label_norm is not None and input_labels is not None and model_labels is not None:
                    print("Running label norm:{}".format(repr(label_norm)))
                    norm_order=label_norm.get('order',3)
                    norm_median=label_norm.get('median',True)
                    hl.label_normalize(output_scan,input_labels,model,model_labels,out=m.tmp('label_norm.mnc'),order=norm_order,median=norm_median)
                    output_scan = m.tmp('label_norm.mnc')
                
                shutil.copyfile(output_scan,output)
        else:
            shutil.copyfile(input,output)
    except mincError as e:
        print("Exception in apply_filter:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in apply_filter:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise


def make_border_mask( input, output, width=1,labels=1):
    '''Extract a border along the edge'''
    try:
        if not os.path.exists(output):
            with mincTools() as m:
                if labels==1:
                    m.binary_morphology(input,"D[{}]".format((width+1)//2),m.tmp('d.mnc'))
                    m.binary_morphology(input,"E[{}]".format(width//2),m.tmp('e.mnc'))
                    m.calc([m.tmp('d.mnc'),m.tmp('e.mnc')],'A[0]>0.5&&A[1]<0.5?1:0',output)
                else: # have to split up labels and then create a mask of all borders
                    split_labels(input,labels, m.tmp('split'))
                    borders=[]
                    for i in range(1,labels):
                        l='{}_{:02d}.mnc'  .format(m.tmp('split'),i)
                        d='{}_{:02d}_d.mnc'.format(m.tmp('split'),i)
                        e='{}_{:02d}_e.mnc'.format(m.tmp('split'),i)
                        b='{}_{:02d}_b.mnc'.format(m.tmp('split'),i)
                        m.binary_morphology(l,"D[{}]".format((width+1)//2),d)
                        m.binary_morphology(l,"E[{}]".format(width//2),e)
                        m.calc([d,e],'A[0]>0.5&&A[1]<0.5?1:0',b)
                        borders.append(b)
                    m.math(borders,'max',m.tmp('max'),datatype='-float')
                    m.reshape(m.tmp('max'),output,datatype='byte',
                              image_range=[0,1],valid_range=[0,1])

    except mincError as e:
        print("Exception in make_border_mask:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in make_border_mask:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise


def split_labels(input, n_labels,output_prefix,
                 antialias=False, blur=None,
                 expit=None, normalize=False ):
    try:
        with mincTools() as m:
            inputs=[ input ]
            outputs=['{}_{:02d}.mnc'.format(output_prefix,i) for i in range(n_labels) ]

            cmd=['itk_split_labels',input,'{}_%02d.mnc'.format(output_prefix),
                       '--missing',str(n_labels)]
            if antialias:
                cmd.append('--antialias')
            if normalize:
                cmd.append('--normalize')
            if blur is not None:
                cmd.extend(['--blur',str(blur)])
            if expit is not None:
                cmd.extend(['--expit',str(expit)])
            m.command(cmd, inputs=inputs, outputs=outputs)
            #return outputs
    except mincError as e:
        print("Exception in split_labels:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in split_labels:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise


def generate_flip_sample(input, labels_datatype='byte'):
    '''generate flipped version of sample'''
    try:
        with mincTools() as m:
            m.flip_volume_x(input.scan,input.scan_f)
            
            for (i,j) in enumerate(input.add):
                m.flip_volume_x(input.add[i],input.add_f[i])
            
            if input.mask is not None:
                m.flip_volume_x(input.mask, input.mask_f, labels=True)

            #for i in input.add:
            #    m.flip_volume_x(i,  input.seg_f, labels=True,datatype=labels_datatype)
    except mincError as e:
        print("Exception in generate_flip_sample:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in generate_flip_sample:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise

def create_unflip_remap(remap,remap_flip):
    if remap is not None and remap_flip is not None:
        # convert both into dict
        _remap=     { int(i[0]):int(i[1]) for i in remap }
        _remap_flip={ int(i[1]):int(i[0]) for i in remap_flip }
        _rr={}
        
        for i,j in _remap.items():
           if j in _remap_flip:
               _rr[j]=j
        return _rr
    else:
        return None

def log_transform_sample(input, output, threshold=1.0):
    try:
        with mincTools() as m:
            m.calc([input.scan],'A[0]>{}?log(A[0]):0.0'.format(threshold),
                output.scan)
    except mincError as e:
        print("Exception in log_transform_sample:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in log_transform_sample:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise
    

def create_patch_norm_db( input_samples, 
                          patch_norm_db, 
                          patch_norm_idx,
                          pct=0.1, 
                          patch=2,
                          sub=1):
    try:
        with mincTools() as m:
            patch_lib=os.path.dirname(input_samples[0].scan)+os.sep+'patch_lib.lst'
            inputs=[]
            outputs=[patch_norm_db]
            
            with open(patch_lib,'w') as f:
                for i in input_samples:
                    f.write( os.path.basename( i.scan ) )
                    f.write("\n")
                    inputs.append(i.scan)
            
            cmd=['create_feature_database', 
                    patch_lib, patch_norm_db,
                    '--patch',    
                    '--patch-radius', str(patch),
                    '--subsample',    str(sub),
                    '--random',       str(pct),
                    '--log',
                    '--threshold',    str(1.0),
                ] 
            
            m.command(cmd, inputs=inputs, outputs=outputs)
            
            cmd=['refine_feature_database', 
                    patch_norm_db, patch_norm_idx
                ]
            m.command(cmd, inputs=[patch_norm_db], outputs=[patch_norm_idx])
            
    except mincError as e:
        print("Exception in create_patch_norm_db:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in create_patch_norm_db:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 
#
# data structures used in segmentation package

import shutil
import os
import sys
import traceback
import json


class MriDataset(object):
    ''' Scan sample with segmentation and mask'''
    def __init__(self, prefix=None, name=None, scan=None, mask=None, seg=None, 
                 scan_f=None, mask_f=None, seg_f=None, protect=False, 
                 add=[], add_n=None, 
                 add_f=[] ):
        self.prefix = prefix
        self.name = name
        self.scan = scan
        self.mask = mask
        self.seg = seg
        self.protect = protect
        self.seg_split = {}
        
        self.scan_f = scan_f
        self.mask_f = mask_f
        self.seg_f  = seg_f
        self.seg_f_split = {}
        self.add    = add
        self.add_f  = add_f
        
        if self.name is None :
            if scan is not None:
                self.name = os.path.basename(scan).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
                if self.prefix is None:
                    self.prefix = os.path.dirname(self.scan)
            else:
                if self.prefix is None:
                    raise("trying to create dataset without name and prefix")
                (_h, _name) = tempfile.mkstemp(suffix='.mnc', dir=prefix)
                os.close(_h)
                self.name=os.path.relpath(_name,prefix)
                os.unlink(_name)

        if scan is None:
            if self.prefix is not None:
                self.scan=self.prefix+os.sep+self.name+'.mnc'
                self.mask=self.prefix+os.sep+self.name+'_mask.mnc'
                self.seg=self.prefix+os.sep+self.name+'_seg.mnc'
                self.scan_f=self.prefix+os.sep+self.name+'_f.mnc'
                self.mask_f=self.prefix+os.sep+self.name+'_f_mask.mnc'
                self.seg_f=self.prefix+os.sep+self.name+'_f_seg.mnc'
                
                if add_n is not None:
                    self.add=[self.prefix+os.sep+self.name+'_{}.mnc'.format(i) for i in range(add_n)]
                    self.add_f=[self.prefix+os.sep+self.name+'_{}_f.mnc'.format(i) for i in range(add_n)]
                else:
                    self.add=[]
                    self.add_f=[]
        #------

    def __repr__(self):
        return "MriDataset(\n prefix=\"{}\",\n name=\"{}\",\n scan=\"{}\",\n scan_f=\"{}\",\n mask=\"{}\",\n mask_f=\"{}\",\n seg=\"{}\",\n seg_f=\"{}\",\n protect={},\n add={},\n add_f={})".\
               format(self.prefix,self.name,self.scan,self.scan_f,self.mask,self.mask_f,self.seg,self.seg_f,repr(self.protect),repr(self.add),repr(self.add_f))

    def cleanup(self):
        if not self.protect:
            for i in (self.scan, self.mask, self.seg, self.scan_f, self.mask_f, self.seg_f ):
                if i is not None and os.path.exists(i):
                    os.unlink(i)
                    
            for (i,j) in self.seg_split.items():
                    if os.path.exists(j):
                        os.unlink(j)
                        
            for (i,j) in self.seg_f_split.items():
                    if os.path.exists(j):
                        os.unlink(j)
                        
            for (i,j) in enumerate(self.add):
                    if os.path.exists(j):
                        os.unlink(j)
    # ------------


class MriTransform(object):
    '''Transformation'''
    def __init__(self, prefix=None, name=None, xfm=None, protect=False, xfm_f=None, xfm_inv=None, xfm_f_inv=None, nl=False ):
        self.prefix=prefix
        self.name=name

        self.xfm=xfm
        self.grid=None

        self.xfm_f=xfm_f
        self.grid_f=None
        
        self.xfm_inv=xfm_inv
        self.grid_inv=None

        self.xfm_f_inv=xfm_f_inv
        self.grid_f_inv=None

        self.protect=protect
        self.nl=nl

        if name is None and xfm is None:
            raise "Undefined name and xfm"

        if name is None and xfm is not None:
            self.name=os.path.basename(xfm).rsplit('.xfm',1)[0]
            
            if self.prefix is None:
                self.prefix=os.path.dirname(self.xfm)

        if xfm is None:
            if self.prefix is not None:
                self.xfm=  self.prefix+os.sep+self.name+'.xfm'
                self.grid= self.prefix+os.sep+self.name+'_grid_0.mnc'

                self.xfm_f= self.prefix+os.sep+self.name+'_f.xfm'
                self.grid_f= self.prefix+os.sep+self.name+'_f_grid_0.mnc'

                self.xfm_inv=  self.prefix+os.sep+self.name+'_invert.xfm'
                self.grid= self.prefix+os.sep+self.name+'_invert_grid_0.mnc'

                self.xfm_f_inv= self.prefix+os.sep+self.name+'_f_invert.xfm'
                self.grid_f_inv= self.prefix+os.sep+self.name+'_f_invert_grid_0.mnc'

    def __repr__(self):
        return 'MriTransform(prefix="{}",name="{}")'.\
               format(self.prefix, self.name )

    def cleanup(self):
        if not self.protect:
            for i in (self.xfm, self.grid, self.xfm_f, self.grid_f, self.xfm_inv, self.grid_inv, self.xfm_f_inv, self.grid_f_inv ):
                if i is not None and os.path.exists(i):
                    os.unlink(i)


class MRIEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MriTransform):
            return {'name': obj.name,
                    'xfm': obj.xfm,
                    'xfm_f': obj.xfm_f,
                    'xfm_inv': obj.xfm_inv,
                    'xfm_f_inv': obj.xfm_f_inv,
                    'prefix': obj.prefix
                   }
        elif isinstance(obj, MriDataset):
            return {'name': obj.name,
                    'scan': obj.scan,
                    'mask': obj.mask,
                    'scan_f': obj.scan_f,
                    'mask_f': obj.mask_f,
                    'prefix': obj.prefix,
                    'add': obj.add,
                    'add_f': obj.add_f
                   }
         # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)



# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

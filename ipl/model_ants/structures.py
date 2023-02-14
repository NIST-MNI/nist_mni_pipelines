# data structures used in model generation package

import shutil
import os
import sys
import traceback
import json

class MriDataset(object):
    def __init__(self, prefix=None, name=None, 
                 iter=None, scan=None, mask=None, 
                 protect=False, par_int=[],par_def=[], has_mask=True):
        self.prefix=prefix
        self.name=name
        self.iter=iter
        self.protect=protect
        self.scan_f=None
        self.mask_f=None
        self.par_int=par_int
        self.par_def=par_def

        if scan is None:
            if self.iter is None:
                self.scan=self.prefix+os.sep+self.name+'.mnc'
                self.mask=self.prefix+os.sep+self.name+'_mask.mnc' if has_mask else None
                self.scan_f=self.prefix+os.sep+self.name+'_f.mnc'
                self.mask_f=self.prefix+os.sep+self.name+'_f_mask.mnc'  if has_mask else None
            else:
                self.scan=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'.mnc'
                self.mask=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'_mask.mnc'  if has_mask else None
                self.scan_f=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'_f.mnc'
                self.mask_f=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'_f_mask.mnc'  if has_mask else None
        else:
            self.scan=scan
            self.mask=mask

        if self.name is None:
            self.name=os.path.basename(self.scan)

        if self.prefix is None:
            self.prefix=os.path.dirname(self.scan)

    def has_mask(self):
        return self.mask is not None

    def __repr__(self):
        return 'MriDataset(prefix="{}",name="{}",iter="{}",scan="{}",mask="{}",protect={},par_int={},par_def={})'.\
               format(self.prefix,self.name,repr(self.iter),self.scan,self.mask,repr(self.protect),repr(self.par_int),repr(self.par_def))

    def exists(self):
        _ex=True
        for i in (self.scan, self.mask, self.scan_f, self.mask_f ):
            if i is not None :
                _ex&=os.path.exists(i)
        return _ex

    def cleanup(self,verbose=False):
        if not self.protect:
            for i in (self.scan, self.mask, self.scan_f, self.mask_f ):
                if i is not None and os.path.exists(i):
                    if verbose:
                        print("Removing:{}".format(i))
                    os.unlink(i)

class MriTransform(object):
    """
    Output from ANTs transform
    """

    def __init__(self, prefix, name, iter=None):
        self.prefix = prefix
        self.name = name
        self.iter = iter
        
        if self.iter is None:
            self.base = self.prefix + os.sep + self.name + '_'
            self.base_f = self.prefix + os.sep + self.name + '_f_'
        else:
            self.base = self.prefix + os.sep + self.name+'.{:03d}_'.format(iter)
            self.base_f = self.prefix + os.sep + self.name+'.{:03d}_f_'.format(iter)

        self.fw   = self.base   + '1_NL.xfm'
        self.fw_f = self.base_f + '1_NL.xfm'
        self.fw_grid    = self.base + '1_NL_grid_0.mnc'
        self.fw_grid_f  = self.base_f + '1_NL_grid_0.mnc'

        self.bw   = self.base   + '1_inverse_NL.xfm'
        self.bw_f = self.base_f + '1_inverse_NL.xfm'
        self.bw_grid    = self.base + '1_inverse_NL_grid_0.mnc'
        self.bw_grid_f  = self.base_f + '1_inverse_NL_grid_0.mnc'

        self.lin_fw   = self.base   + '0_GenericAffine.xfm'
        self.lin_fw_f = self.base_f + '0_GenericAffine.xfm'

    def __repr__(self):
        return 'MriTransform(prefix="{}",name="{}",iter="{}")'.\
               format(self.prefix,self.name,repr(self.iter))

    def cleanup(self,verbose=False):
        for i in (self.fw, self.fw_grid, self.fw_f, self.fw_grid_f,
                  self.bw, self.bw_grid, self.bw_f, self.bw_grid_f,
                  self.lin_fw, self.lin_fw_f):
            if i is not None and os.path.exists(i):
                if verbose:
                    print("Removing:{}".format(i))
                os.unlink(i)

class MRIEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MriTransform):
            return {'name':obj.name,
                    'iter':obj.iter,
                    'fw':  obj.fw,  'fw_grid':obj.fw_grid,
                    'fw_f':obj.fw_f,'fw_grid_f':obj.fw_grid_f,
                    'bw':  obj.bw,  'bw_grid':obj.bw_grid,
                    'bw_f':obj.bw_f,'bw_grid_f':obj.bw_grid_f,
                    'lin_fw':  obj.lin_fw,
                    'lin_fw_f':obj.lin_fw_f,
                   }
        elif isinstance(obj, MriDataset):
            return {'name':obj.name,
                    'iter':obj.iter,
                    'scan':obj.scan,
                    'mask':obj.mask,
                    'scan_f':obj.scan_f,
                    'mask_f':obj.mask_f,
                    'par_def':obj.par_def,
                    'par_int':obj.par_int
                   }
         # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
                
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

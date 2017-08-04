# data structures used in model generation package

import shutil
import os
import sys
import traceback
import json

class MriDataset(object):
    """
    Hold MRI sample together with regression parameters
    """
    def __init__(self, prefix=None, name=None, iter=None, scan=None, mask=None, protect=False, par_int=[],par_vel=[]):
        self.prefix=prefix
        self.name=name
        self.iter=iter
        self.protect=protect
        self.scan_f=None
        self.mask_f=None
        self.par_int=par_int
        self.par_vel=par_vel
                
        if scan is None:
            if self.iter is None:
                self.scan=self.prefix+os.sep+self.name+'.mnc'
                self.mask=self.prefix+os.sep+self.name+'_mask.mnc'
                self.scan_f=self.prefix+os.sep+self.name+'_f.mnc'
                self.mask_f=self.prefix+os.sep+self.name+'_f_mask.mnc'
            else:
                self.scan=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'.mnc'
                self.mask=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'_mask.mnc'
                self.scan_f=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'_f.mnc'
                self.mask_f=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'_f_mask.mnc'
        else:
            self.scan=scan
            self.mask=mask

        if self.name is None:
            self.name=os.path.basename(self.scan)

        if self.prefix is None:
            self.prefix=os.path.dirname(self.scan)

    def __repr__(self):
        return 'MriDataset(prefix="{}",name="{}",iter="{}",scan="{}",mask="{}",protect={},par_int={},par_val={})'.\
               format(self.prefix,self.name,repr(self.iter),self.scan,self.mask,repr(self.protect),repr(self.par_int),repr(self.par_vel))

    def cleanup(self):
        """
        Remove files, use if they are not needed anymore
        """
        if not self.protect:
            for i in (self.scan, self.mask, self.scan_f, self.mask_f ):
                if i is not None and os.path.exists(i):
                    os.unlink(i)
                    
    def exists(self):
        _ex=True
        for i in (self.scan, self.mask, self.scan_f, self.mask_f ):
            if i is not None :
                _ex&=os.path.exists(i)
        return _ex
    

class MriDatasetRegress(object):
    def __init__(self, prefix=None, name=None, iter=None, N=1, protect=False, from_dict=None, nomask=False):
        if from_dict is None:
            self.prefix=prefix
            self.name=name
            self.iter=iter
            self.protect=protect
            self.N=N
            self.volume=[]

            if self.iter is None:
                for n in range(0,N):
                    self.volume.append(self.prefix+os.sep+self.name+'_{}.mnc'.format(n))
                self.mask=self.prefix+os.sep+self.name+'_mask.mnc'
            else:
                for n in range(0,N):
                    self.volume.append(self.prefix+os.sep+self.name+'.{:03d}_{}'.format(iter,n)+'.mnc')
                self.mask=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'_mask.mnc'
            if nomask:
                self.mask=None
        else: # simple hack for now
            self.volume=from_dict["volume"]
            self.iter=from_dict["iter"]
            self.name=from_dict["name"]
            self.mask=from_dict["mask"]
            self.N=len(self.volume)

    def __repr__(self):
        return 'MriDatasetRegress(prefix="{}",name="{}",volume={},mask={},iter="{}",protect={})'.\
               format(self.prefix, self.name, repr(self.volume), self.mask, repr(self.iter), repr(self.protect))

    def cleanup(self):
        if not self.protect:
            for i in self.volume:
                if i is not None and os.path.exists(i):
                    os.unlink(i)
            for i in [self.mask]:
                if i is not None and os.path.exists(i):
                    os.unlink(i)

    def exists(self):
        """
        Check that all files are present
        """
        _ex=True
        for i in self.volume:
            if i is not None :
                _ex&=os.path.exists(i)
                
        for i in [self.mask]:
            if i is not None :
                _ex&=os.path.exists(i)
                
        return _ex

class LDDMriTransform(object):
    """
    Store log-diffemorphic transforation
    """
    def __init__(self,prefix,name,iter=None):
        self.prefix=prefix
        self.name=name
        self.iter=iter
        self.vel_f=None
        
        if self.iter is None:
            self.vel= self.prefix+os.sep+self.name+'_vel.mnc'
            self.vel_f= self.prefix+os.sep+self.name+'_f_vel_0.mnc'
        else:
            self.vel= self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'_vel.mnc'
            self.vel_f= self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'_f_vel.mnc'

    def __repr__(self):
        return 'LDDMriTransform(prefix="{}",name="{}",iter="{}")'.\
               format(self.prefix,self.name,repr(self.iter))

    def cleanup(self):
        for i in (self.vel, self.vel_f):
            if i is not None and os.path.exists(i):
                os.unlink(i)

    def exists(self):
        _ex=True
        for i in (self.vel, self.vel_f):
            if i is not None :
                _ex&=os.path.exists(i)
        return _ex


class LDDMRIEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, LDDMriTransform):
            return {'name':obj.name,
                    'iter':obj.iter,
                    'vel' :obj.vel,
                    'vel_f':obj.vel_f
                   }
        elif isinstance(obj, MriDatasetRegress):
            return {'name':obj.name,
                    'iter':obj.iter,
                    'volume':obj.volume,
                    'mask':obj.mask
                   }
        elif isinstance(obj, MriDataset):
            return {'name':obj.name,
                    'iter':obj.iter,
                    'scan':obj.scan,
                    'mask':obj.mask,
                    'scan_f':obj.scan_f,
                    'mask_f':obj.mask_f,
                    'par_int':obj.par_int,
                    'par_vel':obj.par_vel
                   }
         # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
                
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

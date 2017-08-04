# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 
#

# Longitudinal pipeline data structures

import shutil
import os
import sys
import traceback
import json


class MriScan(object):
    """Represents a 3D volume as an object on disk + (optionally) a mask
    """
    def __init__(self, 
                 prefix =  None, name = None, modality = None,
                 iter =    None, scan = None, mask     = '',
                 protect = False ):
        self.prefix=prefix
        self.name=name
        self.iter=iter
        self.protect=protect
        self.modality=modality

        if scan is None :
            if self.iter is None:
                if self.modality is not None: self.scan=self.prefix+os.sep+self.name+'_'+self.modality+'.mnc'
                else:                         self.scan=self.prefix+os.sep+self.name+'.mnc'
            else:
                if self.modality is not None: self.scan=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'_'+self.modality+'.mnc'
                else:                         self.scan=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'_'+'.mnc'
        else:
            self.scan=scan

        if mask=='':
            if self.iter is None:
                self.mask=self.prefix+os.sep+self.name+'_mask.mnc'
            else:
                self.mask=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'_mask.mnc'
        else:
            self.mask=mask

        if self.name is None:
            self.name=os.path.basename(self.scan)

        if self.prefix is None:
            self.prefix=os.path.dirname(self.scan)

    def __repr__(self):
        return 'MriScan(prefix="{}", name="{}", modality="{}", iter="{}",scan="{}",mask="{}",protect={})'.\
               format(self.prefix,self.name,self.modality,repr(self.iter),self.scan,self.mask,repr(self.protect))

    def cleanup(self,verbose=False):
        if not self.protect:
            for i in (self.scan, self.mask ):
                if i is not None and os.path.exists(i):
                    if verbose:
                        print("Removing:{}".format(i))
                    os.unlink(i)


class MriTransform(object):
    """Represents transformation 
    """
    def __init__(self, prefix, name, iter=None, nl=False, xfm=None, grid=None):
        self.prefix=prefix
        self.name=name
        self.iter=iter
        self.nl=nl
        self.xfm=xfm
        self.grid=grid
        
        if self.xfm is None:
            if self.iter is None:
                self.xfm=  self.prefix+os.sep+self.name+'.xfm'
            else:
                self.xfm=  self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'.xfm'
                
        if self.grid is None and xfm is None and nl:
            if self.iter is None:
                self.grid= self.prefix+os.sep+self.name+'_grid_0.mnc'
            else:
                self.grid= self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+'_grid_0.mnc'
            

    def __repr__(self):
        return 'MriTransform(prefix="{}",name="{}",iter="{}",nl={})'.\
               format(self.prefix,self.name,repr(self.iter),self.nl)

    def cleanup(self, verbose=False):
        for i in (self.xfm, self.grid ):
            if i is not None and os.path.exists(i):
                if verbose:
                    print("Removing:{}".format(i))
                os.unlink(i)


class MriQCImage(object):
    """Represents QC image (.jpg)
    """
    def __init__(self, prefix, name, iter=None, fname=None, suffix='.jpg'):
        self.prefix=prefix
        self.name=name
        self.iter=iter
        self.fname=fname
        self.suffix=suffix
        
        if self.fname is None:
            if self.iter is None:
                self.fname=self.prefix+os.sep+self.name+self.suffix
            else:
                self.fname=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+self.suffix

    def __repr__(self):
        return 'MriQCImage(prefix="{}",name="{}",iter="{}",fname={})'.\
               format(self.prefix,self.name,repr(self.iter),self.fname)

    def cleanup(self, verbose=False):
        #TODO: implement?
        pass


class MriAux(object):
    """Represents an auxiliary file (text)
    """
    def __init__(self, prefix, name, iter=None, fname=None,suffix='.txt'):
        self.prefix=prefix
        self.name=name
        self.iter=iter
        self.fname=fname
        self.suffix=suffix
        
        if self.fname is None:
            if self.iter is None:
                self.fname=self.prefix+os.sep+self.name+self.suffix
            else:
                self.fname=self.prefix+os.sep+self.name+'.{:03d}'.format(iter)+self.suffix

    def __repr__(self):
        return 'MriAux(prefix="{}",name="{}",iter="{}",fname={})'.\
               format(self.prefix,self.name,repr(self.iter),self.fname)

    def cleanup(self, verbose=False):
        #TODO: implement?
        pass


class PipelineEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MriTransform):
            return {'name':obj.name,
                    'iter':obj.iter,
                    'xfm':obj.xfm,
                    'grid':obj.grid,
                    'nl': obj.nl,
                    'type':'transform',
                   }
        
        if isinstance(obj, MriScan):
            return {'name':obj.name,
                    'modality': obj.modality,
                    'iter':obj.iter,
                    'scan':obj.scan,
                    'mask':obj.mask,
                    'modality': obj.modality,
                    'type':'scan',
                   }
        if isinstance(obj, MriQCImage):
            return {'name':obj.name,
                    'iter':obj.iter,
                    'fname':obj.fname,
                    'type':'qc_image',
                   }
        
        if isinstance(obj, MriAux):
            return {'name':obj.name,
                    'iter':obj.iter,
                    'fname':obj.fname,
                    'type':'aux'
                   }
         # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def save_summary(summary,out_file):
    with open(out_file,'w') as f:
        json.dump(summary, f, indent=1, cls=PipelineEncoder, sort_keys=True)

def save_pipeline_output(summary,out_file):
    save_summary(summary,out_file)

def convert_summary(in_dict):
    ret={}
    # iterate over all entries, assuming they should contain only 
    # recognized types
    for i,j in in_dict.iteritems():
        if isinstance(j, dict):
            if j.get('type',None)=='aux':
                ret[i]=MriAux(
                    os.path.dirname(j.get('fname','.')),
                    name=j.get('name',None),
                    iter=j.get('iter',None),
                    fname=j.get('fname',None))

            elif j.get('type',None)=='qc_image':
                ret[i]=MriQCImage(
                    os.path.dirname(j.get('fname','.')),
                    j.get('name',''),
                    iter=j.get('iter',None),
                    fname=j.get('fname',None),
                    suffix='.'+j.get('fname','.jpg').rsplit('.',1)[-1],
                    )

            elif j.get('type',None)=='scan':
                ret[i]=MriScan(
                    prefix=os.path.dirname(j.get('fname','.')),
                    name=j.get('name',''),
                    iter=j.get('iter',None),
                    scan=j.get('scan',None),
                    mask=j.get('mask',''),
                    modality=j.get('modality','')
                    )

            elif j.get('type',None)=='transform':
                ret[i]=MriTransform(
                    os.path.dirname(j.get('fname','.')),
                    j.get('name',''),
                    iter=j.get('iter',None),
                    xfm=j.get('xfm',None),
                    grid=j.get('grid',None),
                    nl=j.get('nl',False)
                    )

            else: # just copy it!
                ret[i]=j

        else:
            ret[i]=j
    return ret

def load_summary(in_file):
    tmp=None
    with open(in_file,'r') as f:
        tmp=json.load(f)
    ret=convert_summary(tmp)
    return ret

def load_pipeline_output(in_file):
    tmp=None
    with open(in_file,'r') as f:
        tmp=json.load(f)
    ret=[]
    for i in tmp:
        o=convert_summary(i)
        o['output']=convert_summary(o['output'])
        ret.append(o)
    return ret
    


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 
#

import copy
import json
import os
import sys
import traceback

def save_library_info(library_description, output,name='library.json'):
    """Save library information into directory, using predfined file structure
    Arguments:
    library_description -- dictionary with library description
    output -- output directory

    Keyword arguments:
    name -- optional name of .json file, relative to the output directory, default 'library.json'
    """
    try:
        tmp_library_description=copy.deepcopy(library_description)
        tmp_library_description.pop('prefix',None)

        for i in ['local_model','local_model_mask', 'local_model_flip', 
                'local_model_mask_flip',
                'local_model_seg','local_model_sd','local_model_avg','local_model_ovl',
                'gco_energy']:
            if tmp_library_description[i] is not None: 
                tmp_library_description[i]=os.path.relpath(tmp_library_description[i],output)
                
        for (j, i) in enumerate(tmp_library_description['local_model_add']):
            tmp_library_description['local_model_add'][j]=os.path.relpath(i, output)

        for (j, i) in enumerate(tmp_library_description['local_model_add_flip']):
            tmp_library_description['local_model_add_flip'][j]=os.path.relpath(i, output)
            
        for i in ['model','model_mask']:
            # if it starts with the same prefix, remove it
            if    os.path.dirname(tmp_library_description[i])==output \
            or tmp_library_description[i][0]!=os.sep:
                tmp_library_description[i]=os.path.relpath(tmp_library_description[i],output)
                
        for (j, i) in enumerate(tmp_library_description['model_add']):
            if os.path.dirname(i)==output:
                tmp_library_description['model_add'][j]=os.path.relpath(i, output)

        for (j, i) in enumerate(tmp_library_description['library']):
            for (k,t) in enumerate(i):
                tmp_library_description['library'][j][k]=os.path.relpath(t, output)

        with open(output+os.sep+name,'w') as f:
            json.dump(tmp_library_description,f,indent=1)
    except :
        print "Error saving library information into:{} {}".format(output,sys.exc_info()[0])
        traceback.print_exc(file=sys.stderr)
        raise

def load_library_info(prefix, name='library.json'):
    """Load library information from directory, using predfined file structure
    Arguments:
    prefix -- directory path
    
    Keyword arguments:
    name -- optional name of .json file, relative to the input directory, default 'library.json'
    """
    try:
        library_description={}
        with open(prefix+os.sep+name,'r') as f:
            library_description=json.load(f)

        library_description['prefix']=prefix

        for i in ['local_model','local_model_mask', 'local_model_flip',
                 'local_model_mask_flip','local_model_seg','gco_energy']:
            if library_description[i] is not None: library_description[i]=prefix+os.sep+library_description[i]
        
        try:
            for (j, i) in enumerate(library_description['local_model_add']):
                library_description['local_model_add'][j]=prefix+os.sep+i
                
            for (j, i) in enumerate(library_description['local_model_add_flip']):
                library_description['local_model_add_flip'][j]=prefix+os.sep+i
        except KeyError:
            pass

        for (j, i) in enumerate(library_description['library']):
            for (k,t) in enumerate(i):
                library_description['library'][j][k]=prefix+os.sep+t

        for i in ['model','model_mask']:
            # if it starts with '/' assume it's absolute path
            if library_description[i] is not None and library_description[i][0]!=os.sep:
                library_description[i]=prefix+os.sep+library_description[i]
        try:
            for (j, i) in enumerate(library_description['model_add']):
                if library_description['model_add'][j][0]!='/':
                    library_description['model_add'][j]=prefix+os.sep+i
        except KeyError:
            pass

        return library_description
    except :
        print "Error loading library information from:{} {}".format(prefix,sys.exc_info()[0])
        traceback.print_exc(file=sys.stderr)
        raise

def make_segmented_label_list(library_description,symmetric=False):
    """ Make a list of labels that are included in the segmentation library
    taking into account flipped labels too if needed
    """
    used_labels=set()

    if isinstance(library_description['map'], dict):
        for i in library_description['map'].iteritems():
            used_labels.add(int(i[1]))
        if symmetric:
            for i in library_description['flip_map'].iteritems():
                used_labels.add(int(i[1]))
    else:
        for i in library_description['map']:
            used_labels.add(int(i[1]))
        if symmetric:
            for i in library_description['flip_map']:
                used_labels.add(int(i[1]))
    return list(used_labels)
    
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

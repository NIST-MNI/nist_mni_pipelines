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

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


from .structures import *


class LibEntry(yaml.YAMLObject):
    """
    Segmentation library sample, closely related to MriDataset
    """
    yaml_tag = '!Entry'

    def __init__(self, lst=None, prefix='.', ent_id=None, relpath=None):
        if relpath is None:
            self.lst = lst
        else:# make relative paths
            self.lst = [os.path.relpath(i, relpath) for i in lst]

        self.ent_id = ent_id

        if ent_id is None and lst is not None and len(lst)>0:
            self.ent_id = lst[0].rsplit('.mnc',1)[0]

        self.prefix = prefix

    def __getitem__(self, item):
        """
        compatibility interface
        """
        if isinstance(item, slice):
            # it's a slice
            return [self.prefix + os.sep + i for i in self.lst[item] ]
        else:
            return self.prefix + os.sep + self.lst[item]

    @classmethod
    def from_yaml(cls, loader, node):
        dat = loader.construct_mapping(node)
        return LibEntry(lst=dat['lst'], ent_id=dat['id'], prefix=None)

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(cls.yaml_tag, {'lst': data.lst, 'id': data.ent_id})


class SegLibrary(yaml.YAMLObject):
    """
    Segmentation library DB
    """
    yaml_tag = '!SegLibrary'

    _rel_paths = {'local_model',
                  'local_model_mask',
                  'local_model_flip',
                  'local_model_mask_flip',
                  'local_model_seg',
                  'local_model_seg_flip',
                  'gco_energy'}

    _abs_paths = {'model',
                  'model_mask' }

    _abs_paths_lst = {'local_model_add', 'local_model_add_flip', 'model_add'}

    _all_visible_tags = _rel_paths | _abs_paths | _abs_paths_lst | {'library'}

    def __init__(self, path=None ):
        # compatibility info
        self.local_model = None
        self.local_model_mask = None
        self.local_model_flip = None,
        self.local_model_mask_flip = None,
        self.local_model_seg = None
        self.local_model_seg_flip = None
        self.local_model_sd  = None
        self.local_model_avg = None
        self.local_model_ovl = None
        self.local_model_add = []
        self.local_model_add_flip = []
        self.model      = None
        self.model_add  = []
        self.model_mask = None
        self.flip_map = {}
        self.map = {}
        self.gco_energy = None
        self.library = []
        self.modalities = 1
        self.classes_number = 2
        self.nl_samples_avail = False
        self.seg_datatype = 'byte'
        self.label_map={}

        # from file:
        self.prefix = None
        if path is not None:
            self.load(path)

    def load(self, path, name=None):
        if name is None:
            if os.path.exists(path + os.sep + 'library.yaml'):
                name = 'library.yaml'
            else:
                name = 'library.json'

        with open(path + os.sep + name, 'r') as f:
            tmp = yaml.load(f)
            self.prefix = path

            if type(tmp) is dict:
                self._load_legacy(tmp)
            elif tmp is not None:
                self.__dict__.update(tmp.__dict__)
            else:
                raise Exception("Can't load yaml from:{}".format(path + os.sep + name))

        # remember the prefix for the library
        self.prefix = path
        # Fix prefixes for all lib entries, as it is not laoded correctly
        for i, _ in enumerate(self.library):
            self.library[i].prefix = self.prefix

    def save(self, path, name='library.yaml'):
        with open(path + os.sep + name, 'w') as f:
            f.write(yaml.dump(self))

    def _load_legacy(self, library_description):
        try:
            for i in {'local_model',
                      'local_model_mask',
                      'local_model_flip',
                      'local_model_mask_flip',
                      'local_model_seg',
                      'gco_energy'}:
                if library_description.get(i, 'None') :
                    self.__dict__[i] = library_description[i]
                else:
                    self.__dict__[i] = None

            for i in ['model', 'model_mask']:
                if library_description.get(i, None):
                    self.__dict__[i] = library_description[i]
                else:
                    self.__dict__[i] = None

            if library_description.get('model_add', None):
                self.model_add = library_description['model_add']

            if library_description.get('local_model_add', None):
                self.local_model_add = library_description['local_model_add']

            if library_description.get('local_model_add_flip', None):
                self.local_model_add_flip = library_description['local_model_add_flip']

            # handle library loading correctly
            # convert samples paths to absolute
            self.library = [LibEntry(i, self.prefix) for i in library_description['library']]

            for i in ['flip_map', 'map','label_map','nl_samples_avail','seg_datatype','modalities']:
                self.__dict__[i] = library_description.get(i, self.__dict__[i])


        except:
            print("Error loading library information from:{} {}".format(prefix, sys.exc_info()[0]))
            traceback.print_exc(file=sys.stderr)
            raise

    def __getitem__(self, item):
        """
        compatibility interface
        """
        return self.get(item, default=None)

    def get(self, item, default=None):
        if item in self.__dict__:
            if item in SegLibrary._rel_paths:
                return self.prefix + os.sep + self.__dict__[item]  if self.__dict__[item] is not None else default
            elif item in SegLibrary._abs_paths:
                return (self.prefix + os.sep + self.__dict__[item] if self.__dict__[item][0] != os.sep else self.__dict__[item] ) if self.__dict__[item] is not None else default
            elif item in SegLibrary._abs_paths_lst:
                return [ (self.prefix + os.sep + i if i[0] != os.sep else i) for i in self.__dict__[item]]
            else:
                return self.__dict__[item]
        else:
            return default

    @classmethod
    def from_yaml(cls, loader, node):
        dat = loader.construct_mapping(node)
        lll = SegLibrary()
        lll.__dict__.update(dat)
        return lll

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(cls.yaml_tag,
             {k: data.__dict__[k] for k in data.__dict__.keys() & SegLibrary._all_visible_tags}
        )


def save_library_info(library_description, output, name='library.json'):
    """Save library information into directory, using predfined file structure
    Arguments:
    library_description -- dictionary with library description
    output -- output directory

    Keyword arguments:
    name -- optional name of .json file, relative to the output directory, default 'library.json'
    """
    try:
        tmp_library_description = copy.deepcopy(library_description)
        tmp_library_description.pop('prefix', None)

        for i in ['local_model', 'local_model_mask', 'local_model_flip',
                  'local_model_mask_flip',
                  'local_model_seg', 'local_model_sd', 'local_model_avg', 'local_model_ovl',
                  'gco_energy']:
            if tmp_library_description[i] is not None: 
                tmp_library_description[i] = os.path.relpath(tmp_library_description[i],output)
                
        for (j, i) in enumerate(tmp_library_description['local_model_add']):
            tmp_library_description['local_model_add'][j] = os.path.relpath(i, output)

        for (j, i) in enumerate(tmp_library_description['local_model_add_flip']):
            tmp_library_description['local_model_add_flip'][j] = os.path.relpath(i, output)
            
        for i in ['model', 'model_mask']:
            # if it starts with the same prefix, remove it
            if os.path.dirname(tmp_library_description[i]) == output \
              or tmp_library_description[i][0] != os.sep:
                 tmp_library_description[i] = os.path.relpath(tmp_library_description[i], output)
                
        for (j, i) in enumerate(tmp_library_description['model_add']):
            if os.path.dirname(i)==output:
                tmp_library_description['model_add'][j] = os.path.relpath(i, output)
        
        # convert samples paths to relative
        for (j, i) in enumerate(tmp_library_description['library']):
            for (k, t) in enumerate(i):
                tmp_library_description['library'][j][k] = os.path.relpath(t, output)

        with open(output+os.sep+name, 'w') as f:
            json.dump(tmp_library_description, f, indent=1)
            #TODO: convert to YAML
    except:
        print("Error saving library information into:{} {}".format(output,sys.exc_info()[0]))
        traceback.print_exc(file=sys.stderr)
        raise


def load_library_info(prefix, name='library.json'):
    """Load library information from directory, using predefined file structure
    Arguments:
    prefix -- directory path
    
    Keyword arguments:
    name -- optional name of .json file, relative to the input directory, default 'library.json'
    """
    try:
        library_description={}
        with open(prefix+os.sep+name, 'r') as f:
            library_description = yaml.load(f)

        library_description['prefix'] = prefix

        for i in ['local_model','local_model_mask', 'local_model_flip',
                  'local_model_mask_flip','local_model_seg','gco_energy']:
            if library_description[i] is not None:
                library_description[i] = prefix+os.sep+library_description[i]
        
        try:
            for (j, i) in enumerate(library_description['local_model_add']):
                library_description['local_model_add'][j] = prefix+os.sep+i
                
            for (j, i) in enumerate(library_description['local_model_add_flip']):
                library_description['local_model_add_flip'][j] = prefix+os.sep+i
        except KeyError:
            pass

        # convert samples paths to absolute
        for (j, i) in enumerate(library_description['library']):
            for (k, t) in enumerate(i):
                library_description['library'][j][k] = prefix+os.sep+t

        for i in ['model', 'model_mask']:
            # if it starts with '/' assume it's absolute path
            if library_description[i] is not None and library_description[i][0]!=os.sep:
                library_description[i] = prefix+os.sep+library_description[i]
        try:
            for (j, i) in enumerate(library_description['model_add']):
                if library_description['model_add'][j][0] != '/':
                    library_description['model_add'][j] = prefix+os.sep+i
        except KeyError:
            pass

        return library_description
    except:
        print("Error loading library information from:{} {}".format(prefix, sys.exc_info()[0]))
        traceback.print_exc(file=sys.stderr)
        raise


def make_segmented_label_list(library_description, symmetric=False):
    """ Make a list of labels that are included in the segmentation library
    taking into account flipped labels too if needed
    """
    used_labels = set()

    if isinstance(library_description['map'], dict):
        for i in library_description['map'].items():
            used_labels.add(int(i[1]))
        if symmetric:
            for i in library_description['flip_map'].items():
                used_labels.add(int(i[1]))
    else:
        for i in library_description['map']:
            used_labels.add(int(i[1]))
        if symmetric:
            for i in library_description['flip_map']:
                used_labels.add(int(i[1]))
    return list(used_labels)


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

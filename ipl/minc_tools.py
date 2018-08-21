#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 11/21/2011
#
# Generic minc tools

from __future__ import print_function

import os
import sys
import shutil
import tempfile
import subprocess
import re
import fcntl
import traceback
import collections
import math

import inspect

# local stuff
from . import registration
from . import ants_registration
from . import dd_registration
from . import elastix_registration

# hack to make it work on Python 3
try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    str = str
    unicode = str
    bytes = bytes
    basestring = (str,bytes)
else:
    # 'unicode' exists, must be Python 2
    str = str
    unicode = unicode
    bytes = str
    basestring = basestring

def get_git_hash():
    _script_dir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    _hash_code=''
    try:
        p=subprocess.Popen(['git', '-C', _script_dir, 'rev-parse', '--short', '--verify', 'HEAD'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output,outerr)=p.communicate()
        _hash_code=output.decode()
        outvalue=p.wait()
    except OSError as e:
        _hash_code='Unknown'
    if not outvalue == 0:
        _hash_code='Unknown'
    return _hash_code.rstrip("\n")
    
class mincError(Exception):
    """MINC tools general error"""
    def __init__(self, value='ERROR'):
        self.value = value
        self.stack = traceback.extract_stack()

    def __repr__(self):
        return "mincError:{}\nAT:{}".format(self.value, self.stack)

    def __str__(self):
        return self.__repr__()
        

class temp_files(object):
    """Class to keep track of temp files"""
    
    def __init__(self, tempdir=None, prefix=None):
        
        self.tempdir = tempdir
        self.clean_tempdir = False
        self.tempfiles = {}
        if not self.tempdir:
            if prefix is None:
                prefix='iplMincTools'
            self.tempdir = tempfile.mkdtemp(prefix=prefix, dir=os.environ.get('TMPDIR',None) )
            self.clean_tempdir = True
            
        if not os.path.exists(self.tempdir):
            os.makedirs(self.tempdir)

    def __enter__(self):
        return self

    def __exit__(
        self,
        type,
        value,
        traceback,
        ):
        self.do_cleanup()

    def __del__(self):
        self.do_cleanup()

    def do_cleanup(self):
        """remove temporary directory if present"""
        if self.clean_tempdir and self.tempdir is not None:
            shutil.rmtree(self.tempdir)
            self.clean_tempdir=False

    def temp_file(self, suffix='', prefix=''):
        """create temporary file"""

        (h, name) = tempfile.mkstemp(suffix=suffix, prefix=prefix,dir=self.tempdir)
        os.close(h)
        os.unlink(name)
        return name

    def tmp(self, name):
        """return path of a temp file named name"""
        try:
            return self.tempfiles[name]
        except KeyError:
            self.tempfiles[name] = self.temp_file(suffix=name)
            return self.tempfiles[name]

    def temp_dir(self, suffix='', prefix=''):
        """ Create temporary directory for processing"""

        name = tempfile.mkdtemp(suffix=suffix, prefix=prefix,
                                dir=self.tempdir)
        return name

    @property
    def dir(self):
        return self.tempdir
    
class cache_files(temp_files):
    """Class to keep track of work files"""
    def __init__(self, work_dir=None, context='',tempdir=None):
        self._locks={}
        super(cache_files,self).__init__(tempdir=tempdir)
        self.work_dir=work_dir
        self.context=context # TODO: something more clever here?
        self.cache_dir=None
        if self.work_dir is not None:
            self.cache_dir=self.work_dir+os.sep+context+os.sep
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

        
    def cache(self,name,suffix=''):
        """Allocate a name in cache, if cache was setup
        also lock the file , so that another process have to wait before using the same file name
        
        Important: call unlock() on result
        """
        #TODO: something more clever here?
        fname=''
        if self.work_dir is not None:
            fname=self.cache_dir+os.sep+name+suffix
            lock_name=fname+'.lock'
            f=self._locks[lock_name]=open(lock_name, 'a')
            fcntl.lockf(f.fileno(), fcntl.LOCK_EX )
        else:
            fname=self.tmp(name+suffix)

        return fname


    def unlock(self,fname):
        #TODO: something more clever here?
        lock_name=fname+'.lock'
        try:
            f=self._locks[lock_name]
            
            if f is not None:
                fcntl.lockf(f.fileno(), fcntl.LOCK_UN)
                f.close()

            del self._locks[lock_name]
            
#            try:
#                os.unlink(lock_name)
#            except OSError:
                #probably somebody else is blocking
#               pass

        except KeyError:
            pass


    #def __del__(self):
        #self.do_cleanup()
    #    pass

    def __enter__(self):
        return self

    def __exit__(
        self,
        type,
        value,
        traceback,
        ):
        self.do_cleanup()


    def do_cleanup(self):
        """unlocking lock files """
        for f in self._locks.keys():
            if self._locks[f] is not None:
                fcntl.flock(self._locks[f].fileno(), fcntl.LOCK_UN)
                self._locks[f].close()
#                try:
#                    os.unlink(f)
#                except OSError:
#                    #probably somebody else is blocking
#                    pass
        self._locks={}
        super(cache_files,self).do_cleanup()

class mincTools(temp_files):
    """minc toolkit interface , mostly basic tools """

    def __init__(self, tempdir=None, resample=None, verbose=0, prefix=None):
        super(mincTools, self).__init__(tempdir=tempdir,prefix=prefix)
        # TODO: add some options?
        self.resample = resample
        self.verbose  = verbose

    def __enter__(self):
        return super(mincTools,self).__enter__()

    def __exit__(
        self,
        type,
        value,
        traceback,
        ):
        return super(mincTools,self).__exit__(type,value,traceback)

    @staticmethod
    def checkfiles(
        inputs=None,
        outputs=None,
        timecheck=False,
        verbose=1,
        ):
        """ Check newer input file """

        itime = -1  # numer of seconds since epoch
        inputs_exist = True

        if inputs is not None:
            if isinstance(inputs, basestring):  # check if input is only string and not list
                if not os.path.exists(inputs):
                    inputs_exist = False
                    raise mincError(' ** Error: Input does not exists! :: {}'.format(str(inputs)))
                else:
                    itime = os.path.getmtime(inputs)
            else:
                for i in inputs:
                    if not os.path.exists(i):
                        inputs_exist = False
                        print(' ** Error: One input does not exists! :: {}'.format(i), file=sys.stderr)
                        raise mincError(' ** Error: One input does not exists! :: {}'.format(i))
                    else:
                        timer = os.path.getmtime(i)
                        if timer < itime or itime < 0:
                            itime = timer

        # Check if outputs exist AND is newer than inputs

        outExists = False
        otime = -1
        exists=[]
        if outputs is not None:
            if isinstance(outputs, basestring):
                outExists = os.path.exists(outputs)
                if outExists:
                    otime = os.path.getmtime(outputs)
                    exists.append(outputs)
            else:
                for o in outputs:
                    outExists = os.path.exists(o)
                    if outExists:
                        exists.append(outputs)
                        timer = os.path.getmtime(o)
                        if timer > otime:
                            otime = timer
                    if not outExists:
                        break

        if outExists:
            if timecheck and itime > 0 and otime > 0 and otime < itime:
                if verbose>1:
                    print(' -- Warning: Output exists but older than input! Redoing command',file=sys.stderr)
                    print('     otime ' + str(otime) + ' < itime ' \
                        + str(itime),file=sys.stderr)
                return True
            else:
                if verbose>1:
                    print(' -- Skipping: Output Exists:{}'.format(repr(exists)),file=sys.stderr)
                return False
        return True

    @staticmethod
    def execute(cmds, verbose=1):
        """
        Execute a command line waiting for the end of it
        Arguments:
        cmds: list containg the command line
        
        Keyword arguments:
        verbose: if false no message will appear

        return : False if error, otherwise the execution output
        """
        output_stderr=""
        output=""
        outvalue=0
        if verbose>0:
            print(repr(cmds))
        try:

            if verbose<2:
                with open(os.devnull, "w") as fnull:
                    p=subprocess.Popen(cmds, stdout=fnull, stderr=subprocess.PIPE)
            else:
                p=subprocess.Popen(cmds, stderr=subprocess.PIPE)
            
            (output,output_stderr)=p.communicate()
            outvalue=p.wait()

        except OSError:
            print("ERROR: command {} Error:{}!\nMessage: {}\n{}".format(str(cmds),str(outvalue),output_stderr,traceback.format_exc()), file=sys.stderr)
            raise mincError("ERROR: command {} Error:{}!\nMessage: {}\n{}".format(str(cmds),str(outerr),output_stderr,traceback.format_exc()))
        if not outvalue == 0:
            print("ERROR: command {} failed {}!\nMessage: {}\n{}".format(str(cmds),str(outvalue),output_stderr,traceback.format_exc()), file=sys.stderr)
            raise mincError("ERROR: command {} failed {}!\nMessage: {}\n{}".format(str(cmds),str(outvalue),output_stderr,traceback.format_exc()))
        return outvalue
        
    @staticmethod
    def execute_w_output(cmds, verbose=0):
        """
        Execute a command line waiting for the end of it

        cmds: list containg the command line
        verbose: if false no message will appear

        return : False if error, otherwise the execution output
        """
        output=''
        outvalue=0

        if verbose>0:
            print(repr(cmds))
        try:
            p=subprocess.Popen(cmds,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (output,outerr)=p.communicate()
            if verbose>0:
                print(output.decode())
            outvalue=p.wait()
        except OSError as e:
            print("ERROR: command {} Error:{}!\n{}".format(repr(cmds),str(e),traceback.format_exc()),file=sys.stderr)
            raise mincError("ERROR: command {} Error:{}!\n{}".format(repr(cmds),str(e),traceback.format_exc()))
        if not outvalue == 0:
            print("Command: {} generated output:{} {}\nError:{}".format(' '.join(cmds),outvalue,output,outerr),file=sys.stderr)
            raise mincError("ERROR: command {} failed {}!\nError:{}\n{}".format(repr(cmds),str(outvalue),outerr,traceback.format_exc()))
        return output.decode()

    @staticmethod
    def command(
        cmds,
        inputs=None,
        outputs=None,
        timecheck=False,
        verbose=1,
        ):
        """
        Execute a command line waiting for the end of it, testing inputs and outputs

        cmds:    list containg the command line
        inputs:  list of files to check if they exist before executing command
        outputs: list of files that should be when finishing
        verbose: if 0 no message will appear
        outputlines: store the output as a string 
        timecheck: The command won't be executed if the output exists and is newer than the input file.

        return : False if error, otherwise the execution output
        """

        if verbose>0:
            print(repr(cmds))

        if not mincTools.checkfiles(inputs=inputs, outputs=outputs,
                                    verbose=verbose,
                                    timecheck=timecheck):
            return 0
        outvalue=0
        output_stderr=""
        output=""
        use_shell=not isinstance(cmds, list)
        try:
            if verbose<2:
                with open(os.devnull, "w") as fnull:
                    p=subprocess.Popen(cmds, stdout=fnull, stderr=subprocess.PIPE,shell=use_shell)
            else:
                p=subprocess.Popen(cmds, stderr=subprocess.PIPE,shell=use_shell)
            
            (output,output_stderr)=p.communicate()
            outvalue=p.wait()
            
        except OSError:
            print("ERROR: command {} Error:{}!\nMessage: {}\n{}".format(str(cmds),str(outvalue),output_stderr,traceback.format_exc()), file=sys.stderr)
            raise mincError("ERROR: command {} Error:{}!\nMessage: {}\n{}".format(str(cmds),str(outvalue),output_stderr,traceback.format_exc()))
        if not outvalue == 0:
            print("ERROR: command {} failed {}!\nMessage: {}\n{}".format(str(cmds),str(outvalue),output_stderr,traceback.format_exc()), file=sys.stderr)
            raise mincError("ERROR: command {} failed {}!\nMessage: {}\n{}".format(str(cmds),str(outvalue),output_stderr,traceback.format_exc()))
        
        outExists = False
        if outputs is None:
            outExists = True
        elif isinstance(outputs, basestring):
            outExists = os.path.exists(outputs)
        else:
            for o in outputs:
                outExists = os.path.exists(o)
                if not outExists:
                    break

        if not outExists:
            raise mincError('ERROR: Command didn not produce output: {}!'.format(str(cmds)))

        return outvalue

    @staticmethod
    def qsub(
        comm,
        queue='all.q',
        name=None,
        logfile=None,
        depends=None,
        ):
        """ 
        Send the job into the sge queue
        TODO: improve dependencies and so on
        """

        if not name:
            name = comm[0]
        try:
            qsub_comm = [
                'qsub','-cwd',
                '-N', name,
                '-j', 'y',
                '-V', '-q',
                queue,
                ]
            path = ''
            if logfile:
                path = os.path.abspath(logfile)
                qsub_comm.extend(['-o', path])
            if depends:
                qsub_comm.extend(['-hold_jid', depends])

            print(' - Name    ' + name)
            print(' - Queue   ' + queue)
            print(' - Cmd     ' + ' '.join(comm))
            print(' - logfile ' + path)

            #qsub_comm.append(tmpscript)

            cmds="#!/bin/bash\nhostname\n"
            cmds+=' '.join(comm)+"\n"

            p=subprocess.Popen(qsub_comm,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.STDOUT)

            p.communicate(cmds)
            # TODO: check error code?
        finally:
            pass
            
    @staticmethod
    def qsub_pe(
        comm,
        pe='all.pe',
        slots=1,
        name=None,
        logfile=None,
        depends=None,
        ):
        """ 
        Send the job into the sge queue
        TODO: improve dependencies and so on
        """

        if not name:
            name = comm[0]
        try:
            qsub_comm = [
                'qsub','-cwd',
                '-N', name,
                '-j', 'y',
                '-V', '-pe',
                pe,str(slots)
                ]
            path = ''
            if logfile:
                path = os.path.abspath(logfile)
                qsub_comm.extend(['-o', path])
            if depends:
                qsub_comm.extend(['-hold_jid', depends])

            print(' - Name    ' + name)
            print(' - PE      ' + pe)
            print(' - Slots   ' + str(slots))
            print(' - Cmd     ' + ' '.join(comm))
            print(' - logfile ' + path)

            cmds="#!/bin/bash\nhostname\n"
            cmds+=' '.join(comm)+"\n"

            p=subprocess.Popen(qsub_comm,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.STDOUT)

            p.communicate(cmds)
            # TODO: check error code?
        finally:
            pass

    @staticmethod
    def query_dimorder(input):
        '''read a value of an attribute inside minc file'''

        i = subprocess.Popen(['mincinfo', '-vardims', 'image', input],
                             stdout=subprocess.PIPE).communicate()
        return i[0].decode().rstrip('\n').split(' ')
    
    @staticmethod
    def query_attribute(input, attribute):
        '''read a value of an attribute inside minc file'''

        i = subprocess.Popen(['mincinfo', '-attvalue', attribute,
                             input],
                             stdout=subprocess.PIPE).communicate()
        return i[0].decode().rstrip('\n').rstrip(' ')
        
    @staticmethod
    def set_attribute(input, attribute, value):
        '''set a value of an attribute inside minc file
        if value=None - delete the attribute
        '''
        if value is None:
            mincTools.execute(['minc_modify_header', input, '-delete', attribute])
        elif isinstance(value, basestring):
            mincTools.execute(['minc_modify_header', input, '-sinsert', attribute + '='
                    + value])
        else:
            # assume that it's a number
            mincTools.execute(['minc_modify_header', input, '-dinsert', attribute + '='
                    + str(value)])

    @staticmethod
    def mincinfo(input):
        """read a basic information about minc file
        Arguments:
            input -- input minc file
        Returns dict with entries per dimension
        """
        # TODO: make this robust to errors!
        _image_dims = subprocess.Popen(['mincinfo', '-vardims', 'image', input],
                             stdout=subprocess.PIPE).communicate()[0].decode().rstrip('\n').rstrip(' ').split(' ')
        
        _req=['mincinfo']
        for i in _image_dims:
            _req.extend(['-dimlength',i,
                         '-attvalue', '{}:start'.format(i),
                         '-attvalue', '{}:step'.format(i), 
                         '-attvalue', '{}:direction_cosines'.format(i)])
        _req.append(input)
        _info= subprocess.Popen(_req,
                             stdout=subprocess.PIPE).communicate()[0].decode().rstrip('\n').rstrip(' ').split("\n")

        diminfo=collections.namedtuple('dimension',['length','start','step','direction_cosines'])
        
        _result={}
        for i,j in enumerate(_image_dims):
            _result[j]=diminfo(length=int(_info[i*4]),
                               start=float(_info[i*4+1]),
                               step=float(_info[i*4+2]),
                               direction_cosines=[float(k) for k in _info[i*4+3].rstrip(' ').split(' ') ])
            
        return _result
        
    def ants_linear_register(
        self,
        source,
        target,
        output_xfm, 
        **kwargs
        ):
        """perform linear registration with ANTs, obsolete"""
        return ants_registration.ants_linear_register(source,target,output_xfm,**kwargs)
                                                 

    def linear_register(
        self,
        source,
        target,
        output_xfm,
        **kwargs
        ):
        """perform linear registration"""

        return registration.linear_register(source,target,output_xfm,**kwargs)
    
    def linear_register_to_self(
        self,
        source,
        target,
        output_xfm,
        **kwargs
        ):
        """perform linear registration"""
        
        return registration.linear_register_to_self(source,target,output_xfm,**kwargs)
    
    def nl_xfm_to_elastix(self , xfm, elastix_par):
        """Convert MINC style xfm into elastix style registration parameters"""
        return elastix_registration.nl_xfm_to_elastix(sfm,elastix_par)

    def nl_elastix_to_xfm(self , elastix_par, xfm, **kwargs ):
        """Convert elastix style parameter file into a nonlinear xfm file"""
        return elastix_registration.nl_elastix_to_xfm(elastix_par,xfm,**kwargs)

    def register_elastix( self, source, target, **kwargs ):
        """Perform registration with elastix """
        return elastix_registration.register_elastix(source,target,**kwargs)

    def non_linear_register_ants(
        self, source, target, output_xfm, **kwargs
        ):
        """perform non-linear registration using ANTs, 
        WARNING: will create inverted xfm  will be named output_invert.xfm
        """
        return ants_registration.non_linear_register_ants(source, target, output_xfm, **kwargs)

    def non_linear_register_ldd(
        self,
        source, target,
        output_velocity,
        **kwargs    ):
        """Use log-diffeomorphic demons to run registration"""
        return dd_registration.non_linear_register_ldd(source,target,output_velocity,**kwargs)
        
    def non_linear_register_full(
        self, 
        source, target, output_xfm,
        **kwargs
        ):
        """perform non-linear registration"""
        return registration.non_linear_register_full(source,target,output_xfm,**kwargs)

    def non_linear_register_increment(
        self, source, target, output_xfm,** kwargs
        ):
        """perform incremental non-linear registration"""
        return registration.non_linear_register_increment(source, target, output_xfm,** kwargs)
        
    def resample_smooth(
        self,
        input,
        output,
        transform=None,
        like=None,
        order=4,
        uniformize=None,
        unistep=None,
        invert_transform=False,
        resample=None,
        datatype=None,
        tfm_input_sampling=False,
        labels=False
        ):
        """resample an image, interpreting voxels as intnsities 
        
        Arguments:
        input -- input minc file
        output -- output minc file
        transform -- (optional) transformation file
        like -- (optional) reference file for sampling
        order -- interpolation order for B-Splines , default 4
        uniformize -- (optional) uniformize volume to have identity direction 
                      cosines and uniform sampling
        unistep -- (optional) resample volume to have uniform steps
        invert_transform  -- invert input transform, default False
        resample  -- (optional) resample type, variants: 
                     'sinc', 'linear', 'cubic','nearest' - mincresample
                     otherwise use itk_resample
        datatype -- output minc file data type, variants 
                     'byte','short','long','float','double'
        labels   -- assume scan contains integer labels, only works with nearest neignour
        tfm_input_sampling -- apply linear xfm to sampling parameters, assumes mincresample is used
        """
        if os.path.exists(output):
            return

        if not resample:
            resample = self.resample
        if resample == 'sinc':
            cmd = ['mincresample', input, output, '-sinc', '-q']
            if transform:
                cmd.extend(['-transform', transform])
            if like:
                cmd.extend(['-like', like])
            elif tfm_input_sampling:
                cmd.append('-tfm_input_sampling')
            else:
                cmd.append('-use_input_sampling')
            if invert_transform:
                cmd.append('-invert_transform')
            if uniformize:
                raise mincError('Not implemented!')
            if datatype:
                cmd.append('-' + datatype)
            self.command(cmd, inputs=[input], outputs=[output])
        elif resample == 'linear':
            cmd = ['mincresample', input, output, '-trilinear', '-q']
            if transform:
                cmd.extend(['-transform', transform])
            if like:
                cmd.extend(['-like', like])
            elif tfm_input_sampling:
                cmd.append('-tfm_input_sampling')
            else:
                cmd.append('-use_input_sampling')
            if invert_transform:
                cmd.append('-invert_transform')
            if uniformize:
                raise mincError('Not implemented!')
            if datatype:
                cmd.append('-' + datatype)
            self.command(cmd, inputs=[input], outputs=[output])
        elif resample == 'cubic':
            cmd = ['mincresample', input, output, '-tricubic', '-q']
            if transform:
                cmd.extend(['-transform', transform])
            if like:
                cmd.extend(['-like', like])
            elif tfm_input_sampling:
                cmd.append('-tfm_input_sampling')
            else:
                cmd.append('-use_input_sampling')
            if invert_transform:
                cmd.append('-invert_transform')
            if uniformize:
                raise mincError('Not implemented!')
            if datatype:
                cmd.append('-' + datatype)
            self.command(cmd, inputs=[input], outputs=[output], verbose=self.verbose)
        elif resample == 'nearest' or labels:
            cmd = ['mincresample', input, output, '-nearest', '-q']
            if transform:
                cmd.extend(['-transform', transform])
            if like:
                cmd.extend(['-like', like])
            elif tfm_input_sampling:
                cmd.append('-tfm_input_sampling')
            else:
                cmd.append('-use_input_sampling')
            if invert_transform:
                cmd.append('-invert_transform')
            if uniformize:
                raise mincError('Not implemented!')
            if datatype:
                cmd.append('-' + datatype)
            if labels:
                cmd.append('-labels')
            self.command(cmd, inputs=[input], outputs=[output], verbose=self.verbose)
        else:
            cmd = ['itk_resample', input, output, '--order', str(order)]
            if transform:
                cmd.extend(['--transform', transform])
            if like:
                cmd.extend(['--like', like])
            if invert_transform:
                cmd.append('--invert_transform')
            if uniformize:
                cmd.extend(['--uniformize', str(uniformize)])
            if unistep:
                cmd.extend(['--unistep', str(unistep)])
            if datatype:
                cmd.append('--' + datatype)
            self.command(cmd, inputs=[input], outputs=[output], verbose=self.verbose)

    def resample_labels(
        self,
        input,
        output,
        transform=None,
        like=None,
        invert_transform=False,
        order=None,
        datatype=None,
        remap=None,
        aa=None,
        baa=False,
        uniformize=None,
        unistep=None,
        ):
        """resample an image with discrete labels"""
        if datatype is None:
            datatype='byte'

        cmd = ['itk_resample', input, output, '--labels']
        
        
        if remap is not None:
            if isinstance(remap, list): 
                remap=dict(remap)
            
            if isinstance(remap, dict):
                if any(remap):
                    _remap=""
                    for (i,j) in remap.items(): _remap+='{} {};'.format(i,j)
                    cmd.extend(['--lut-string', _remap ])
            else:
                cmd.extend(['--lut-string', str(remap) ])
        if transform is not None:
            cmd.extend(['--transform', transform])
        if like is not None:
            cmd.extend(['--like', like])
        if invert_transform:
            cmd.append('--invert_transform')
        if order is not None:
            cmd.extend(['--order',str(order)])
        if datatype is not None:
            cmd.append('--' + datatype)
        if aa is not None:
            cmd.extend(['--aa',str(aa)])
        if baa :
            cmd.append('--baa')
        if uniformize:
            cmd.extend(['--uniformize', str(uniformize)])
        if unistep:
            cmd.extend(['--unistep', str(unistep)])
            
        self.command(cmd, inputs=[input], outputs=[output], verbose=self.verbose)


    def resample_smooth_logspace(
        self,
        input,
        output,
        velocity=None,
        like=None,
        order=4,
        invert_transform=False,
        datatype=None,
        ):
        """resample an image """
        if os.path.exists(output):
            return

        cmd = ['log_resample', input, output, '--order', str(order)]
        if velocity:
            cmd.extend(['--log_transform', velocity])
        if like:
            cmd.extend(['--like', like])
        if invert_transform:
            cmd.append('--invert_transform')
        if datatype:
            cmd.append('--' + datatype)
        self.command(cmd, inputs=[input], outputs=[output], verbose=self.verbose)

    def resample_labels_logspace(
        self,
        input,
        output,
        velocity=None,
        like=None,
        invert_transform=False,
        order=None,
        datatype=None,
        ):
        """resample an image with discrete labels"""
        if datatype is None:
            datatype='byte'

        cmd = ['log_resample', input, output, '--labels']


        if velocity is not None:
            cmd.extend(['--log_transform', velocity])
        if like is not None:
            cmd.extend(['--like', like])
        if invert_transform:
            cmd.append('--invert_transform')
        if order is not None:
            cmd.extend(['--order',str(order)])
        if datatype is not None:
            cmd.append('--' + datatype)

        self.command(cmd, inputs=[input], outputs=[output], verbose=self.verbose)

        
    def xfminvert(self, input, output):
        """invert transformation"""

        self.command(['xfminvert', input, output], inputs=[input],
                     outputs=[output],verbose=self.verbose)

    def xfmavg(
        self,
        inputs,
        output,
        nl=False,
        ):
        """average transformations"""

        cmd = ['xfmavg']
        cmd.extend(inputs)
        cmd.append(output)
        if nl:
            cmd.append('-ignore_linear')
        self.command(cmd, inputs=inputs, outputs=[output], verbose=self.verbose)


    def xfmconcat(self, inputs, output):
        """concatenate transformations"""

        cmd = ['xfmconcat']
        cmd.extend(inputs)
        cmd.append(output)
        self.command(cmd, inputs=inputs, outputs=[output], verbose=self.verbose)


    def xfm_v0_scaling(self, inputs, output):
        """concatenate transformations"""

        cmd = ['xfm_v0_scaling.pl']
        cmd.extend(inputs)
        cmd.append(output)
        self.command(cmd, inputs=inputs, outputs=[output], verbose=self.verbose)


    def average(
        self,
        inputs,
        output,
        sdfile=None,
        datatype=None,
        ):
        """average images"""

        cmd = ['mincaverage', '-q', '-clob']
        cmd.extend(inputs)
        cmd.append(output)

        if sdfile:
            cmd.extend(['-sdfile', sdfile])
        if datatype:
            cmd.append(datatype)
        cmd.extend(['-max_buffer_size_in_kb', '1000000', '-copy_header'])
        self.command(cmd, inputs=inputs, outputs=[output], verbose=self.verbose)

    def median(
        self,
        inputs,
        output,
        madfile=None,
        datatype=None,
        ):
        """average images"""

        cmd = ['minc_median', '--clob']
        cmd.extend(inputs)
        cmd.append(output)

        if madfile:
            cmd.extend(['--mad', madfile])
        if datatype:
            cmd.append(datatype)
            
        self.command(cmd, inputs=inputs, outputs=[output], verbose=self.verbose)


    def calc(
        self,
        inputs,
        expression,
        output,
        datatype=None,
        labels=False
        ):
        """apply mathematical expression to image(s)"""

        cmd = ['minccalc', '-copy_header','-q', '-clob', '-express', expression]
        
        if datatype:
            cmd.append(datatype)
        if labels:
            cmd.append('-labels')
        
        cmd.extend(inputs)
        cmd.append(output)
        
        self.command(cmd, inputs=inputs, outputs=[output], verbose=self.verbose)
        
    def math(
        self,
        inputs,
        operation,
        output,
        datatype=None,
        labels=False
        ):
        """apply mathematical operation to image(s)"""

        cmd = ['mincmath', '-q', '-clob', '-copy_header', '-'+operation]
        
        if datatype:
            cmd.append(datatype)
        if labels:
            cmd.append('-labels')
        cmd.extend(inputs)
        cmd.append(output)
        
        self.command(cmd, inputs=inputs, outputs=[output], verbose=self.verbose)


    def stats(self, input, 
              stats, mask=None,
              mask_binvalue=1,
              val_floor=None,
              val_ceil=None,
              val_range=None,
              single_value=True):
        args=['mincstats',input,'-q']
        
        if isinstance(stats, list): 
            args.extend(stats)
        else:
            args.append(stats)
        
        if mask is not None:
            args.extend(['-mask',mask,'-mask_binvalue',str(mask_binvalue)])
        if val_floor is not None:
            args.extend(['-floor',str(val_floor)])
        if val_ceil is not None:
            args.extend(['-ceil',str(val_ceil)])
        if val_range is not None:
            args.extend(['-range',str(val_range[0]),str(val_range[1])])
            
        r=self.execute_w_output(args,verbose=self.verbose)
        if single_value :
            return float(r)
        else:
            return [float(i) for i in r.split(' ')]

    def similarity(self, reference, sample, ref_mask=None, sample_mask=None,method="msq"):
        """Calculate image similarity metric"""
        args=['itk_similarity',reference,sample,'--'+method]

        if ref_mask is not None:
            args.extend(['--src_mask',ref_mask])
        if sample_mask is not None:
            args.extend(['--target_mask',sample_mask])

        r=self.execute_w_output(args,verbose=self.verbose)
        return float(r)

    def label_similarity(self, reference, sample, method="gkappa"):
        """Calculate image similarity metric"""
        args=['volume_gtc_similarity',reference, sample,'--'+method]
        r=self.execute_w_output(args,verbose=self.verbose)
        return float(r)
        
    def noise_estimate(self, input, mask=None):
        '''Estimate file noise (absolute)'''
        args=['noise_estimate',input]
        if mask is not None:
            args.extend(['--mask',mask])
        r=self.execute_w_output(args,verbose=self.verbose)
        return float(r)
    
    def snr_estimate(self, input, mask=None):
        '''Estimate file SNR'''
        args=['noise_estimate',input,'--snr']
        if mask is not None:
            args.extend(['--mask',mask])
        r=self.execute_w_output(args,verbose=self.verbose)
        return float(r)

    def log_average(self, inputs, output):
        """perform log-average (geometric average)"""
        tmp = ['log(A[%d])' % i for i,_ in enumerate(inputs)]
        self.calc(inputs, 'exp((%s)/%d)' % ('+'.join(tmp), len(inputs)), 
                  output, datatype='-float')


    def param2xfm(self, output, scales=None, translation=None, rotations=None, shears=None):
        cmd = ['param2xfm','-clobber',output]

        if translation is not None:
            cmd.extend(['-translation',str(translation[0]),str(translation[1]),str(translation[2])])
        if rotations is not None:
            cmd.extend(['-rotations',str(rotations[0]),str(rotations[1]),str(rotations[2])])
        if scales is not None:
            cmd.extend(['-scales',str(scales[0]),str(scales[1]),str(scales[2])])
        if shears is not None:
            cmd.extend(['-shears',str(shears[0]),str(shears[1]),str(shears[2])])
        self.command(cmd, inputs=[], outputs=[output], verbose=self.verbose)


    def flip_volume_x(self,input,output, labels=False, datatype=None):
        '''flip along x axis'''
        if not os.path.exists(self.tmp('flip_x.xfm')):
            self.param2xfm(self.tmp('flip_x.xfm'),
                           scales=[-1.0,1.0,1.0])
        if labels:
            self.resample_labels(input,output,order=0,transform=self.tmp('flip_x.xfm'),datatype=datatype)
        else:
            self.resample_smooth(input,output,order=0,transform=self.tmp('flip_x.xfm'),datatype=datatype)


    def volume_pol(
        self,
        source,target,
        output,
        source_mask=None,
        target_mask=None,
        order=1,
        expfile=None,
        datatype=None,
        ):
        """normalize intensities"""

        if (expfile is None or os.path.exists(expfile) ) and os.path.exists(output):
            return

        rm_expfile = False
        if not expfile:
            expfile = self.temp_file(suffix='.exp')
            rm_expfile = True
        try:
            cmd = ['volume_pol',
                source, target,
                '--order',  str(order),
                '--expfile', expfile,
                '--noclamp','--clob', ]
            if source_mask:
                cmd.extend(['--source_mask', source_mask])
            if target_mask:
                cmd.extend(['--target_mask', target_mask])
            self.command(cmd, inputs=[source, target],
                         outputs=[expfile], verbose=self.verbose)
            exp = open(expfile).read().rstrip()
            cmd = ['minccalc', '-q' ,'-expression', exp, source, output]
            if datatype:
                cmd.append(datatype)
            self.command(cmd, inputs=[source, target], outputs=[output], verbose=self.verbose)
        finally:
            if rm_expfile and os.path.exists(expfile):
                os.unlink(expfile)

    def nuyl_normalize(
        self,
        source,target,
        output,
        source_mask=None,
        target_mask=None,
        linear=False,
        steps=10
        ):
        """normalize intensities
        Arguments:
        source - input image 
        target - reference image
        output - output image
        
        Optional Arguments:
        souce_mask - input image mask (used for calculating intensity mapping)
        target_mask - reference image mask
        linear - use linear intensity model (False)
        steps - number of steps in linear-piece-wise approximatation (10)
        
        """
        cmd = ['minc_nuyl', source, target,'--clob', output,'--steps',str(steps) ]
        if source_mask:
            cmd.extend(['--source-mask', source_mask])
        if target_mask:
            cmd.extend(['--target-mask', target_mask])
        if linear:
            cmd.append('--linear')

        self.command(cmd, inputs=[source, target],
                        outputs=[output], verbose=self.verbose)
        

    def nu_correct(
        self,
        input,
        output_imp=None,
        output_field=None,
        output_image=None,
        mask=None,
        mri3t=False,
        normalize=False,
        distance=None,
        downsample_field=None,
        datatype=None
        ):
        """apply N3"""

        if (output_image is None or os.path.exists(output_image)) and \
           (output_imp   is None or os.path.exists(output_imp)) and \
           (output_field is None or os.path.exists(output_field)):
               return

        output_imp_ = output_imp

        if not output_imp_  is not None:
            output_imp_ = self.temp_file(suffix='.imp')
            
        if output_field is not None:
            output_field_tmp=self.temp_file(suffix='.mnc')
            
        output_image_ = output_image

        if not output_image_:
            output_image_ = self.temp_file(suffix='nuc.mnc')

        cmd = [
            'nu_estimate',
            '-stop', '0.00001',
            '-fwhm', '0.1',
            '-iterations','1000',
            input, output_imp_,
            ]

        if normalize:
            cmd.append('-normalize_field')

        if mask is not None:
            cmd.extend(['-mask', mask])

        if distance is not None:
            cmd.extend(['-distance', str(distance)])
        elif mri3t:
            cmd.extend(['-distance', '50'])

        try:
            self.command(cmd, inputs=[input], outputs=[output_imp_], 
                         verbose=self.verbose)

            cmd=['nu_evaluate', 
                 input, '-mapping',
                 output_imp_,
                 output_image_]

            if mask is not None:
                cmd.extend(['-mask', mask])

            if output_field is not None:
                cmd.extend(['-field', output_field_tmp] )

            self.command(cmd,inputs=[input], outputs = [output_image_],
                         verbose=self.verbose)
            
            if output_field is not None:
                self.resample_smooth(output_field_tmp, output_field, datatype=datatype,unistep=downsample_field)
                
        finally:
            if output_imp is None :
                os.unlink(output_imp_)
            if output_image is None :
                os.unlink(output_image_)

    def n4(self, input, 
           output_corr=None, output_field=None,
           mask=None,        distance=200, 
           shrink=None,      weight_mask=None,
           datatype=None,    iter=None,
           sharpening=None,  threshold=None,
           downsample_field =None
           ):
        
        outputs=[]
        if output_corr is not None:
            outputs.append(output_corr)
        
        if output_field is not None:
            outputs.append(output_field)
        
        if not self.checkfiles(inputs=[input],outputs=outputs,
                                    verbose=self.verbose):
            return

        _out=self.temp_file(suffix='.mnc')
        _out_fld=self.temp_file(suffix='.mnc')
        
        cmd=[ 'N4BiasFieldCorrection', '-d', '3',
              '-i', input,'--rescale-intensities', '1',
              '--bspline-fitting', str(distance),
              '--output','[{},{}]'.format(_out,_out_fld) ]
        
        if mask is not None:
            cmd.extend(['--mask-image',mask])
        if weight_mask is not None:
            cmd.extend(['--weight-image',weight_mask])
        if shrink is not None:
            cmd.extend(['--shrink-factor',str(shrink)])
        if iter is not None:
            if threshold is None: threshold=0.0
            cmd.extend(['--convergence','[{}]'.format( ','.join([str(iter),str(threshold)]) )])
        if sharpening is not None:
            cmd.extend(['--histogram-sharpening','[{}]'.format(str(sharpening))])
        self.command(cmd,inputs=[input],outputs=[_out,_out_fld],
                    verbose=self.verbose)
        
        if output_corr is not None:
            if datatype is not None:
                self.reshape(_out, output_corr, datatype=datatype)
                os.unlink(_out)
            else:
                shutil.move(_out, output_corr)
        
        if output_field is not None:
            if downsample_field is not None:
                self.resample_smooth(_out_fld, output_field, datatype=datatype, unistep=downsample_field)
            else:
                if datatype is not None:
                    self.reshape(_out_fld, output_field, datatype=datatype)
                    os.unlink(_out_fld)
                else:
                    shutil.move(_out_fld, output_field)

    def difference_n4(
        self,
        input,
        model,
        output,
        mask=None,
        distance=None,
        iter=None
        ):

        diff = self.temp_file(suffix='.mnc')
        _output = self.temp_file(suffix='_out.mnc')
        try:
            if mask:
                self.calc([input, model, mask],
                          'A[2]>0.5?A[0]-A[1]+100:0', diff)
            else:
                self.calc([input, model], 
                          'A[0]-A[1]+100', diff)
            
            self.n4(diff, mask=mask, output_field=_output, 
                    distance=distance,
                    iter=iter)
            # fix , because N4 doesn't preserve dimension order
            self.resample_smooth(_output, output, like=diff)
            
        finally:
            os.unlink(diff)
            os.unlink(_output)
            
          
    def apply_fld(self,input,fld,output):
        '''Apply inhomogeniety correction field'''
        _res_fld=self.temp_file(suffix='.mnc')
        if not self.checkfiles(inputs=[input],outputs=[output],
                                verbose=self.verbose):
            return
        try:
            self.resample_smooth(fld, _res_fld, like=input,order=1)
            self.calc([input, _res_fld], 
                        'A[1]>0.0?A[0]/A[1]:A[0]', output)
        finally:
            os.unlink(_res_fld)
            
        
        
    def apply_n3_vol_pol(
        self,
        input,
        model,
        output,
        source_mask=None,
        target_mask=None,
        bias=None,
        ):

        intermediate = input
        try:
            if bias:
                intermediate = self.temp_file(suffix='.mnc')
                self.calc([input, bias],
                          'A[1]>0.5&&A[1]<1.5?A[0]/A[1]:A[0]',
                          intermediate, datatype='-float')
            self.volume_pol(
                intermediate,
                model,
                output,
                source_mask=source_mask,
                target_mask=target_mask,
                datatype='-short',
                )
        finally:
            if bias:
                os.unlink(intermediate)

    def difference_n3(
        self,
        input,
        model,
        output,
        mask=None,
        mri3t=False,
        distance=None,
        normalize=True,
        ):

        diff = self.temp_file(suffix='.mnc')

        try:
            if mask:
                self.calc([input, model, mask],
                          'A[2]>0.5?A[0]-A[1]+100:0', diff)
            else:
                self.calc([input, model], 
                          'A[0]-A[1]+100', diff)

            self.nu_correct(diff, mask=mask, output_field=output, 
                                  mri3t=mri3t, distance=distance, 
                                  normalize=normalize)
        finally:
            os.unlink(diff)


    def xfm_normalize(
        self, input,
        like, output,
        step=None,
        exact=False,
        invert=False,
        ):

        # TODO: convert xfm_normalize.pl to python
        cmd = ['xfm_normalize.pl', input, '--like', like, output]
        if step:
            cmd.extend(['--step', str(step)])
        if exact:
            cmd.extend(['--exact'])
        if invert:
            cmd.extend(['--invert'])

        self.command(cmd, inputs=[input, like], outputs=[output], verbose=self.verbose)


    def xfm_noscale(self, input, output, unscale=None):
        """remove scaling from linear part of XFM"""

        scale = self.temp_file(suffix='scale.xfm')
        _unscale=unscale
        if unscale is None:
            _unscale = self.temp_file(suffix='unscale.xfm')
        try:
            (out, err) = subprocess.Popen(['xfm2param', input],
                    stdout=subprocess.PIPE).communicate()
            scale_ = list(filter(lambda x: re.match('^\-scale', x),
                            out.decode().split('\n')))
            if len(scale_) != 1:
                raise mincError("Can't extract scale from " + input)
            scale__ = re.split('\s+', scale_[0])
            cmd = ['param2xfm']
            cmd.extend(scale__)
            cmd.extend([scale])
            self.command(cmd, verbose=self.verbose)
            self.xfminvert(scale, _unscale)
            self.xfmconcat([input, _unscale], output)
        finally:
            if os.path.exists(scale):
                os.unlink(scale)
            if unscale!=_unscale and os.path.exists(_unscale):
                os.unlink(_unscale)


    def blur(
        self,
        input,
        output,
        fwhm,
        gmag=False,
        dx=False,
        dy=False,
        dz=False,
        output_float=False,
        ):
        """Apply gauissian blurring to the input image"""

        cmd = ['fast_blur', input, output, '--fwhm', str(fwhm)]
        if gmag:
            cmd.append('--gmag')
        if dx:
            cmd.append('--dx')
        if dy:
            cmd.append('--dy')
        if dz:
            cmd.append('--dz')
        if output_float:
            cmd.append('--float')
        self.command(cmd, inputs=[input], outputs=[output], verbose=self.verbose)

    def blur_orig(
        self,
        input,
        output,
        fwhm,
        gmag=False,
        dx=False,
        dy=False,
        dz=False,
        output_float=False,
        ):
        """Apply gauissian blurring to the input image"""
        with temp_files() as tmp:
            p=tmp.tmp('blur_orig')
            cmd = ['mincblur', input, p, '-fwhm', str(fwhm),'-no_apodize']
            if gmag:
                cmd.append('-gradient')
            if output_float:
                cmd.append('-float')
            self.command(cmd, inputs=[input], outputs=[p+'_blur.mnc'], verbose=2)
            
            if gmag:
                shutil.move(p+'_dxyz.mnc',output)
            else:
                shutil.move(p+'_blur.mnc',output)


    def blur_vectors(
        self,
        input,
        output,
        fwhm,
        gmag=False,
        output_float=False,
        dim=3
        ):
        """Apply gauissian blurring to the input vector field """
        
        if not self.checkfiles(inputs=[input], outputs=[output],
                                    verbose=self.verbose):
            return
        
        with temp_files() as tmp:
            b=[]
            dimorder=self.query_dimorder(input)
            for i in range(dim):
                self.reshape(input,tmp.tmp(str(i)+'.mnc'),dimrange='vector_dimension={}'.format(i))
                self.blur(tmp.tmp(str(i)+'.mnc'),tmp.tmp('blur_'+str(i)+'.mnc'),fwhm=fwhm,output_float=output_float,gmag=gmag)
                b.append(tmp.tmp('blur_'+str(i)+'.mnc'))
            # assemble
            cmd=['mincconcat','-concat_dimension','vector_dimension','-quiet']
            cmd.extend(b)
            cmd.append(tmp.tmp('output.mnc'))
            self.command(cmd,inputs=b,outputs=[],verbose=self.verbose)
            self.command(['mincreshape','-dimorder',','.join(dimorder),tmp.tmp('output.mnc'),output,'-quiet'],
                         inputs=[],outputs=[output],verbose=self.verbose)
        # done
            
            
    def nlm(self,
        input,output,
        beta=0.7,
        patch=3,
        search=1,
        sigma=None,
        datatype=None,
        ):
        
        if sigma is None:
            sigma=self.noise_estimate(input)
            
        cmd=['itk_minc_nonlocal_filter',
              input, output,
              #'--beta', str(beta), 
              '--patch',str(patch),
              '--search',str(search) 
          ]
        
        cmd.extend(['--sigma',str(sigma*beta)])
        if datatype   is not None: cmd.append('--' + datatype)

        self.command(cmd,
            inputs=[input], outputs=[output], verbose=self.verbose)


    def anlm(self,
        input, output,
        beta=0.7,
        patch=None,
        search=None,
        regularize=None,
        datatype=None,
        ):
        cmd=['itk_minc_nonlocal_filter', '--clobber', '--anlm',
            input, output,'--beta', str(beta),]
        
        if patch      is not None: cmd.extend(['--patch',     str(patch)]     )
        if search     is not None: cmd.extend(['--search',    str(search)]    )
        if regularize is not None: cmd.extend(['--regularize',str(regularize)])
        if datatype   is not None: cmd.append('--' + datatype)
        
        self.command(cmd,  inputs=[input], outputs=[output], verbose=self.verbose)


    def qc(
        self,
        input,
        output,
        image_range=None,
        mask=None,
        mask_range=None,
        title=None,
        labels=False,
        labels_mask=False,
        spectral_mask=False,
        big=False,
        clamp=False,
        bbox=False,
        discrete=False,
        discrete_mask=False,
        red=False,
        green_mask=False,
        cyanred=False,
        cyanred_mask=False,
        mask_lut=None
        ):
        """
        Generate QC image
        """

        cmd = ['minc_qc.pl', input, output, '--verbose']

        if image_range is not None:
            cmd.extend(['--image-range', str(image_range[0]),
                       str(image_range[1])])
        if mask is not None:
            cmd.extend(['--mask', mask])
        if mask_range is not None:
            cmd.extend(['--mask-range', str(mask_range[0]),
                       str(mask_range[1])])
        if title is not None:
            cmd.extend(['--title', title])
        if labels:
            cmd.append('--labels')
        if labels_mask:
            cmd.append('--labels-mask')
        if spectral_mask:
            cmd.append('--spectral-mask')
        if big:
            cmd.append('--big')
        if clamp:
            cmd.append('--clamp')
        if bbox:
            cmd.append('--bbox')
        if labels:
            cmd.append('--labels')
        if labels_mask:
            cmd.append('--labels-mask')
        if discrete:
            cmd.append('--discrete')
        if discrete_mask:
            cmd.append('--discrete-mask')
        if red:
            cmd.append('--red')
        if green_mask:
            cmd.append('--green-mask')
        if cyanred:
            cmd.append('--cyanred')
        if cyanred_mask:
            cmd.append('--cyanred-mask')
        if mask_lut is not None:
            cmd.extend(['--mask-lut',mask_lut])
        self.command(cmd, inputs=[input], outputs=[output], verbose=self.verbose)

    def aqc(
        self,
        input,
        output_prefix,
        slices=3
        ):

        cmd = ['minc_aqc.pl', input, output_prefix, '--slices', str(slices) ]

        self.command(cmd, inputs=[input], outputs=[output_prefix+'_0.jpg'], verbose=self.verbose)


    def grid_determinant(
        self,
        input,
        output,
        datatype=None
        ):
        """
        Calculate jacobina determinant using grid file
        """
        cmd=['grid_proc','--det',input,output]
        if datatype is not None:
            cmd.append('--'+datatype)
        self.command(cmd, inputs=[input], outputs=[output], verbose=self.verbose)

    def grid_2_log(
        self,
        input,
        output,
        datatype=None,
        exp=False,
        factor=None,
        ):
        cmd=['grid_2_log',input,output]
        if datatype is not None:
            cmd.append('--'+datatype)
        if exp:
            cmd.append('--exp')
        if factor is not None:
            cmd.extend(['--factor',str(factor)])
        self.command(cmd, inputs=[input], outputs=[output], verbose=self.verbose)

    def grid_magnitude(
        self,
        input,
        output,
        datatype=None
        ):
        cmd=['grid_proc','--mag',input,output]
        if datatype is not None:
            cmd.append('--'+datatype)
        self.command(cmd, inputs=[input], outputs=[output], verbose=self.verbose)

    def reshape(
        self,
        input,
        output,
        normalize=False,
        datatype=None,
        image_range=None,
        valid_range=None,
        dimorder=None,
        signed=False,
        unsigned=False,
        dimrange=None
        ):
        """reshape minc files, #TODO add more options to fully support mincreshape"""
        if signed and unsigned:
            raise mincError('Attempt to reshape file to have both signed and unsigned datatype')
        cmd = ['mincreshape', input, output, '-q']
        if image_range:
            cmd.extend(['-image_range', str(image_range[0]),
                       str(image_range[1])])
        if valid_range:
            cmd.extend(['-valid_range', str(image_range[0]),
                       str(image_range[1])])
        if dimorder:
            cmd.extend(['-dimorder', ','.join(dimorder)])
        if datatype:
            cmd.append('-' + datatype)
        if normalize:
            cmd.append('-normalize')
        if signed:
            cmd.append('-signed')
        if unsigned:
            cmd.append('-unsigned')
        if dimrange is not None:
            if type(dimrange) is list:
                [ cmd.extend(['-dimrange',i]) for i in dimrange ]
            else:
                cmd.extend(['-dimrange',dimrange])
                
        self.command(cmd, inputs=[input], outputs=[output], verbose=self.verbose)

    def split_labels(self, input, output_prefix, normalize=True, lut=None, aa=True, expit=4.0):
        '''split a miltilabel file into a set of files possibly with anti-aliasing filter applied'''
        output_file_pattern=output_prefix+'_%03d.mnc'
        cmd=['itk_split_labels',input,output_file_pattern]
        if aa : cmd.append('--aa')
        if expit> 0: cmd.extend(['--expit',expit])
        if normalize: cmd.append('--normalize')
        if lut is not None:
            if isinstance(lut, list): 
                lut=dict(remap)
            
            if isinstance(lut, dict):
                if any(lut):
                    _lut=""
                    for (i,j) in lut.items(): _lut+='{} {};'.format(i,j)
                    cmd.extend(['--lut-string', _lut ])
            else:
                cmd.extend(['--lut-string', str(lut) ])
        # TODO: figure out how to effectively predict output file names
        # and report it to the calling programm
        out_=self.execute_w_output(cmd).split("\n")
        return dict( [int(i[0]),i[1]] for i in [ j.split(',') for j in out_] )

    def merge_labels(self,input,output):
        '''merge labels using voting'''
        try:
            data_type='--byte'
            inputs=[self.temp_file(suffix='merge.csv')]
            with open(self.temp_file(suffix='merge.csv'),'w') as f:
                for (i,j) in input.items():
                    f.write("{},{}\n".format(i,j))
                    inputs.append(j)
                    if int(i)>255: data_type='--short'

            cmd=['itk_merge_labels', '--csv', self.temp_file(suffix='merge.csv'), output, '--clob', data_type]
            self.command(cmd,inputs=inputs,outputs=[output])
        finally:
            os.unlink(self.temp_file(suffix='merge.csv'))

    def label_stats(self, input, 
                    bg=False, 
                    label_defs=None, 
                    volume=None, 
                    median=False, 
                    mask=None):
        ''' calculate label statistics : label_id, volume, mx, my, mz,[mean/median] '''
        _label_file=label_defs
        cmd=['itk_label_stats',input]
        if bg: cmd.append('--bg')
        if label_defs is not None: 
            if isinstance(label_defs, list) :
                _label_file=self.temp_file(suffix='.csv')
                with open(_label_file,'w') as f:
                    for i in label_defs:
                        f.write("{},{}\n".format(i[0],i[1]))
            elif isinstance(label_defs, dict) :
                _label_file=self.temp_file(suffix='.csv')
                with open(_label_file,'w') as f:
                    for i, j in label_defs.items():
                        f.write("{},{}\n".format(i,j))
        
            cmd.extend(['--labels',_label_file])
        if volume is not None:
            cmd.extend(['--volume',volume])
            if median:
                cmd.append('--median')
        
        if mask is not None:
            cmd.extend(['--mask',mask])
        
        _out=self.execute_w_output(cmd).split("\n")
        _out.pop(0)# remove header
        if _label_file != label_defs:
            os.unlink(_label_file)
        out=[]
        
        if label_defs is not None:
            out=[ [ ( float(j) if k>0 else j )      for k,j in enumerate(i.split(',')) ] for i in _out if len(i)>0 ]
        else:
            out=[ [ ( float(j) if k>0 else int(j) ) for k,j in enumerate(i.split(',')) ] for i in _out if len(i)>0 ]
        return out

    def skullregistration(
        self,
        source,
        target,
        source_mask,
        target_mask,
        output_xfm,
        init_xfm=None,
        stxtemplate_xfm=None,
        ):
        """perform linear registration based on the skull segmentaton"""

        temp_dir = self.temp_dir(prefix='skullregistration') + os.sep
        fit = '-xcorr'
        try:
            if init_xfm:
                resampled_source = temp_dir + 'resampled_source.mnc'
                resampled_source_mask = temp_dir \
                    + 'resampled_source_mask.mnc'
                self.resample_smooth(source, resampled_source,
                        like=target, transform=init_xfm)
                self.resample_labels(source_mask,
                        resampled_source_mask, like=target,
                        transform=init_xfm)
                source = resampled_source
                source_mask = resampled_source_mask
            if stxtemplate_xfm:
                resampled_target = temp_dir + 'resampled_target.mnc'
                resampled_target_mask = temp_dir \
                    + 'resampled_target_mask.mnc'
                self.resample_smooth(target, resampled_target,
                        transform=stxtemplate_xfm)
                self.resample_labels(target_mask,
                        resampled_target_mask,
                        transform=stxtemplate_xfm)
                target = resampled_target
                target_mask = resampled_target_mask

            self.command(['itk_morph', '--exp', 'D[3]', source_mask,
                         temp_dir + 'dilated_source_mask.mnc'], verbose=self.verbose)
            self.calc([temp_dir + 'dilated_source_mask.mnc', source],
                      'A[0]<=0.1 && A[0]>=-0.1 ? A[1]:0', temp_dir
                      + 'non_brain_source.mnc' )
            self.command(['mincreshape', '-dimrange', 'zspace=48,103',
                         temp_dir + 'non_brain_source.mnc', temp_dir
                         + 'non_brain_source_crop.mnc'], verbose=self.verbose )
            self.command(['itk_morph', '--exp', 'D[3]', target_mask,
                         temp_dir + 'dilated_target_mask.mnc'], verbose=self.verbose)
            self.calc([temp_dir + 'dilated_target_mask.mnc', target],
                      'A[0]<=0.1 && A[0]>=-0.1 ? A[1]:0', temp_dir
                      + 'non_brain_target.mnc')
            self.command(['mincreshape', '-dimrange', 'zspace=48,103',
                         temp_dir + 'non_brain_target.mnc', temp_dir
                         + 'non_brain_target_crop.mnc'], verbose=self.verbose )
            self.command([
                'bestlinreg_s2',
                '-clobber', '-lsq12', source, target,
                temp_dir + '1.xfm',
                ], verbose=self.verbose )
            self.command([
                'minctracc',
                '-quiet','-clobber',
                fit,
                '-step', '2', '2', '2',
                '-simplex','1',
                '-lsq12',
                '-model_mask', target_mask,
                source,
                target,
                temp_dir + '2.xfm',
                '-transformation', temp_dir + '1.xfm',
                ], verbose=self.verbose)

            self.command([
                'minctracc',
                '-quiet','-clobber',
                fit,
                '-step', '2', '2','2',
                '-simplex', '1',
                '-lsq12','-transformation', temp_dir + '2.xfm',
                temp_dir + 'non_brain_source_crop.mnc',
                temp_dir + 'non_brain_target_crop.mnc',
                temp_dir + '3.xfm',
                ], verbose=self.verbose)

            self.command([
                'minctracc',
                '-quiet', '-clobber',
                fit,
                '-step', '2', '2', '2',
                '-transformation',
                temp_dir + '3.xfm',
                '-simplex', '1',
                '-lsq12',
                '-w_scales', '0', '0', '0',
                '-w_shear',  '0', '0', '0',
                '-model_mask', target_mask,
                source,
                target,
                temp_dir + '4.xfm',
                ], verbose=self.verbose)

            if init_xfm:
                self.command(['xfmconcat', init_xfm, temp_dir + '4.xfm'
                             , output_xfm, '-clobber'], verbose=self.verbose)
            else:
                shutil.move(temp_dir + '4.xfm', output_xfm)
        finally:
            shutil.rmtree(temp_dir)

    def binary_morphology(self, source, expression, target , binarize_bimodal=False, binarize_threshold=None):
        cmd=['itk_morph',source,target]
        if expression is not None and expression!='':
            cmd.extend(['--exp',expression])
        if binarize_bimodal:
            cmd.append('--bimodal')
        elif binarize_threshold is not None :
            cmd.extend(['--threshold',str(binarize_threshold) ])
        self.command(cmd,inputs=[source],outputs=[target], verbose=2)

    def grayscale_morphology(self, source, expression, target ):
        cmd=['itk_g_morph',source,'--exp',expression,target]
        self.command(cmd,inputs=[source],outputs=[target], verbose=self.verbose)


    def patch_norm(self, input, output, 
                   index=None,  db=None, threshold=0.0, 
                   spline=None, median=None, field=None, 
                   subsample=2, iterations=None ): 

        cmd=['flann_patch_normalize.pl',input,output]
        if index is None or db is None:
            raise mincError("patch normalize need index and db")
        cmd.extend(['--db',db,'--index',index])
        
        if median is not None:
            cmd.extend(['--median',str(median)])
        
        if spline is not None:
            cmd.extend(['--spline',str(spline)])
            
        if iterations is not None:
            cmd.extend(['--iter',str(iterations)])
        
        cmd.extend(['--subsample',str(subsample)])
        
        if field is not None:
            cmd.extend(['--field',field])
        
        self.command(cmd,inputs=[input],outputs=[output], verbose=self.verbose)

    def autocrop(self,input,output,
                 isoexpand=None,isoextend=None):
        # TODO: repimplement in python
        cmd=['autocrop',input,output]
        if isoexpand: cmd.extend(['-isoexpand',str(isoexpand)])
        if isoextend: cmd.extend(['-isoextend',str(isoextend)])
        self.command(cmd,inputs=[input],outputs=[output], verbose=self.verbose)


    def run_mincbeast(self, input_scan, output_mask, 
                      beast_lib=None, beast_conf=None, beast_res=2):
        if beast_lib is None:
            raise mincError('mincbeast needs location of library')
        if beast_conf is None:
            beast_conf=beast_lib+os.sep+'default.{}mm.conf'.format(beast_res)
        
            
        cmd = [
            'mincbeast',
            beast_lib,
            input_scan,
            output_mask,
            '-median',
            '-fill',
            '-conf',
            beast_conf,
            '-same_resolution']
        
        self.command(cmd,inputs=[input_scan],outputs=[output_mask], verbose=2)
        
        
        
    def classify_clean( 
        self, input_scans, output_cls,
        mask=None, xfm=None, model_dir=None, model_name=None
        ):
        """
        run classify_clean
        """
        # TODO reimplement in python?
        
        cmd = ['classify_clean', '-clean_tags']
        
        cmd.extend(input_scans)
        
        if mask is not None: cmd.extend(['-mask',mask,'-mask_tag','-mask_classified'])
        if xfm  is not None: cmd.extend(['-tag_transform',xfm])
        
        if model_dir is not None and model_name is not None:
            cmd.extend([
                '-tagdir',     model_dir,
                '-tagfile',   "{}_ntags_1000_prob_90_nobg.tag".format(model_name),
                '-bgtagfile', "{}_ntags_1000_bg.tag".format(model_name)
                ])
        cmd.append(output_cls)
        self.command(cmd,inputs=input_scans,outputs=[output_cls], verbose=self.verbose)
        
    def lobe_segment(self,in_cls,out_lobes,
                     nl_xfm=None,lin_xfm=None,
                     atlas_dir=None,template=None):
        """
        Run lobe_segment script
        """
        # TODO convert to python
        identity=self.tmp('identity.xfm')
        self.param2xfm(identity)
        
        if nl_xfm is None:
            nl_xfm=identity
        if lin_xfm is None:
            lin_xfm=identity
            
        # TODO: setup sensible defaults here?
        if atlas_dir is None or template is None:
            raise mincError('lobe_segment needs atlas_dir and template')
        
        cmd = [
            'lobe_segment',
            nl_xfm,
            lin_xfm,
            in_cls,
            out_lobes,
            '-modeldir', atlas_dir,
            '-template', template,
            ]
        
        self.command(cmd, inputs=[in_cls],outputs=[out_lobes], verbose=self.verbose)
        
    def xfm2param(self, input):
        """extract transformation parameters"""

        out=self.execute_w_output(['xfm2param', input])
        
        params_=[ [ float(k) if s>0 else k for s,k in enumerate(re.split('\s+', l))] for l in out.split('\n') if re.match('^\-', l) ]
            
        return { k[0][1:] :[k[1],k[2],k[3]] for k in params_ }
        
        
    def defrag(self,input,output,stencil=6,max_connect=None,label=1):
        cmd = [
            'mincdefrag',
            input,output, str(label),str(stencil)
            ]
        if max_connect is not None:
            cmd.append(str(max_connect))
        self.command(cmd, inputs=[input],outputs=[output], verbose=self.verbose)

    def winsorize_intensity(self,input,output,pct1=1,pct2=95):
        # obtain percentile
        _threshold_1=self.stats(input,['-pctT',str(pct1)])
        _threshold_2=self.stats(input,['-pctT',str(pct2)])
        self.calc([input],"clamp(A[0],{},{})".format(_threshold_1,_threshold_2),output)
        
    def relx_fit(self,inputs,output,mask=None,t2_max=1.0):
        cmd = ['t2_fit']
        cmd.extend(inputs)
        cmd.append(output)
        if mask is not None:
            cmd.extend(['--mask',mask])
        self.command(cmd, inputs=inputs,outputs=[output], verbose=self.verbose)
        
        
    def downsample_registration_files(self, sources, targets, source_mask, target_mask, downsample=None):
        
        sources_lr=sources
        targets_lr=targets
        
        source_mask_lr=source_mask
        target_mask_lr=target_mask
        
        modalities=len(sources)

        if downsample is not None:
            for _s in range(modalities):
                s_base=os.path.basename(sources[_s]).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
                t_base=os.path.basename(targets[_s]).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
                
                source_lr=self.tmp(s_base+'_'+str(downsample)+'_'+str(_s)+'.mnc')
                target_lr=self.tmp(t_base+'_'+str(downsample)+'_'+str(_s)+'.mnc')

                self.resample_smooth(sources[_s],source_lr,unistep=downsample)
                self.resample_smooth(targets[_s],target_lr,unistep=downsample)
                
                sources_lr.append(source_lr)
                targets_lr.append(target_lr)
                
                if _s==0:
                    if target_mask is not None:
                        target_mask_lr=self.tmp(s_base+'_mask_'+str(downsample)+'.mnc')
                        self.resample_labels(target_mask,target_mask_lr,unistep=downsample,datatype='byte')
                    if target_mask is not None:
                        target_mask_lr=self.tmp(s_base+'_mask_'+str(downsample)+'.mnc')
                        self.resample_labels(target_mask,target_mask_lr,unistep=downsample,datatype='byte')
        
        return (sources_lr, targets_lr, source_mask_lr, target_mask_lr)
        
if __name__ == '__main__':
    pass

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

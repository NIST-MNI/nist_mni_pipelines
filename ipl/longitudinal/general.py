# -*- coding: utf-8 -*-

#
# @author Daniel
# @date 10/07/2011
#
# Generic python functions for scripting

import os
import sys
from   subprocess import *
import subprocess

from shutil import rmtree
import tempfile


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


class IplError(Exception):
    def __init__(self, value=''):
        self.value = value

    def __str__(self):
        return "IplError({})".format(repr(self.value))


def setENV(
    env,
    value,
    append=True,
    verbose=True,
    ):
    ''' Add a value to the enviroment
      It can also print the variables
  '''

    shell = os.environ['SHELL']

    if env in os.environ and append:
        os.environ[env] = value + ':' + os.environ[env]

        if verbose:
            if shell.count('bash'):
                print('export ' + env + '=' + value + ':${' + env + '}')
            else:
                print('setenv  ' + env + ' ' + value + ':${' + env + '}')
    else:
        os.environ[env] = value

        if verbose:
            if shell.count('bash'):
                print('export  ' + env + '=' + value)
            else:
                print('setenv  ' + env + ' ' + value)

def checkMINC():
    """
      Test is minc is available
  """

  # choose one executable

    comm = ['mincresample', '-help']
    try:
        tmp = tempfile.mkstemp('check')
        call(comm, stdout=tmp[0], stderr=tmp[0])
        os.remove(tmp[1])
    except OSError:
        print(' -- Please check PATH variable: we could not find mincresample')
        return False
    return True


def checkNIHPD_PIPELINE():
    """
    NIPHD_PIPELINE CONFIGURATION
  """

  # choose one executable

    testbin = 'pipeline_classify.pl'
    comm = [testbin, '-help']
    try:
        tmp = tempfile.mkstemp('check')
        call(comm, stdout=tmp[0], stderr=tmp[0])
        os.remove(tmp[1])
    except OSError:
        raise IplError(' -- Please check PATH variable: we could not find pipeline_classify.pl'
                       )
    return True


def execute(
    commandline,
    clfile=None,
    logfile=None,
    verbose=True,
    ):
    """
      Execute a command line waiting for the end of it. Use command() instead of execute

      commdandline: either a string or a list containg the command line
      clfile : save the executed command line in a text file
      logfile: save the execution output in a text file*
      verbose: if false no message will appear

      return : False if error
  """

    if clfile is not None:
        f = open(clfile, 'a')
        f.write(' '.join(commandline) + '\n')
        f.close()

    cline = commandline
    retv = 0
    try:

    # if logfile is not None:
    #  f=open(logfile,'a')
    #  retv=call(cline,stdout=f)
    #  f.close()
    # else:

        print('Calling:' + ','.join(commandline) + '\n')
        retv = call(cline)
    except OSError:
        raise IplError('ERROR: unable to find executable %s!'
                       % str(commandline))
    return retv


def cmdWoutput(commandline, clfile=None, verbose=True):
    """
      Execute a command line, the output is return as a string

      This is useful to obtain information from the command line (e.x. when using mincinfo)

      commdandline: either a string or a list containg the command line
      clfile : save the executed command line in a text file
      verbose: if false no message will appear

      return : False if error
  """

    if verbose:
        print(' '.join(commandline))

    if clfile is not None:
        f = open(clfile, 'a')
        f.write(' '.join(commandline) + '\n')
        f.close()

    cline = commandline
    lines = []
    try:
        lines = Popen(cline, stdout=PIPE).communicate()
    except OSError:
        raise IplError('ERROR: unable to find executable %s!'
                       % str(commandline))
    return lines[0]  # we ignore the error output :: lines[1]


def checkfiles(
    inputs=None,
    outputs=None,
    timecheck=False,
    verbose=False,
    ):

  # Check newer input file

    itime = -1  # numer of seconds since epoch
    inputs_exist = True
    if inputs is not None:
        if isinstance(inputs, basestring):  # check if input is only string and not list
            if not os.path.exists(inputs):
                inputs_exist = False
                raise IplError(' ** Error: Input does not exists! :: '
                               + str(inputs))
            else:

                itime = os.path.getmtime(inputs)
        else:
            for i in inputs:
                if not os.path.exists(i):
                    inputs_exist = False
                    raise IplError(' ** Error: One input does not exists! :: '
                                    + i)
                else:
                    timer = os.path.getmtime(i)
                    if timer < itime or itime < 0:
                        itime = timer

  # Check if outputs exist AND is newer than inputs

    outExists = False
    otime = -1
    if outputs is not None:
        if isinstance(outputs, basestring):
            outExists = os.path.exists(outputs)
            if outExists:
                otime = os.path.getmtime(outputs)
        else:
            for o in outputs:
                outExists = os.path.exists(o)
                if outExists:
                    timer = os.path.getmtime(o)
                    if timer > otime:
                        otime = timer
                if not outExists:
                    break

    if outExists:
        if timecheck and itime > 0 and otime > 0 and otime < itime:
            if verbose:
                print(' -- Warning: Output exists but older than input! Redoing command')
                print('     otime ' + str(otime) + ' < itime ' + str(itime) )
        else:
            if verbose:
                print(' -- Skipping: Output Exists')
            return False

    return True


def command(
    commandline,
    inputs=None,
    outputs=None,
    clfile=None,
    logfile=None,
    verbose=True,
    timecheck=False,
    ):
    """
      Execute a command line waiting for the end of it, testing inputs and outputs

      commdandline: list containg the command line
      inputs: list of files to check if they exist before executing command
      outputs: list of files that should be when finishing
      clfile : save the executed command line in a text file
      logfile: save the execution output in a text file
      verbose: if false no message will appear
      outputlines: store the output as a string
      timecheck: The command won't be executed if the output exists and is newer than the input file.


      return : False if error, otherwise the execution output
  """

    if verbose:
        print(' '.join(commandline))

    if not checkfiles(inputs=inputs, outputs=outputs, verbose=verbose,
                      timecheck=timecheck):
        return 0

  # run command

    outvalue = execute(commandline, clfile, logfile, verbose)

    if not outvalue == 0:
        if verbose:
            print(' ** Error: Executable output was ' + str(outvalue))
        return outvalue

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
        if verbose:
            print(' -- Error: output does not exist!')
            return -1

  # return command output

    return outvalue


def mkdir(path):
    """
      create dir if it doesn't exist
  """

    if not os.path.isdir(path):
        os.makedirs(path)


def changename(
    name,
    suffix='',
    output='',
    extension=None,
    ):
    """
      Create name from the original minc image
      extension: None: does not change extension;
  """

    tmp = name
    ext = ''

  # Remove minc extension

    if name.rfind('.mnc') > 0:  # this includes .mnc.gz
        tmp = name[:name.rfind('.mnc')]
        ext = name[name.rfind('.mnc'):]
    elif name.rfind('.xfm') > 0:
        tmp = name[:name.rfind('.xfm')]
        ext = name[name.rfind('.xfm'):]
    elif name.rfind('.nii') > 0:
        tmp = name[:name.rfind('.nii')]
        ext = name[name.rfind('.nii'):]

  # Change output dir

    if len(output) > 0:
        tmp = output + os.sep + os.path.basename(tmp)

  # Add suffix

    if len(suffix) > 0:
        tmp = tmp + suffix

  # Add extension

    if extension is None:
        tmp = tmp + ext
        pass
    else:
        tmp = tmp + extension

    return tmp

def qsub_pe(
    comm,
    pe,
    peslots,
    name=None,
    logfile=None,
    depends=None,
    queue=None
    ):
    """
    Send the job into the sge queue using paralle environment
    TODO: improve dependencies and so on
    """

    if not name:
        name = comm[0]
    try:
        qsub_comm = [
            'qsub','-cwd',
            '-N', name,
            '-j', 'y',
            '-l', 'h_vmem=6G',
            '-V', '-pe', pe, str(peslots)
            ]
        path = ''

        if logfile is not None:
            path = os.path.abspath(logfile)
            qsub_comm.extend(['-o', path])
        if depends is not None:
            qsub_comm.extend(['-hold_jid', depends])
        if queue is not None:
            qsub_comm.extend(['-q', queue])

        print(' - Name    ' + name)
        print(' - PE      ' + pe)
        print(' - PESLOTS ' + str(peslots))
        print(' - Cmd     ' + ' '.join(comm))
        print(' - logfile ' + path)

        cmds="#!/bin/bash\nhostname\n"
        cmds+="\n".join(comm)+"\n"

        p=subprocess.Popen(qsub_comm,
                stdin=subprocess.PIPE,
                stderr=subprocess.STDOUT)

        p.communicate(cmds.encode())
        # TODO: check error code?
        # Encoded cmds to bytes solved the TypeError -SoF
    finally:
        pass


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
        tmpscript = tempfile.mkstemp()[1]
        p = open(tmpscript, 'w')
        p.write('''#! /bin/bash
hostname
''')
        p.write(' '.join(comm))
        p.write('\n')  # added for compliance, SFE
        p.close()
        os.chmod(tmpscript, 777)

        qsub_comm = [
            'qsub',
            '-cwd',
            '-N',
            name,
            '-j',
            'y',
            '-V',
            '-q',
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

        qsub_comm.append(tmpscript)
        if not execute(qsub_comm):
            print(' -- Submitted job ' + name)
    finally:

        pass


# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

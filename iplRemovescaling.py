#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# @author Guizz
# @date 2012/09/24

version = '1.0'

#
# Remove visit scaling in stx2 space
#

from iplGeneral import *
from ipl.minc_tools import mincTools,mincError


# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_removescaling(patient, tp, t0):

    class params:

        pass

  # params.qc_title=patient[tp].qc_title
  # params.qc_stx2v0_mask=patient[tp].qc_jpg['stx_mask']

    if os.path.exists(patient[tp].stx2v0_mnc['masknoles']):
        print ' -- pipeline_removescaling already done!'
        return True
    if patient.pipeline_version == '1.0':
        removescaling_v10(patient, tp, t0)
    else:
        print ' -- Chosen version not found!'


def removescaling_v10(patient, tp, t0):

  # # doing the processing

    tmpdir = tempfile.mkdtemp(os.path.basename(sys.argv[0])) + os.sep

  # TODO: convert to mincTools

    try:

    # atlas

        atlas = patient.modeldir + os.sep + patient.modelname + '.mnc'

        comm = ['xfm_v0_scaling.pl', patient[tp].stx2_xfm['t1'],
                patient[t0].stx2_xfm['t1'], patient[tp].stx2v0_xfm['t1'
                ]]
        if command(comm, [patient[tp].stx2_xfm['t1'],
                   patient[t0].stx2_xfm['t1']],
                   [patient[tp].stx2v0_xfm['t1']], patient.cmdfile,
                   patient.logfile):
            raise IplError(' -- ERROR : remove scaling :: ' + comm[0])

    # reformat image into stx2

        comm = [
            'mincresample',
            patient[tp].clp2['t1'],
            patient[tp].stx2v0_mnc['t1'],
            '-transform',
            patient[tp].stx2v0_xfm['t1'],
            '-like',
            atlas,
            ]
        if command(comm, [patient[tp].clp2['t1'],
                   patient[tp].stx2v0_xfm['t1']],
                   [patient[tp].stx2v0_mnc['t1']], patient.cmdfile,
                   patient.logfile):
            raise IplError(' -- ERROR: resampling image')
        comm = [
            'mincresample',
            patient[tp].clp2['mask'],
            patient[tp].stx2v0_mnc['mask'],
            '-transform',
            patient[tp].stx2v0_xfm['t1'],
            '-like',
            atlas,
            ]
        if command(comm, [patient[tp].clp2['mask'],
                   patient[tp].stx2v0_xfm['t1']],
                   [patient[tp].stx2v0_mnc['mask']], patient.cmdfile,
                   patient.logfile):
            raise IplError(' -- ERROR: resampling image')
        if 't2les' in patient[tp].native:
            comm = [
                'mincresample',
                '-clobber',
                '-nearest',
                patient[tp].native['t2les'],
                patient[tp].stx_mnc['t2les'],
                '-like',
                patient[tp].stx_mnc['t1'],
                '-transform',
                patient[tp].stx_xfm['t2'],
                ]
            if command(comm, [patient[tp].native['t2les'], model,
                       patient[tp].stx_xfm['t2']],
                       [patient[tp].stx_mnc['t2les']], patient.cmdfile,
                       patient.logfile):
                raise IplError(' -- ERROR : preprocessing:: ' + comm[0])

      # if command(comm,[patient[tp].clp2['masknoles'],patient[tp].stx2v0_xfm['t1']],[patient[tp].stx2v0_mnc['masknoles']],patient.cmdfile,patient.logfile):
        # raise IplError( " -- ERROR: resampling image")
      # if not os.path.exists(patient[tp].stx2_mnc['masknoles']) :
        # # dilate lesions
        # tmpdir=tempfile.mkdtemp(os.path.basename(sys.argv[0]))+os.sep
        # mkdir(tmpdir)
        # tmpdilated=tmpdir+'dilated.mnc'
        # comm=['mincmorph','-clobber','-dilation',patient[tp].stx2_mnc["t2les"],tmpdilated]
        # if command(comm,[patient[tp].stx2_mnc["t2les"]],[tmpdilated],patient.cmdfile,patient.logfile):
          # raise IplError(  " -- ERROR : preprocessing:: "+comm[0])
        # #. Remove lesions from mask
        # comm=['minccalc','-clobber','-expression','if(A[0]>0.5 && A[1]<0.5){1}else{0}',patient[tp].stx2_mnc["mask"],tmpdilated,patient[tp].stx2_mnc["masknoles"]]
        # if command(comm,[patient[tp].stx2_mnc["mask"],tmpdilated],[patient[tp].stx2_mnc["masknoles"]],patient.cmdfile,patient.logfile):
          # raise IplError(  " -- ERROR : preprocessing:: "+comm[0])

        rmtree(tmpdir, True)
    finally:

      # finally used to clean tmpfiles

        rmtree(tmpdir, True)
        pass


if __name__ == '__main__':

  # Concat not very useful in stand-alone i guess

    pass

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

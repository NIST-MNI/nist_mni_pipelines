#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# @author Daniel
# @date 10/07/2011

version = '1.0'

#
# Skull stripping in t1-w
# - Using bet or beast

from iplGeneral import *
from optparse import OptionParser  # to change when python updates in the machines for argparse
from optparse import OptionGroup  # to change when python updates in the machines for argparse
from iplMincTools import *


# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_stx_skullstripping(patient, tp):

  # creation of a structure to pass the information

    class params:
        pass

    params.pipeline_version = patient.pipeline_version

    params.cmdfile = patient.cmdfile
    params.logfile = patient.logfile
    params.qc_title = patient[tp].qc_title
    params.resample = patient.resample

    params.novolpol = True

    params.final = patient.beastresolution  # This one we do not need it to be accurate (unless cross-sectional)
    params.beastdir = patient.beastdir

    #  if len(patient)==1:
    #    params.final="1" # In cross-sectional there is not a second version of the skullstripping

    # setting inputs

    params.clpt1 = patient[tp].clp['t1']
    params.stxt1 = patient[tp].stx_mnc['t1']
    params.xfmt1 = patient[tp].stx_xfm['t1']
    params.ns_stxt1 = patient[tp].stx_ns_mnc['t1']
    params.ns_xfmt1 = patient[tp].stx_ns_xfm['t1']

    # setting outputs

    params.stx_mask = patient[tp].stx_mnc['mask']
    params.ns_stx_mask = patient[tp].stx_ns_mnc['mask']
    params.clp_mask = patient[tp].clp['mask']
    params.qc_stx_mask = patient[tp].qc_jpg['stx_mask']

    if os.path.exists(params.stx_mask) \
        and os.path.exists(params.ns_stx_mask) \
        and os.path.exists(params.qc_stx_mask):
        print ' -- pipeline_stx_skullstripping is done'
    else:

        runSkullstripping(params)

    return True


  # @todo write it into the history

# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_stx2_skullstripping(patient, tp):

  # creation of a structure to pass data to the function

    class params:

        pass

    params.pipeline_version = patient.pipeline_version

    params.cmdfile = patient.cmdfile
    params.logfile = patient.logfile
    params.qc_title = patient[tp].qc_title
    params.resample = patient.resample

    params.novolpol = True
    params.final = patient.beastresolution

  # timepoint files
  # params.t2=patient[tp].clp["t2"]
  # params.pd=patient[tp].clp["pd"]

    params.clpt1 = patient[tp].clp['t1']
    params.stxt1 = patient[tp].stx2_mnc['t1']
    params.xfmt1 = patient[tp].stx2_xfm['t1']
    params.ns_stxt1 = None  # patient[tp].stx_ns_mnc["t1"]
    params.ns_xfmt1 = None  # patient[tp].stx_ns_xfm["t1"]
    params.beastdir = patient.beastdir

  # output files
  # ##############

    clpdir = patient[tp].tpdir + 'clp/'
    mkdir(clpdir)
    stx2dir = patient[tp].tpdir + 'stx2/'
    mkdir(stx2dir)

    params.stx_mask = patient[tp].stx2_mnc['mask']
    params.ns_stx_mask = None  # stx2dir+'nsstx_'+patient.id+"_"+tp+"_mask.mnc"
    params.clp_mask = patient[tp].clp2['mask']
    params.qc_stx_mask = patient[tp].qc_jpg['stx2_mask']

    if not os.path.exists(params.stx_mask) \
        or not os.path.exists(params.qc_stx_mask):
        runSkullstripping(params)
    else:
        print ' -- pipeline_stx2_skullstripping is done'

    # copy output files into the structure!
    # # CHECK IF IT WORKS!
    # print " -- Copying to patient!"
    # patient[tp].stx2_mnc["mask"]=params.stx_mask
    # #patient[tp].stx_ns_mnc["mask"]=None
    # patient[tp].clp["mask2"]=params.clp_mask
    # patient[tp].qc_jpg['stx2_mask']=params.qc_stx_mask

    if 't2les' in patient[tp].native:
        if not os.path.exists(patient[tp].stx2_mnc['masknoles']):

      # dilate lesions

            tmpdir = tempfile.mkdtemp(os.path.basename(sys.argv[0])) \
                + os.sep
            mkdir(tmpdir)
            tmpdilated = tmpdir + 'dilated.mnc'
            comm = ['mincmorph', '-clobber', '-dilation',
                    patient[tp].stx2_mnc['t2les'], tmpdilated]
            if command(comm, [patient[tp].stx2_mnc['t2les']],
                       [tmpdilated], patient.cmdfile, patient.logfile):
                raise IplError(' -- ERROR : preprocessing:: ' + comm[0])

      # . Remove lesions from mask

            comm = [
                'minccalc',
                '-clobber',
                '-expression',
                'if(A[0]>0.5 && A[1]<0.5){1}else{0}',
                patient[tp].stx2_mnc['mask'],
                tmpdilated,
                patient[tp].stx2_mnc['masknoles'],
                ]
            if command(comm, [patient[tp].stx2_mnc['mask'],
                       tmpdilated], [patient[tp].stx2_mnc['masknoles'
                       ]], patient.cmdfile, patient.logfile):
                raise IplError(' -- ERROR : preprocessing:: ' + comm[0])
            rmtree(tmpdir, True)


# Last preprocessing (or more common one)

def runSkullstripping(params):
    if params.pipeline_version == '1.0':
        skullstripping_v10(params)  # beast by simon fristed
    else:
        print ' -- Chosen version not found!'


# function using beast
# needs image in standard space

def skullstripping_v10(params):

    tmpdir = tempfile.mkdtemp(os.path.basename(sys.argv[0])) + os.sep
    minc = mincTools(tempdir=tmpdir, resample=params.resample)

    try:

    # temporary images in the dimensions of beast database

        tmpstxt1 = tmpdir + 'beast_stx_t1w.mnc'
        tmpmask = tmpdir + 'beast_stx_mask.mnc'

        beast_v10_template = params.beastdir + os.sep \
            + 'intersection_mask.mnc'
        beast_v10_margin = params.beastdir + os.sep + 'margin_mask.mnc'

        beast_v10_conffile = {'1': params.beastdir + os.sep \
                              + 'default.1mm.conf',
                              '2': params.beastdir + os.sep \
                              + 'default.2mm.conf'}
        beast_v10_intersect = params.beastdir + os.sep \
            + 'intersection_mask.mnc'

        if not os.path.exists(params.stx_mask):

      # changing the size of stx if necessary to fit with the beast images dimensions

            minc.resample_smooth(params.stxt1, tmpstxt1,
                                 like=beast_v10_template)

      # perform segmentation

            comm = [
                'mincbeast',
                params.beastdir,
                tmpstxt1,
                tmpmask,
                '-median',
                '-fill',
                '-conf',
                beast_v10_conffile[params.final],
                '-same_resolution',
                ]
            if command(comm, [tmpstxt1], [tmpmask], params.cmdfile,
                       params.logfile):
                raise IplError(' -- ERROR : skullstripping:: '
                               + comm[0])

            # reformat into the orginial stx size
            minc.resample_labels(tmpmask, params.stx_mask,
                                 like=params.stxt1)

        # reformat mask into native space

        if params.clp_mask is not None and os.path.exists(params.xfmt1) \
            and os.path.exists(params.clpt1):
            minc.resample_labels(params.stx_mask, params.clp_mask,
                                 like=params.clpt1,
                                 invert_transform=True,
                                 transform=params.xfmt1)

        # reformat mask into ns space
        if params.ns_stx_mask is not None \
            and os.path.exists(params.ns_xfmt1) \
            and os.path.exists(params.ns_stxt1):
            minc.resample_labels(params.clp_mask, params.ns_stx_mask,
                                 like=params.ns_stxt1,
                                 transform=params.ns_xfmt1)

        if params.qc_stx_mask is not None:
            minc.qc(
                params.stxt1,
                params.qc_stx_mask,
                title=params.qc_title,
                image_range=[0, 120],
                mask=params.stx_mask,
                big=True,
                clamp=True,
                )
    finally:

      # finally used to clean tmpfiles

        rmtree(tmpdir, True)
        pass


if __name__ == '__main__':

  # Using script as a stand-alone script

    usage = """usage: %prog 
   """

    parser = OptionParser(usage=usage, version=version)

  # variables in the parser are the same as in the pipeline

    group = OptionGroup(parser, ' -- Launch options V',
                        ' Options to start processing')
    group.add_option('-d', '--beastdir', dest='beastdir',
                     help='Beast library location,default=%default',
                     default='/ipl/quarantine/models/beastlib')
    group.add_option('-i', '--stx', dest='stxt1',
                     help='Input image in stx space')
    group.add_option('-t', '--template', dest='beast_template',
                     help='Normalize to template')
    group.add_option('-c', '--clamp', dest='clpt1',
                     help='Input in native space')
    group.add_option('-x', '--xfm', dest='xfmt1',
                     help='Linear transformation to stx space (to use with native space image)'
                     )
    group.add_option('', '--no-volpol', dest='novolpol',
                     help=" Don't normalize image intensities towards the atlas (already done)"
                     )
    group.add_option('-f', '--final', dest='final',
                     help=' Final resolution for BeAST (default 2 mm)',
                     default='2')
    parser.add_option_group(group)
    group = OptionGroup(parser, ' -- Output Options',
                        ' Decide output options')
    group.add_option('-p', '--print', dest='print',
                     help='Print BEaST libreries and template')
    group.add_option('-s', '--save-stx', dest='savestx',
                     help='Save native image in stx space')
    group.add_option('-o', '--output', dest='stx_mask',
                     help='Output mask in stx space')
    group.add_option('-m', '--mask', dest='clp_mask',
                     help='Output mask in native space')
    group.add_option('-r', '--resample', dest='resample',
                     help='Resample algorithm: itk,sinc,linear,cubic [%default]'
                     , default='itk')

    parser.add_option_group(group)

    group = OptionGroup(parser, ' -- Script parameters ',
                        ' BEaST parameters')
    group.add_option('-u', '--use-version', dest='pipeline_version',
                     help=' Version of the pipeline to run (BETA)',
                     default=version)
    parser.add_option_group(group)

    (opts, args) = parser.parse_args()

    if opts.stxt1 is None and opts.clpt1 is None:
        print ' -- ERROR: -i, or -c are mandatory'
        sys.exit(-1)

    if opts.stx_mask is None and opts.clp_mask is None:
        print ' -- ERROR: no output was given'
        sys.exit(-1)

  # The skull stripping requires images in the standard space.
  # If native image is not in standard space a 9 linear registration is performed
  # If image is not already in the intensity space the image is normalized too

    opts.cmdfile = None
    opts.logfile = None
    opts.ns_stx_mask = None
    opts.qc_stx_mask = None
    tmpdir = tempfile.mkdtemp(os.path.basename(sys.argv[0])) + os.sep

  # mkdir(tmpdir)

    try:

        if opts.stxt1 is None:
            opts.stxt1 = tmpdir + 'stx_t1.mnc'
        if opts.xfmt1 is None:
            opts.xfmt1 = tmpdir + 'xfm_t1.mnc'
        if opts.stx_mask is None:
            opts.stx_mask = tmpdir + 'stx_mask.mnc'

        xfmparam = []
        if not os.path.exists(opts.stxt1):

       # apply transformation to native image

            if os.path.exists(opts.xfmt1):

        # apply the given transformation

                pass
            else:

        # compute the linear resgistration to atlas

                comm = ['bestlinreg.pl', opts.clpt1,
                        beast_v10.template, opts.xfmt1, '-lsq9']
                if command(comm, [opts.clpt1, beast_v10.template],
                           [opts.xfmt1], opts.cmdfile, opts.logfile):
                    print ' -- ERROR : skullstripping:: ' + comm[0]
                    sys.exit(1)

      # register to stx space

            comm = [
                'itk_resample',
                opts.clpt1,
                opts.stxt1,
                '--clobber',
                '--transform',
                opts.xfmt1,
                '--order',
                '3',
                '--like',
                beast_v10.template,
                ]
            if command(comm, [opts.clpt1, beast_v10.template,
                       opts.xfmt1], [opts.stxt1], opts.cmdfile,
                       opts.logfile):
                print ' -- ERROR : skullstripping:: ' + comm[0]
                sys.exit(1)

    # beast needs intensity normalization, if not done before, done here

        if not opts.novolpol and opts.beast_template:
            tmpvolpol = tmpdir + 'beast_volpol_t1w.mnc'
            tmpstats = tmpdir + 'beast_stats.exp'

            comm = [
                'volume_pol',
                '--clobber',
                '--order',
                '1',
                '--min',
                '0',
                '--max',
                '100',
                '--noclamp',
                opts.stxt1,
                opts.beast_template,
                '--expfile',
                tmpstats,
                ]
            if command(comm, [opts.stxt1, opts.beast_template],
                       [tmpstats], opts.cmdfile, opts.logfile):
                print ' -- ERROR : skullstripping:: ' + comm[0]
                sys.exit(1)

            comm = [
                'minccalc',
                '-clobber',
                opts.stxt1,
                '-expfile',
                tmpstats,
                '-short',
                tmpvolpol,
                ]
            if command(comm, [opts.stxt1, tmpstats], [tmpvolpol],
                       opts.cmdfile, opts.logfile):
                print ' -- ERROR : skullstripping:: ' + comm[0]
                sys.exit(1)

      # replace the stxt1 image

            opts.stxt1 = tmpvolpol

        runSkullstripping(opts)

        if opts.savestx is not None:
            comm = ['mv', '-f', opts.stxt1, opts.savestx]
            command(comm, [opts.stxt1], [opts.savestx])
    finally:

      # finally used to clean tmpfiles

        rmtree(tmpdir, True)
        pass


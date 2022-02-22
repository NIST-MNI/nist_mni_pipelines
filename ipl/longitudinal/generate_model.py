# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 10/07/2011
#
# Generate average linear model

version = '1.0'

from iplGeneral import *  # functions to call binaries and general functions

# Classes to parse input

from optparse import OptionParser  # to change when python updates in the machines for argparse
from optparse import OptionGroup  # to change when python updates in the machines for argparse

import shutil
import os
import sys
from ipl.minc_tools import mincTools,mincError


class MriSample(object):

    """An MRI sample with ROI"""

    def __init__(
        self,
        mri='',
        mask='',
        name='',
        temp=False,
        ):
        self.mri = mri
        self.mask = mask
        self.name = name
        self.temp_sample = temp
        if not self.name:
            self.name = os.path.basename(self.mri)

    def do_cleanup(self):
        """cleanup temporary files"""

        if self.temp_sample:
            for i in (self.mri, self.mask):
                if os.access(i, os.F_OK):
                    os.unlink(i)


class ModelIterationSample(MriSample):

    """An MRI sample with ROI, and information for current iteration"""

    def _gen_names(self, prefix):
        """generate temporary file names bases on prefix"""

        self.transform = '%s_xfm.xfm' % prefix
        self.inv_transform = '%s_inv_xfm.xfm' % prefix
        self.transform_grid = '%s_xfm_grid_0.mnc' % prefix
        self.inv_transform_grid = '%s_inv_xfm_grid_0.mnc' % prefix
        self.grid = '%s_grid_0.mnc' % prefix

        self.mask = '%s_mask.mnc' % prefix
        self.mri = '%s_mri.mnc' % prefix
        self.corr_mri = '%s_corr_mri.mnc' % prefix
        self.corr_transform = '%s_corr_xfm.xfm' % prefix

        self.corr_transform_grid0 = '%s_corr_xfm_grid_0.mnc' % prefix
        self.corr_transform_grid1 = '%s_corr_xfm_grid_1.mnc' % prefix
        self.diff_bias = '%s_diff_bias.mnc' % prefix
        self.corr_bias = '%s_corr_bias.mnc' % prefix

        self.regu_transform = '%s_regu_xfm.xfm' % prefix
        self.regu_inv_transform = '%s_regu_inv_xfm.xfm' % prefix

        self.model_regu1 = '%smodel_param1.mnc' % prefix
        self.model_regu0 = '%smodel_param0.mnc' % prefix

    def __init__(
        self,
        mri='',
        mask='',
        model_regu1='',
        model_regu0='',
        name='',
        base=None,
        tempdir='',
        it=0,
        temp=True,
        ):
        self.model_regu1 = model_regu1
        self.model_regu0 = model_regu0
        self.temp_sample = temp
        if base:
            self.name = base.name
            self.mask = base.mask
            self.mri = base.mri
        else:
            self.name = name
            self.mri = mri
            self.mask = mask

    # now let's create file names

        if not self.name:
            self.name = os.path.basename(self.mri)

        self.transform = ''
        self.inv_transform = ''
        self.corr_transform = ''
        self.temp = temp

        if tempdir:
            self.prefix = '%s/%02d_%s' % (tempdir, it, self.name)
            self._gen_names(self.prefix)

    def do_cleanup(self):
        """cleanup temporary files"""

        super(ModelIterationSample, self).do_cleanup()

        if self.temp:
            for i in (
                self.transform,
                self.inv_transform,
                self.corr_transform,
                self.corr_mri,
                self.diff_bias,
                self.corr_bias,
                self.transform_grid,
                self.inv_transform_grid,
                self.corr_transform_grid0,
                self.corr_transform_grid1,
                ):
                if os.access(i, os.F_OK):
                    os.unlink(i)

    def create_qc_image(self):
        """Create QC image"""
        with mincTools() as minc:
            minc.qc(self.mri, self.prefix + '_qc.jpg',mask=self.mask)


# read MRI samples from a csv file

def readCSV(input_file):
    out = []
    for i in open(input_file):
        ll = i.rstrip('\n').split(',')
        out.append([ll[0], ll[1]])
    return out


class LinearModelGenerator(mincTools):

    def __init__(
        self,
        tempdir=None,
        registration_parameters='-lsq12',
        cleanup=True,
        qc=False,
        iterations=4,
        mt=1,
        biascorr=False,
        cleanup_final=True,
        resample='itk',
        temporalregu=False,
        ):
        super(LinearModelGenerator, self).__init__(tempdir=tempdir,
                resample=resample)
        self.registration_parameters = registration_parameters

    # self.tempdir=tempdir

        self.iterations = iterations
        self.cleanup = cleanup
        self.cleanup_final = cleanup_final
        self.qc = qc
        self.mt = mt
        self.biascorr = biascorr
        self.temporalregu = temporalregu

    def temporal_regularization(
        self,
        samples,
        visits,
        regumodelparam_1,
        regumodelparam_0,
        temporalregu,
        ):
        """Regularize the deformation fields"""

        transforms = []
        for i in samples:
            transforms.append(i.inv_transform)
        regutransforms = []
        for i in samples:
            regutransforms.append(i.regu_inv_transform)
        if temporalregu == 'taylor':
            self.temporal_regularization_nlxfm_taylor(transforms,
                    visits, regutransforms)
        else:
            self.temporal_regularization_nlxfm(transforms, visits,
                    regutransforms, regumodelparam_1.grid,
                    regumodelparam_0.grid)

    def normalize_sample(
        self,
        original,
        sample,
        model,
        bias_field=None,
        ):
        """Normalize sample intensity"""

        self.apply_n3_vol_pol(
            original.mri,
            model.mri,
            sample.corr_mri,
            source_mask=original.mask,
            target_mask=model.mask,
            bias=bias_field,
            )

    def linear_register_step(
        self,
        sample,
        model,
        output,
        init_xfm=None,
        ):
        """perform linear registration to the model, and calculate inverse"""

        self.linear_register(
            output.corr_mri,
            model.mri,
            output.transform,
            parameters=self.registration_parameters,
            source_mask=sample.mask,
            target_mask=model.mask,
            init_xfm=init_xfm,
            )
        self.xfminvert(output.transform, output.inv_transform)

    def average_transforms(
        self,
        samples,
        output,
        nl=False,
        ):
        """average given transformations"""

        avg = []
        for i in samples:
            if self.temporalregu:
                avg.append(i.regu_inv_transform)
            else:
                avg.append(i.inv_transform)
        self.xfmavg(avg, output.transform, nl=nl)

    def concat_resample(
        self,
        original,
        sample,
        model,
        ):
        """apply correction transformation and resample input"""

        if self.temporalregu:
            self.xfmconcat([sample.regu_transform, model.transform],
                           sample.corr_transform)
        else:
            self.xfmconcat([sample.transform, model.transform],
                           sample.corr_transform)
        self.resample_smooth(sample.corr_mri, sample.mri,
                             transform=sample.corr_transform)
        if original.mask:
            self.resample_labels(original.mask, sample.mask,
                                 transform=sample.corr_transform)

    def average_samples(
        self,
        samples,
        model,
        model_sd=None,
        ):
        """average individual samples"""

        avg = []
        for s in samples:
            avg.append(s.mri)

        if model_sd:
            self.average(avg, model.mri, sdfile=model_sd.mri)
        else:
            self.average(avg, model.mri)

        # average masks

        if samples[0].mask:
            avg = []
            for s in samples:
                avg.append(s.mask)
            self.average(avg, model.prefix + '_mask_.mnc')
            command([
                'minccalc',
                '-byte',
                '-express',
                'A[0]>0.5?1:0',
                model.prefix + '_mask_.mnc',
                model.mask,
                ])
            os.unlink(model.prefix + '_mask_.mnc')

    def calculate_diff_bias_field(self, sample, model):
        self.difference_n3(sample.mri, model.mri, sample.diff_bias,
                           mask=model.mask)

    def average_bias_fields(self, samples, model):
        avg = []
        for s in samples:
            avg.append(s.diff_bias)
        self.log_average(avg, model.diff_bias)

  # improvements by Daniel:

    def resample_and_correct_bias(
        self,
        sample,
        model,
        previous=None,
        ):

    # resample bias field and apply previous estimate

        tmp_bias = self.temp_file(prefix=sample.name,
                                  suffix='corr_bias.mnc')
        tmp_bias2 = self.temp_file(prefix=sample.name,
                                   suffix='corr_bias2.mnc')
        try:
            self.calc([sample.diff_bias, model.diff_bias],
                      'A[1]>0.5?A[0]/A[1]:1.0', tmp_bias)
            self.resample_smooth(tmp_bias, tmp_bias2, like=sample.mri,
                                 transform=sample.corr_transform,
                                 invert_transform=True)
            if previous:
                self.calc([previous.corr_bias, tmp_bias2], 'A[0]*A[1]',
                          sample.corr_bias, datatype='-float')
            else:
                shutil.copy(tmp_bias2, sample.corr_bias)
        finally:
            if os.access(tmp_bias, os.F_OK):
                os.unlink(tmp_bias)
            if os.access(tmp_bias2, os.F_OK):
                os.unlink(tmp_bias2)

    def execute(
        self,
        samples,
        initial_model,
        output_model,
        output_model_sd=None,
        ):
        """ perform iterative model creation"""

        tempfiles = []

        # use first sample as initial model

        if not initial_model:
            initial_model = samples[0]

        current_model = []  # current estimate of template
        current_model_sd = []

        # go through all the iterations

        for it in range(0, self.iterations):

            # this will be a model for next iteration actually

            current_model.append(ModelIterationSample(name='avg',
                                 tempdir=self.tempdir, it=it + 1))
            current_model_sd.append(ModelIterationSample(name='sd',
                                    tempdir=self.tempdir, it=it + 1))
            tempfiles.append([])

            # 1 register all subjects to current template

            model = initial_model
            if it > 0:
                model = current_model[it - 1]

            for (i, s) in enumerate(samples):
                tempfiles[it].append(ModelIterationSample(base=s,
                        tempdir=self.tempdir, it=it))
                prev_transform = None
                prev_bias_field = None
                if it > 0:
                    prev_transform = tempfiles[it - 1][i].corr_transform
                    if self.biascorr:
                        prev_bias_field = tempfiles[it - 1][i].corr_bias

                self.normalize_sample(s, tempfiles[it][i], model,
                        bias_field=prev_bias_field)
                self.linear_register_step(s, model, tempfiles[it][i],
                        init_xfm=prev_transform)

            if self.cleanup and it > 0:
                current_model[it - 1].do_cleanup()

            # 2 average all transformations

            self.average_transforms(tempfiles[it], current_model[it])

            # 3 concatenate correction and resample

            for (i, s) in enumerate(samples):
                self.concat_resample(s, tempfiles[it][i],
                        current_model[it])
                if self.qc:
                    s.create_qc_image()

            # 4 average resampled samples to create new estimate

            if self.cleanup and not output_model_sd:
                self.average_samples(tempfiles[it], current_model[it])
            else:
                self.average_samples(tempfiles[it], current_model[it],
                        current_model_sd[it])

            if self.qc:
                current_model[it].create_qc_image()

            # 5 calculate residual bias

            if self.biascorr:

                # calculate difference-based bias field

                for (i, s) in enumerate(samples):
                    self.calculate_diff_bias_field(tempfiles[it][i],
                            current_model[it])

                # average all of them

                self.average_bias_fields(tempfiles[it],
                        current_model[it])

                # resample into native space and compensate for the drift

                prev_sample = None
                for (i, s) in enumerate(samples):
                    if it > 0:
                        prev_sample = tempfiles[it - 1][i]
                    self.resample_and_correct_bias(tempfiles[it][i],
                            current_model[it], prev_sample)

            # cleanup

            if self.cleanup:

                # remove files we used from previous iteration

                if it > 0:
                    for t in tempfiles[it - 1]:
                        t.do_cleanup()
                    current_model[it - 1].do_cleanup()
                    current_model_sd[it - 1].do_cleanup()
                if self.biascorr:
                    os.unlink(current_model[it].diff_bias)

        # remove unneded files from this iteration

                for t in tempfiles[it]:
                    if self.biascorr:
                        os.unlink(t.diff_bias)
                    os.unlink(t.corr_mri)

        # copy output to the destination

        shutil.copy(current_model[self.iterations - 1].mri,
                    output_model.mri)
        if output_model.mask:
            shutil.copy(current_model[self.iterations - 1].mask,
                        output_model.mask)
        if output_model_sd and output_model_sd.mri:
            shutil.copy(current_model_sd[self.iterations - 1].mri,
                        output_model_sd.mri)

        # we are done with iterations, remove unneeded files

        if self.cleanup_final:
            for s in tempfiles[self.iterations - 1]:
                s.do_cleanup()
            current_model[self.iterations - 1].do_cleanup()
            current_model_sd[it - 1].do_cleanup()


class NonLinearModelGenerator(LinearModelGenerator):

    """Perform non-linear iterative model creation"""

    def __init__(
        self,
        tempdir=None,
        registration_parameters={4: 4},
        cleanup=True,
        qc=False,
        mt=1,
        biascorr=False,
        cleanup_final=True,
        resample='itk',
        temporalregu=False,
        ):
        super(NonLinearModelGenerator, self).__init__(
            tempdir=tempdir,
            cleanup=cleanup,
            qc=qc,
            mt=mt,
            biascorr=biascorr,
            iterations=0,
            cleanup_final=cleanup_final,
            resample=resample,
            temporalregu=temporalregu,
            )
        self.registration_parameters = registration_parameters

    def non_linear_register_step(
        self,
        sample,
        model,
        output,
        init_xfm=None,
        level=4,
        ):
        """perform non-linear registration to the model, and calculate inverse"""

        tmp_input_xfm = None
        tmp_input_grid = None
        try:
            if init_xfm:  # we need to normalize it
                ttt = self.temp_file(prefix=sample.name, suffix='nl_xfm'
                        )
                tmp_input_xfm = ttt + '.xfm'
                tmp_input_grid = ttt + '_grid_0.mnc'
                self.xfm_normalize(init_xfm, model.mri, tmp_input_xfm,
                                   step=level)
                self.non_linear_register_increment(
                    output.corr_mri,
                    model.mri,
                    output.transform,
                    source_mask=sample.mask,
                    target_mask=model.mask,
                    level=level,
                    init_xfm=tmp_input_xfm,
                    )
            else:
                self.non_linear_register_full(
                    output.corr_mri,
                    model.mri,
                    output.transform,
                    source_mask=sample.mask,
                    target_mask=model.mask,
                    level=level,
                    )

      #

            self.xfm_normalize(output.transform, model.mri,
                               output.inv_transform, step=level,
                               invert=True)
        finally:
            if tmp_input_xfm:
                os.unlink(tmp_input_xfm)
                os.unlink(tmp_input_grid)

    def execute(
        self,
        samples,
        visits,
        initial_model,
        output_model,
        output_model_sd=None,
        temporalregu=None,
        output_regu_param_0=None,
        output_regu_param_1=None,
        ):
        """ perform iterative model creation"""

        tempfiles = []

        # use first sample as initial model

        if not initial_model:
            initial_model = samples[0]

        current_model = []  # current estimate of template
        current_model_sd = []
        current_model_regumodel_0 = []
        current_model_regumodel_1 = []
        it = 0
        print( 'Registration parameters:%s' \
            % self.registration_parameters)

        # go through all the iterations

        for level in sorted(self.registration_parameters.keys(),
                            reverse=True):
            iterations = self.registration_parameters[level]
            print('Level:%d iterations:%s Total iterations:%d' \
                   % (level, iterations, it))
            for it_i in range(iterations):
                # this will be a model for next iteration actually

                current_model.append(ModelIterationSample(name='avg',
                        tempdir=self.tempdir, it=it + 1))
                current_model_sd.append(ModelIterationSample(name='sd',
                        tempdir=self.tempdir, it=it + 1))

                tempfiles.append([])
                current_model_regumodel_0.append(ModelIterationSample(name='regumodel_0'
                        , tempdir=self.tempdir, it=it))
                current_model_regumodel_1.append(ModelIterationSample(name='regumodel_1'
                        , tempdir=self.tempdir, it=it))

                if it == 8:
                    self.temporalregu = None

                # 1 register all subjects to current template

                model = initial_model
                if it > 0:
                    model = current_model[it - 1]

                for (i, s) in enumerate(samples):
                    tempfiles[it].append(ModelIterationSample(base=s,
                            tempdir=self.tempdir, it=it))
                    prev_transform = None
                    prev_bias_field = None
                    if it > 0:
                        prev_transform = tempfiles[it
                                - 1][i].corr_transform
                        if self.biascorr:
                            prev_bias_field = tempfiles[it
                                    - 1][i].corr_bias

                    self.normalize_sample(s, tempfiles[it][i], model,
                            bias_field=prev_bias_field)
                    self.non_linear_register_step(s, model,
                            tempfiles[it][i], init_xfm=prev_transform,
                            level=level)


                if self.cleanup and it > 0:
                    current_model[it - 1].do_cleanup()


                if self.temporalregu:

                    # 1bis regularization spatio-temporal
                    # (ModelIterationSample(name='regumodel_0',tempdir=self.tempdir,it=it))

                    self.temporal_regularization(tempfiles[it], visits,
                            current_model_regumodel_1[it],
                            current_model_regumodel_0[it], temporalregu)
                    for (i, s) in enumerate(samples):
                        self.xfminvert(tempfiles[it][i].regu_inv_transform,
                                tempfiles[it][i].regu_transform)

                #
                # 2 average all transformations

                self.average_transforms(tempfiles[it],
                        current_model[it], nl=True)

                #
                # 3 concatenate correction and resample

                for (i, s) in enumerate(samples):
                    self.concat_resample(s, tempfiles[it][i],
                            current_model[it])
                    if self.qc:
                        tempfiles[it][i].create_qc_image()

                #
                # 4 average resampled samples to create new estimate

                self.average_samples(tempfiles[it], current_model[it],
                        current_model_sd[it])
                if self.qc:
                    current_model[it].create_qc_image()

                #
                # 5 calculate residual bias

                if self.biascorr:

                    # calculate difference-based bias field

                    for (i, s) in enumerate(samples):
                        self.calculate_diff_bias_field(tempfiles[it][i],
                                current_model[it])

                    # average all of them

                    self.average_bias_fields(tempfiles[it], current_model[it])

                    # resample into native space and compensate for the drift

                    prev_sample = None
                    for (i, s) in enumerate(samples):
                        if it > 0:
                            prev_sample = tempfiles[it - 1][i]
                        self.resample_and_correct_bias(tempfiles[it][i],
                                current_model[it], prev_sample)

                # cleanup

                if self.cleanup:
                    # remove files we used from previous iteration

                    if it > 0:
                        for t in tempfiles[it - 1]:
                            t.do_cleanup()
                        current_model[it - 1].do_cleanup()
                        current_model_sd[it - 1].do_cleanup()

                    if self.biascorr:
                        os.unlink(current_model[it].diff_bias)

                    # remove unneded files from this iteration

                    for t in tempfiles[it]:
                        if self.biascorr:
                            os.unlink(t.diff_bias)
                        os.unlink(t.corr_mri)

                # end of internal loop, going to next iteration

                it += 1

                # ## end of iterative model creation
                # copy output to the destination

        shutil.copy(current_model[it - 1].mri, output_model.mri)

        if temporalregu == 'linear':
            shutil.copy(current_model_regumodel_0[it - 1].grid,
                        output_regu_param_0.mri)
            shutil.copy(current_model_regumodel_1[it - 1].grid,
                        output_regu_param_1.mri)

        if output_model.mask:
            shutil.copy(current_model[it - 1].mask, output_model.mask)
        if output_model_sd and output_model_sd.mri:
            shutil.copy(current_model_sd[self.iterations - 1].mri,
                        output_model_sd.mri)

        # we are done with iterations, remove unneeded files

        if self.cleanup_final:
            for s in tempfiles[it - 1]:
                s.do_cleanup()
            current_model[it - 1].do_cleanup()
            current_model_sd[it - 1].do_cleanup()


def generate_average_model(options):
    """execute model creation"""

    initial_model = None
    output_model_sd = None
    temporalregu = None
    output_regu_param_0 = None
    output_regu_param_1 = None

    if options.model:
        initial_model = MriSample(options.model, options.model_mask)
    if options.nonlinear and options.temporalregu:
        temporalregu = options.temporalregu
        output_model = MriSample(mri=options.output_model,
                                 mask=options.output_model_mask)
        if options.temporalregu == 'linear':
            output_regu_param_1 = MriSample(mri=options.output_regu_1)
            output_regu_param_0 = MriSample(mri=options.output_regu_0)
    else:
        output_model = MriSample(mri=options.output_model,
                                 mask=options.output_model_mask)

    visits = options.visits

  # visits=sorted(visits)

  # options.input=sorted(options.input)

    inp = [MriSample(ll[0], ll[1]) for ll in options.input]

    delete_temp = False
    try:
        if not options.tempdir:
            options.tempdir = \
                tempfile.mkdtemp(prefix='iplGenerateModel_')
            delete_temp = True

        if options.nonlinear:
            param = None
            if options.nonlinear:
                param = dict([[int(j) for j in i.split(',')] for i in
                             options.nonlinear.split(':')])
            else:
                param = options.nonlinear_param

            generator = NonLinearModelGenerator(
                tempdir=options.tempdir,
                registration_parameters=param,
                cleanup=options.cleanup,
                qc=options.qc,
                biascorr=options.biascorr,
                cleanup_final=options.cleanup_final,
                resample=options.resample,
                temporalregu=options.temporalregu,
                )

            generator.execute(
                inp,
                visits,
                initial_model,
                output_model,
                output_model_sd,
                temporalregu,
                output_regu_param_0,
                output_regu_param_1,
                )
        else:

            generator = LinearModelGenerator(
                tempdir=options.tempdir,
                iterations=options.iter,
                registration_parameters=options.parameters,
                cleanup=options.cleanup,
                qc=options.qc,
                biascorr=options.biascorr,
                cleanup_final=options.cleanup_final,
                resample=options.resample,
                )
            generator.execute(inp, initial_model, output_model,
                              output_model_sd)
    finally:
        if delete_temp:
            shutil.rmtree(options.tempdir)


## If used in a stand-alone application on one patient

if __name__ == '__main__':

    usage = \
        """usage: %prog -l <input.list> -o <outputdir> 
   or: %prog -h
   
   The list have this structure:
      anatomical_scan.mnc[,mask.mnc]
      
   """

    parser = OptionParser(usage=usage, version=version)

    group = OptionGroup(parser, ' -- Mandatory options ', ' Necessary')
    group.add_option('-l', '--list', dest='list',
                     help='CSV file with the list of files: (format) anatomical[,mask]'
                     )
    group.add_option('-o', '--output', dest='output',
                     help='Output prefix')
    group.add_option('--output-model', dest='output_model',
                     help='Output model')
    group.add_option('--output-model-mask', dest='output_model_mask',
                     help='Output model mask')
    group.add_option('--output-model-sd', dest='output_model_sd',
                     help='Output model sd')
    parser.add_option_group(group)

    group = OptionGroup(parser, ' -- Algorithm Options ', '')
    group.add_option('--model', dest='model', help='Initial model',
                     default='')
    group.add_option('--model-mask', dest='model_mask',
                     help='Initial model mask', default='')
    group.add_option(
        '-i',
        '--iter',
        dest='iter',
        help='Number of iterations (linear model generation)',
        default=4,
        type='int',
        )
    group.add_option('-L', '--linear', dest='linear',
                     help='Run linear mode [%default]',
                     default=False,action='store_true')
    group.add_option('-n', '--nonlinear', dest='nonlinear',
                     help='Nonlinear mode program [%default]',
                     default='32,4:16,4:8,4')
    group.add_option('-p', '--parameters', dest='parameters',
                     help='Linear registration parameters  [%default]',
                     default='-lsq12')
    group.add_option('-r', '--resample', dest='resample',
                     help='Resample algorithm: itk,sinc,linear,cubic [%default]'
                     , default='itk')
    parser.add_option_group(group)

    group = OptionGroup(parser, ' -- General Options ', '')

  # group.add_option("-S", "--sge",       dest="sge",   help="Run using SGE jobs",action="store_true",default=False)

    group.add_option('--qc', dest='qc', help='Create QC images',
                     default=False, action='store_true')
    group.add_option(
        '-c',
        '--cleanup',
        dest='cleanup',
        help='Cleanup files',
        default=False,
        action='store_true',
        )
    group.add_option('--cleanup-final', dest='cleanup_final',
                     help='Cleanup files from the final iteration',
                     default=False, action='store_true')
    group.add_option(
        '-b',
        '--biascorr',
        dest='biascorr',
        help='Perform bias correction as part of model creation',
        default=False,
        action='store_true',
        )

  # group.add_option("-q", "--queue",     dest="queue", help="Specifiy sge queue",default="all.q")

    group.add_option(
        '-v',
        '--verbose',
        dest='verbose',
        help='Verbose mode',
        action='store_true',
        default=False,
        )
    group.add_option('-t', '--tmpdir', dest='tempdir',
                     help='Location of temporary directory')
    parser.add_option_group(group)

    (opts, args) = parser.parse_args()
    if opts.list is not None:
        opts.input = readCSV(opts.list)

        if opts.output is not None and opts.output_model is None:
            opts.output_model = opts.output + '_avg.mnc'
            opts.output_model_mask = opts.output + '_avg_mask.mnc'
            opts.output_model_sd = opts.output + '_sd.mnc'

        if opts.output_model is None:
            print(' -- Please specify output prefix (-o) or output model (--output-model) ')
            sys.exit(1)

        opts.temporalregu=None
        opts.visits=None
        if opts.linear:
            opts.nonlinear=None

        generate_average_model(opts)
    else:
        print(' -- Error: Option -l or -o are mandatory')
        print('')
        parser.print_help()

    print('')
    print(" ... and that's all!")

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

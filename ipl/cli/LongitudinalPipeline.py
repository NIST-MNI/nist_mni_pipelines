# -*- coding: utf-8 -*-

#
# @author Daniel,Berengere,Vladimir,Nicolas
# @date 10/07/2011

from __future__ import print_function

import os
import traceback
import json
import six

import argparse

# Generic python functions for scripting

from ipl.longitudinal.general            import *  # functions to call binaries and general functions
from ipl.longitudinal.patient            import *  # class to store all the data


from   ipl.minc_tools import mincTools,mincError

# files storing all processing
from ipl.longitudinal.t1_preprocessing   import pipeline_t1preprocessing
from ipl.longitudinal.t2pd_preprocessing import pipeline_t2pdpreprocessing
from ipl.longitudinal.skull_stripping    import pipeline_stx_skullstripping
from ipl.longitudinal.skull_stripping    import pipeline_stx2_skullstripping
from ipl.longitudinal.linear_template    import pipeline_linearlngtemplate
from ipl.longitudinal.nonlinear_template import pipeline_lngtemplate
from ipl.longitudinal.stx2_registration  import pipeline_linearatlasregistration
from ipl.longitudinal.atlas_registration import pipeline_atlasregistration
from ipl.longitudinal.concat             import pipeline_concat


# possibly deprecate
from ipl.longitudinal.classification     import pipeline_classification
from ipl.longitudinal.lng_classification import pipeline_lng_classification

# refactor ?
from ipl.longitudinal.vbm                import pipeline_vbm
from ipl.longitudinal.dbm                import pipeline_lngDBM

#
from ipl.longitudinal.add                import pipeline_run_add_tp,pipeline_run_add

# to be deprecated
from ipl.longitudinal.lobe_segmentation  import pipeline_lobe_segmentation

# parallel processing
from scoop import futures, shared


version = '1.0'

# a hack to use environment stored in a file
# usefull when scoop spans across multiple nodes
# and default environment is not the same
if os.path.exists('lng_environment.json'):
    env={}
    with open('lng_environment.json','r') as f:
        env=json.load(f)
    for i in env.iterkeys():
        os.environ[i]=env[i]


def launchPipeline(options):
    '''
    INPUT: options are the parsed information from the command line
    TASKS
    - Read the patients lit
    - Create a pickle file for each patient to store all the image information
    - Run pipeline in each pickle files
    '''

    _opts    = {}

    if options.json is not None:
        with open(options.json,'r') as f:
            _opts=json.load(f)

        # population options if empty:
        if 'modelname' in _opts:
            options.modelname=_opts['modelname']

        if 'modeldir' in _opts:
            options.modeldir=_opts['modeldir']

        #if 'temporalregu' in _opts:
            #options.temporalregu=_opts['temporalregu']
        options.temporalregu = False # VF: not implemented in the public release

        if 'skullreg' in _opts:
            options.skullreg=_opts['skullreg']

        if 'large_atrophy' in _opts:
            options.large_atrophy=_opts['large_atrophy']

        if 'manual' in _opts:
            options.manual=_opts['manual']

        if 'mask_n3' in _opts:
            options.mask_n3=_opts['mask_n3']

        if 'n4' in _opts:
            options.n4=_opts['n4']

        if 'les' in _opts:
            options.les=_opts['les']

        if 'dobiascorr' in _opts:
            options.dobiascorr=_opts['dobiascorr']

        if 'geo' in _opts:
            options.geo=_opts['geo']

        if 'dodbm' in _opts:
            options.dodbm=_opts['dodbm']

        if 'lngcls' in _opts:
            options.lngcls=_opts['lngcls']

        if 'donl' in _opts:
            options.donl=_opts['donl']

        if 'denoise' in _opts:
            options.denoise=_opts['denoise']


        if 'vbm_options' in _opts:
            options.vbm_blur = _opts['vbm_options'].get('vbm_blur',4.0)
            options.vbm_res  = _opts['vbm_options'].get('vbm_res',2 )
            options.vbm_nl   = _opts['vbm_options'].get('vbm_nl',None)
            options.dovbm    = True

        if 'linreg' in _opts:
            options.linreg = _opts['linreg']

        if 'add' in _opts:
            options.add = _opts['add']

        if 'rigid' in _opts:
            options.rigid = _opts['rigid']

        # TODO: add more options
    # patients dictionary
    patients = {}

    # open list
    # create output dir

    # load additional steps and store them inside option structure
    # if they are strings
    # otherwise assume they are already loaded properly
    _add=[]
    for i,j in enumerate(options.add):
        try:
            _par=j
            if isinstance(j, six.string_types):
                with open(j,'r') as f:
                    _par=json.load(f)
            _add.append(_par)
        except :
            print("Error loading JSON:{}\n{}".format(j, sys.exc_info()[0]),file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            exit(1)

    options.add=_add

    mkdir(options.output)
    options.output = os.path.abspath(options.output) + os.sep  # always use abs paths for sge
    if options.workdir is not None:
        options.workdir = os.path.abspath(options.workdir) + os.sep  # always use abs paths for sge
        if not os.path.exists(options.workdir):
          os.makedirs(options.workdir)

    # for each patient
    with open(options.list) as p:
        for line in p:

            # remove the last character of the line... the '\n'

            sp = line[:-1].split(',')

            size = len(sp)  # depending the number of items not all information was given
            if size < 3:
                print( ' -- Line error: ' + str(len(sp)) )
                print( '     - Minimum format is :  id,visit,t1' )
                continue

            id = sp[0]  # set id
            visit = sp[1]  # set visit

            # ## Add patient id if not found

            if id not in patients:  # search key in the dictionary

                patients[id] = LngPatient(id)  # create new LngPatient

                if size > 6:
                    patients[id].sex = sp[6]

                patients[id].pipeline_version = version
                patients[id].geo_corr = options.geo

                # create patient's dir

                patients[id].patientdir = options.output + os.sep + id + os.sep
                mkdir(patients[id].patientdir)

                if options.workdir is None:
                    patients[id].workdir = patients[id].patientdir + os.sep + 'tmp' + os.sep
                    mkdir(patients[id].workdir)
                else:
                    patients[id].workdir=options.workdir

                if options.manual is not None:
                    patients[id].manualdir = options.manual + os.sep + id + os.sep
                else:
                    patients[id].manualdir = None

                # create pickle name
                # this file saves all the names and processing information

                patients[id].pickle = patients[id].patientdir + id + '.pickle'

                if os.path.exists(patients[id].pickle):
                    print( ' -- PICKLE already exists!! ')
                    print( '    TODO: compare options, now skipping!! ')
                    continue

                # file storing the output of the processing

                patients[id].logfile = patients[id].patientdir + id + '.log'

                # file storing only the comand lines employed

                patients[id].cmdfile = patients[id].patientdir + id + '.commands'

                # model information

                patients[id].modeldir = options.modeldir
                patients[id].modelname = options.modelname
                patients[id].beastdir = options.beastdir

                # PIPELINE OPTIONS

                patients[id].denoise = options.denoise

                # patients[id].beastresolution=options.beastres

                patients[id].mask_n3  = options.mask_n3
                patients[id].n4       = options.n4
                patients[id].donl     = options.donl
                patients[id].dolngcls = options.dolngcls
                patients[id].dodbm    = options.dodbm
                patients[id].dovbm    = options.dovbm
                patients[id].deface   = options.deface

                patients[id].mri3T    = options.mri3T
                patients[id].fast     = options.fast
                patients[id].temporalregu = options.temporalregu
                patients[id].skullreg = options.skullreg
                patients[id].large_atrophy = options.large_atrophy
                patients[id].dobiascorr = options.dobiascorr
                patients[id].linreg   = options.linreg
                patients[id].rigid    = options.rigid
                patients[id].add      = options.add

                patients[id].vbm_options = { 'vbm_fwhm':options.vbm_blur,
                                             'vbm_resolution':options.vbm_res,
                                             'vbm_nl_level':options.vbm_nl,
                                             'vbm_nl_method':'minctracc' }

                #if options.sym == True:
                    #patients[id].nl_method = 'bestsym1stepnlreg.pl'

                # end of creating a patient

            # ## Add timepoint to the patient

            if visit in patients[id]:
                raise IplError(' -- ERROR : Timepoint ' + visit
                            + ' repeated in patient ' + id)
            else:
                print('     - ' + id + '::' + visit)
                patients[id][visit] = TP(visit)  # creating a timepoint for the patient[id]

                # create visit's dir

                patients[id][visit].tpdir = patients[id].patientdir + visit \
                    + os.sep

                patients[id][visit].qc_title = id + '_' + visit

                # Reading available information depending on the size of arguments
                # VF: check existence of file

                if not os.path.exists(sp[2]):
                    raise IplError('-- ERROR : Patient %s Timepoint %s missing file:%s '
                                    % (id, visit, sp[2]))

                patients[id][visit].native['t1'] = sp[2]

                if size > 3 and len(sp[3]) > 0:

                    # VF: check existence of file

                    if not os.path.exists(sp[3]):
                        raise IplError('-- ERROR : Patient %s Timepoint %s missing file:%s '
                                        % (id, visit, sp[3]))
                    patients[id][visit].native['t2'] = sp[3]
                if size > 4 and len(sp[4]) > 0:

                    # VF: check existence of file

                    if not os.path.exists(sp[4]):
                        raise IplError('-- ERROR : Patient %s Timepoint %s missing file:%s '
                                        % (id, visit, sp[4]))
                    patients[id][visit].native['pd'] = sp[4]
                if size > 5:
                    patients[id][visit].age = sp[5]
                if size > 6 and len(sp[6])>0:
                    patients[id].sex = sp[6]
                if size > 7 and len(sp[7]) > 0:

                    # VF: check existence of file

                    if not os.path.exists(sp[7]):
                        raise IplError('-- ERROR : Patient %s Timepoint %s missing file:%s '
                                        % (id, visit, sp[7]))
                    patients[id][visit].geo['t1'] = sp[7]

                if size > 8 and len(sp[8]) > 0:
                    if not os.path.exists(sp[7]):
                        raise IplError('-- ERROR : Patient %s Timepoint %s missing file:%s '
                                        % (id, visit, sp[8]))
                    patients[id][visit].geo['t2'] = sp[8]

                if size > 9 and options.les and len(sp[9]) > 0:
                    if not os.path.exists(sp[9]):
                        raise IplError('-- ERROR : Patient %s Timepoint %s missing file:%s '
                                        % (id, visit, sp[9]))
                    patients[id][visit].native['t2les'] = sp[9]

            # end of adding timepoint
            print('{} - {}'.format(id,visit) )
            # store patients in the pickle

    if options.pe is None: # use SCOOP to run all subjects in parallel
        pickles = []

        for (id, i) in patients.items():
            # writing the pickle file
            if not os.path.exists(i.pickle):
                i.write(i.pickle)
            pickles.append(i.pickle)

        jobs=[futures.submit(runPipeline,i) for i in pickles] # assume workdir is properly set in pickle...
        futures.wait(jobs, return_when=futures.ALL_COMPLETED)
        print('All subjects finished:%d' % len(jobs))

    else: # USE SGE to submit one job per subject, using required peslots
        pickles = []

        for (id, i) in patients.items():
            # writing the pickle file
            slots=options.peslots
            # don't submit jobs with too many requested slots, when only limited number of
            # timepoints is available

            if len(i)<slots:
                slots=len(i)
            if slots<1:
                slots=1

            if not os.path.exists(i.pickle):
                i.write(i.pickle)


            # tell python to use SCOOP module to run program
            comm=['unset PE_HOSTFILE'] # HACK SCOOP not to rely on PE setting to prevent it from SSHing
            comm.extend(['export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={}'.format(slots)])
            comm.extend(['export OMP_NUM_THREADS={}'.format(slots)])
            comm.extend(['export OMP_DYNAMIC=TRUE'])
            comm.extend(['python -m scoop -n {} {} -p {}'.format(str(slots),os.path.abspath(sys.argv[0]),i.pickle)])

            qsub_pe(comm,options.pe,
                    options.peslots,
                    name='LNG_{}'.format(str(id)),
                    logfile=i.patientdir+os.sep+str(id)+".sge.log",
                    queue=options.queue)

def runTimePoint_FirstStage(tp, patient):
    '''
    Process one timepoint for cross-sectional analysis
    First Stage : preprocessing and initial stereotaxic registration
    '''

    print( 'runTimePoint_FirstStage %s %s' % (patient.id, tp))

    try:
        # preprocessing
        # ##############

        if not pipeline_t1preprocessing(patient, tp):
            raise IplError(' XX Error in the preprocessing of '
                        + patient.id + ' ' + tp)

        # writing images to file
        # skull stripping
        # ################
        # This first mask is done in 2mm unless crossectional version

        pipeline_stx_skullstripping(patient, tp)  # change to stx_skullstripping

        # writing images to file

        # t2/pd preprocessing
        # ################

        pipeline_t2pdpreprocessing(patient, tp)

        return True
    except mincError as e:
        print("Exception in runTimePoint_FirstStage:{}".format(repr(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in runTimePoint_FirstStage:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise


def runTimePoint_SecondStage(tp, patient, vbm_options):
    '''
    Process one timepoint for cross-sectional analysis
    Second Stage, run in case of a single time point
    '''

    try:
        pipeline_linearatlasregistration(patient, tp)

        pipeline_stx2_skullstripping(patient, tp)
        patient.write(patient.pickle)  # copy new images in the pickle

        print("pipeline_atlasregistration")
        pipeline_atlasregistration(patient, tp)
        patient.write(patient.pickle)  # copy new images in the pickle

        # tissue classification
        # ######################
        pipeline_classification(patient, tp)
        patient.write(patient.pickle)  # copy new images in the pickle

        # lobe segmentation
        # ######################
        pipeline_lobe_segmentation(patient, tp)
        patient.write(patient.pickle)  # copy new images in the pickle


        if len(patient.add)>0:
            pipeline_run_add(patient)
            pipeline_run_add_tp(patient,tp)

        # vbm images
        # ###########
        pipeline_vbm(patient, tp, vbm_options)

    except mincError as e:
        print("Exception in runTimePoint_SecondStage:{}".format(repr(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in runTimePoint_SecondStage:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise


def runSkullStripping(tp, patient):
    try:
        pipeline_stx2_skullstripping(patient, tp)
    except mincError as e:
        print("Exception in runSkullStripping:{}".format(repr(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in runSkullStripping:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise


def runTimePoint_ThirdStage(tp, patient):
    # calculate full NL registration in multiple TP case
    try:
        #########################
        pipeline_concat(patient, tp)
        patient.write(patient.pickle)  # copy new images in the pickle

        pipeline_classification(patient, tp)
        patient.write(patient.pickle)  # copy new images in the pickle

    except mincError as e:
        print("Exception in runTimePoint_ThirdStage:{}".format(repr(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in runTimePoint_ThirdStage:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise


def runTimePoint_FourthStage(tp, patient, vbm_options):
    # perform steps that requre full NL registration in multi tp case
    try:
        #pipeline_classification(patient, tp)
        #patient.write(patient.pickle)  # copy new images in the pickle
        if patient.dodbm:
            pipeline_lngDBM(patient, tp)

        # lobe segmentation
        # ######################
        pipeline_lobe_segmentation(patient, tp)
        patient.write(patient.pickle)  # copy new images in the pickle

        # vbm images
        # ###########
        pipeline_vbm(patient, tp, vbm_options)

        if len(patient.add)>0:
            print("Running add step:{}".format(patient.add))
            pipeline_run_add_tp(patient,tp)


    except mincError as e:
        print("Exception in runTimePoint_FourthStage:{}".format(repr(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in runTimePoint_FourthStage:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise

def runPipeline(pickle, workdir=None):
    '''
    RUN PIPELINE
    Process selected pickle file
    '''
    # TODO: make VBM options part of initialization parameters

    patient = None
    try:
        if not os.path.exists(pickle):
            raise IplError(' -- Pickle does not exists ' + pickle)
        # # Read patient

        patient = LngPatient.read(pickle)
        if not version == patient.pipeline_version:
            print(' #### NOT THE SAME PIPELINE VERSION!! ')
            raise IplError('       - Change the pipeline version or restart all processing'
                        )

        setFilenames(patient)
        print("Processing:")
        patient.printself()


        if workdir is not None:
          patient.workdir=workdir
        # prepare qc folder

        tps = sorted(patient.keys())
        jobs=[]
        for tp in tps:
            jobs.append(futures.submit(runTimePoint_FirstStage,tp, patient))

        futures.wait(jobs, return_when=futures.ALL_COMPLETED)

        print('First stage finished!')

        patient.write(patient.pickle)  # copy new images in the pickle

        jobs=[]

        if len(tps) == 1:
            for tp in tps:
                runTimePoint_SecondStage( tp, patient, patient.vbm_options  )
        else:
            # create longitudinal template
            # ############################
            # it creates a new stx space (stx2) registering the linear template to the atlas
            # all images are aligned using this new template and the bias correction used in the template creation

            pipeline_linearlngtemplate(patient)

            for tp in tps:
                jobs.append(futures.submit(runSkullStripping, tp , patient))
            # wait for all jobs to finish
            futures.wait(jobs, return_when=futures.ALL_COMPLETED)

            # using the stx2 space, we do the non-linear template
            # ################################################
            pipeline_lngtemplate(patient)

            # non-linear registration of the template to the atlas
            # ##########################
            pipeline_atlasregistration(patient)

            if len(patient.add)>0:
                pipeline_run_add(patient)

            # Concatenate xfm files for each timepoint.
            # run per tp tissue classification
            jobs=[]
            for tp in tps:
                jobs.append(futures.submit(
                    runTimePoint_ThirdStage, tp, patient
                    ))
            futures.wait(jobs, return_when=futures.ALL_COMPLETED)

            # longitudinal classification
            # ############################
            if patient.dolngcls:
                pipeline_lng_classification(patient)
            else:
                print(' -- Skipping Lng classification')

            jobs=[]
            for tp in tps:
                jobs.append(futures.submit(runTimePoint_FourthStage, tp, patient, patient.vbm_options))
            futures.wait(jobs, return_when=futures.ALL_COMPLETED)

        patient.write(patient.pickle)  # copy new images in the pickle

        return patient.id
    except mincError as e:
        print("Exception in runPipeline:{}".format(repr(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in runPipeline:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise


###### STATUS PIPELINE
# - Print pickle file in human readable form

def statusPipeline(pickle):
    patient = LngPatient.read(pickle)
    patient.printself()


###### CLEAN PICKLE
# - Remove all not created images

def cleanPickle(pickle):
    patient = LngPatient.read(pickle)
    patient.clean()
    patient.write(pickle)




def parse_options():
    usage = \
    """%(prog)s -l <patients.list> -o <outputdir> [--run]
   or: %(prog)s -p <patient.pickle> [--status|--run]
   or: %(prog)s -h

   The list have this structure:
      id,visit,t1w(,t2w,pdw,sex,age,geot1,geot2,lesions)

      - id,visit,t1w are mandatory.
      - if the data do not exist, no space should be left
          id,visit,t1w,,,sex,age -> to include sex and age in the pipeline

   -- alternative folder:
      It is a folder with the same structure as the processing <alt_folder>/<id>/<timepoint>
      Three types of files can be found here:
          - Clp image  :              clp_manual_<id>_<timepoint>_<sequence>.mnc
          - Initial xfm:              stx_manual_<id>_<timepoint>_<sequence>.xfm
          - Manual mask in stx space: stx_manual_<id>_<timepoint>_mask.mnc
          If the image exists it will replace part of the processing.
   """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                      usage=usage,
                                      description="Longitudinal pipeline")


    group = parser.add_argument_group('Mondatory options ')

    group.add_argument('-l', '--list', dest='list',
                     help='CSV file with the list of subjects: (format) id,visit,t1w,t2w,pdw,sex,age,geot1,geot2,lesions'
                     )

    group.add_argument('-o', '--output-dir', dest='output',
                     help='Output dir')


    group = group.add_argument_group('Pipeline options ',
                         ' Options to start processing')

    group.add_argument('-j', '--json', dest='json',
                     help='Json file with processing options',
                     default=None)

    group.add_argument('-w', '--work-dir', dest='workdir',
                     help='Work dir',default=None)

    group = parser.add_argument_group('Processing options ')

    group.add_argument(
        '-d',
        '--deface',
        dest='deface',
        help='Deface images (NOT YET)',
        action='store_true',
        default=False,
        )

    group.add_argument(
        '-D',
        '--denoise',
        dest='denoise',
        help='Denoise first images',
        action='store_true',
        default=False,
        )

    group.add_argument(
        "-I",
        "--inpaint",
        dest="inpaint",
        help="Inpaint T1W images based on the lesion mask",
        action="store_true",
        default=False)

    group.add_argument(
        '-N',
        '--no-nonlinear',
        dest='donl',
        help="Don't do non linear registration (NOT YET)",
        action='store_false',
        default=True,
        )

    group.add_argument('-3', '--3T', dest='mri3T',
                     help='Parameters for 3T scans', action='store_true',
                     default=False
                     )

    group.add_argument(
        '-L',
        '--lngcls',
        dest='dolngcls',
        help='Do longitudinal clasification',
        action='store_true',
        default=False,
        )

    group.add_argument(
        "-1",
        "--onlyt1",
        dest="onlyt1",
        help="Use only T1 in classification",
        action="store_true",
        default=False)

    group.add_argument(
        '--DBM',
        dest='dodbm',
        help='Do longitudinal dbm',
        action='store_true',
        default=False,
        )

    group.add_argument(
        '--VBM',
        dest='dovbm',
        help='Run VBM',
        action='store_true',
        default=False,
        )

    group.add_argument(
        '--vbm_blur',
        dest='vbm_blur',
        help='VBM blurring',
        default=4.0,
        )

    group.add_argument(
        '--vbm_res',
        dest='vbm_res',
        help='VBM resolution',
        default=2.0,
        )

    group.add_argument(
        '--vbm_nl_level',
        dest='vbm_nl',
        help='VBM nl level'
        )


    # group.add_argument("-b", "--beast-res", dest="beastres", help="Beast resolution (def: 1mm)",default="1")

    group.add_argument(
        '--nogeo',
        dest='geo',
        help='Disable distorsion correction, default enabled if present',
        action='store_false',
        default=True,
        )

    group.add_argument(
        '--no_mask_n3',
        dest='mask_n3',
        help='Disable masking of MRI for N3',
        action='store_false',
        default=False,
        )

    group.add_argument(
        '--n4',
        dest='n4',
        help='Use N4 + advanced masking',
        action='store_true',
        default=False,
        )

    group.add_argument(
        '--noles',
        dest='les',
        help='Disable lesion masks',
        action='store_false',
        default=True,
        )
    group.add_argument(
        '--biascorr',
        dest='dobiascorr',
        help='Perform longitudinal bias correction',
        action='store_true',
        default=False)

    group.add_argument('--model-dir', dest='modeldir',
                     help='Directory with the model ',
                     default='/ipl/quarantine/models/icbm152_model_09c/'
                     )

    group.add_argument('--model-name', dest='modelname',
                     help='Model name',
                     default='mni_icbm152_t1_tal_nlin_sym_09c')

    group.add_argument('--beast-dir', dest='beastdir',
                     help='Directory with the beast library ',
                     default='/ipl/quarantine/models/beast')

    #group.add_argument( '--sym', dest='sym',
                     #help='Use symmetric minctracc (NOT YET)')

    # group.add_argument('-4', '--4Dregu', dest='temporalregu',
    #                  help="Use 4D regularization for the individual template creation with a linear regressor ('linear' or 'taylor' serie decomposition)",
    #                  default=False)

    group.add_argument(
        '--skullreg',
        dest='skullreg',
        help='Run skull registration in stx2',
        action='store_true',
        default=False,
        )

    group.add_argument(
        '--large_atrophy',
        dest='large_atrophy',
        help='Remove the ventricles for the linear template creation',
        action='store_true',
        default=False,
        )

    group.add_argument('--manual', dest='manual',
                     help='Manual or alternative processing path to find auxiliary data (look info)'
                     )

    group.add_argument(
        '--linreg',
        dest='linreg',
        help='Linear registration method',
        default='bestlinreg_20180117',
        )

    group.add_argument(
        '--rigid',
        dest='rigid',
        help='Use lsq6 for linear average',
        action='store_true',
        default=False
        )

    group.add_argument(
        '--add',
        action='append',
        dest='add',
        help='Add custom step with description in .json file',
        default=[],
        )


    group = parser.add_argument_group('Execution options ',
                         ' Once the picke files are created')

    group.add_argument('-p', '--pickle', dest='pickle',
                     help=' Open a pickle file')

    group.add_argument(
        '-s',
        '--status',
        dest='pstatus',
        help=' Status of the pickle file (print object in a readable form)',
        action='store_true',
        default=False,
        )

    group.add_argument(
        '-c',
        '--clean',
        dest='pclean',
        help=' Clean all the images in the pickle that do not exist',
        action='store_true',
        default=False,
        )

    # parser.add_option_group(group)

    group = parser.add_argument_group('General Options ')

    group.add_argument(
        '-v',
        '--verbose',
        dest='verbose',
        help='Verbose mode',
        action='store_true',
        default=False,
        )

    group.add_argument('-f', '--fast', dest='fast',
                     help='Fast mode : quick & dirty mostly for testing pipeline'
                     , action='store_true')

    group.add_argument('--pe', dest='pe',
                     help='Submit jobs using parallel environment'
                     )

    group.add_argument('--peslots', dest='peslots',
                     help='PE slots ', default=4, type=int)

    group.add_argument('-q','--queue', dest='queue',
                     help='Specify SGE queue for submission'
                     )

    options = parser.parse_args()

    return options


## If used in a stand-alone application on one patient

def main():
    opts = parse_options()
    # VF: disabled in public release
    opts.temporalregu = False

    if opts.list is not None :
        if opts.output is None:
            print('Please specify and output dir (-o)')
            sys.exit(1)
        launchPipeline(opts)
    elif opts.pickle is not None:
        runPipeline(opts.pickle,workdir=opts.workdir)
    else:
        print("missing something...")
        sys.exit(1)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

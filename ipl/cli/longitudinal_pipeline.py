# -*- coding: utf-8 -*-

#
# @author Daniel,Berengere,Vladimir,Nicolas
# @date 10/07/2011

import os
from threading import local
import traceback
import json
import six
import sys

import argparse

# Generic python functions for scripting

from ipl.longitudinal.general            import *  # functions to call binaries and general functions
from ipl.longitudinal.patient            import *  # class to store all the data


from ipl.minc_tools import mincTools,mincError

from ipl.longitudinal.main import runPipeline

# parallel processing
import ray


def setup_patient(id, options):
    '''
    INPUT: patient is a LngPatient object
           options are the parsed information from the command line
    TASKS
    - Set all the options in the patient object
    - Create the pickle file
    - Run the pipeline
    '''
    patient = LngPatient(id)

    patient.geo_corr = options.geo

    patient.patientdir = options.output + os.sep + patient.id + os.sep
    os.makedirs(patient.patientdir,exist_ok=True)

    if options.workdir is None:
        patient.workdir = patient.patientdir + os.sep + 'tmp' + os.sep
        os.makedirs(patient.workdir,exist_ok=True)
    else:
        patient.workdir=options.workdir

    if options.manual is not None:
        patient.manualdir = options.manual + os.sep + patient.id + os.sep
    else:
        patient.manualdir = None

    patient.pickle = patient.patientdir + patient.id + '.pickle'

    # file storing the output of the processing
    patient.logfile = patient.patientdir + id + '.log'
    # file storing only the comand lines employed
    patient.cmdfile = patient.patientdir + id + '.commands'

    # model information

    patient.modeldir = options.modeldir
    patient.modelname = options.modelname
    patient.beastdir = options.beastdir

    # PIPELINE OPTIONS

    patient.denoise = options.denoise

    patient.mask_n3  = options.mask_n3
    patient.n4       = options.n4
    #patient.dolngcls = options.dolngcls
    patient.dodbm    = options.dodbm
    patient.dovbm    = options.dovbm
    #patient.deface   = options.deface

    patient.mri3T    = options.mri3T
    patient.fast     = options.fast
    patient.temporalregu = options.temporalregu
    patient.skullreg = options.skullreg
    patient.redskull_onnx = options.redskull_onnx
    patient.redskull_var = options.redskull_var
    patient.synthstrip_onnx = options.synthstrip_onnx

    patient.bison_pfx = options.bison_pfx
    patient.bison_atlas_pfx = options.bison_atlas_pfx
    patient.bison_method = options.bison_method
    patient.wmh_bison_pfx = options.wmh_bison_pfx
    patient.wmh_bison_atlas_pfx = options.wmh_bison_atlas_pfx
    patient.wmh_bison_method = options.wmh_bison_method
    patient.threads  = options.threads


    patient.large_atrophy = options.large_atrophy
    patient.dobiascorr = options.dobiascorr
    patient.linreg   = options.linreg
    patient.rigid    = options.rigid
    patient.add      = options.add

    patient.vbm_options = { 'vbm_fwhm':      options.vbm_blur,
                                'vbm_resolution':options.vbm_res,
                                'vbm_nl_level':  options.vbm_nl,
                                'vbm_nl_method':'minctracc' }

    patient.nl_step = options.nl_step

    if options.nl_ants :
        patient.nl_method = 'ANTS'
        patient.vbm_options['vbm_nl_method'] = 'ANTS'

    patient.nl_cost_fun = options.nl_cost_fun
    patient.do_cleanup = options.cleanup

    # end of creating a patient
    return patient

def setup_visit(patient,visit,t1=None,t2=None,pd=None,age=None,geo_t1=None,geo_t2=None,t2les=None):
    assert visit not in patient , f' -- ERROR : Timepoint {visit} repeated in patient {patient.id}'

    patient[visit] = TP(visit)  # creating a timepoint for the patient[id]

    # create visit's dir

    patient[visit].tpdir = patient.patientdir + visit + os.sep

    patient[visit].qc_title = patient.id + '_' + visit

    # Reading available information depending on the size of arguments
    # VF: check existence of file

    if not os.path.exists(t1):
        raise IplError(f'-- ERROR : Patient {patient.id} Timepoint {visit} missing file:{t1} ')
    
    patient[visit].native['t1'] = t1

    if t2 is not None and len(t2) > 0:
        patient[visit].native['t2'] = t2
    
    if pd is not None and len(pd) > 0:
        patient[visit].native['pd'] = pd

    if age is not None:
        patient[visit].age = age

    if geo_t1 is not None and len(geo_t1) > 0:
        patient[visit].geo['t1'] = geo_t1

    if geo_t2 is not None and len(geo_t2) > 0:
        patient[visit].geo['t2'] = geo_t2

    if t2les is not None and len(t2les) > 0:
        patient[visit].native['t2les'] = t2les

           

def launchPipeline(options):
    '''
    INPUT: options are the parsed information from the command line
    TASKS
    - Read the patients lit
    - Create a pickle file for each patient to store all the image information
    - Run pipeline in each pickle files
    '''

    _opts    = {}

    patients = {}

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

    os.makedirs(options.output, exist_ok=True)
    options.output = os.path.abspath(options.output) + os.sep  # always use abs paths for sge
    if options.workdir is not None:
        options.workdir = os.path.abspath(options.workdir) + os.sep  # always use abs paths for sge
        if not os.path.exists(options.workdir):
          os.makedirs(options.workdir)


    if options.json is not None: 
        # all information is stored in .json file
        # format: list of dicts: {subject:<s> , visit:<v> , t1w:<t1w> , pdw:<pdw> , t2w:<t2w> , flr:<flr>, age: <age>, sex: <sex>}

        with open(options.json,'r') as f:
            patient_list=json.load(f)
        assert isinstance(patient_list,list), "JSON file should contain a list of patients"
        for p in patient_list:
            assert 'subject' in p, "JSON file should contain a list of patients with 'subject' field"
            assert 'visit' in p, "JSON file should contain a list of patients with 'visit' field"
            assert 't1w' in p, "JSON file should contain a list of patients with 't1w' field"

            id=str(p['subject'])
            visit=str(p['visit'])

            if id not in patients:
                patients[id] = setup_patient(id,options)
                if 'sex' in p:
                    patients[id].sex = p['sex']

            setup_visit(patients[id], visit,
                        t1=p['t1w'],
                        t2=p.get('t2w',None),
                        pd=p.get('pdw',None),
                        age=p.get('age',None),
                        geo_t1=p.get('geot1',None),
                        geo_t2=p.get('geot2',None),
                        t2les=p.get('t2les',None),
                        )

    elif options.list is not None: # legacy option
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

                # ## Add patient if not found
                if id not in patients:  # search key in the dictionary
                    patients[id] = setup_patient(id,options)  # create new LngPatient

                    if size > 6:
                        patients[id].sex = sp[6]


                # ## Add timepoint to the patient
                setup_visit(patients[id], visit,
                            t1=sp[2],
                            t2=sp[3]     if size > 2 and len(sp[3])>0 else None,
                            pd=sp[4]     if size > 3 and len(sp[4])>0 else None,
                            age=float(sp[5])  if size > 4 and len(sp[5])>0 else None,
                            geo_t1=sp[7] if size > 6 and len(sp[7])>0 else None,
                            geo_t2=sp[8] if size > 7 and len(sp[8])>0 else None,
                            t2les=sp[9]  if size > 8 and len(sp[9])>0 else None,
                            )
                # end of adding timepoint
                print('{} - {}'.format(id,visit) )
                # store patients in the pickle

    # use ray to run all subjects in parallel
    pickles = []

    for (id, i) in patients.items():
        # writing the pickle file
        if not os.path.exists(i.pickle):
            i.write(i.pickle)
        pickles.append(i.pickle)
    
    if options.ray_batch==0:
        options.ray_batch=len(pickles)

    n_fail=0
    jobs_done = []
    while len(pickles)>0:

        jobs=[runPipeline.remote(i) for j,i in enumerate(pickles) if j<options.ray_batch]
        pickles=pickles[len(jobs):]
        print(f"waiting for {len(jobs)} jobs")

        while jobs:
            try:
                ready_jobs, jobs = ray.wait(jobs, num_returns=1)
                jobs_done += ray.get(ready_jobs)
            except ray.exceptions.RayTaskError as e:
                n_fail+=1
                print("Exception in runPipeline:{}".format(sys.exc_info()[0]) )
                print(e.traceback_str,flush=True)
                ee=e.as_instanceof_cause()
                print(ee,flush=True)
            except KeyboardInterrupt:
                print("Aborting")
                exit(1)

    print(f'Work finished {len(jobs_done)}, failed:{n_fail} ')


###### CLEAN PICKLE
# - Remove all not created images

def cleanPickle(pickle):
    patient = LngPatient.read(pickle)
    patient.clean()
    patient.write(pickle)


def parse_options():
    usage = \
    """%(prog)s -l <patients.list> -o <outputdir> [--run]
   or: %(prog)s --json <patients.json> -o <outputdir> [--run]
   or: %(prog)s -p <patient.pickle> [--status|--run]
   or: %(prog)s -h

   The list have this structure:
      id,visit,t1w(,t2w,pdw,age,sex,geot1,geot2,lesions)

      - id,visit,t1w are mandatory.
      - if the data do not exist, no space should be left
          id,visit,t1w,,,age,sex -> to include sex and age in the pipeline

   Alternatively, the json file should contain a list of dictionaries with the following fields:
      id,visit,t1w[,t2w,pdw,age,sex,geot1,geot2,t2les]
      - id,visit,t1w are mandatory.

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


    group = parser.add_argument_group('Required options ')

    group.add_argument('-j', '--json', dest='json',
                     help='Json file with a list of datapoints to process',
                     default=None)
    
    group.add_argument('-l', '--list', dest='list',
                     help='CSV file without header with the list of subjects: (format) id,visit,t1w,t2w,pdw,sex,age,geot1,geot2,lesions'
                     )

    group.add_argument('-o', '--output-dir', dest='output',
                     help='Output dir')

    group = group.add_argument_group('Pipeline options ',
                         ' Options to start processing')


    group.add_argument('-w', '--work-dir', dest='workdir',
                     help='Work dir',default=None)

    group = parser.add_argument_group('Processing options ')


    group.add_argument(
        '-D',
        '--denoise',
        dest='denoise',
        help='Denoise first images',
        action='store_true',
        default=False,
        )

    group.add_argument('-3', '--3T', dest='mri3T',
                     help='Parameters for 3T scans', action='store_true',
                     default=False
                     )

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
        type=float,
        default=2.0,
        )

    group.add_argument(
        '--vbm_nl_level',
        dest='vbm_nl',
        help='VBM nl level'
        )

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

    group.add_argument(
        '--skullreg',
        dest='skullreg',
        help='Run skull registration in stx (REDSKULL)',
        action='store_true',
        default=False,
        )

    group.add_argument('--redskull_onnx', 
                     help='omnivision library for redskull brain+skull',
                     default=None
                     )
    
    group.add_argument('--redskull_var', 
                     dest='redskull_var',
                     help='Redskull variant',
                     default='seg'
                     )

    group.add_argument('--synthstrip_onnx', 
                     dest='synthstrip_onnx',
                     help='onnx library for synthstrip segmentation'
                     )
    
    group.add_argument('--bison_pfx', 
                     help='Bison tissue classification model prefix'
                     )
    group.add_argument('--bison_atlas_pfx', 
                     help='Bison atlas prefix'
                     )
    group.add_argument('--bison_method', 
                     help='Bison method'
                     )
    group.add_argument('--wmh_bison_pfx', 
                     help='Bison tissue classification model prefix'
                     )
    group.add_argument('--wmh_bison_atlas_pfx', 
                     help='Bison atlas prefix'
                     )
    group.add_argument('--wmh_bison_method', 
                     help='Bison method'
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
        '--nl_ants',
        dest='nl_ants',
        help='Use ANTs for nonlinear registration',
        action='store_true',
        default=False,
        )

    group.add_argument(
        '--nl_cost_fun',
        dest='nl_cost_fun',
        help='ANTs cost function',
        default='CC',
        choices=['CC', 'MI', 'Mattes']
        )

    group.add_argument(
        '--nl_step',
        dest='nl_step',
        help='Nonlinear registration step',
        type=float,
        default=2.0
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

    group = parser.add_argument_group('Parallel execution options ')

    group.add_argument('--ray_start',type=int,
                        help='start local ray instance')
    group.add_argument('--ray_local',action='store_true',
                        help='local ray (single process)')
    group.add_argument('--ray_host',
                        help='ray host address')
    group.add_argument('--ray_batch',default=0,type=int,
                        help='Submit ray jobs in batches')
    group.add_argument('--threads',default=1,type=int,
                        help='Number of threads to use inside some ray jobs')

    group = parser.add_argument_group('General Options ')

    group.add_argument(
        '-v',
        '--verbose',
        dest='verbose',
        help='Verbose mode',
        action='store_true',
        default=False,
        )
    group.add_argument(
        '-q',
        '--quiet',
        help='Suppress some logging messages',
        action='store_true',
        default=False,
        )

    group.add_argument('-f', '--fast', dest='fast',
                     help='Fast mode : quick & dirty mostly for testing pipeline', 
                     action='store_true')
    
    group.add_argument('--cleanup', 
                     help='Remove intermediate files to save disk space', 
                     action='store_true',
                     default=False)


    options = parser.parse_args()


    return options


## If used in a stand-alone application on one patient
def main():
    opts = parse_options()
    # VF: disabled in public release
    opts.temporalregu = False

    if opts.ray_start is not None: # HACK?
        ray.init(num_cpus=opts.ray_start,log_to_driver=not opts.quiet)
    elif opts.ray_local:
        ray.init(local_mode=True,log_to_driver=not opts.quiet)
    elif opts.ray_host is not None:
        ray.init(address=opts.ray_host+':6379',log_to_driver=not opts.quiet)
    else:
        ray.init(address='auto',log_to_driver=not opts.quiet)

    if opts.list is not None or opts.json is not None:
        if opts.output is None:
            print('Please specify and output dir (-o)')
            sys.exit(1)
        launchPipeline(opts)
    elif opts.pickle is not None:
        runPipeline(opts.pickle, workdir=opts.workdir)
    else:
        print("missing something...")
        sys.exit(1)

    if opts.ray_start is not None:
        ray.shutdown()
    

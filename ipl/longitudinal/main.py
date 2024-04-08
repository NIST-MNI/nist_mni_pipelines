# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 12/03/2024

import os
import traceback
import sys

# Generic python functions for scripting

from ipl.longitudinal.general            import *  # functions to call binaries and general functions
from ipl.longitudinal.patient            import *  # class to store all the data


from ipl.minc_tools import mincTools,mincError

# files storing all processing
from ipl.longitudinal.t1_preprocessing   import pipeline_t1preprocessing,pipeline_t1preprocessing_s0
from ipl.longitudinal.t2pd_preprocessing import pipeline_t2pdpreprocessing,pipeline_t2pdpreprocessing_s0
from ipl.longitudinal.flr_preprocessing  import pipeline_flrpreprocessing,pipeline_flrpreprocessing_s0
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
import ray


@ray.remote
def runTimePoint_FirstStageA(tp, patient):
    '''
    Process one timepoint for cross-sectional analysis
    First Stage : preprocessing and initial stereotaxic registration
    '''

    try:
        # preprocessing
        # ##############

        if not pipeline_t1preprocessing_s0(patient, tp):
            raise IplError(' XX Error in the preprocessing of '
                        + patient.id + ' ' + tp)

        pipeline_t2pdpreprocessing_s0(patient, tp)
        pipeline_flrpreprocessing_s0(patient, tp)

        return True
    except mincError as e:
        print("Exception in runTimePoint_FirstStageA:{}".format(repr(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in runTimePoint_FirstStageA:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise


@ray.remote
def runTimePoint_FirstStageB(tp, patient):
    '''
    Process one timepoint for cross-sectional analysis
    First Stage : preprocessing and initial stereotaxic registration
    '''

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

        # flr preprocessing
        # ################
        pipeline_flrpreprocessing(patient, tp)

        return True
    except mincError as e:
        print("Exception in runTimePoint_FirstStageB:{}".format(repr(e)))
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in runTimePoint_FirstStageB:{}".format(sys.exc_info()[0]))
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote
def runTimePoint_SecondStage(tp, patient, vbm_options):
    '''
    Process one timepoint for cross-sectional analysis
    Second Stage, run in case of a single time point
    '''
    try:
        pipeline_linearatlasregistration(patient, tp)

        pipeline_stx2_skullstripping(patient, tp)
        patient.write(patient.pickle)  # copy new images in the pickle

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

        # Additional steps because there is only one timepoint actually
        # ######################
        if len(patient.add)>0:
            pipeline_run_add(patient)
            pipeline_run_add_tp(patient,tp,single_tp=True)

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


@ray.remote
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


@ray.remote
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


@ray.remote
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
            pipeline_run_add_tp(patient,tp)


    except mincError as e:
        print("Exception in runTimePoint_FourthStage:{}".format(repr(e)) )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in runTimePoint_FourthStage:{}".format(sys.exc_info()[0]) )
        traceback.print_exc(file=sys.stdout)
        raise

@ray.remote
def runPipeline(pickle=None, patient=None, workdir=None):
    '''
    RUN PIPELINE
    Process selected pickle file
    '''
    try:
        if patient is None:
            if not os.path.exists(pickle):
                raise IplError(' -- Pickle does not exists ' + pickle)
            patient = LngPatient.read(pickle)

        setFilenames(patient)

        if workdir is not None:
            patient.workdir=workdir
        
        # prepare qc folder
        tps=sorted(list(patient.keys()))
        # first stage A, multithreading steps
        ray.get([runTimePoint_FirstStageA.remote(tp, patient) for tp in tps])
        patient.write(patient.pickle)  # copy new images in the pickle

        # first stage B, single threading steps
        ray.get([runTimePoint_FirstStageB.remote(tp, patient) for tp in tps])
        patient.write(patient.pickle)  # copy new images in the pickle

        jobs=[]

        if len(tps) == 1:
            for tp in tps:
                ray.get([runTimePoint_SecondStage.remote( tp, patient, patient.vbm_options  )])
        else:
            
            # create longitudinal template
            # ############################
            # it creates a new stx space (stx2) registering the linear template to the atlas
            # all images are aligned using this new template and the bias correction used in the template creation
            pipeline_linearlngtemplate(patient)

            # wait for all jobs to finish
            ray.get([runSkullStripping.remote(tp , patient) for tp in tps])

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
            ray.get([runTimePoint_ThirdStage.remote( tp, patient) for tp in tps])

            # longitudinal classification
            # ############################
            if patient.dolngcls:
                pipeline_lng_classification(patient)

            ray.get([runTimePoint_FourthStage.remote( tp, patient, patient.vbm_options) for tp in tps])

        if patient.do_cleanup:
            patient.cleanup()
        else:
            # no need to write it, if we will cleanup
            patient.write(patient.pickle)  # copy new images in the pickle

        return patient.id
    except mincError as e:
        print("Exception in runPipeline:{}".format(repr(e)),flush=True )
        traceback.print_exc(file=sys.stdout)
        raise
    except :
        print("Exception in runPipeline:{}".format(sys.exc_info()[0]),flush=True )
        traceback.print_exc(file=sys.stdout)
        raise
    
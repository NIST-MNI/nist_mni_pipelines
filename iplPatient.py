#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# @author Daniel
# @date 01/08/2011

###### CREATING STORING CLASSES
###############################

import pickle  # to store the class
import os
import copy
from iplGeneral import *  # functions to call binaries and general functions


def printImages(images):
    if len(images) > 0:
        for (i, j) in images.iteritems():
            print('         - ' + i + '  -  ' + j)
    else:
        print('         - EMPTY!!')


def cleanImages(images):
    removekeys = []
    for (key, i) in images.iteritems():
        if not os.path.exists(i):
            removekeys.append(key)
    for k in removekeys:

    # print(" -- Removing "+k+" --> "+images[k])

        del images[k]


class LngPatient(dict):

    def toto(self):
        print(self.__dict__.keys())

    def __init__(self, id):

        # processing parameters

        self.pipeline_version = ''  # version of the pipeline employed
        self.commmandline = ''  # save the command line (for check ups)
        self.patientdir = ''  # folder with all the data
        self.manualdir = ''  # folder with manual processing to add to the automatic pipeline
        self.qcdir = ''  # qc images directory
        self.logfile = ''  # output of all executables
        self.cmdfile = ''  # file with all command lines
        self.pickle = ''  # pickles storing this object
        self.lock = ''  # lock file
        self.workdir = '' # patient-specific scratch space, default self.patientdir+'/tmp'

        # processing options

        self.onlyt1 = False  # Use only t1 in classification
        self.denoise = False  # Do denoising
        self.inpaint = False  # Do inpainting
        self.dobiascorr = False  # Do longitudinal bias correction
        self.dolngcls = True  # Do longitudinal classification
        self.donl = True  # Do non linear registration
        self.mask_n3 = False  # Use brain mask for N3
        self.n4 = True  # Use N4 for non-uniformity correction
        self.nl_method = 'nlfit_s'  # Do non-linear registration
        self.deface = False  # Do defacing (not implemented yet)
        self.mri3T  = True  # Using 3T
        self.beastresolution = '2'  # Beast resolution
        self.fast = False  # default - do everything thoroughly
        self.temporalregu = False  # default no temporal regularization for template creations
        self.run_face = False  # default - do not attempt to run FACE
        self.run_deep = False  # default - do not perform deep structures segmentation
        self.run_skullreg = False  # default - do not attempt to run skull registration
        self.large_atrophy = False  # default - do not use the ventricle mask for the linear template creation
        self.geo_corr = False  # default - do not perform distortion correction
        self.dodbm = False  #  default - do not create dbm files
        self.dovbm = False  #  default - do not create vbm files
        self.vbm_options = {} # VBM options

        # model used in the processing

        self.modeldir = ''  # model directory
        self.modelname = ''  # model name
        self.beastdir = ''  # beast library directory

        # patient data

        self.id = id  # subject id
        self.sex = ''  # subject gender
        self.clinicaldata = {}  # additional info

        # common processing

        self.template = {}  # containing longitudinal templates
        self.stx_mnc = {}  # containing images in the template (stx) space
        self.stx2_mnc = {}  # containing images in the improved template (stx2) space

        # self.stx2v0_mnc={}  # containing images in the improved template (stx2) space
        # self.stx2v0_mnc={}  # containing images in the improved template (stx2) space

        self.nl_xfm = ''  # transformations of the lng template to stx space
        self.qc_jpg = {}  # qc images
        
        # additional outputs
        self.add = {}

    def clean(self):
        cleanImages(self.template)
        cleanImages(self.stx_mnc)
        cleanImages(self.stx2_mnc)

        # cleanImages(self.stx2v0_mnc)

        cleanImages(self.qc_jpg)

        if not os.path.exists(self.nl_xfm):
            # self.stx2v0_xfm={}
            self.nl_xfm = ''

        # do it in the timepoints to

        if len(self) > 0:
            for (i, j) in self.iteritems():
                j.clean()

    @staticmethod  # static function to load pickle
    def read(filename):
        if os.path.exists:
            p = open(filename)
            try:
                newpatient = pickle.load(p)
            except:
                raise IplError(' -- Problem reading the pickle %s !'
                               % filename)
            return newpatient
        else:
            raise IplError(' -- Pickle %s does not exists!' % filename)

    def write(self, filename):

        # deep copy of the patient
        tmppatient = copy.deepcopy(self)
        # clean patient... removing non existing images
        tmppatient.clean()

        # write into the pickle

        p = open(filename, 'w')
        pickle.dump(tmppatient, p)
        p.close()

    def printself(self):
        print(' -- Printing LngPatient ' + self.id)
        print('')
        print('     - Processing options ')
        print('        - version      = ' + self.pipeline_version)
        print('        - commmandline = ' + self.commmandline)
        print('        - patientdir   = ' + self.patientdir)
        print('        - qcdir        = ' + self.qcdir)
        print('        - logfile      = ' + self.logfile)
        print('        - cmdfile      = ' + self.cmdfile)
        print('        - pickle       = ' + self.pickle)
        print('        - lock         = ' + self.lock)
        print('        - denoise      = ' + str(self.denoise))
        print('        - mask N3      = ' + str(self.mask_n3))
        print('        - advanced N4  = ' + str(self.n4))
        print('        - donl         = ' + str(self.donl))
        print("        - dolngcls     = " + str(self.dolngcls))
        print("        - onlyt1       = " + str(self.onlyt1))
        print('        - dodbm        = ' + str(self.dodbm))
        print('        - dovbm        = ' + str(self.dovbm))
        print('        - deface       = ' + str(self.deface))
        print('        - mri3T        = ' + str(self.mri3T))
        print('        - modeldir     = ' + self.modeldir)
        print('        - modelname    = ' + self.modelname)
        print('        - beastdir     = ' + self.beastdir)
        print('        - run Skull rgistration     = ' + str(self.skullreg))
        print('        - Geo Corr     = ' + str(self.geo_corr))
        print('')
        print('    - Patient Info ' + self.id)
        print('        - sex = ' + self.sex)
        print('')
        print('    - Common processing ')
        print('      - stx_mnc ')
        printImages(self.stx_mnc)
        print('      - stx2_mnc ')
        printImages(self.stx2_mnc)

        # print("      - stx2v0_mnc ")
        # printImages(self.stx2v0_mnc)
        # print("      - stx2v0_xfm ")
        # printImages(self.stx2v0_xfm)

        print('      - template ')
        printImages(self.template)
        print('      - nl_xfm ')
        print('               ' + self.nl_xfm)
        print('      - qc_jpg ')
        printImages(self.qc_jpg)
        print('')
        print(' -- TIMEPOINTS ')
        if len(self) > 0:
            for (i, j) in self.iteritems():
                print('')
                j.printself()


class TP:

    def __init__(self, visit):

        # patient data

        self.tp = visit  # visit name
        self.age = 0  # age of visit
        self.clinicaldata = {}  # additional info for the timepoint
        self.tpdir = ''  # timepoint directory

        # native space

        self.native = {}  # native images
        self.geo = {}  # geometrical distorsion correction

        self.clp = {}  # after N3, and normalization
        self.clp2 = {}  # after the linear N3 correction
        self.den = {} # after denoising
        self.nuc = {} # N3 field
        self.corr = {} # after distortion correction

        # native space

        self.manual = {}  # manually corrected files

        # standard space

        self.stx_mnc = {}  # images in stereotaxic space
        self.stx_xfm = {}  # registration to stereotaxic space
        self.stx_ns_mnc = {}  # images in stereotaxic space (non-scaled)
        self.stx_ns_xfm = {}  # images in stereotaxic space (non-scaled)

        # improved stereotatic space
        # - images are aligned to the template
        # - bias correction is improved, using the template difference

        self.stx2_mnc = {}  # images in stereotaxic space
        self.stx2_xfm = {}  # registration to stereotaxic space

        # self.stx2v0_mnc={}    # images in stereotaxic space
        # self.stx2v0_xfm={}    # registration to stereotaxic space

        self.lng_mnc = {}  # newly corrected images @TODO useless?

        self.vbm = {}  # vbm files
        self.vol = {}  # text files with volume data

        # non linear images

        self.lng_xfm = {}  # registration towards the lng template
        self.lng_ixfm = {}  # registration from lng templat to scan
        self.lng_igrid = {}  # registration from lng templat to scan
        self.lng_det = {}  # registration from lng templat to scan  determinant
        self.nl_xfm = ''  # registration towards the atlas

        # qc images

        self.qc_jpg = {}  # qc images in jpg
        self.qc_title = ''

        # FACE

        self.face = {}
        
        # Additional outputs
        self.add = {}

    # .... and so on

    def clean(self):
        cleanImages(self.native)
        cleanImages(self.geo)
        cleanImages(self.clp)
        cleanImages(self.clp2)
        cleanImages(self.stx_mnc)
        cleanImages(self.stx_xfm)
        cleanImages(self.stx_ns_mnc)
        cleanImages(self.stx_ns_xfm)
        cleanImages(self.stx2_mnc)
        cleanImages(self.stx2_xfm)
        cleanImages(self.lng_mnc)

        # cleanImages(self.stx2v0_mnc)
        # cleanImages(self.stx2v0_xfm)

        cleanImages(self.vbm)
        cleanImages(self.vol)
        cleanImages(self.lng_xfm)
        cleanImages(self.lng_ixfm)
        cleanImages(self.lng_igrid)
        cleanImages(self.lng_det)
        cleanImages(self.manual)

    # #

        if not os.path.exists(self.nl_xfm):
            self.nl_xfm = ''
        cleanImages(self.qc_jpg)

    def printself(self):
        print('    -- Timepoint ' + self.tp + '  tpdir: ' + self.tpdir)
        print('       + age = ' + str(self.age))
        print('       + native ')
        printImages(self.native)
        print('       + manual ')
        printImages(self.manual)
        print('       + geo ')
        printImages(self.geo)
        print('       + clp ')
        printImages(self.clp)
        print('       + clp2 ')
        printImages(self.clp2)
        print('       + stx_mnc ')
        printImages(self.stx_mnc)
        print('       + stx_xfm ')
        printImages(self.stx_xfm)
        print('       + stx_ns_mnc ')
        printImages(self.stx_ns_mnc)
        print('       + stx_ns_xfm ')
        printImages(self.stx_ns_xfm)
        print('       + stx2_mnc ')
        printImages(self.stx2_mnc)
        print('       + stx2_xfm ')
        printImages(self.stx2_xfm)

    # print("       + stx2v0_xfm ")
    # printImages(self.stx2_mnc)
    # print("       + stx2v0_mnc ")
    # printImages(self.stx2_xfm)

        print('       + lng_mnc ')
        printImages(self.lng_mnc)
        print('       + vbm ')
        printImages(self.vbm)
        print('       + vol ')
        printImages(self.vol)
        print('       + lng_xfm ')
        printImages(self.lng_xfm)
        print('       + lng_ixfm ')
        printImages(self.lng_ixfm)
        print('       + lng_igrid ')
        printImages(self.lng_igrid)
        print('       + lng_det ')
        printImages(self.lng_det)
        print('       + nl_xfm')
        print('         - ' + self.nl_xfm)
        print('       + QC images')
        printImages(self.qc_jpg)
        print('       + FACE')
        printImages(self.face)


###### SET FILENAMES

def setFilenames(patient):

    patient.qcdir = patient.patientdir + 'qc' + os.sep
    mkdir(patient.qcdir)
    
    if patient.workdir is None:
        patient.workdir = patient.patientdir + 'tmp' + os.sep
    # # For each time point

    for tp in patient.keys():

        # # Create directories

        clpdir = patient[tp].tpdir + 'clp' + os.sep
        mkdir(clpdir)
        clp2dir = patient[tp].tpdir + 'clp2' + os.sep
        mkdir(clp2dir)
        stxdir = patient[tp].tpdir + 'stx' + os.sep
        mkdir(stxdir)
        stx2dir = patient[tp].tpdir + 'stx2' + os.sep
        mkdir(stx2dir)
        nldir = patient[tp].tpdir + 'nl' + os.sep
        mkdir(nldir)
        vbmdir = patient[tp].tpdir + 'vbm' + os.sep
        mkdir(vbmdir)
        clsdir = patient[tp].tpdir + 'cls' + os.sep
        mkdir(clsdir)
        adddir = patient[tp].tpdir + 'add' + os.sep
        mkdir(adddir)
        voldir = patient[tp].tpdir + 'vol' + os.sep
        mkdir(voldir)
        lngdir = patient[tp].tpdir + 'lng' + os.sep
        mkdir(lngdir)

        # take the sequences of the patient from the native images
        # this includes t1,t2,pd and t2les

        seqs = patient[tp].native.keys()
        for s in seqs:

            # clp space, after denoising
            patient[tp].den[s] = clpdir + 'den_' + patient.id + '_' \
                + tp + '_' + s + '.mnc'

            # clp space
            patient[tp].clp[s] = clpdir + 'clp_' + patient.id + '_' \
                + tp + '_' + s + '.mnc'

            patient[tp].nuc[s] = clpdir + 'nuc_' + patient.id + '_' \
                + tp + '_' + s + '.mnc'


            # clp space, after [denoising] and [distortion] correction
            patient[tp].corr[s] = clpdir + 'corr_' + patient.id + '_' \
                + tp + '_' + s + '.mnc'

            patient[tp].clp2[s] = clp2dir + 'clp2_' + patient.id + '_' \
                + tp + '_t1.mnc'

            # stx space
            patient[tp].stx_mnc[s] = stxdir + 'stx_' + patient.id + '_' \
                + tp + '_' + s + '.mnc'
            patient[tp].stx_xfm[s] = stxdir + 'stx_' + patient.id + '_' \
                + tp + '_' + s + '.xfm'

            # stx_ns  space
            patient[tp].stx_ns_mnc[s] = stxdir + 'nsstx_' + patient.id \
                + '_' + tp + '_' + s + '.mnc'
            patient[tp].stx_ns_xfm[s] = stxdir + 'nsstx_' + patient.id \
                + '_' + tp + '_' + s + '.xfm'

            # stx2 space
            patient[tp].stx2_mnc[s] = stx2dir + 'stx2_' + patient.id \
                + '_' + tp + '_' + s + '.mnc'
            patient[tp].stx2_xfm[s] = stx2dir + 'stx2_' + patient.id \
                + '_' + tp + '_' + s + '.xfm'

            # stx2 space
            # patient[tp].stx2v0_mnc[s] = stx2dir+"stx2v0_"+patient.id+"_"+tp+"_"+s+".mnc"
            # patient[tp].stx2v0_xfm[s] = stx2dir+"stx2v0_"+patient.id+"_"+tp+"_"+s+".xfm"
            # if len(patient)==1: # cross sectional
            #  patient[tp].stx2_mnc[s]  =patient[tp].stx_mnc[s]
            #  patient[tp].stx2_xfm[s]  =patient[tp].stx_xfm[s]
            # qc

            patient[tp].qc_jpg['stx_' + s] = patient.qcdir + 'qc_stx_' + s + '_' + patient.id + '_' + tp + '.jpg'
            patient[tp].qc_jpg['stx2_' + s] = patient.qcdir + 'qc_stx2_' + s + '_' + patient.id + '_' + tp + '.jpg'
            patient[tp].qc_jpg['nl_' + s] = patient.qcdir + 'qc_nl_' + s + '_' + patient.id + '_' + tp + '.jpg'

            # patient[tp].qc_jpg["stx2v0_"+s]=patient.qcdir+"qc_stx2v0_"+s+"_"+patient.id+"_"+tp+".jpg"

            if not s == 't1':
                patient[tp].qc_jpg['t1' + s] = patient.qcdir + 'qc_t1' \
                    + s + '_' + patient.id + '_' + tp + '.jpg'

            # MANUAL: manually corrected native space image

            if patient.manualdir is not None:

                patient[tp].manual['clp_' + s] = patient.manualdir \
                    + os.sep + tp + os.sep + 'clp_manual_' + patient.id + '_' \
                    + tp + '_' + s + '.mnc'
                if not os.path.exists(patient[tp].manual['clp_' + s]):

                    # if it does not exists, we remove it from the dictionary
                    del patient[tp].manual['clp_' + s]

                    # DLC 2016.06.16 , the line was:
                    # + os.sep tp + 'stx_manual_' + patient.id + '_'
                    # which gave two '//' before the timepoint, and no '/' afterwards.
                    
                patient[tp].manual['stx_' + s] = patient.manualdir \
                    + os.sep + tp + os.sep + 'stx_manual_' + patient.id + '_' \
                    + tp + '_' + s + '.xfm'

                # print(out manual stuff...)
                print('manual xfm is: ' )
                print(patient[tp].manual['stx_' + s])
                print('\n')

                
                if not os.path.exists(patient[tp].manual['stx_' + s]):

                    # if it does not exists, we remove it from the dictionary
                    del patient[tp].manual['stx_' + s]

            # end of sequences

        # masks
        patient[tp].clp["mask"]        = clpdir+'clp_'+patient.id+"_"+tp+"_mask.mnc"
        patient[tp].clp2["mask"]       = clp2dir+'clp2_'+patient.id+"_"+tp+"_mask.mnc"
        patient[tp].stx_mnc["mask"]    = stxdir+"stx_"+patient.id+"_"+tp+"_mask.mnc"
        patient[tp].stx2_mnc["mask"]   = stx2dir+"stx2_"+patient.id+"_"+tp+"_mask.mnc"
        
        # depricated files
        patient[tp].stx2_mnc["rhc"]   = stx2dir+"stx2_"+patient.id+"_"+tp+"_rhc.mnc"
        patient[tp].stx2_mnc["lhc"]   = stx2dir+"stx2_"+patient.id+"_"+tp+"_lhc.mnc"
        patient[tp].stx2_mnc["ram"]   = stx2dir+"stx2_"+patient.id+"_"+tp+"_ram.mnc"
        patient[tp].stx2_mnc["lam"]   = stx2dir+"stx2_"+patient.id+"_"+tp+"_lam.mnc"
        patient[tp].stx2_mnc["vent"]   = stx2dir+"stx2_"+patient.id+"_"+tp+"_vent.mnc"
        
        #patient[tp].stx2v0_mnc["mask"]   = stx2dir+"stx2v0_"+patient.id+"_"+tp+"_mask.mnc"
        patient[tp].stx_ns_mnc["mask"] = stxdir+"nsstx_"+patient.id+"_"+tp+"_mask.mnc"
        patient[tp].qc_jpg['stx_mask'] = patient.qcdir+"qc_stx_mask_"+patient.id+"_"+tp+".jpg"
        patient[tp].qc_jpg['stx2_mask']= patient.qcdir+"qc_stx2_mask_"+patient.id+"_"+tp+".jpg"
        patient[tp].qc_jpg['stx2_rhc']= patient.qcdir+"qc_stx2_rhc_"+patient.id+"_"+tp+".jpg"
        patient[tp].qc_jpg['stx2_lhc']= patient.qcdir+"qc_stx2_lhc_"+patient.id+"_"+tp+".jpg"
        patient[tp].qc_jpg['stx2_ram']= patient.qcdir+"qc_stx2_ram_"+patient.id+"_"+tp+".jpg"
        patient[tp].qc_jpg['stx2_lam']= patient.qcdir+"qc_stx2_lam_"+patient.id+"_"+tp+".jpg"
        patient[tp].qc_jpg['stx2_vent']= patient.qcdir+"qc_stx2_vent_"+patient.id+"_"+tp+".jpg"
        patient[tp].qc_jpg['stx2_deep']= patient.qcdir+"qc_stx2_deep_"+patient.id+"_"+tp+".jpg"
        patient[tp].qc_jpg['lng_det']= patient.qcdir+"qc_lngdet_"+patient.id+"_"+tp+".jpg"
        patient[tp].qc_jpg['nl_det']= patient.qcdir+"qc_nldet_"+patient.id+"_"+tp+".jpg"
        
        if 't2les' in patient[tp].native:
            patient[tp].stx_mnc['masknoles'] = stxdir + 'stx_' \
                + patient.id + '_' + tp + '_masknoles.mnc'
            patient[tp].stx2_mnc['masknoles'] = stx2dir + 'stx2_' \
                + patient.id + '_' + tp + '_masknoles.mnc'
            patient[tp].stx_ns_mnc['masknoles'] = stxdir + 'nsstx_' \
                + patient.id + '_' + tp + '_masknoles.mnc'
        else:

            # patient[tp].stx2v0_mnc["masknoles"]   = stx2dir+"stx2v0_"+patient.id+"_"+tp+"_masknoles.mnc"
            # if no lesions we copy the mask into masknoles

            patient[tp].stx_mnc['masknoles'] = \
                patient[tp].stx_mnc['mask']
            patient[tp].stx2_mnc['masknoles'] = \
                patient[tp].stx2_mnc['mask']
            patient[tp].stx_ns_mnc['masknoles'] = \
                patient[tp].stx_ns_mnc['mask']

        # patient[tp].stx2v0_mnc["masknoles"]   = patient[tp].stx2v0_mnc["mask"]
        # if len(patient)==1: # cross sectional
        #  patient[tp].stx2_mnc['mask']=patient[tp].stx_mnc['mask']
        #  patient[tp].stx2_mnc['masknoles']=patient[tp].stx_mnc['masknoles']

        # t2-> t1 registration

        patient[tp].clp['t2t1xfm'] = clpdir + 'clp_' + patient.id + '_' \
            + tp + '_t2t1.xfm'
        patient[tp].clp['pdt1xfm'] = clpdir + 'clp_' + patient.id + '_' \
            + tp + '_pdt1.xfm'

        # MANUAL: manually corrected native space image

        if patient.manualdir is not None:
            patient[tp].manual['stx_mask'] = patient.manualdir + os.sep \
                + tp + 'stx_manual_' + patient.id + '_' + tp \
                + '_mask.mnc'
            if not os.path.exists(patient[tp].manual['stx_mask']):
            # if it does not exists, we remove it from the dictionary
                del patient[tp].manual['stx_mask']

        # non linear images
        # longitudinal registration

        patient[tp].lng_xfm['t1'] = lngdir + 'lng_' + patient.id + '_' \
            + tp + '_t1.xfm'
        patient[tp].lng_ixfm['t1'] = lngdir + 'lng_' + patient.id + '_' \
            + tp + '_t1_inv.xfm'
        patient[tp].lng_igrid['t1'] = lngdir + 'lng_' + patient.id \
            + '_' + tp + '_t1_inv_grid_0.mnc'
        patient[tp].lng_det['t1'] = lngdir + 'lng_' + patient.id + '_' \
            + tp + '_t1_inv_det.mnc'
        patient[tp].lng_mnc['t1'] = lngdir + 'lng_' + patient.id + '_' \
            + tp + '_t1.mnc'

        patient[tp].nl_xfm = nldir + 'nl_' + patient.id + '_' + tp \
            + '.xfm'

        # classification

        patient[tp].stx2_mnc['classification'] = clsdir + 'cls_' \
            + patient.id + '_' + tp + '.mnc'
        patient[tp].stx2_mnc['lng_classification'] = clsdir + 'lngcls_' \
            + patient.id + '_' + tp + '.mnc'

        patient[tp].stx2_mnc['lobes'] = clsdir + 'lob_' + patient.id \
            + '_' + tp + '.mnc'
        patient[tp].vol['lobes'] = voldir + 'vol_' + patient.id + '_' \
            + tp + '.txt'

        patient[tp].stx2_mnc["add_prefix"] = adddir + 'add_' + patient.id + '_' + tp 
        
        
        patient[tp].stx2_mnc['lng_lobes'] = clsdir + 'lnglob_' \
            + patient.id + '_' + tp + '.mnc'
        patient[tp].vol['lng_lobes'] = voldir + 'lngvol_' + patient.id \
            + '_' + tp + '.txt'

        patient[tp].qc_jpg['classification'] = patient.qcdir \
            + 'qc_cls_' + patient.id + '_' + tp + '.jpg'
        patient[tp].qc_jpg['lobes'] = patient.qcdir + 'qc_lob_' \
            + patient.id + '_' + tp + '.jpg'

        patient[tp].qc_jpg['lngclassification'] = patient.qcdir \
            + 'qc_lngcls_' + patient.id + '_' + tp + '.jpg'
        patient[tp].qc_jpg['lnglobes'] = patient.qcdir + 'qc_lnglob_' \
            + patient.id + '_' + tp + '.jpg'

        # vbm analysis

        patient[tp].vbm['csf']  = vbmdir + 'vbm_imp_csf_' + patient.id + '_' + tp + '.mnc'
        patient[tp].vbm['gm']   = vbmdir + 'vbm_imp_gm_'  + patient.id + '_' + tp + '.mnc'
        patient[tp].vbm['wm']   = vbmdir + 'vbm_imp_wm_'  + patient.id + '_' + tp + '.mnc'
        patient[tp].vbm['idet'] = vbmdir + 'vbm_idet_'    + patient.id + '_' + tp + '.mnc'

        patient[tp].vbm['xfm']  = vbmdir + 'vbm_xfm_'     + patient.id + '_' + tp + '.xfm'
        patient[tp].vbm['grid'] = vbmdir + 'vbm_xfm_'     + patient.id + '_' + tp + '_grid_0.mnc'

        patient[tp].vbm['ixfm']  = vbmdir + 'vbm_xfm_'    + patient.id + '_' + tp + '_inverse.xfm'
        patient[tp].vbm['igrid'] = vbmdir + 'vbm_xfm_'    + patient.id + '_' + tp + '_inverse_grid_0.mnc'


        # TODO: add more needed files
        # end of timepoints

    # # COMMON IMAGES
    # directories

    lngtmpldir = patient.patientdir + 'lng_template' + os.sep
    mkdir(lngtmpldir)

    # ## template images
    # a) linear

    patient.template['linear_template'] = lngtmpldir + 'lin_template_' \
        + patient.id + '_t1.mnc'
    patient.template['linear_template_sd'] = lngtmpldir \
        + 'lin_template_' + patient.id + '_t1_sd.mnc'
    patient.template['linear_template_mask'] = lngtmpldir \
        + 'lin_template_' + patient.id + '_mask.mnc'
    patient.template['stx2_xfm'] = lngtmpldir + 'lin_template_' \
        + patient.id + '_t1.xfm'
    patient.qc_jpg['linear_template'] = patient.qcdir \
        + 'qc_lin_template_' + patient.id + '.jpg'

    # b) non-linear

    patient.qc_jpg['nl_template_prefix'] = patient.qcdir     + 'qc_nl_template_' + patient.id 
    patient.qc_jpg['nl_template']    = patient.qc_jpg['nl_template_prefix'] + '.jpg'
    patient.qc_jpg['nl_template_nl'] = patient.qc_jpg['nl_template_prefix'] + '_nl.jpg'
    
    patient.template['nl_template_prefix'] = lngtmpldir + 'nl_template_' + patient.id 
    patient.template['nl_template']     = patient.template['nl_template_prefix'] + '_t1.mnc'
    
    patient.template['nl_template_sd']  = patient.template['nl_template_prefix'] + '_t1_sd.mnc'
    patient.template['nl_template_mask']= patient.template['nl_template_prefix'] + '_mask.mnc'
    
    patient.template['regu_0'] = patient.template['nl_template_prefix'] + '_t1_regu_param_0_grid.mnc'
    patient.template['regu_1'] = patient.template['nl_template_prefix'] + '_t1_regu_param_1_grid.mnc'
    
    patient.nl_xfm = lngtmpldir + 'nl_' + patient.id + '.xfm'
    patient.nl_det = lngtmpldir + "nl_" + patient.id + "_det.mnc"
    patient.lock = patient.patientdir + os.sep + patient.id + '.sge.lock'

    if len(patient) == 1:
        tp = patient.keys()[0]  # taking the only timepoint

        # a) linear
        # patient.template["linear_template"] = patient[tp].stx_mnc["t1"]
        # patient.template["linear_template_mask"]     = patient[tp].stx_mnc["masknoles"]

        patient.template['stx2_xfm'] = lngtmpldir + 'lin_template_' \
            + patient.id + '_t1.xfm'

        # b) non-linear

        patient.template['nl_template'] = patient[tp].stx2_mnc['t1']
        patient.template['nl_template_mask'] = patient[tp].stx2_mnc['masknoles']
        patient.nl_xfm = patient[tp].tpdir + 'nl/nl_' + patient.id + '_' + tp + '.xfm'

        # setting the patient and the timepoint to be the same nl_xfm

        patient[tp].nl_xfm = patient.nl_xfm
  # end of setting names

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

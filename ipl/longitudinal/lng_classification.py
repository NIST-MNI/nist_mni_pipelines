# -*- coding: utf-8 -*-

#
# @author Daniel
# @date 10/07/2011

#
# Longitudinal classification
#

from .general import *
from ipl.minc_tools import mincTools,mincError
from ipl import minc_qc
import shutil

version = '1.0'


# Run preprocessing using patient info
# - Function to read info from the pipeline patient
# - pipeline_version is employed to select the correct version of the pipeline

def pipeline_lng_classification(patient):

    if len(patient) == 1:
        print(' -- No lng_classification for one timepoint!')
        return 1

    # make a vector with all output images

    allDone = True
    for (i, tp) in patient.items():
        if not os.path.exists(tp.stx2_mnc['lng_classification']) \
            or not os.path.exists(tp.qc_jpg['lngclassification']):
            allDone = False
            break

    if allDone:
        print(' -- pipeline_lng_classification is done!')
        return 1

    if patient.pipeline_version == '1.0':
        lng_classification_v10(patient)  # beast by simon fristed
    else:
        print(' -- Chosen version not found!')

    # @todo add history



def create_prior(priors,masks,outs,fwhm=8.0):
    
    with mincTools() as minc:
        wms=[]
        gms=[]
        csfs=[]
        bkgs=[]

        for i,j in enumerate(priors):
            wm  = minc.tmp("wm_{}.mnc".format(i))
            gm  = minc.tmp("gm_{}.mnc".format(i))
            csf = minc.tmp("csf_{}.mnc".format(i))
            bkg = minc.tmp("bkg_{}.mnc".format(i))
        
            if len(outs)==4:
                minc.calc([priors[i],masks[i]],'if(A[0]<0.5 && A[1]>0.5){1}else{0}', bkg)
            
            minc.calc([priors[i]],'A[0]>0.5&&A[0]<1.5?1:0',csf,labels=True)
            minc.calc([priors[i]],'A[0]>1.5&&A[0]<2.5?1:0',gm ,labels=True)
            minc.calc([priors[i]],'A[0]>2.5&&A[0]<3.5?1:0',wm ,labels=True)
            
            wms.append(wm)
            gms.append(gm)
            csfs.append(csf)
            
            if len(outs)==4:
                bkgs.append(bkg)
        
        ave_wm =minc.tmp("average_wm.mnc")
        ave_gm =minc.tmp("average_gm.mnc")
        ave_csf=minc.tmp("average_csf.mnc")
        ave_bkg=minc.tmp("average_bkg.mnc")
        
        minc.average(wms, ave_wm, datatype='-float')
        minc.average(gms, ave_gm, datatype='-float')
        minc.average(csfs,ave_csf,datatype='-float')
        
        if len(outs)==4:
            minc.average(bkgs,ave_bkg,datatype='-float')

        ## Option A: Use directly the blurring
        usebkg=0
        if len(outs)==4:
            minc.blur(ave_bkg,outs[0],fwhm=fwhm)
            usebkg=1

        minc.blur(ave_wm, outs[2+usebkg],fwhm=fwhm)
        minc.blur(ave_gm, outs[1+usebkg],fwhm=fwhm)
        minc.blur(ave_csf,outs[0+usebkg],fwhm=fwhm)
        print("created {}".format(repr(outs)))

# pipeline_classify_prior.pl --mask /export01/data/vfonov/src1/nihpd_pipeline/validation_dataset/test_n4//subject04/1/stx2/stx2_subject04_1_mask.mnc \
#                      --model_dir /home/vfonov/data/viola03/models/icbm152_model_09c 
#                      --model_name mni_icbm152_t1_tal_nlin_sym_09c 
#                      --xfm /export01/data/vfonov/src1/nihpd_pipeline/validation_dataset/test_n4//subject04/1/nl/nl_subject04_1.xfm 
#                      --prior /tmp/tmpDHQFyeiplLongitudinalPipeline.py/prior_csf.mnc,/tmp/tmpDHQFyeiplLongitudinalPipeline.py/prior_gm.mnc,/tmp/tmpDHQFyeiplLongitudinalPipeline.py/prior_wm.mnc /export01/data/vfonov/src1/nihpd_pipeline/validation_dataset/test_n4//subject04/1/stx2/stx2_subject04_1_t1.mnc /export01/data/vfonov/src1/nihpd_pipeline/validation_dataset/test_n4//subject04/1/cls/lngcls_subject04_1.mnc

def classify_prior(inputs,output,mask=None,model_dir=None,model_name=None,xfm=None,priors=None,mrf=False):
    with mincTools() as minc:
        tags="{modeldir}/{modelname}_ntags_1000_prob_90_nobg.tag".format(modeldir=model_dir, modelname=model_name)
        nltags=minc.tmp("nltags.tag")
        
        if xfm is not None: 
            argsreg=['transform_tags',tags,xfm,nltags,'invert']
            minc.command(argsreg)
        else:
            nltags=tags

        cleantags=minc.tmp("clean.tag")
        argsclean=['cleantag','-clobber','-oldtag',nltags,'-newtag',cleantags,'-mode','101','-threshold','0.9',
                   '-difference','0.5',priors[0],'1',priors[1],'2',priors[2],'3','-comment','clean_apriori']
        minc.command(argsclean)

        # 3. Bayesian classificationregis
        ################
        tmpcls=minc.tmp("cls.mnc")

        # NOTE: classify have to have fixed to correctly work with priors
        argsclass=['classify','-clobber','-nocache','-bayes','-apriori','-volume',"{},{},{}".format(priors[0],priors[1],priors[2]),
                    '-tagfile',cleantags]
        argsclass.extend(inputs)
        argsclass.append(tmpcls)
        
        if mask is not None:
         argsclass.extend(['-mask',mask,'-user_mask_value','0.5'])
         
        minc.command(argsclass);


        ## Adding MRF
        ###################
        if mrf:
            #TODO run PVE!
            #if len(inputs)>1:
            #    do_cmd('pve3','-image',tmpcls,'-mask',mask,@ARGV,"$tmpdir/pve")
            #else:
            #    do_cmd('pve','-image',tmpcls,'-mask',mask,@ARGV,"$tmpdir/pve")
            #do_cmd('minccalc','-express','(A[0]>A[1]?(A[0]>A[2]?1:3):(A[1]>A[2]?2:3))*A[3]','-byte',"$tmpdir/pve_csf.mnc","$tmpdir/pve_gm.mnc","$tmpdir/pve_wm.mnc",$mask_file,$outfile_clean,'-clobber');
            pass
        else:
            shutil.copyfile(tmpcls,output)



def lng_classification_v10(patient):

    lng_cls = 'doBayes'

    # lng_cls="doGC4D"
    # lng_cls="doEM4D"

    # # doing the processing
    # ######################
    
    with mincTools() as minc:
        tmpdir  = minc.tempdir
        tmp_bkg = tmpdir + 'prior_bkg.mnc'
        tmp_csf = tmpdir + 'prior_csf.mnc'
        tmp_gm  = tmpdir + 'prior_gm.mnc'
        tmp_wm  = tmpdir + 'prior_wm.mnc'

        # take all cross sectional segmentations

        masks = []
        clss = []
        for (i, tp) in patient.items():
            if os.path.exists(tp.stx2_mnc['classification']):
                clss.append(tp.stx2_mnc['classification'])
                masks.append(tp.stx2_mnc['masknoles'])

        # create priors - background prior removed by SFE because of syntax problems
        # comm=['pipeline_create_prior.pl','--o',tmp_bkg+","+tmp_csf+","+tmp_gm+","+tmp_wm,'--mask',",".join(masks)]
        
        create_prior(clss, masks, [tmp_bkg, tmp_csf, tmp_gm, tmp_wm])

        # final classification method

        if lng_cls == 'doGC4D':

            # use GC4D for regularization
            # ############################
            # NOT READY!!
            # @warning only using T1-w for GC segmentation
            # @todo INCLUDE A MASK FOR EVERY TIMEPOINT

            # # @todo include the other sequences ing gcut4D
            # @todo compute for each timepoint, (mask-tmp_tissue)

            t1 = []
            masks = []

            gcut1 = []
            gcut2 = []
            lngclassif = []
            for (i, tp) in patient.items():
                if os.path.exists(tp.stx2_mnc['t1']):
                    t1.append(tp.stx2_mnc['t1'])
                if os.path.exists(tp.stx2_mnc['masknoles']):
                    masks.append(tp.stx2_mnc['masknoles'])

                # constructing tmpoutput

                gcut1.append(tmpdir + 'gcut1_' + i + '.mnc')
                gcut2.append(tmpdir + 'gcut2_' + i + '.mnc')
                lngclassif.append(tp.stx2_mnc['lng_classification'])

            # constructing output list

            # 1. Sum gw, wm

            tmp_black = tmpdir + 'dark_apriori.mnc'
            tmp_bright = tmpdir + 'bright_apriori.mnc'
            minc.command(['mincmath', '-add', tmp_wm, tmp_gm, tmp_bright],inputs=[tmp_wm, tmp_gm], outputs=[tmp_bright])

            # 2. Sum csf+gm
            minc.command(['mincmath', '-add', tmp_gm, tmp_csf, tmp_black],inputs=[tmp_gm, tmp_csf],outputs=[tmp_black])

            # 3. Do gc csf vs. gm+wm

            tmp_tissue = tmpdir + 'tmp_tissue.mnc'
            minc.command([ 'gcut4D', '-i', ','.join(t1), '-m',masks,
                '-t', tmp_bright, '-b', tmp_csf,
                '-a', '10', '-s', '10', '-n','2', '-p', '-o', ','.join(gcut1) ],
                inputs=t1 + masks + [tmp_bright, tmp_csf],
                outputs=gcut1)

            # 4. Do gc gm vs.wm in the target of 3
            #    We include CSF in the GM proba to avoid errors with the csf
            #    The masks are the output of the first gc

            binary_wm = tmpdir + 'binary_wm.mnc'
            minc.command(['gcut4D', '-i', ','.join(t1),
                '-m', ','.join(gcut1),'-t', tmp_wm,'-b', tmp_black, 
                '-a','10','-s','10', '-n', '2', '-p',
                '-o', ','.join(gcut2)],t1 + gcut1 + [tmp_wm, tmp_black],outputs=gcut2)

            # There is no background class so a third gc is not necessary
            # tmp_nottissue=tmpdir+"tmp_nottissue.mnc"
            # 5. Do gc csf vs. bkg in mask-target of 3.
            # binary_csf=tmpdir+"binary_csf.mnc"
            # comm=["gcut4D","-i",",".join(t1),"-m",tmp_nottissue,"-t",tmp_csf,"-b",tmp_bkg,"-a","10","-s","10","-n","2","-o",binary_csf]

            # 6 Combine results
            # for each timepoint!!!

            for i in range(len(lngclassif)):
                minc.calc([gcut2[i],gcut1[i],masks[i]],'if(A[0]>0.5){3}else if(A[1]>0.5){2} else if(A[2]>0.5){1} else{0}',
                    lngclassif[i])
        
        elif lng_cls == 'doEM4D':

            # use EM in 4D with trimmed likelihood
            # #####################################
            # taking all images, all modalities

            t1 = []
            t2 = []
            pd = []
            masks = []
            tps = []

            for (i, tp) in patient.items():
                tps.append(i)  # add for later
                if os.path.exists(tp.stx2_mnc['t1']):
                    t1.append(tp.stx2_mnc['t1'])
                if os.path.exists(tp.stx2_mnc['masknoles']):
                    masks.append(tp.stx2_mnc['masknoles'])

                if not patient.onlyt1 and 't2' in tp.stx2_mnc \
                    and os.path.exists(tp.stx2_mnc['t2']):
                    t2.append(tp.stx2_mnc['t2'])

                if not patient.onlyt1 and 'pd' in tp.stx2_mnc \
                    and os.path.exists(tp.stx2_mnc['pd']):
                    pd.append(tp.stx2_mnc['pd'])

            # Check all images are here

            timepoints = len(t1)
            modalities = 3
            uset2 = True
            usepd = True
            if patient.onlyt1:
                modalities = 1
                uset2 = False
                usept = False
            else:
                if len(t1) is not len(t2):
                    print(' -- Not the same number of t1 and t2: not using t2 images')
                    uset2 = False
                    modalities = modalities - 1
                if len(t1) is not len(pd):
                    print(' -- Not the same number of t1 and t2: not using t2 images')
                    usepd = False
                    modalities = modalities - 1

            # ordering images

            images = []
            for t in range(timepoints):
                images.append(t1[t])
                if uset2:
                    images.append(t2[t])
                if usepd:
                    images.append(pd[t])

            # Create command line
            # @todo change implementation to accept more than one linear mask
            # the robust is the number of outliers removed from the segmentation
            # the more sequences the more likely this will be necessary

            if modalities == 1:
                robust = '0.01'
            elif modalities == 2:
                robust = '0.02'
            else:
                robust = '0.05'
            tmpclassif = tmpdir + 'tmpclassif.mnc'
            comm = [
                'Classification',
                '-init',
                '200:10:1',
                '-iter',
                '20:100:80',
                '-m',
                ','.join(masks),
                '-t',
                str(timepoints),
                '-n',
                str(modalities),
                '-o',
                tmpclassif,
                '-r',
                robust,
                '-a',
                tmp_csf,
                tmp_gm,
                tmp_wm,
                ]
            comm.extend(images)
            classifN = tmpclassif[:-4] + '1.mnc'
            
            minc.command(comm,inputs=images,outputs=[classifN])

    # rename the output from classification

            for t in range(timepoints):
                classifN = tmpclassif[:-4] + str(t + 1) + '.mnc'
                shutil.move(classifN, patient[tps[t]].stx2_mnc['lng_classification'])
        else:
            # use bayesian classification
            # ###################################################
            for (i, tp) in patient.items():
                
                inputs=[tp.stx2_mnc['t1']]
                if not patient.onlyt1 and 't2' in tp.stx2_mnc:
                    inputs.append(tp.stx2_mnc['t2'])
                if not patient.onlyt1 and 'pd' in tp.stx2_mnc:
                    inputs.append(tp.stx2_mnc['pd'])
                
                classify_prior(inputs,tp.stx2_mnc['lng_classification'],
                            mask=tp.stx2_mnc['masknoles'],
                            model_dir=patient.modeldir,
                            model_name=patient.modelname,
                            xfm=tp.nl_xfm,
                            priors=[tmp_csf,tmp_gm,tmp_wm])

        # create QC images for all timepoints

        for (i, tp) in patient.items():
            minc_qc.qc(
                tp.stx2_mnc['t1'],
                tp.qc_jpg['lngclassification'],
                title=tp.qc_title,
                image_range=[0,120],dpi=200,use_max=True,
                mask=tp.stx2_mnc['lng_classification'],
                samples=20,bg_color="black",fg_color="white")
            

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 10/07/2011
#
# Generate average model
from __future__ import print_function

import shutil
import os
import sys
import csv
import traceback
import argparse
import json

from ipl.minc_tools    import mincTools,mincError

# 
from ipl.lp.pipeline        import standard_pipeline
#from ipl.lp.iter_pipeline   import iter_step
from ipl.lp.structures      import MriScan,MriTransform,MriQCImage,MriAux
from ipl.lp.structures      import save_pipeline_output,load_pipeline_output

from scoop import futures, shared


default_pipeline_parameters= { 
                    # 0 iteration
                    'model':     'mni_icbm152_t1_tal_nlin_sym_09c',
                    'model_dir': '/opt/minc/share/icbm152_model_09c',
                    't1w_nuc':   {},
                    't2w_nuc':   {},
                    'pdw_nuc':   {},
                    'beast':     { 'beastlib':  '/opt/minc/share/beast-library-1.1' },
                    'tissue_classify': {},
                    'lobe_segment': {},
                    'nl':        False,
                    'lobes':     False,
                    'cls'  :     False,
                    'qc':        True,
                    'denoise':   {},
                    't1w_stx':   {
                        'type':'elx',
                        'resample':False,
                        #'options': {
                            #'levels': 2,
                            #'conf':  {'2':1000,'1':1000}, 
                            #'blur':  {'2':4, '1': 2 }, 
                            #'shrink':{'2':4, '1': 2 },
                            #'convergence':'1.e-8,20',
                            #'cost_function':'MI',
                            #'cost_function_par':'1,32,random,0.3',
                            #'transformation':'similarity[ 0.3 ]',
                            #}
                    },
                    'nl_reg': {
                        'type':'ants',
                        'start': 16,
                        'level': 2,
                        'options': {
                            'conf':{'16':100,'8':100,'4':20,'2':10},
                            'blur':{'16':16,'8':8,'4':4,'2':2},
                            'shrink':{'16':16,'8':8,'4':4,'2':2},
                            'cost_function':'CC',
                            'cost_function_par':'1,3,Regular,1.0',
                        }
                    }
                }

preprocessing_pipeline_parameters={
                    # 0 iteration
                    'model':     'mni_icbm152_t1_tal_nlin_sym_09c',
                    'model_dir': '/opt/minc/share/icbm152_model_09c',
                    't1w_nuc':   {'distance':50},
                    't1w_clp':   {},
                    't2w_nuc':   {'distance':50},
                    'pdw_nuc':   {'distance':50},
                    'beast':     { 'beastlib':  '/opt/minc/share/beast-library-1.1' },
                    'tissue_classify': {},
                    'lobe_segment': {},
                    'nl':        False,
                    'lobes':     False,
                    'cls'  :     False,
                    'qc':        True,
                    'denoise':   {'anlm':True, },
                    't1w_stx':   {
                        'type':'elx' ,
                        'resample':False,
                        #'options': {
                            #'levels': 2,
                            #'conf':  {'2':1000,'1':1000}, 
                            #'blur':  {'2':4, '1': 2 }, 
                            #'shrink':{'2':4, '1': 2 },
                            #'convergence':'1.e-8,20',
                            #'cost_function':'MI',
                            #'cost_function_par':'1,32,random,0.3',
                            #'transformation':'similarity[ 0.3 ]',
                            #}
                    }
                }



def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Run pipeline')

    parser.add_argument("--debug",
                    action="store_true",
                    dest="debug",
                    default=False,
                    help="Print debugging information" )

    parser.add_argument("--options",
                    help="Segmentation options in json format",
                    dest="options")

    parser.add_argument("--iterations",
                    help="Use iterative algorithm with this number of iterations",
                    dest="iterations",
                    type=int, default=None)

    parser.add_argument("--load",
                    help="Load result of previous run",
                    dest="load")

    parser.add_argument("--load-qc",
                    help="load QC result of previous run",
                    dest="load_qc")

    parser.add_argument("--output",
                    help="Output directory, required for application of method",
                    dest="output",
                    default=".")
    
    parser.add_argument("--manual",
                    help="Directory with manually created files")

    parser.add_argument("--csv",
                         help="Input data, in standard form:id,visit,t1w[,t2w,pdw,sex,age,geot1,geot2,lesions]",
                         dest="csv")

    parser.add_argument("-M","--modalities",
                    help="Additional Processing modalities",
                    dest="modalities",
                    default="t2w,pdw")

    #parser.add_argument("--json",
                        #help="load json description")

    #parser.add_argument("--save",
                        #help="Save information to json file")

    parser.add_argument("subject",
                        help="Subject id",
                        nargs="?")

    parser.add_argument("visit",
                        help="Subject visit",
                        nargs="?")

    parser.add_argument("scans",
                        help="Subject scan",
                        nargs="*")
    
    parser.add_argument("--corr",
                        help="Distortion correct",
                        nargs="*")
    
    options = parser.parse_args()

    if options.debug:
        print(repr(options))

    return options


def main():
    options = parse_options()
    pipeline_parameters=default_pipeline_parameters
    pipeline_info={}
    modalities=options.modalities.split(',')
    try:
        if options.options is not None:
            try:
                with open(options.options,'r') as f:
                    pipeline_parameters=json.load(f)
            except :
                print("Error reading:{}".format(options.options))
                raise
        
        
        if (options.csv is not None) or (options.load is not None):
            inputs=[]
            
            if options.load is not None:
                inputs=load_pipeline_output(options.load)
            else:
                with open(options.csv, 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
                    for row in reader:
                        if len(row)>=3:
                            data_name='{}_{}'.format(row[0],row[1])
                            
                            t1w=MriScan(name=data_name, 
                                        scan=row[2], 
                                        modality='t1w', 
                                        mask=None)
                            
                            t2w=None
                            pdw=None
                            corr_t1w=None
                            corr_t2w=None
                            age=None
                            sex=None
                            
                            add=[]
                            
                            for l,ll in enumerate(modalities):
                                if len(row)>(3+l) and row[3+l]!='':
                                    add.append(MriScan(name=data_name, scan=row[3+l], modality=ll, mask=None))
                                    
                            if len(row)>(4+len(modalities)) and row[(4+len(modalities))]!='':
                                age=float(row[(4+len(modalities))])
                            if len(row)>(5+len(modalities)) and row[(5+len(modalities))]!='':
                                sex=float(row[(5+len(modalities))])
                            if len(row)>(6+len(modalities))  and row[(6+len(modalities))]!='':
                                corr_t1w=MriTransform(None,'corr_t1w',xfm=row[(6+len(modalities))]) # corr_t1w
                            if len(row)>(7+len(modalities)) and  row[(7+len(modalities))]!='':
                                corr_t2w=MriTransform(None,'corr_t2w',xfm=row[(7+len(modalities))]) # corr_t1w

                            line={  'subject':row[0],
                                    'visit':  row[1],
                                    # MRI 
                                    't1w':t1w, 
                                    # demographic info
                                    'age':age,
                                    'sex':sex , 
                                    # distortion correction
                                    'corr_t1w':corr_t1w, 
                                    'corr_t2w':corr_t2w,
                                    # timepoint specific model
                                    'model_name':None,
                                    'model_dir':None,
                                }
                            # 
                            if len(add)>0:
                                line['add']=add
                            
                            inputs.append( line )
                        else:
                            print("Error, unexpected line format:{}".format(repr(row)))
                            raise Exception()
            
            pipeline_parameters['debug']=options.debug
            if options.debug:
                print(repr(inputs))
            
            run_pipeline=[]
            for (i, s) in enumerate(inputs):
                output_dir=options.output+os.sep+s['subject']+os.sep+s['visit']
                manual_dir=None
                
                if options.manual is not None:
                    manual_dir=options.manual+os.sep+s['subject']+os.sep+s['visit']
                
                
                run_pipeline.append( futures.submit( 
                    standard_pipeline,
                        s,
                        output_dir, 
                        options=pipeline_parameters ,
                        work_dir=output_dir,
                        manual_dir=manual_dir
                    ))
            #
            # wait for all to finish
            #
            futures.wait(run_pipeline, return_when=futures.ALL_COMPLETED)

            for j,i in enumerate(run_pipeline):
                inputs[j]['output']=i.result()

            save_pipeline_output(inputs,options.output+os.sep+'summary.json')

        elif options.scans   is not None and \
             options.subject is not None and \
             options.visit   is not None:
            # run on a single subject
            data_name='{}_{}'.format(options.subject,options.visit)
            pipeline_parameters['debug']=options.debug
            output_dir=options.output+os.sep+options.subject+os.sep+options.visit
            manual_dir=None
            
            if options.manual is not None:
                manual_dir=options.manual+os.sep+options.subject+os.sep+options.visit
            
            add=[]
            
            for l,ll in enumerate(modalities):
                if len(options.scans)>(l+1):
                    add.append(MriScan(name=data_name, 
                                    scan=options.scans[(l+1)], 
                                    modality=ll, 
                                    mask=None))
            
            if len(add)==0: add=None
            
            info={  'subject':options.subject, 
                    'visit':  options.visit, 
            
                    't1w':    MriScan(name=data_name, 
                                      scan=options.scans[0], 
                                      modality='t1w', 
                                      mask=None),
                    'add':   add
                 }
            
            if options.corr is not None:
                
                info['corr_t1w']=MriTransform(None,'corr_t1w',xfm=options.corr[0])
                
                if len(options.corr)>1:
                    info['corr_t2w']=MriTransform(None,'corr_t2w',xfm=options.corr[1])
            
            ret=standard_pipeline( info,
                               output_dir, 
                               options=pipeline_parameters, 
                               work_dir=output_dir,
                               manual_dir=manual_dir
                             )
            # TODO: make a check if there is a summary file there already?
            #save_pipeline_output([info],options.output+os.sep+'summary.json')
            
        else:
            print("Refusing to run without input data, run --help")
            exit(1)
    except :
        print("Exception :{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

import shutil
import os
import sys
import csv
import copy
import json
import random

# MINC stuff
from ipl.minc_tools import mincTools,mincError

# scoop parallel execution
from scoop import futures, shared

from .fuse             import *
from .structures       import *
from .resample         import *
from .filter           import *
from .analysis         import *

def run_grading_experiment( input_scan,
                            input_seg,
                            grading_library,
                            output_experiment,
                            grading_parameters={},
                            debug=False, 
                            mask=None,
                            work_dir=None,
                            fuse_variant='fuse',
                            add=[],
                            cleanup=False,
                            group=None,
                            grading=None
                            ):
    """run a grading experiment: perform grading and compare with ground truth

    Arguments:
    input_scan -- input scan object MriDataset
    input_seg -- input segmentation file name (ground truth)
    grading_library -- segmntation library object
    output_experiment -- prefix for output

    Keyword arguments:
    grading_parameters -- paramteres for segmentation algorithm, 
    debug -- debug flag, (default False)
    mask -- mask file name to restrict segmentation , (default None)
    work_dir -- work directory, (default None - use output_experiment)
    fuse_variant -- name of fusion parameters, (default 'fuse' )
    add -- additional modalities [T2w,PDw etc]
    cleanup -- flag to clean most of the temporary files
    
    """
    try:
        relabel=grading_library.get("label_map",None)
        
        if relabel is not None and isinstance(relabel, list) :
            _r={i[0]:i[1] for i in relabel}
            relabel=_r
        
        if debug:
            if not os.path.exists(os.path.dirname(output_experiment)):
                os.makedirs(os.path.dirname(output_experiment))
            with open(output_experiment+'_par.json','w') as f:
                json.dump(grading_parameters,f,indent=1)

        (output_seg, output_grad, output_volumes, output_info) = fusion_grading(
                    input_scan, 
                    grading_library,
                    output_experiment,
                    input_mask=mask,
                    parameters=grading_parameters,
                    debug=debug,
                    work_dir=work_dir,
                    fuse_variant=fuse_variant,
                    add=add,
                    cleanup=cleanup)
        
        stats = calc_similarity_stats( input_seg, 
                                       output_seg, 
                                       output_stats = output_experiment+'_stats.csv',
                                       relabel =      relabel)
        
        stats['group']=group
        stats['grading']=grading
        stats['result']=output_volumes
        
        name=os.path.basename(input_scan).rsplit('.mnc',1)[0]
        grading_map=work_dir+os.sep+fuse_variant+'_'+name+'_grading_nl.mnc'

        lin_xfm=None
        nl_xfm=None
        
        if output_info['bbox_initial_xfm'] is not None:
            lin_xfm=output_info['bbox_initial_xfm'].xfm
            
        if output_info['nonlinear_xfm'] is not None:
            nl_xfm=output_info['nonlinear_xfm'].xfm
            
        create_grading_map(output_grad, grading_map, 
                         lin_xfm=lin_xfm,
                         nl_xfm=nl_xfm,
                         template=grading_library.get('local_model',None))

        output_info['stats']=stats
        output_info['output']=output_seg
        output_info['ground_truth']=input_seg
        output_info['grading_map']=grading_map
        output_info['group']=group
        output_info['grading']=grading
        output_info['volumes']=output_volumes
        
        with open(output_experiment+'_out.json','w') as f:
            json.dump(output_info,f,indent=1, cls=GMRIEncoder)
            
        with open(output_experiment+'_stats.json','w') as f:
            json.dump(stats,f,indent=1, cls=GMRIEncoder)
        
        return (stats, output_info)

    except mincError as e:
        print("Exception in run_grading_experiment:{}".format( str(e)) )
        traceback.print_exc( file=sys.stderr )
        raise

    except :
        print("Exception in run_grading_experiment:{}".format( sys.exc_info()[0]) )
        traceback.print_exc( file=sys.stderr )
        raise


def loo_cv_fusion_grading(validation_library, 
                          grading_library, 
                          output, 
                          grading_parameters,
                          debug=False,
                          fuse_variant='fuse',
                          cv_variant='cv',
                          cleanup=False,
                          cv_iter=None):
    '''Run leave-one-out cross-validation experiment'''
    # for each N subjects run segmentation and compare
    # Right now run LOOCV 
    if not os.path.exists(output):
        try:
            os.makedirs(output)
        except:
            pass # assume directory was created by competing process

    results=[]
    results_json=[]
    
    modalities=grading_library.get('modalities',1)-1
    
    print("cv_iter={}".format(repr(cv_iter)))
    
    for (i,j) in enumerate(validation_library):
        n = os.path.basename(j[0]).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
        output_experiment = output+os.sep+n+'_'+cv_variant
        
        validation_sample  = j[0]
        validation_segment = j[1]
        
        validation_group   = int(  j[-2] )
        validation_grading = float(j[-1] )
        
        add=j[2:2+modalities]
        
        experiment_grading_library=copy.deepcopy(grading_library)

        # remove sample
        experiment_grading_library['library']=[ _i for _i in grading_library['library'] if _i[2].find(n)<0 ]
        
        if (cv_iter is None) or (i == cv_iter):
            results.append( futures.submit( 
                run_grading_experiment, 
                validation_sample, validation_segment, 
                experiment_grading_library,
                output_experiment,
                grading_parameters=grading_parameters,
                debug=debug,
                work_dir=output+os.sep+'work_'+n+'_'+fuse_variant,
                fuse_variant=fuse_variant,
                add=add,
                cleanup=cleanup,
                group=validation_group,
                grading=validation_grading
                ))
        else:
            results_json.append( (output_experiment+'_stats.json',
                                  output_experiment+'_out.json') )
    
    futures.wait(results, return_when=futures.ALL_COMPLETED)

    stat_results=[]
    output_results=[]
    
    if cv_iter is None:
        stat_results  = [ _i.result()[0] for _i in results ]
        output_results= [ _i.result()[1] for _i in results ]
        
    elif cv_iter==-1:
        # TODO: load from json files
        for _i in results_json:
            with open(_i[0],'r') as _f:
                stat_results.append(json.load(_f))
            with open(_i[1],'r') as _f:
                output_results.append(json.load(_f))
        
    return (stat_results, output_results)

def full_cv_fusion_grading(validation_library, 
                           grading_library, 
                           output,
                           grading_parameters, 
                           cv_iterations,
                           cv_exclude,
                           debug=False,
                           fuse_variant='fuse',
                           cv_variant='cv',
                           cleanup=False,
                           cv_iter=None):
    if cv_iter is not None:
        raise "Not Implemented!"
    
    validation_library_idx=range(len(validation_library))
    # randomly exlcude samples, repeat 
    results=[]
    if not os.path.exists(output):
        try:
            os.makedirs(output)
        except:
            pass # assume directory was created by competing process
    
    modalities=grading_library.get('modalities',1)-1
    
    for i in range( cv_iterations ):
        #TODO: save this list in a file
        rem_list=[]
        ran_file=output+os.sep+ ('random_{}_{}.json'.format(cv_variant,i))
        
        if not os.path.exists( ran_file ):
            rem_list=random.sample( validation_library_idx, cv_exclude )
            
            with open( ran_file , 'w') as f:
                json.dump(rem_list,f)
        else:
            with open( ran_file ,'r') as f:
                rem_list=json.load(f)
                
        # list of subjects 
        rem_items=[ validation_library[j] for j in rem_list ]
        
        rem_n=[os.path.basename(j[0]).rsplit('.gz',1)[0].rsplit('.mnc',1)[0] for j in rem_items]
        rem_lib=[]
        
        for j in rem_n:
            rem_lib.extend( [ k for (k,t) in enumerate( grading_library['library'] ) if t[2].find(j)>=0 ] )

        if debug: print(repr(rem_lib))
        rem_lib=set(rem_lib)
        #prepare exclusion list
        experiment_grading_library=copy.deepcopy(grading_library)
        
        experiment_grading_library['library']=\
            [ k for j,k in enumerate( grading_library['library'] )  if j not in rem_lib ]
        
        for j,k in enumerate(rem_items):
            output_experiment=output+os.sep+('{}_{}_{}'.format(i,rem_n[j],cv_variant))
            work_dir=output+os.sep+('work_{}_{}_{}'.format(i,rem_n[j],fuse_variant))

            results.append( futures.submit( 
                run_grading_experiment, k[0], k[1], 
                experiment_grading_library,
                output_experiment,
                grading_parameters=grading_parameters,
                debug=debug,
                work_dir=work_dir,
                fuse_variant=fuse_variant,
                add=k[4:4+modalities],
                cleanup=cleanup,
                group=int(k[-2]),
                grading=float(k[-1])
                ))
                
    futures.wait(results, return_when=futures.ALL_COMPLETED)
    stat_results   = [ i.result()[0] for i in results ]
    output_results = [ i.result()[1] for i in results ]
    
    return ( stat_results, output_results )

    
def cv_fusion_grading( cv_parameters,
                       grading_library,
                       output,
                       grading_parameters,
                       debug=False,
                       cleanup=False,
                       cv_iter=None):
    '''Run cross-validation experiment
    for each N subjects run segmentation and compare
    Right now run LOOCV or random CV
    '''
    
    # TODO: implement more realistic, random schemes
    validation_library=cv_parameters['validation_library']
    
    # maximum number of iterations
    cv_iterations=cv_parameters.get('iterations',-1)
    
    # number of samples to exclude
    cv_exclude=cv_parameters.get('cv',1)

    # use to distinguish different versions of label fusion
    fuse_variant=cv_parameters.get('fuse_variant','fuse')

    # use to distinguish different versions of cross-validation
    cv_variant=cv_parameters.get('cv_variant','cv')
    
    cv_output=output+os.sep+cv_variant+'_stats.json'
    res_output=output+os.sep+cv_variant+'_res.json'
    
    if validation_library is not list:
        with open(validation_library,'r') as f:
            validation_library=list(csv.reader(f))

    print("Validation library:",validation_library)
    stat_results=None
    output_results=None
    
    if cv_iter is not None:
        cv_iter=int(cv_iter)

    if cv_iterations==-1 and cv_exclude==1: # simle LOO cross-validation
        (stat_results, output_results) = loo_cv_fusion_grading(validation_library, 
                                             grading_library,
                                             output, grading_parameters,
                                             debug=debug,
                                             cleanup=cleanup,
                                             fuse_variant=fuse_variant,
                                             cv_variant=cv_variant,
                                             cv_iter=cv_iter)
    else: # arbitrary number of iterations
        (stat_results, output_results) = full_cv_fusion_grading(validation_library, 
                                              grading_library,
                                              output, grading_parameters,
                                              cv_iterations, cv_exclude,
                                              debug=debug,
                                              cleanup=cleanup,
                                              fuse_variant=fuse_variant,
                                              cv_variant=cv_variant,
                                              cv_iter=cv_iter)

    if cv_iter is None or cv_iter==-1:
        # build glim-image tables (?)
        results=[]
        
        #for GLIM image
        with open(output+os.sep+cv_variant+'_grading.glim','w') as f:
            for k in output_results:
                group=k['group']
                grading=k['grading']
                grading_map=k['grading_map']
                f.write("{} {} {}\n".format(grading_map,1.0,grading))
        
        #for RMINC image
        with open(output+os.sep+cv_variant+'_grading.csv','w') as f:
            f.write("grading_map,group,grading\n")
            for k in output_results:
                group=k['group']
                grading=k['grading']
                grading_map=k['grading_map']
                f.write("{},{},{}\n".format(grading_map,group,grading))

        #TODO: run glim-image or RMINC here
        
        with open(cv_output,'w') as f:
            json.dump(stat_results, f, indent=1 )

        with open(res_output,'w') as f:
            json.dump(output_results, f, indent=1, cls=GMRIEncoder)

        return stat_results
    else:
        # we assume that results will be available later
        return None

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

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
from .train_ec         import *
from .filter           import *
from .analysis         import *

def run_segmentation_experiment( input_scan,
                                 input_seg,
                                 segmentation_library,
                                 output_experiment,
                                 segmentation_parameters={},
                                 debug=False, 
                                 mask=None,
                                 work_dir=None,
                                 ec_parameters=None,
                                 ec_variant='ec',
                                 fuse_variant='fuse',
                                 regularize_variant='gc',
                                 add=[],
                                 cleanup=False,
                                 presegment=None,
                                 train_list=None):
    """run a segmentation experiment: perform segmentation and compare with ground truth

    Arguments:
    input_scan -- input scan object MriDataset
    input_seg -- input segmentation file name (ground truth)
    segmentation_library -- segmntation library object
    output_experiment -- prefix for output

    Keyword arguments:
    segmentation_parameters -- paramteres for segmentation algorithm, 
    debug -- debug flag, (default False)
    mask -- mask file name to restrict segmentation , (default None)
    work_dir -- work directory, (default None - use output_experiment)
    ec_parameters -- error correction paramters, (default None)
    ec_variant -- name of error correction parameters setting , (default 'ec')
    fuse_variant -- name of fusion parameters, (default 'fuse' )
    regularize_variant -- name of regularization parameters, (default 'gc')
    add -- additional modalities [T2w,PDw etc]
    cleanup -- flag to clean most of the temporary files
    presegment -- use pre-segmented result (when comparing with external tool)
    """
    try:
        relabel=segmentation_library.get("label_map",None)
        
        if relabel is not None and isinstance(relabel, list) :
            _r={i[0]:i[1] for i in relabel}
            relabel=_r

        if ec_parameters is not None:
            _ec_parameters=copy.deepcopy(ec_parameters)
            # let's train error correction!

            if work_dir is not None:
                fuse_output=work_dir+os.sep+fuse_variant+'_'+regularize_variant
            else:
                fuse_output=output_experiment+os.sep+fuse_variant+'_'+regularize_variant

            _ec_parameters['work_dir']=fuse_output
            _ec_parameters['output']=ec_output=fuse_output+os.sep+ec_variant+'.pickle'
            _ec_parameters['variant']=ec_variant
            
            train_ec_loo(   segmentation_library,
                            segmentation_parameters=copy.deepcopy(segmentation_parameters),
                            ec_parameters=_ec_parameters,
                            debug=debug,
                            fuse_variant=fuse_variant,
                            regularize_variant=regularize_variant,
                            cleanup=cleanup,
                            ext=(presegment is not None),
                            train_list=train_list  )

            segmentation_parameters['ec_options']=copy.deepcopy(ec_parameters)
            segmentation_parameters['ec_options']['training']=ec_output

        if debug:
            if not os.path.exists(os.path.dirname(output_experiment)):
                os.makedirs(os.path.dirname(output_experiment))
            with open(output_experiment+'_par.json','w') as f:
                json.dump(segmentation_parameters,f,indent=1)
        
        (output_file, output_info) = fusion_segment(
                    input_scan, 
                    segmentation_library,
                    output_experiment,
                    input_mask=mask,
                    parameters=segmentation_parameters,
                    debug=debug,
                    work_dir=work_dir,
                    ec_variant=ec_variant,
                    fuse_variant=fuse_variant,
                    regularize_variant=regularize_variant,
                    add=add,
                    cleanup=cleanup,
                    presegment=presegment)
        
        stats = calc_similarity_stats( input_seg, output_file, 
                                       output_stats = output_experiment+'_stats.csv',
                                       use_labels   = output_info['used_labels'],
                                       relabel      = relabel )
        
        remap = segmentation_library.get('map',{})
        labels_used=[]
        error_maps=[]
        
        if any(remap):
            for (i,j) in remap.items():
                labels_used.append( int(j) )
        else:
            # assume binary mode
            labels_used=[1]
        
        for i in labels_used: 
            error_maps.append( work_dir+os.sep+fuse_variant+'_'+regularize_variant+'_error_{:03d}.mnc'.format(i) )
            
        lin_xfm=None
        nl_xfm=None
        if output_info['bbox_initial_xfm'] is not None:
            lin_xfm=output_info['bbox_initial_xfm'].xfm
            
        if output_info['nonlinear_xfm'] is not None:
            nl_xfm=output_info['nonlinear_xfm'].xfm
            
        create_error_map( input_seg, output_file, error_maps, 
                          lin_xfm=lin_xfm,
                          nl_xfm=nl_xfm,
                          template=segmentation_library.get('local_model',None),
                          label_list=labels_used )

        output_info['stats']       = stats
        output_info['output']      = output_file
        output_info['ground_truth']= input_seg
        output_info['error_maps']  = error_maps
        
        if presegment is not None:
            output_info['presegment']=presegment

        with open(output_experiment+'_out.json','w') as f:
            json.dump(output_info,f,indent=1, cls=MRIEncoder)
            
        with open(output_experiment+'_stats.json','w') as f:
            json.dump(stats,f,indent=1, cls=MRIEncoder)

        return (stats, output_info)

    except mincError as e:
        print("Exception in run_segmentation_experiment:{}".format( str(e)) )
        traceback.print_exc( file=sys.stdout )
        raise

    except:
        print("Exception in run_segmentation_experiment:{}".format( sys.exc_info()[0]) )
        traceback.print_exc( file=sys.stdout )
        raise


def loo_cv_fusion_segment(validation_library, 
                          segmentation_library, 
                          output, 
                          segmentation_parameters,
                          ec_parameters=None,
                          debug=False,
                          ec_variant='ec',
                          fuse_variant='fuse',
                          cv_variant='cv',
                          regularize_variant='gc',
                          cleanup=False,
                          ext=False,
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
    
    modalities=segmentation_library.get('modalities',1)-1
    print("cv_iter={}".format(repr(cv_iter)))
    
    for (i,j) in enumerate(validation_library):
        
        n = os.path.basename(j[0]).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
        output_experiment = output+os.sep+n+'_'+cv_variant
        
        validation_sample  = j[0]
        validation_segment = j[1]
        
        presegment=None
        add=[]
        
        if ext:
            presegment=j[2]
            add=j[3:3+modalities]
        else:
            add=j[2:2+modalities]
        
        # remove training sample (?)
        _validation_library=validation_library[0:i]
        _validation_library.extend(validation_library[i+1:len(validation_library)])
        
        experiment_segmentation_library = copy.deepcopy(segmentation_library)

        # remove sample
        experiment_segmentation_library.library=[ _i for _i in segmentation_library.library if _i[0].find(n)<0 ]
        
        if (cv_iter is None) or (i == cv_iter):
            results.append( futures.submit( 
                run_segmentation_experiment, 
                validation_sample, validation_segment, 
                experiment_segmentation_library,
                output_experiment,
                segmentation_parameters=segmentation_parameters,
                debug=debug,
                work_dir=output+os.sep+'work_'+n+'_'+fuse_variant,
                ec_parameters=ec_parameters,
                ec_variant=ec_variant, 
                fuse_variant=fuse_variant,
                regularize_variant=regularize_variant,
                add=add,
                cleanup=cleanup,
                presegment=presegment,
                train_list=_validation_library
                ))
        else:
            results_json.append( (output_experiment+'_stats.json',
                                  output_experiment+'_out.json') )
    
    print("Waiting for {} jobs".format(len(results)))
    futures.wait(results, return_when=futures.ALL_COMPLETED)
    
    stat_results=[]
    output_results=[]
    
    if cv_iter is None:
        stat_results  = [ _i.result()[0] for _i in results ]
        output_results= [ _i.result()[1] for _i in results ]
    elif cv_iter==-1:
        # TODO: load from json files
        for _i in results_json:
            if os.path.exists(_i[0]) and os.path.exists(_i[1]):# VF: a hack
                with open(_i[0],'r') as _f:
                    stat_results.append(json.load(_f))
                with open(_i[1],'r') as _f:
                    output_results.append(json.load(_f))
            else:
               if not os.path.exists(_i[0]):
                   print("Warning: missing file:{}".format(_i[0]))
               if not os.path.exists(_i[1]):
                   print("Warning: missing file:{}".format(_i[1]))

    return (stat_results, output_results)
    

def full_cv_fusion_segment(validation_library, 
                           segmentation_library, 
                           output,
                           segmentation_parameters, 
                           cv_iterations,
                           cv_exclude,
                           ec_parameters=None,
                           debug=False,
                           ec_variant='ec',
                           fuse_variant='fuse',
                           cv_variant='cv',
                           regularize_variant='gc',
                           cleanup=False,
                           ext=False,
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
        
    modalities = segmentation_library.modalities-1
    
    for i in range( cv_iterations ):
        #TODO: save this list in a file
        rem_list = []
        ran_file = output+os.sep+ ('random_{}_{}.json'.format(cv_variant, i))
        
        if not os.path.exists( ran_file ):
            rem_list = random.sample(validation_library_idx, cv_exclude)
            
            with open( ran_file , 'w') as f:
                json.dump(rem_list,f)
        else:
            with open( ran_file ,'r') as f:
                rem_list = json.load(f)
                
        # list of subjects 
        rem_items = [validation_library[j] for j in rem_list]
        
        rem_n = [os.path.basename(j[0]).rsplit('.gz', 1)[0].rsplit('.mnc', 1)[0] for j in rem_items]
        rem_lib = []
        val_lib = []
        
        for j in rem_n:
            rem_lib.extend([k for (k, t) in enumerate(segmentation_library.library) if t[0].find(j) >= 0])
            val_lib.extend([k for (k, t) in enumerate(validation_library)           if t[0].find(j) >= 0])
            
            
        if debug: print(repr(rem_lib))
        rem_lib = set(rem_lib)
        val_lib = set(val_lib)
        
        #prepare exclusion list
        experiment_segmentation_library = copy.deepcopy(segmentation_library)
        
        experiment_segmentation_library.library = \
            [k for j, k in enumerate(segmentation_library.library) if j not in rem_lib]
        
        _validation_library = \
            [k for j, k in enumerate( validation_library ) if j not in val_lib ]
        
        for j, k in enumerate(rem_items):
            
            output_experiment = output+os.sep+('{}_{}_{}'.format(i, rem_n[j], cv_variant))
            work_dir          = output+os.sep+('work_{}_{}_{}'.format(i, rem_n[j], fuse_variant))
            
            validation_sample  = k[0]
            validation_segment = k[1]
            
            presegment = None
            shift = 2
            
            if ext:
                presegment = k[2]
                shift = 3
            
            results.append( futures.submit( 
                run_segmentation_experiment, validation_sample, validation_segment, 
                experiment_segmentation_library,
                output_experiment,
                segmentation_parameters=segmentation_parameters,
                debug=debug,
                work_dir=work_dir,
                ec_parameters=ec_parameters,
                ec_variant=ec_variant, 
                fuse_variant=fuse_variant, 
                regularize_variant=regularize_variant,
                add=k[shift:shift+modalities],
                cleanup=cleanup,
                presegment=presegment,
                train_list=_validation_library
                ))
                
    futures.wait(results, return_when=futures.ALL_COMPLETED)
    stat_results   = [ i.result()[0] for i in results ]
    output_results = [ i.result()[1] for i in results ]
    
    return ( stat_results, output_results )

    
def cv_fusion_segment( cv_parameters,
                       segmentation_library,
                       output,
                       segmentation_parameters,
                       ec_parameters=None,
                       debug=False,
                       cleanup=False,
                       ext=False,
                       extlib=None,
                       cv_iter=None ):
    '''Run cross-validation experiment
    for each N subjects run segmentation and compare
    Right now run LOOCV or random CV
    '''
    
    # TODO: implement more realistic, random schemes
    validation_library = cv_parameters['validation_library']
    
    # maximum number of iterations
    cv_iterations = cv_parameters.get('iterations',-1)
    
    # number of samples to exclude
    cv_exclude = cv_parameters.get('cv',1)

    # use to distinguish different versions of error correction
    ec_variant = cv_parameters.get('ec_variant','ec')
    
    # use to distinguish different versions of label fusion
    fuse_variant = cv_parameters.get('fuse_variant','fuse')

    # use to distinguish different versions of cross-validation
    cv_variant = cv_parameters.get('cv_variant','cv')
    
    # different version of label regularization
    regularize_variant=cv_parameters.get('regularize_variant','gc')
    
    cv_output = output+os.sep+cv_variant+'_stats.json'
    res_output = output+os.sep+cv_variant+'_res.json'
    
    if extlib is not None:
        validation_library = extlib
    
    if validation_library is not list:
        with open(validation_library,'r') as f:
            validation_library = list(csv.reader(f))
        
    if cv_iter is not None:
        cv_iter=int(cv_iter)

    stat_results = None
    output_results = None

    if ext:
        # TODO: move pre-rpcessing here?
        # pre-process presegmented scans here!
        # we only neeed to re-create left-right flipped segmentation
        pass

    if cv_iterations==-1 and cv_exclude==1: # simle LOO cross-validation
        (stat_results, output_results) = loo_cv_fusion_segment(validation_library, 
                                             segmentation_library,
                                             output, segmentation_parameters,
                                             ec_parameters=ec_parameters, 
                                             debug=debug,
                                             cleanup=cleanup,
                                             ec_variant=ec_variant, 
                                             fuse_variant=fuse_variant,
                                             cv_variant=cv_variant,
                                             regularize_variant=regularize_variant,
                                             ext=ext,
                                             cv_iter=cv_iter)
    else: # arbitrary number of iterations
        (stat_results, output_results) = full_cv_fusion_segment(validation_library, 
                                              segmentation_library,
                                              output, segmentation_parameters,
                                              cv_iterations, cv_exclude,
                                              ec_parameters=ec_parameters, 
                                              debug=debug,
                                              cleanup=cleanup,
                                              ec_variant=ec_variant, 
                                              fuse_variant=fuse_variant,
                                              cv_variant=cv_variant,
                                              regularize_variant=regularize_variant,
                                              ext=ext,
                                              cv_iter=cv_iter)

    # average error maps
    
    if cv_iter is None or cv_iter == -1:
        results=[]
        output_results_all={'results':output_results}
        output_results_all['cv_stats']=cv_output
        output_results_all['error_maps']={}
        all_error_maps=[]
        
        for (i, j) in enumerate(output_results[0]['error_maps']):
            out_avg=output+os.sep+cv_variant+'_error_{:03d}.mnc'.format(i)
            output_results_all['error_maps'][i]=out_avg
            all_error_maps.append(out_avg)
            maps=[ k['error_maps'][i] for k in output_results ]
            results.append(futures.submit(
                average_error_maps,maps,out_avg))
        
        futures.wait(results, return_when=futures.ALL_COMPLETED)

        output_results_all['max_error']=output+os.sep+cv_variant+'_max_error.mnc'.format(i)
        max_error_maps(all_error_maps,output_results_all['max_error'])
        
        with open(cv_output, 'w') as f:
            json.dump(stat_results, f, indent=1 )

        with open(res_output, 'w') as f:
            json.dump(output_results_all, f, indent=1, cls=MRIEncoder)

        return stat_results
    else:
        # we assume that results will be available later
        return None

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

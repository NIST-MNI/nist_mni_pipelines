# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 29/06/2015
#
# registration tools
import os
import sys
import shutil
import tempfile
import subprocess
import re
import fcntl
import traceback
import collections
import math

# local stuff
import ipl.minc_tools
logger=ipl.minc_tools.get_logger()

# hack to make it work on Python 3
try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    str = str
    unicode = str
    bytes = bytes
    basestring = (str,bytes)
else:
    # 'unicode' exists, must be Python 2
    str = str
    unicode = unicode
    bytes = str
    basestring = basestring


def ants_linear_register(
    source,
    target,
    output_xfm,
    parameters=None,
    source_mask=None,
    target_mask=None,
    init_xfm=None,
    objective=None,
    conf=None,
    debug=False,
    close=False,
    work_dir=None,
    downsample=None,
    verbose=0
    ):
    """perform linear registration with ANTs"""

    # TODO: make use of parameters
    
    if parameters is None:
        parameters={}
    
    with ipl.minc_tools.mincTools(verbose=verbose) as minc:
      if not minc.checkfiles(inputs=[source,target], outputs=[output_xfm]):
          return

      prev_xfm = None

      s_base=os.path.basename(source).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
      t_base=os.path.basename(target).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]

      source_lr=source
      target_lr=target
      
      source_mask_lr=source_mask
      target_mask_lr=target_mask
      # figure out what to do here:
      with ipl.minc_tools.cache_files(work_dir=work_dir,context='reg') as tmp:
          
          if downsample is not None:
              source_lr=tmp.cache(s_base+'_'+str(downsample)+'.mnc')
              target_lr=tmp.cache(t_base+'_'+str(downsample)+'.mnc')
              
              minc.resample_smooth(source,source_lr,unistep=downsample)
              minc.resample_smooth(target,target_lr,unistep=downsample)
              
              if source_mask is not None:
                  source_mask_lr=tmp.cache(s_base+'_mask_'+str(downsample)+'.mnc')
                  minc.resample_labels(source_mask,source_mask_lr,unistep=downsample,datatype='byte')
              if target_mask is not None:
                  target_mask_lr=tmp.cache(s_base+'_mask_'+str(downsample)+'.mnc')
                  minc.resample_labels(target_mask,target_mask_lr,unistep=downsample,datatype='byte')
          
          iterations=parameters.get('affine-iterations','10000x10000x10000x10000x10000')
          
          default_gradient_descent_option='0.5x0.95x1.e-5x1.e-4'
          if close:default_gradient_descent_option='0.05x0.5x1.e-4x1.e-4'
          gradient_descent_option=parameters.get('gradient_descent_option',default_gradient_descent_option)
          
          mi_option=parameters.get('mi-option','32x16000')
          use_mask=parameters.get('use_mask',True)
          use_histogram_matching=parameters.get('use_histogram_matching',False)
          affine_metric=parameters.get('metric_type','MI')
          affine_rigid=parameters.get('rigid',False)
          
          cost_function_par='1,4'

          cmd=['ANTS','3']

          s_base=os.path.basename(source).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
          t_base=os.path.basename(target).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]

          source_lr=source
          target_lr=target

          target_mask_lr=target_mask

          if downsample is not None:
              source_lr=tmp.cache(s_base+'_'+str(downsample)+'.mnc')
              target_lr=tmp.cache(t_base+'_'+str(downsample)+'.mnc')

              minc.resample_smooth(source,source_lr,unistep=downsample)
              minc.resample_smooth(target,target_lr,unistep=downsample)

              if target_mask is not None:
                  target_mask_lr=tmp.cache(s_base+'_mask_'+str(downsample)+'.mnc')
                  minc.resample_labels(target_mask,target_mask_lr,unistep=downsample,datatype='byte')


          cmd.extend(['-m','{}[{},{},{}]'.format('CC',source_lr,target_lr,cost_function_par)])
          cmd.extend(['-i','0'])
          cmd.extend(['--number-of-affine-iterations',iterations])
          cmd.extend(['--affine-gradient-descent-option', gradient_descent_option])
          cmd.extend(['--MI-option', mi_option]) 
          cmd.extend(['--affine-metric-type', affine_metric])
          
          if affine_rigid:
              cmd.append('--rigid-affine')
          
          cmd.extend(['-o',output_xfm])
          
          inputs=[source_lr,target_lr]
          if target_mask_lr is not None and use_mask:
              inputs.append(target_mask_lr)
              cmd.extend(['-x',target_mask_lr])
          
          if use_histogram_matching:
              cmd.append('--use-Histogram-Matching')
        
          if winsorize_intensity is not None:
            if isinstance(winsorize_intensity, dict):
                cmd.extend(['--winsorize-image-intensities',str(winsorize_intensity.get('low',0.05)),str(winsorize_intensity.get('high',0.95))])
            else:
                cmd.append('--winsorize-image-intensities')
              
          if init_xfm is not None:
              cmd.extend(['--initial-affine',init_xfm])

          outputs=[output_xfm ] # TODO: add inverse xfm ?
          minc.command(cmd, inputs=inputs, outputs=outputs)
        
        

def non_linear_register_ants(
    source, target, output_xfm,
    target_mask=None,
    init_xfm   =None,
    parameters =None,
    downsample =None,
    verbose=0
    ):
    """perform non-linear registration using ANTs, WARNING: will create inverted xfm  will be named output_invert.xfm"""

    with ipl.minc_tools.mincTools(verbose=verbose) as minc:

        if parameters is None:
            parameters={}
            
        
        if not minc.checkfiles(inputs=[source,target], 
                                outputs=[output_xfm ]):
            return
            
        cost_function=parameters.get('cost_function','CC')
        cost_function_par=parameters.get('cost_function_par','1,2')
        
        reg=parameters.get('regularization','Gauss[2,0.5]')
        iterations=parameters.get('iter','20x20x0')
        transformation=parameters.get('transformation','SyN[0.25]')
        affine_iterations=parameters.get('affine-iterations','0x0x0')
        use_mask=parameters.get('use_mask',True)
        use_histogram_matching=parameters.get('use_histogram_matching',False)

        cmd=['ANTS','3']

        s_base=os.path.basename(source).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
        t_base=os.path.basename(target).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]

        source_lr=source
        target_lr=target

        target_mask_lr=target_mask

        if downsample is not None:
            source_lr=tmp.cache(s_base+'_'+str(downsample)+'.mnc')
            target_lr=tmp.cache(t_base+'_'+str(downsample)+'.mnc')

            minc.resample_smooth(source,source_lr,unistep=downsample)
            minc.resample_smooth(target,target_lr,unistep=downsample)

            if target_mask is not None:
                target_mask_lr=tmp.cache(s_base+'_mask_'+str(downsample)+'.mnc')
                minc.resample_labels(target_mask,target_mask_lr,unistep=downsample,datatype='byte')


        cmd.extend(['-m','{}[{},{},{}]'.format(cost_function,source_lr,target_lr,cost_function_par)])
        cmd.extend(['-i',iterations])
        cmd.extend(['-t',transformation])
        cmd.extend(['-r',reg])
        cmd.extend(['--number-of-affine-iterations',affine_iterations])
        cmd.extend(['-o',output_xfm])
        
        inputs=[source_lr,target_lr]
        if target_mask_lr is not None and use_mask:
            inputs.append(target_mask_lr)
            cmd.extend(['-x',target_mask_lr])
        
        if use_histogram_matching:
            cmd.append('--use-Histogram-Matching')
        
        outputs=[output_xfm ] # TODO: add inverse xfm ?
        
        #print(repr(cmd))
        
        minc.command(cmd, inputs=inputs, outputs=outputs)


def non_linear_register_ants2(
    source, target, output_xfm,
    target_mask=None,
    source_mask=None,
    init_xfm   =None,
    parameters =None,
    downsample =None,
    start      =None,
    level      =32.0,
    verbose    =0
    ):
    """perform non-linear registration using ANTs, WARNING: will create inverted xfm  will be named output_invert.xfm"""
    if start is None:
        start=level
    
    with ipl.minc_tools.mincTools(verbose=verbose) as minc:

        sources = []
        targets = []
        
        if isinstance(source, list):
            sources.extend(source)
        else:
            sources.append(source)
        
        if isinstance(target, list):
            targets.extend(target)
        else:
            targets.append(target)
        if len(sources)!=len(targets):
            raise ipl.minc_tools.mincError(' ** Error: Different number of inputs ')
        
        modalities=len(sources)


        if parameters is None:
            #TODO add more options here
            parameters={'conf':{},
                        'blur':{}, 
                        'shrink':{} 
                       }
        else:
            if not 'conf'   in parameters: parameters['conf']   = {}
            if not 'blur'   in parameters: parameters['blur']   = {}
            if not 'shrink' in parameters: parameters['shrink'] = {}
            
        prog=''
        shrink=''
        blur=''
        for i in range(int(math.log(start)/math.log(2)),-1,-1):
            res=2**i
            if res>=level:
                prog+=  str(parameters['conf'].  get(res,parameters['conf'].  get(str(res),20)))
                shrink+=str(parameters['shrink'].get(res,parameters['shrink'].get(str(res),2**i)))
                blur+=  str(parameters['blur'].  get(res,parameters['blur'].  get(str(res),2**i)))
            if res>level:
                prog+='x'
                shrink+='x'
                blur+='x'
                
        if not minc.checkfiles(inputs=sources+targets, 
                                outputs=[output_xfm ]):
            return
        
        prog+=','+parameters.get('convergence','1.e-6,10')
        
        output_base=output_xfm.rsplit('.xfm',1)[0]
            
        cost_function=parameters.get('cost_function','CC')
        cost_function_par=parameters.get('cost_function_par','1,2,Regular,1.0')
        
        transformation=parameters.get('transformation','SyN[ .25, 2, 0.5 ]')
        use_mask=parameters.get('use_mask',True)
        use_histogram_matching=parameters.get('use_histogram_matching',False)
        use_float=parameters.get('use_float',False)
        
        winsorize_intensity=parameters.get('winsorize-image-intensities',None)
        
        cmd=['antsRegistration','--minc','1','-a','--dimensionality','3']


        (sources_lr, targets_lr, source_mask_lr, target_mask_lr)=minc.downsample_registration_files(sources,targets,source_mask,target_mask, downsample)

        # generate modalities
        for _s in range(modalities):
            if isinstance(cost_function, list): 
                cost_function_=cost_function[_s]
            else:
                cost_function_=cost_function
            #    
            if isinstance(cost_function_par, list): 
                cost_function_par_=cost_function_par[_s]
            else:
                cost_function_par_=cost_function_par
            #
            cmd.extend(['--metric','{}[{},{},{}]'.format(cost_function_, sources_lr[_s], targets_lr[_s], cost_function_par_)])

        
        cmd.extend(['--convergence','[{}]'.format(prog)])
        cmd.extend(['--shrink-factors',shrink])
        cmd.extend(['--smoothing-sigmas',blur])
        cmd.extend(['--transform',transformation])
        
        cmd.extend(['--output',output_base])
        #cmd.extend(['--save-state',output_xfm])

        if init_xfm is not None:
            cmd.extend(['--initial-fixed-transform',init_xfm])
        
        inputs=sources_lr+targets_lr
        
        if target_mask_lr is not None and source_mask_lr is not None and use_mask:
            inputs.extend([source_mask_lr, target_mask_lr])
            cmd.extend(['-x','[{},{}]'.format(source_mask_lr, target_mask_lr)])
        
        if use_histogram_matching:
            cmd.append('--use-histogram-matching')
            
        if winsorize_intensity is not None:
            if isinstance(winsorize_intensity, dict):
                cmd.extend(['--winsorize-image-intensities',str(winsorize_intensity.get('low', 0.01)),str(winsorize_intensity.get('high',0.99))])
            else:
                cmd.append( '--winsorize-image-intensities')
            
        if use_float:
            cmd.append('--float')
        
        if verbose>0:
            cmd.extend(['--verbose','1'])
            
        outputs=[output_xfm ] # TODO: add inverse xfm ?
        
        
        print(">>>\n{}\n>>>>".format(' '.join(cmd)))
        
        minc.command(cmd, inputs=inputs, outputs=outputs)

def linear_register_ants2(
    source, target, output_xfm,
    target_mask= None,
    source_mask= None,
    init_xfm   = None,
    parameters = None,
    downsample = None,
    close      = False,
    verbose=0
    ):
    """perform linear registration using ANTs"""
    #TODO:implement close
    
    
    with ipl.minc_tools.mincTools(verbose=verbose) as minc:

        
        if parameters is None:
            #TODO add more options here
            parameters={ 
                        'conf':  {},
                        'blur':  {},
                        'shrink':{}
                        }
        else:
            if not 'conf'   in parameters: parameters['conf']   = {}
            if not 'blur'   in parameters: parameters['blur']   = {}
            if not 'shrink' in parameters: parameters['shrink'] = {}

        levels=parameters.get('levels',3)
        prog=''
        shrink=''
        blur=''
        
        for i in range(levels,0,-1):
            _i=str(i)
            prog+=  str(parameters['conf'].  get(i,parameters['conf'].  get(_i,10000)))
            shrink+=str(parameters['shrink'].get(i,parameters['shrink'].get(_i,2**i)))
            blur+=  str(parameters['blur'].  get(i,parameters['blur'].  get(_i,2**i)))
            
            if i>1:
                prog+='x'
                shrink+='x'
                blur+='x'
        # TODO: make it a parameter?
        prog+=','+parameters.get('convergence','1.e-8,20')
        
        sources = []
        targets = []
        
        if isinstance(source, list):
            sources.extend(source)
        else:
            sources.append(source)
        
        if isinstance(target, list):
            targets.extend(target)
        else:
            targets.append(target)
            
            
        if len(sources)!=len(targets):
            raise ipl.minc_tools.mincError(' ** Error: Different number of inputs ')
        
        modalities=len(sources)
        
        if not minc.checkfiles(inputs=sources+targets,
                               outputs=[output_xfm ]):
            return
        
        output_base            = output_xfm.rsplit('.xfm',1)[0]
        
        cost_function          = parameters.get('cost_function',    'Mattes')
        cost_function_par      = parameters.get('cost_function_par','1,32,regular,0.3')
        
        transformation         = parameters.get('transformation','affine[ 0.1 ]')
        use_mask               = parameters.get('use_mask',True)
        use_histogram_matching = parameters.get('use_histogram_matching',False)
        winsorize_intensity    = parameters.get('winsorize-image-intensities',None)
        use_float              = parameters.get('use_float',False)
        intialize_fixed        = parameters.get('initialize_fixed',None)
        intialize_moving       = parameters.get('intialize_moving',None)
        
        cmd=['antsRegistration','--collapse-output-transforms', '0', '--minc','1','-a','--dimensionality','3']

        (sources_lr, targets_lr, source_mask_lr, target_mask_lr)=minc.downsample_registration_files(sources,targets,source_mask,target_mask, downsample)
        
        # generate modalities
        for _s in range(modalities):
            if isinstance(cost_function, list): 
                cost_function_=cost_function[_s]
            else:
                cost_function_=cost_function
            #    
            if isinstance(cost_function_par, list): 
                cost_function_par_=cost_function_par[_s]
            else:
                cost_function_par_=cost_function_par
            #
            cmd.extend(['--metric','{}[{},{},{}]'.format(cost_function_, sources_lr[_s], targets_lr[_s], cost_function_par_)])
        #
        # 
        cmd.extend(['--convergence','[{}]'.format(prog)])
        cmd.extend(['--shrink-factors',shrink])
        cmd.extend(['--smoothing-sigmas',blur])
        cmd.extend(['--transform',transformation])
        cmd.extend(['--output',output_base])
        #cmd.extend(['--save-state',output_xfm])

        if init_xfm is not None:
            cmd.extend(['--initial-fixed-transform',init_xfm])
            # this is  a hack in attempt to make initial linear transform to work as expected
            # currently, it looks like the center of the transform (i.e center of rotation) is messed up :(
            # and it causes lots of problems 
            cmd.extend(['--initialize-transforms-per-stage','1'])
        elif intialize_fixed is not None:
            cmd.extend(['--initial-fixed-transform',"[{},{},{}]".format(sources_lr[0],targets_lr[0],str(intialize_fixed))])
        elif not close:
            cmd.extend(['--initial-fixed-transform',"[{},{},{}]".format(sources_lr[0],targets_lr[0],'0')])
        
        if intialize_moving is not None:
            cmd.extend(['--initial-moving-transform',"[{},{},{}]".format(sources_lr[0],targets_lr[0],str(intialize_moving))])
        elif not close:
            cmd.extend(['--initial-moving-transform',"[{},{},{}]".format(sources_lr[0],targets_lr[0],'0')])
        #
        inputs=sources_lr+targets_lr
        #
        if target_mask_lr is not None and source_mask_lr is not None and use_mask:
            inputs.extend([source_mask_lr, target_mask_lr])
            cmd.extend(['-x','[{},{}]'.format(source_mask_lr, target_mask_lr)])
        
        if use_histogram_matching:
            cmd.append('--use-histogram-matching')

        if winsorize_intensity is not None:
            if isinstance(winsorize_intensity, dict):
                cmd.extend(['--winsorize-image-intensities',str(winsorize_intensity.get('low',0.01)),str(winsorize_intensity.get('high',0.99))])
            else:
                cmd.append('--winsorize-image-intensities')
            
        if use_float:
            cmd.append('--float')
        
        if verbose>0:
            cmd.extend(['--verbose','1'])
        
        outputs=[output_xfm ] # TODO: add inverse xfm ?
        minc.command(cmd, inputs=inputs, outputs=outputs,verbose=verbose)

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80

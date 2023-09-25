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

__lin_template="""
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
(UseDirectionCosines "true")

(AutomaticTransformInitialization "{automatic_transform_init}")
(AutomaticTransformInitializationMethod "{automatic_transform_init_method}")
(AutomaticScalesEstimation "true")
(AutomaticParameterEstimation "true")
(MaximumStepLength {max_step})

(Registration         "MultiResolutionRegistration")
(Interpolator         "BSplineInterpolator" )
(ResampleInterpolator "FinalBSplineInterpolator" )
(Resampler            "DefaultResampler" )
(ShowExactMetricValue {exact_metric})

(FixedImagePyramid  "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

(Optimizer "{optimizer}")
(Transform "{transform}")
(Metric    "{metric}")

(HowToCombineTransforms "Compose")

(ErodeMask "false")

(NumberOfResolutions {resolutions})

(ImagePyramidSchedule {pyramid} )

(MaximumNumberOfIterations {iterations} )
(RequiredRatioOfValidSamples 0.01)
(MaximumNumberOfSamplingAttempts 10)

(NumberOfSpatialSamples  {samples} )

(NewSamplesEveryIteration "{new_samples}")
(ImageSampler             "{sampler}" )

(BSplineInterpolationOrder 1)

(FinalBSplineInterpolationOrder 1)

(DefaultPixelValue    0)

(WriteResultImage     "false")

// The pixel type and format of the resulting deformed moving image
(ResultImagePixelType  "float")
(ResultImageFormat     "mnc")
"""

__nl_template='''
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
(UseDirectionCosines "true")

(Registration "MultiResolutionRegistration")
(Interpolator "BSplineInterpolator" )
(ResampleInterpolator "FinalBSplineInterpolator" )
(Resampler "DefaultResampler" )
(ShowExactMetricValue {exact_metric})

(FixedImagePyramid  "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

(Optimizer "{optimizer}")
(Transform "{transform}")
(Metric "{metric}")
(MaximumStepLength {max_step})

(FinalGridSpacingInPhysicalUnits {grid_spacing})

(HowToCombineTransforms "Compose")

(ErodeMask "false")

(NumberOfResolutions {resolutions})

(ImagePyramidSchedule {pyramid} )

(MaximumNumberOfIterations {iterations} )
(MaximumNumberOfSamplingAttempts 10)

(NumberOfSpatialSamples {samples} )

(NewSamplesEveryIteration "{new_samples}")
(ImageSampler "{sampler}" )

(BSplineInterpolationOrder 1)

(FinalBSplineInterpolationOrder 1)

(DefaultPixelValue 0)

(WriteResultImage "false")

// The pixel type and format of the resulting deformed moving image
(ResultImagePixelType "float")
(ResultImageFormat "mnc")
'''

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

def parse_tags(tag):
    tags=[]
    #p = re.compile(r'\S')
    volumes=1
    with open(tag,'r') as f:
        started=False
        for line in f:
            line=line.rstrip('\r\n')
            
            if not started:
                m = re.match(".*Volumes = (\S)",line)
                
                if re.match(".*Points =",line): 
                    started=True
                    continue
                elif m is not None : 
                    volumes=int(m.group(1))
            else:
                if re.match('.*;',line) is not None: # this is the last line
                    line=line.replace(';','')
                    # last line?
                c=line.split(' ')
                if len(c[0])==0:
                    c.pop(0)
                    
                #print(','.join(c))
                #shift @c unless $c[0]; #protection against empty first parameter
                #push(@tags, [$c[0], $c[1], $c[2], $c[3], $c[4], $c[5]] );
                tags.append([float(i) for i in c])
                
    return (volumes,tags)

def tag2elx(tags,out1,out2): 
    (vols,tags)=parse_tags(tags)
    
    with open(out1,'w') as f:
        f.write("point\n{}\n".format(len(tags)))
        for i in tags:
            f.write("{} {} {}\n".format(i[0],i[1],i[2]))
    
    if vols>1:
        with open(out2,'w') as f:
            f.write("point\n{}\n".format(len(tags)))
            for i in tags:
                f.write("{} {} {}\n".format(i[3],i[4],i[5]))
                
    return vols


def nl_xfm_to_elastix(xfm, elastix_par):
    """Convert MINC style xfm into elastix style registration parameters
    Assuming that xfm file is strictly non-linear, with a single non-linear deformation field
    """
    # TODO: make a proper parsing of XFM file
    with ipl.minc_tools.mincTools() as minc:
        grid=xfm.rsplit('.xfm',1)[0]+'_grid_0.mnc'
        if not os.path.exists(grid):
          logger.error("nl_xfm_to_elastix error!")
          raise ipl.minc_tools.mincError("Unfortunately currently only a very primitive way of dealing with Minc XFM files is implemented\n{}".format(traceback.format_exc()))
        
        with open(elastix_par,'w') as f:
            f.write("(Transform \"DeformationFieldTransform\")\n")
            f.write("(DeformationFieldInterpolationOrder 0)\n")
            f.write("(DeformationFieldFileName \"{}\")\n".format(grid))
        return elastix_par


def lin_xfm_to_elastix(xfm,elastix_par):
    """Convert MINC style xfm into elastix style registration parameters
    Assuming that xfm fiel is strictly linear
    """
    with ipl.minc_tools.mincTools() as minc:
        minc.command(['itk_convert_xfm',xfm,minc.tmp('input.txt')],
                     inputs=xfm,outputs=[minc.tmp('input.txt')])
        # parsing text transformation
        param=None
        fix_param=None
        
        with open(minc.tmp('input.txt'),'r') as f:
            for ln in f:
                if re.match('^Parameters: ', ln):
                    param=ln.split(' ')
                if re.match('^FixedParameters: ', ln):
                    fix_param=ln.split(' ')
        param.pop(0)
        fix_param.pop(0)
        with open(minc.tmp('elastix_par'),'w') as f:
            f.write('''(Transform "AffineTransform")
(NumberOfParameters 12)
(TransformParameters {})
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")

// EulerTransform specific
(CenterOfRotationPoint {})
'''.format(' '.join(param),' '.join(fix_param)))

    
def nl_elastix_to_xfm(elastix_par, xfm, downsample_grid=None, nl=True ):
    """Convert elastix transformation file into minc XFM file"""
    with ipl.minc_tools.mincTools() as minc:
        threads=os.environ.get('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS',1)
        cmd=['transformix', '-tp',  elastix_par, '-out',  minc.tempdir,'-xfm', xfm, '-q', '-threads', str(threads)]
        
        if nl:
          cmd.extend(['-def',  'all'])
          if downsample_grid is not None:
            cmd.extend(['-sub',str(downsample_grid)])
        
        minc.command(cmd, inputs=[elastix_par], outputs=[xfm]);
        return xfm


def gen_config_nl(parameters, output_txt, template=__nl_template, def_iterations=4000):
    with open(output_txt,'w') as p:
        p.write(template.format(
                            optimizer= parameters.get('optimizer','AdaptiveStochasticGradientDescent'),
                            transform= parameters.get('transform','BSplineTransform'),
                            metric=    parameters.get('metric','AdvancedNormalizedCorrelation'),
                            resolutions=parameters.get('resolutions',3),
                            pyramid=   parameters.get('pyramid','8 8 8 4 4 4 2 2 2'),
                            iterations=parameters.get('iterations',def_iterations),
                            samples=   parameters.get('samples',4096),
                            sampler=   parameters.get('sampler',"Random"),
                            grid_spacing=parameters.get('grid_spacing',10),
                            max_step  =parameters.get('max_step',"1.0"),
                            exact_metric=str(parameters.get("exact_metric",False)).lower(),
                            new_samples=str(parameters.get("new_samples",True)).lower(),
                            ))    
    
def gen_config_lin(parameters,output_txt,template=__lin_template,def_iterations=4000):
    with open(output_txt,'w') as p:
        p.write(template.format(
                            optimizer=parameters.get('optimizer','CMAEvolutionStrategy'),
                            transform=parameters.get('transform','SimilarityTransform'),
                            metric=parameters.get('metric','AdvancedNormalizedCorrelation'),
                            resolutions=parameters.get('resolutions', 3 ),
                            pyramid=parameters.get('pyramid','8 8 8  4 4 4  2 2 2'),
                            iterations=parameters.get('iterations',def_iterations),
                            samples=parameters.get('samples',4096),
                            sampler=parameters.get('sampler',"Random"),
                            max_step=parameters.get('max_step',"1.0"),
                            automatic_transform_init=str(parameters.get("automatic_transform_init",True)).lower(), # to convert True to true
                            automatic_transform_init_method=parameters.get("automatic_transform_init_method","CenterOfGravity"),
                            exact_metric=str(parameters.get("exact_metric",False)).lower(),
                            new_samples=str(parameters.get("new_samples",True)).lower(),
                            ))
        if 'grid_spacing' in parameters: p.write("(SampleGridSpacing {})\n".format(parameters['grid_spacing']))
        #if 'exact_metric' in parameters: p.write("(ShowExactMetricValue {})\n".format(parameters['exact_metric']))
        if 'exact_metric_spacing' in parameters: p.write("(ExactMetricSampleGridSpacing {})\n".format(parameters['exact_metric_spacing']))


def register_elastix( 
                    source, target, 
                    output_par = None,
                    output_xfm = None,
                    source_mask= None,
                    target_mask= None,
                    init_xfm   = None,
                    init_par   = None,
                    parameters = None,
                    work_dir   = None,
                    downsample = None,
                    downsample_grid=None,
                    nl         = True,
                    output_log = None,
                    tags       = None,
                    verbose    = 0,
                    iterations = None):
    """Run elastix with given parameters
    Arguments:
    source -- source image (fixed image in Elastix notation)
    target -- target, or reference image (moving image in Elastix notation)
    
    Keyword arguments:
    output_par -- output transformation in elastix format
    output_xfm -- output transformation in MINC XFM format
    source_mask -- source mask
    target_mask -- target mask
    init_xfm    -- initial transform in XFM format
    init_par    -- initial transform in Elastix format
    parameters  -- parameters for transformation
                   if it is a string starting with @ it's a text file name that contains 
                   parameters in elastix format
                   if it any other string - it should be treated as transformation parameters in elastix format
                   if it is a dictionary:
                   for non-linear mode (nl==True):
                        "optimizer" , "AdaptiveStochasticGradientDescent" (default for nonlinear)
                                      "CMAEvolutionStrategy" (default for linear)
                                      "ConjugateGradient"
                                      "ConjugateGradientFRPR"
                                      "FiniteDifferenceGradientDescent"
                                      "QuasiNewtonLBFGS"
                                      "RegularStepGradientDescent"
                                      "RSGDEachParameterApart"
                                      
                        "transform", "BSplineTransform" (default for nonlinear mode)
                                     "SimilarityTransform" (default for linear)
                                     "AffineTransform"
                                     "AffineDTITransform"
                                     "EulerTransform"
                                     "MultiBSplineTransformWithNormal"
                                     "TranslationTransform"
                                     
                        "metric"   , "AdvancedNormalizedCorrelation"  (default)
                                     "AdvancedMattesMutualInformation"
                                     "NormalizedMutualInformation"
                                     "AdvancedKappaStatistic"
                                     "KNNGraphAlphaMutualInformation"
                                     
                        "resolutions", 3   - number of resolution steps
                        "pyramid","8 8 8 4 4 4 2 2 2" - downsampling schedule
                        "iterations",4000 - number of iterations
                        "samples",4096  - number of samples
                        "sampler", "Random" (default)
                                   "Full"
                                   "RandomCoordinate"
                                   "Grid"  TODO: add SampleGridSpacing
                                   "RandomSparseMask"
                                   
                        "grid_spacing",10  - grid spacing in mm
                        "max_step","1.0" - maximum step (mm)
                        
                   for linear mode (nl==False):
                        "optimizer","CMAEvolutionStrategy" - optimizer
                        "transform","SimilarityTransform"  - transform
                        "metric","AdvancedNormalizedCorrelation" - cost function
                        "resolutions", 3  - number of resolutions
                        "pyramid","8 8 8  4 4 4 2 2 2" - resampling schedule
                        "iterations",4000  - number of iterations
                        "samples",4096 - number of samples
                        "sampler","Random" - sampler
                        "max_step","1.0" - max step
                        "automatic_transform_init",True - perform automatic transform initialization
                        "automatic_transform_init_method", - type of automatic transform initalization method, 
                                                          "CenterOfGravity" (default)
                                                          "GeometricalCenter" - center of the image based
    work_dir    -- Work directory
    downsample  -- Downsample input images
    downsample_grid -- Downsample output nl-deformation
    nl          -- flag to show that non-linear version is running
    output_log  -- output log
    iterations  -- run several iterations (restarting elastix), will be done automatically if parameters is a list
    :rtype:
    """
    with ipl.minc_tools.mincTools() as minc:
        
        def_iterations=4000
        
        s_base=os.path.basename(source).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]
        t_base=os.path.basename(target).rsplit('.gz',1)[0].rsplit('.mnc',1)[0]

        source_lr=source
        target_lr=target

        source_mask_lr=source_mask
        target_mask_lr=target_mask
        use_mask=True

        if (init_par is not None) and (init_xfm is not None):
            logger.error("register_elastix: init_xfm={} init_par={}".format(repr(init_xfm),repr(init_par)))
            raise ipl.minc_tools.mincError("Specify either init_xfm or init_par")

        outputs=[]
        if output_par is not None: outputs.append(output_par)
        if output_xfm is not None: outputs.append(output_xfm)

        if len(outputs)>0 and (not minc.checkfiles( inputs=[source,target], 
                                outputs=outputs )):
            return
        
        threads=os.environ.get('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS',1)
        
        if parameters is None:
            parameters={}
        #print("Running elastix with parameters:{}".format(repr(parameters)))
        # figure out what to do here:
        with ipl.minc_tools.cache_files(work_dir=work_dir,context='elastix') as tmp:
            
            if init_xfm is not None:
                if nl:
                    init_par=nl_xfm_to_elastix(init_xfm, tmp.cache('init.txt'))
                else:
                    init_par=lin_xfm_to_elastix(init_xfm, tmp.cache('init.txt'))
            
            # a fitting we shall go...
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
            
            _iterations=1
            
            if isinstance(parameters,list):
                _iterations=len(parameters)
            
            try:
                for it in range(_iterations):

                    if isinstance(parameters,list):
                        _par=parameters[it]
                    else:
                        _par=parameters
                    
                    par_file=tmp.cache('parameters_{}.txt'.format(it))
                    measure_mode=False
                    # paramters could be stored in a file
                    
                    if isinstance(_par, dict):
                        use_mask=_par.get('use_mask',True)
                        measure_mode=_par.get('measure',False)
                        
                        if measure_mode:
                            def_iterations=1
                            _par['iterations']=1
                        
                        if nl:
                            gen_config_nl(_par,par_file,def_iterations=def_iterations)
                        else:
                            gen_config_lin(_par,par_file,def_iterations=def_iterations)
                    else:
                        if _par[0]=="@":
                            par_file=_par.split("@",1)[1]
                        else:
                            with open(par_file,'w') as p:
                                p.write(_par)
                    it_output_dir=tmp.tempdir+os.sep+str(it)
                    if not os.path.exists(it_output_dir):
                        os.makedirs(it_output_dir)
                            
                    cmd=['elastix', 
                        '-f',       source_lr , 
                        '-m',       target_lr, 
                        '-out',     it_output_dir+os.sep , 
                        '-p',       par_file,
                        '-threads', str(threads)] # , '-q'
                    
                    if measure_mode:
                        cmd.append('-M')
                        
                    if verbose<1:
                        cmd.append('-q')
                    
                    inputs=[source_lr , target_lr]

                    if init_par is not None:
                        cmd.extend(['-t0',init_par])
                        inputs.append(init_par)

                    if source_mask is not None and use_mask:
                        cmd.extend( ['-fMask',source_mask_lr] )
                        inputs.append(source_mask_lr)

                    if target_mask is not None and use_mask:
                        cmd.extend( ['-mMask',target_mask_lr] )
                        inputs.append(target_mask_lr)

                    if tags is not None:
                        vols=tag2elx(tags,tmp.cache(s_base+'_tags.txt'),tmp.cache(t_base+'_tags.txt'))
                        inputs.append(tmp.cache(s_base+'_tags.txt') )
                        cmd.extend(['-fp',tmp.cache(s_base+'_tags.txt')] )
                        shutil.copyfile(tmp.cache(s_base+'_tags.txt'),"source.tag")
                        
                        if vols>1:
                            inputs.append(tmp.cache(t_base+'_tags.txt') )
                            cmd.extend(['-mp',tmp.cache(t_base+'_tags.txt')] )
                            shutil.copyfile(tmp.cache(t_base+'_tags.txt'),"target.tag")
                    
                    
                    outputs=[ it_output_dir+os.sep+'TransformParameters.0.txt' ]

                    outcome=None
                    
                    if measure_mode:
                        # going to read the output of iterations
                        out_=minc.execute_w_output(cmd).split("\n")
                        for l,j in enumerate(out_):
                            if re.match("^1\:ItNr\s2\:Metric\s.*",j):
                                outcome=float(out_[l+1].split("\t")[1])
                                break
                        else:
                            #
                            logger.error("Elastix output:\n{}".format("\n".join(out_)))
                            raise ipl.minc_tools.mincError("Elastix didn't report measure")
                    else:
                        try:
                            minc.command(cmd, inputs=inputs, outputs=outputs, verbose=verbose)
                        except ipl.minc_tools.mincError as e:
                            with open(it_output_dir+os.sep+'elastix.log','r') as f:
                                print("Elastix output:\n{}".format(f.read()))
                            raise e
                    
                    init_par = it_output_dir +os.sep+'TransformParameters.0.txt'
                    # end of iterations
                    
                if  output_par is not None:
                    shutil.copyfile( it_output_dir +os.sep+'TransformParameters.0.txt' , output_par )

                if output_xfm is not None:
                    nl_elastix_to_xfm( it_output_dir +os.sep+'TransformParameters.0.txt', 
                                    output_xfm, 
                                    downsample_grid=downsample_grid, 
                                    nl=nl)
                    
            finally:
                if output_log is not None:
                    shutil.copyfile(it_output_dir+os.sep+'elastix.log', output_log)

        return outcome

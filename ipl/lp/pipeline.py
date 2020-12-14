# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 14/08/2015
#
# Longitudinal pipeline preprocessing

import shutil
import os
import sys
import csv
import traceback

# MINC stuff
from ipl.minc_tools import mincTools,mincError,temp_files

#Hippocampus segmentation
from ..segment.fuse import fusion_segment

# local stuff
from .structures   import *
from .preprocess   import *
from .utils        import *
from .registration import *
from .resample     import *
from .segment      import *
from .qc           import *
from .aqc          import *

# for xml manipulation
from string import Template

def save_ibis_summary(iter_summary,
                 fname):
    """
    Save a scene containing the volumes calculated by the pipeline.
    The volumes are the registered head image, the brain mask, the cortex
    surface, and the skin surface. The scene is stored in fname.

    Arguments: iter_summary Dictionary containing all the filenames
               fname Path of the output scene file
    """

    #Set xml structure

    default_xml = Template("""<!DOCTYPE configML>
    <configuration>
     <SaveScene>
      <IbisVersion value="3.0.0  Dev"/>
      <IbisRevision value="4a50db1"/>
      <Version value="6.0"/>
      <NextObjectID value="5"/>
      <NumberOfSceneObjects value="7"/>
      <ObjectList>
       <ObjectInScene_0>
        <ObjectClass value="WorldObject"/>
        <FullFileName value="none"/>
        <ObjectID value="-2"/>
        <ParentID value="-1"/>
        <ObjectName value="World"/>
        <AllowChildren value="1"/>
        <AllowChangeParent value="0"/>
        <ObjectManagedBySystem value="1"/>
        <ObjectHidden value="0"/>
        <AllowHiding value="0"/>
        <ObjectDeletable value="0"/>
        <NameChangeable value="0"/>
        <ObjectListable value="1"/>
        <AllowManualTransformEdit value="0"/>
        <LocalTransform value="1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 "/>
       </ObjectInScene_0>
       <ObjectInScene_1>
        <ObjectClass value="TripleCutPlaneObject"/>
        <FullFileName value="none"/>
        <ObjectID value="-3"/>
        <ParentID value="-2"/>
        <ViewPlanes value="1 1 1 "/>
        <PlanesPosition value="-9.0635992929070994e+00 -2.8457999184123580e+01 2.7191258146407492e+01 "/>
        <SliceThickness value="1"/>
        <SliceMixMode>
         <NumberOfElements value="2"/>
         <Element_0 value="2"/>
         <Element_1 value="2"/>
        </SliceMixMode>
        <BlendingModeIndices>
         <NumberOfElements value="2"/>
         <Element_0 value="2"/>
         <Element_1 value="2"/>
        </BlendingModeIndices>
       </ObjectInScene_1>
       <ObjectInScene_2>
        <ObjectClass value="VolumeRenderingObject"/>
        <FullFileName value="none"/>
        <ObjectID value="-5"/>
        <ParentID value="-2"/>
        <ObjectName value="PRISM Volume Render"/>
        <AllowChildren value="0"/>
        <AllowChangeParent value="0"/>
        <ObjectManagedBySystem value="1"/>
        <ObjectHidden value="1"/>
        <AllowHiding value="1"/>
        <ObjectDeletable value="0"/>
        <NameChangeable value="0"/>
        <ObjectListable value="1"/>
        <AllowManualTransformEdit value="0"/>
        <LocalTransform value="1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 "/>
        <IsAnimating value="0"/>
        <SamplingDistance value="1.0000000000000000e+00"/>
        <ShowInteractionWidget value="0"/>
        <InteractionWidgetLine value="0"/>
        <InteractionPoint1 value="0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 "/>
        <InteractionPoint2 value="2.0000000000000000e+02 0.0000000000000000e+00 0.0000000000000000e+00 "/>
        <RayInitShaderTypeName value="None"/>
        <StopConditionShaderTypeName value="ERT alpha 99%"/>
        <ImageSlots>
         <NumberOfElements value="1"/>
         <Element_0>
          <VolumeEnabled value="1"/>
          <VolumeIs16Bits value="0"/>
          <LinearSampling value="1"/>
          <ShaderContributionTypeName value="Add"/>
          <LastImageObjectId value="0"/>
          <ColorTransferFunction>
           <NbColorPoints value="2"/>
           <ColorPoint_0 value="0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
           <ColorPoint_1 value="2.5500000000000000e+02 1.0000000000000000e+00 1.0000000000000000e+00 1.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
          </ColorTransferFunction>
          <OpacityTransferFunction>
           <NbPoints value="2"/>
           <Point_0 value="1.1316568047337279e+00 4.0000000000000001e-02 5.0000000000000000e-01 0.0000000000000000e+00 "/>
           <Point_1 value="2.5349112426035504e+02 9.7599999999999998e-01 5.0000000000000000e-01 0.0000000000000000e+00 "/>
          </OpacityTransferFunction>
         </Element_0>
        </ImageSlots>
       </ObjectInScene_2>
       <ObjectInScene_3>
        <ObjectClass value="ImageObject"/>
        <FullFileName value="./$replace_main_image_path"/>
        <ObjectID value="0"/>
        <ParentID value="-2"/>
        <ObjectName value="$replace_main_image_name"/>
        <AllowChildren value="1"/>
        <AllowChangeParent value="1"/>
        <ObjectManagedBySystem value="0"/>
        <ObjectHidden value="0"/>
        <AllowHiding value="1"/>
        <ObjectDeletable value="1"/>
        <NameChangeable value="1"/>
        <ObjectListable value="1"/>
        <AllowManualTransformEdit value="1"/>
        <LocalTransform value="1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 "/>
        <LabelImage value="0"/>
        <ViewOutline value="0"/>
        <LutIndex value="0"/>
        <LutRange value="-9.3311071395874023e+00 3.1488708496093750e+02 "/>
        <IntensityFactor value="9.8999999999999999e-01"/>
        <VolumeRenderingEnabled value="0"/>
        <ColorWindow value="1.0000000000000000e+00"/>
        <ColorLevel value="5.0000000000000000e-01"/>
        <EnableShading value="0"/>
        <Ambiant value="1.0000000000000001e-01"/>
        <Diffuse value="6.9999999999999996e-01"/>
        <Specular value="2.0000000000000001e-01"/>
        <SpecularPower value="1.0000000000000000e+01"/>
        <EnableGradientOpacity value="1"/>
        <AutoSampleDistance value="1"/>
        <SampleDistance value="1.0000000000000000e+00"/>
        <ShowVolumeClippingBox value="0"/>
        <VolumeRenderingBounds value="-9.6000000000000000e+01 9.6000000000000000e+01 -1.3200000000000000e+02 9.6000000000000000e+01 -7.8000000000000000e+01 1.1400000000000000e+02 "/>
        <ScalarOpacity>
         <NbPoints value="2"/>
         <Point_0 value="0.0000000000000000e+00 0.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
         <Point_1 value="2.5500000000000000e+02 1.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
        </ScalarOpacity>
        <GradientOpacity>
         <NbPoints value="2"/>
         <Point_0 value="0.0000000000000000e+00 1.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
         <Point_1 value="2.5500000000000000e+02 1.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
        </GradientOpacity>
        <ColorTransferFunction>
         <NbColorPoints value="2"/>
         <ColorPoint_0 value="0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
         <ColorPoint_1 value="2.5500000000000000e+02 1.0000000000000000e+00 1.0000000000000000e+00 1.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
        </ColorTransferFunction>
       </ObjectInScene_3>
       <ObjectInScene_4>
        <ObjectClass value="ImageObject"/>
        <FullFileName value="./$replace_mask_path"/>
        <ObjectID value="1"/>
        <ParentID value="-2"/>
        <ObjectName value="$replace_mask_name"/>
        <AllowChildren value="1"/>
        <AllowChangeParent value="1"/>
        <ObjectManagedBySystem value="0"/>
        <ObjectHidden value="0"/>
        <AllowHiding value="1"/>
        <ObjectDeletable value="1"/>
        <NameChangeable value="1"/>
        <ObjectListable value="1"/>
        <AllowManualTransformEdit value="1"/>
        <LocalTransform value="1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 "/>
        <LabelImage value="0"/>
        <ViewOutline value="0"/>
        <LutIndex value="0"/>
        <LutRange value="-9.2964763641357422e+00 1.7405364990234375e+02 "/>
        <IntensityFactor value="9.8999999999999999e-01"/>
        <VolumeRenderingEnabled value="0"/>
        <ColorWindow value="1.0000000000000000e+00"/>
        <ColorLevel value="5.0000000000000000e-01"/>
        <EnableShading value="0"/>
        <Ambiant value="1.0000000000000001e-01"/>
        <Diffuse value="6.9999999999999996e-01"/>
        <Specular value="2.0000000000000001e-01"/>
        <SpecularPower value="1.0000000000000000e+01"/>
        <EnableGradientOpacity value="1"/>
        <AutoSampleDistance value="1"/>
        <SampleDistance value="1.0000000000000000e+00"/>
        <ShowVolumeClippingBox value="0"/>
        <VolumeRenderingBounds value="-9.6000000000000000e+01 9.6000000000000000e+01 -1.3200000000000000e+02 9.6000000000000000e+01 -7.8000000000000000e+01 1.1400000000000000e+02 "/>
        <ScalarOpacity>
         <NbPoints value="2"/>
         <Point_0 value="0.0000000000000000e+00 0.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
         <Point_1 value="2.5500000000000000e+02 1.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
        </ScalarOpacity>
        <GradientOpacity>
         <NbPoints value="2"/>
         <Point_0 value="0.0000000000000000e+00 1.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
         <Point_1 value="2.5500000000000000e+02 1.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
        </GradientOpacity>
        <ColorTransferFunction>
         <NbColorPoints value="2"/>
         <ColorPoint_0 value="0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
         <ColorPoint_1 value="2.5500000000000000e+02 1.0000000000000000e+00 1.0000000000000000e+00 1.0000000000000000e+00 5.0000000000000000e-01 0.0000000000000000e+00 "/>
        </ColorTransferFunction>
       </ObjectInScene_4>
       <ObjectInScene_5>
        <ObjectClass value="PolyDataObject"/>
        <FullFileName value="./$replace_cortex_surface_path"/>
        <ObjectID value="2"/>
        <ParentID value="-2"/>
        <ObjectName value="$replace_cortex_surface_name"/>
        <AllowChildren value="1"/>
        <AllowChangeParent value="1"/>
        <ObjectManagedBySystem value="0"/>
        <ObjectHidden value="0"/>
        <AllowHiding value="1"/>
        <ObjectDeletable value="1"/>
        <NameChangeable value="1"/>
        <ObjectListable value="1"/>
        <AllowManualTransformEdit value="1"/>
        <LocalTransform value="1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 "/>
        <RenderingMode value="2"/>
        <LutIndex value="0"/>
        <ScalarsVisible value="0"/>
        <VertexColorMode value="0"/>
        <ScalarSourceObjectId value="-1"/>
        <Opacity value="1.0000000000000000e+00"/>
        <ObjectColor value="1.0000000000000000e+00 1.0000000000000000e+00 1.0000000000000000e+00 "/>
        <CrossSectionVisible value="0"/>
        <ClippingEnabled value="0"/>
        <ClippingPlanesOrientation value="1 1 1 "/>
        <ShowTexture value="0"/>
        <TextureFileName value=""/>
       </ObjectInScene_5>
       <ObjectInScene_6>
        <ObjectClass value="PolyDataObject"/>
        <FullFileName value="./$replace_skin_surface_path"/>
        <ObjectID value="4"/>
        <ParentID value="-2"/>
        <ObjectName value="$replace_skin_surface_name"/>
        <AllowChildren value="1"/>
        <AllowChangeParent value="1"/>
        <ObjectManagedBySystem value="0"/>
        <ObjectHidden value="1"/>
        <AllowHiding value="1"/>
        <ObjectDeletable value="1"/>
        <NameChangeable value="1"/>
        <ObjectListable value="1"/>
        <AllowManualTransformEdit value="1"/>
        <LocalTransform value="1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 "/>
        <RenderingMode value="2"/>
        <LutIndex value="0"/>
        <ScalarsVisible value="0"/>
        <VertexColorMode value="0"/>
        <ScalarSourceObjectId value="-1"/>
        <Opacity value="1.0000000000000000e+00"/>
        <ObjectColor value="1.0000000000000000e+00 1.0000000000000000e+00 1.0000000000000000e+00 "/>
        <CrossSectionVisible value="0"/>
        <ClippingEnabled value="0"/>
        <ClippingPlanesOrientation value="1 1 1 "/>
        <ShowTexture value="0"/>
        <TextureFileName value=""/>
       </ObjectInScene_6>
      </ObjectList>
      <Plugins>
       <USAcquisitionDoubleView/>
       <SEEGAtlas/>
      </Plugins>
      <SceneManager>
       <CurrentObjectID value="4"/>
       <ReferenceObjectID value="0"/>
       <ViewBackgroundColor value="0.0000000000000000e+00 0.0000000000000000e+00 4.9803921568627452e-01 "/>
       <View3DBackgroundColor value="0.0000000000000000e+00 0.0000000000000000e+00 4.9803921568627452e-01 "/>
       <Views>
        <NumberOfViews value="4"/>
        <View_0>
         <ViewID value="-2"/>
         <ViewType value="2"/>
         <Name value="Transverse"/>
         <Position value="0.0000000000000000e+00 -1.8000000000000000e+01 5.9383439472723376e+02 "/>
         <FocalPoint value="0.0000000000000000e+00 -1.8000000000000000e+01 1.8000000000000000e+01 "/>
         <Scale value="6.3166315072106393e+01"/>
         <ViewUp value="0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00 "/>
         <ViewAngle value="3.0000000000000000e+01"/>
        </View_0>
        <View_1>
         <ViewID value="-3"/>
         <ViewType value="3"/>
         <Name value="ThreeD"/>
         <Position value="5.2109475696103266e+02 -1.2685995307647572e+02 4.8483440068776019e+01 "/>
         <FocalPoint value="0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 "/>
         <Scale value="2.5980762113533160e+02"/>
         <ViewUp value="0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 "/>
         <ViewAngle value="3.0000000000000000e+01"/>
        </View_1>
        <View_2>
         <ViewID value="-4"/>
         <ViewType value="1"/>
         <Name value="Coronal"/>
         <Position value="0.0000000000000000e+00 -5.4255375505322445e+02 1.8000000000000000e+01 "/>
         <FocalPoint value="0.0000000000000000e+00 -1.8000000000000000e+01 1.8000000000000000e+01 "/>
         <Scale value="6.5009256279320482e+01"/>
         <ViewUp value="0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 "/>
         <ViewAngle value="3.0000000000000000e+01"/>
        </View_2>
        <View_3>
         <ViewID value="-5"/>
         <ViewType value="0"/>
         <Name value="Sagittal"/>
         <Position value="-5.7583439472723376e+02 -1.8000000000000000e+01 1.8000000000000000e+01 "/>
         <FocalPoint value="0.0000000000000000e+00 -1.8000000000000000e+01 1.8000000000000000e+01 "/>
         <Scale value="9.5406885948819266e+01"/>
         <ViewUp value="0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00 "/>
         <ViewAngle value="3.0000000000000000e+01"/>
        </View_3>
       </Views>
      </SceneManager>
      <AxesHidden value="0"/>
      <CursorVisible value="1"/>
      <CutPlanesCursorColor_r value="50"/>
      <CutPlanesCursorColor_g value="50"/>
      <CutPlanesCursorColor_b value="50"/>
      <QuadViewWindow>
       <CurrentViewWindow value="1"/>
       <ViewExpanded value="0"/>
      </QuadViewWindow>
      <Plugins>
       <GeneratedSurface/>
       <LabelVolumeToSurfaces/>
       <LandmarkRegistrationObject/>
       <PRISMVolumeRender/>
       <USAcquisitionDoubleView/>
       <SEEGAtlas/>
      </Plugins>
     </SaveScene>
    </configuration>
    """)

    #Replace correspoding image names in the xml scene

    xml_modified = default_xml.substitute(
                        replace_main_image_path = os.path.relpath(iter_summary['t1w_tal_noscale'].scan, iter_summary['output_dir']),
                        replace_main_image_name = iter_summary['t1w_tal_noscale'].name,
                        replace_mask_path = os.path.relpath(iter_summary['t1w_tal_noscale_mask'].scan, iter_summary['output_dir']),
                        replace_mask_name = iter_summary['t1w_tal_noscale_mask'].name,
                        replace_cortex_surface_path = os.path.relpath(iter_summary['cortex_surface'].fname, iter_summary['output_dir']),
                        replace_cortex_surface_name = iter_summary['cortex_surface'].name,
                        replace_skin_surface_path = os.path.relpath(iter_summary['skin_surface'].fname, iter_summary['output_dir']),
                        replace_skin_surface_name = iter_summary['skin_surface'].name
                        )

    file_out = open(fname, "wt")
    file_out.write(xml_modified)
    file_out.close()

default_pipeline_options = {
                'model':     'mni_icbm152_t1_tal_nlin_sym_09c',
                'model_dir': '/opt/minc/share/icbm152_model_09c',

                't1w_nuc':   {"distance":200.0},
                'add_nuc':   {"distance":200.0},

                't1w_clp':   {},
                'add_clp':   {},

                't1w_stx':   {  # options for linear registration
                        'type': '-lsq9',
                        'objective': '-nmi',
                        'options': 'bestlinreg_20180117',
                        'resample':False, # only used to resample to 1x1x1 when needed (if org data is very hig res)
                        #'type':'ants',
                        #'options': {     # options for linear regisration engine;  in this case for ants.
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

                'stx': {
                            'noscale':True, # this will give (size) scaled and unscaled data
                            'nuc': None,
                    },
                'beast':     { 'beastlib':  '/opt/minc/share/beast-library-1.1' },

                'brain_nl_seg':  None,
                'tissue_classify': {},
                'lobe_segment': {},

                'nl':        False, # to examine later... we might need these.
                'lobes':     False,
                'cls'  :     False,

                'qc':        {  # for QC images
                    'nu':      False,
                    't1w_stx': True,
                    'add_stx': True, # for T2 or PD is available
                    'cls':     True,
                    'lob':     True
                    },

                'aqc':        { # automated QC; none done here... for now.
                    'nu':      False,
                    't1w_stx': False,
                    'add_stx': False,
                    'cls':     False,
                    'lob':     False,
                    'slices':  3
                    },

                'denoise':   {}, # run standard patch-based denoising
                'ibis_output': False # export files for IBIS
            }

def standard_pipeline(info,
                      output_dir,
                      options =None,
                      work_dir=None,
                      manual_dir=None):
    """
    drop-in replacement for the standard pipeline

    Argumets: t1w_scan `MriScan` for T1w scan
            output_dir string pointing to output directory

    Kyword arguments:
            work_dir string pointing to work directory , default None - use output_dir
    """
    try:
        with temp_files() as tmp:
            if options is None:
                # try to use default options for 1.5T scan
                options = default_pipeline_options

            # setup parameters
            subject_id       = info['subject']
            timepoint_id     = info.get('visit', None)
            t1w_scan         = info['t1w']
            add_scans        = info.get('add', None)
            init_t1w_lin_xfm = info.get('init_t1w_lin_xfm', None)
            manual           = options.get('manual',None)
            ibis_output      = options.get('ibis_output',False)


            corr_t1w = info.get('corr_t1w', None)
            corr_add = info.get('corr_add', None)

            dataset_id=subject_id

            if dataset_id is None:
                dataset_id=t1w_scan.name

            if timepoint_id is not None:
                dataset_id=dataset_id+'_'+timepoint_id

            model_name=None
            model_dir=None

            # generate model reference
            if info.get('model_dir',None) is not None:
                model_dir =info['model_dir']
                model_name=info['model']
            else:
                model_dir =options['model_dir']
                model_name=options['model']

            model_t1w=MriScan(scan=model_dir+os.sep+options['model']+'.mnc',
                              mask=model_dir+os.sep+options['model']+'_mask.mnc')

            model_outline=MriScan(scan=model_dir+os.sep+options['model']+'_outline.mnc',
                                mask=None)

            lobe_atlas_dir =options.get('lobe_atlas_dir',None)
            lobe_atlas_defs=options.get('lobe_atlas_defs',None)

            if lobe_atlas_dir is None:
                lobe_atlas_dir=model_dir + os.sep + model_name + '_atlas'+os.sep

            if lobe_atlas_defs is None:
                lobe_atlas_defs=model_dir + os.sep + model_name + '_atlas'+os.sep+'lobe_defs.csv'
                if not os.path.exists(lobe_atlas_defs):
                    lobe_atlas_defs=None

            if work_dir is None:
                work_dir=output_dir+os.sep+'work_'+dataset_id

            run_qc    = options.get('qc',{})
            run_aqc   = options.get('aqc',None)
            run_nl    = options.get('nl',True)
            run_cls   = options.get('cls',True)
            run_lobes = options.get('lobes',True)

            if isinstance(run_qc, bool): # fix for old version of options
                run_qc={}
            if isinstance(run_aqc, bool): # fix for old version of options
                run_aqc={}

            denoise_parameters = options.get('denoise',None)
            nuc_parameters     = options.get('t1w_nuc',{})
            clp_parameters     = options.get('t1w_clp',{})
            stx_parameters     = options.get('t1w_stx',{})

            surfaces_parameters = options.get('surfaces',
                                  {'skin':True, 'cortex':True, 'hippocampus':True})

            create_unscaled    = stx_parameters.get('noscale',False)
            #stx_nuc            = stx_parameters.get('nuc',None)
            stx_disable        = stx_parameters.get('disable',False)

            clp_dir = work_dir+os.sep+'clp'
            tal_dir = work_dir+os.sep+'tal'
            nl_dir  = work_dir+os.sep+'nl'
            cls_dir = work_dir+os.sep+'tal_cls'
            qc_dir  = work_dir+os.sep+'qc'
            aqc_dir = work_dir+os.sep+'aqc'
            lob_dif = work_dir+os.sep+'lob'
            vol_dir = work_dir+os.sep+'vol'
            obj_dir = work_dir+os.sep+'obj'

            manual_clp_dir = None
            manual_tal_dir = None

            if manual_dir is not None:
                manual_clp_dir = manual_dir+os.sep+'clp'
                manual_tal_dir = manual_dir+os.sep+'tal'

            # create all subdirs (even if we don't need them because some outputs are not required)
            create_dirs([clp_dir,tal_dir,nl_dir,cls_dir,qc_dir,aqc_dir,lob_dif,vol_dir,obj_dir])

            # files produced by pipeline
            # native space
            t1w_den=MriScan(prefix=clp_dir,  name='den_'+dataset_id,   modality='t1w', mask=None)
            t1w_field=MriScan(prefix=clp_dir,name='fld_'+dataset_id,   modality='t1w', mask=None)
            t1w_nuc=MriScan(prefix=clp_dir,  name='n4_'+dataset_id,    modality='t1w', mask=None)
            t1w_clp=MriScan(prefix=clp_dir,  name='clamp_'+dataset_id, modality='t1w', mask=None)

            # stereotaxic space
            #    all filenames for internal use
            t1w_tal_xfm=MriTransform(prefix=tal_dir,name='tal_xfm_'+dataset_id)

            #      if auto registration does not work, we can start it with a manual seed
            if manual_dir is not None:
                manual_t1w_tal_xfm=MriTransform(prefix=manual_tal_dir,name='tal_xfm_'+dataset_id)

                if os.path.exists(manual_t1w_tal_xfm.xfm) and init_t1w_lin_xfm is None: # HACK ish...
                    init_t1w_lin_xfm=manual_t1w_tal_xfm
                else:
                    print("Missing manual xfm:{}".format(manual_t1w_tal_xfm.xfm))

            t1w_tal_noscale_xfm=MriTransform(prefix=tal_dir,name='tal_noscale_xfm_'+dataset_id)
            unscale_xfm=MriTransform(prefix=tal_dir,name='unscale_xfm_'+dataset_id)

            t1w_tal=MriScan(prefix=tal_dir, name='tal_'+dataset_id, modality='t1w')
            t1w_tal_fld=MriScan(prefix=tal_dir, name='tal_fld_'+dataset_id, modality='t1w') # to xform nonuniformity correction field into stx space

            t1w_tal_noscale=MriScan(prefix=tal_dir, name=dataset_id+'_full_head_image',modality='t1w')

            t1w_tal_noscale_masked=MriScan(prefix=tal_dir, name=dataset_id+'_brain_image',modality='t1w')

            t1w_tal_noscale_cortex=MriAux(prefix=obj_dir, name=dataset_id+'_cortex_surface', suffix='.obj')
            t1w_tal_noscale_skin=MriAux(prefix=obj_dir, name=dataset_id+'_skin_surface', suffix='.obj')
            t1w_tal_noscale_hippocampus=MriAux(prefix=obj_dir, name=dataset_id+'_hippocampus_surface', suffix='.obj')

            t1w_tal_par=MriAux(prefix=tal_dir,name='tal_par_t1w_'+dataset_id) # for elastics only...
            t1w_tal_log=MriAux(prefix=tal_dir,name='tal_log_t1w_'+dataset_id)

            # tissue classification results (if requested)
            tal_cls=MriScan(prefix=cls_dir, name='cls_'+dataset_id)
            native_t1w_cls=MriScan(prefix=clp_dir,  name='cls_'+dataset_id, modality='t1w')
            # lobe segmentation results (if requested)
            tal_lob=MriScan(prefix=lob_dif, name='lob_'+dataset_id)

            # nl space  (if requested)
            nl_xfm=MriTransform(prefix=nl_dir, name='nl_'+dataset_id)

            # QC files(if requested)
            qc_tal= MriQCImage(prefix=qc_dir,name='tal_t1w_'+dataset_id)
            qc_mask=MriQCImage(prefix=qc_dir,name='tal_mask_'+dataset_id)
            qc_cls= MriQCImage(prefix=qc_dir,name='tal_cls_'+dataset_id)
            qc_lob= MriQCImage(prefix=qc_dir,name='tal_lob_'+dataset_id)
            qc_nu=  MriQCImage(prefix=qc_dir,name='nu_'+dataset_id)

            # QC files (if requested)
            aqc_tal= MriQCImage(prefix=aqc_dir,name='tal_t1w_'+dataset_id,suffix='')
            aqc_mask=MriQCImage(prefix=aqc_dir,name='tal_mask_'+dataset_id,suffix='')
            aqc_cls= MriQCImage(prefix=aqc_dir,name='tal_cls_'+dataset_id,suffix='')
            aqc_lob= MriQCImage(prefix=aqc_dir,name='tal_lob_'+dataset_id,suffix='')
            aqc_nu=  MriQCImage(prefix=aqc_dir,name='nu_'+dataset_id,suffix='')

            # AUX files (filenames for segmentation volumes and other measurements)
            lob_volumes=MriAux(prefix=vol_dir,name='vol_'+dataset_id)
            lob_volumes_json=MriAux(prefix=vol_dir,name='vol_'+dataset_id,suffix='.json')

            ibis_summary_file=MriAux(prefix=work_dir,name='summary_'+dataset_id,suffix='.xml')
            summary_file=MriAux(prefix=work_dir,name='summary_'+dataset_id,suffix='.json')



            iter_summary={      # dictionary containing all the filenames (for other scripts eventually)
                        'subject':      subject_id,
                        'timepoint':    timepoint_id,
                        'dataset_id':   dataset_id,

                        'input_t1w':    t1w_scan,
                        'input_add':    add_scans,

                        'output_dir':   output_dir,

                        "t1w_field":    t1w_field,
                        "t1w_nuc":      t1w_nuc,
                        "t1w_clp":      t1w_clp,

                        "t1w_tal_xfm":  t1w_tal_xfm,
                        "t1w_tal":      t1w_tal,
                        "t1w_tal_noscale":t1w_tal_noscale,

                        "corr_t1w": corr_t1w,
                        "corr_add": corr_add
                        }

            # actual processing steps
            # 1. preprocessing
            if denoise_parameters is not None:
                denoise(t1w_scan, t1w_den, parameters=denoise_parameters)
                t1w_den.mask=t1w_scan.mask
            else:
                t1w_den=t1w_scan

            iter_summary["t1w_den"]=t1w_den

            if nuc_parameters is not None:
                # non-uniformity correction
                print("Running N4")

                estimate_nu(t1w_den, t1w_field,
                            parameters=nuc_parameters,
                            model=model_t1w)
                if run_qc is not None and run_qc.get('nu',False):
                    draw_qc_nu(t1w_field,qc_nu,options=run_qc)
                    iter_summary["qc_nu"]=qc_nu
                if run_aqc is not None and run_aqc.get('nu',False):
                    make_aqc_nu(t1w_field,aqc_nu,options=run_aqc)
                    iter_summary["aqc_nu"]=aqc_nu

                # apply field
                apply_nu(t1w_den, t1w_field, t1w_nuc,
                        parameters=nuc_parameters)
                t1w_nuc.mask=t1w_den.mask
            else:
                t1w_nuc=t1w_den
                t1w_field=None

            iter_summary["t1w_field"] = t1w_field
            iter_summary["t1w_nuc"]   = t1w_nuc

            ################
            # normalize intensity

            if clp_parameters is not None:
                normalize_intensity(t1w_nuc, t1w_clp,
                                    parameters=options.get('t1w_clp',{}),
                                    model=model_t1w)
                t1w_clp.mask=t1w_nuc.mask
            else:
                t1w_clp=t1w_nuc

            iter_summary["t1w_clp"]   = t1w_clp

            ####
            if add_scans is not None: # any additional modalities to worry about?
                iter_summary["add_den"]   = []
                iter_summary["add_field"] = []
                iter_summary["add_nuc"]   = []
                iter_summary["add_clp"]   = []
                iter_summary["add_xfm"]   = []

                prev_co_xfm=None

                for i,c in enumerate(add_scans): # if so, deal with each modality, mapping to t1.
                    # get add options
                    #TODO do it per modality
                    add_options            = options.get('add',options)

                    add_denoise_parameters = add_options.get('denoise',denoise_parameters)
                    add_nuc_parameters     = add_options.get('nuc'    ,nuc_parameters)
                    add_clp_parameters     = add_options.get('clp'    ,clp_parameters)
                    add_stx_parameters     = add_options.get('stx'    ,stx_parameters)
                    add_model_dir          = add_options.get('model_dir',model_dir)
                    add_model_name         = add_options.get('model'  ,model_name)

                    add_denoise_parameters = add_options.get('{}_denoise'.format(c.modality),add_denoise_parameters)
                    add_nuc_parameters     = add_options.get('{}_nuc'    .format(c.modality),add_nuc_parameters)
                    add_clp_parameters     = add_options.get('{}_clp'    .format(c.modality),add_clp_parameters)
                    add_stx_parameters     = add_options.get('{}_stx'    .format(c.modality),add_stx_parameters)
                    add_model_dir          = add_options.get('{}_model_dir'.format(c.modality),add_model_dir)
                    add_model_name         = add_options.get('{}_model'  .format(c.modality),add_model_name)

                    add_model              = MriScan(scan=add_model_dir+os.sep+add_model_name+'.mnc',
                                                      mask=model_t1w.mask)

                    den   = MriScan(prefix=clp_dir,  name='den_'  +dataset_id, modality=c.modality, mask=None)
                    field = MriScan(prefix=clp_dir,  name='fld_'  +dataset_id, modality=c.modality, mask=None)
                    nuc   = MriScan(prefix=clp_dir,  name='n4_'   +dataset_id, modality=c.modality, mask=None)
                    clp   = MriScan(prefix=clp_dir,  name='clamp_'+dataset_id, modality=c.modality, mask=None)

                    add_qc_nu = MriQCImage(prefix=qc_dir,    name='nu_' + c.modality+'_' + dataset_id)
                    add_aqc_nu= MriQCImage(prefix=aqc_dir,   name='nu_' + c.modality+'_' + dataset_id)
                    co_xfm= MriTransform(prefix=clp_dir,     name='xfm_'+ c.modality+'_' + dataset_id)

                    manual_co_xfm = None
                    if manual_dir is not None:
                        manual_co_xfm=MriTransform(prefix=manual_clp_dir,name='xfm_'+ c.modality+'_' + dataset_id)
                        if not os.path.exists(manual_co_xfm.xfm):
                            print("Missing manual xfm:{}".format(manual_co_xfm.xfm))
                            manual_co_xfm=None

                    co_par=MriAux(prefix=clp_dir, name='xfm_par_'+ c.modality+'_'+dataset_id)
                    co_log=MriAux(prefix=clp_dir, name='xfm_log_'+ c.modality+'_'+dataset_id)

                    corr_xfm=None
                    if corr_add is not None:
                        corr_xfm=corr_add[i]

                    # denoising
                    if add_denoise_parameters is not None:
                        denoise(c, den, parameters=add_denoise_parameters)
                        iter_summary["add_den"].append(den)
                        den.mask=c.mask # maybe transfer mask from t1w ?
                    else:
                        den=c

                    # non-uniformity correction
                    if add_nuc_parameters is not None:
                        estimate_nu(den, field, parameters=add_nuc_parameters,model=add_model)
                        if run_qc is not None and run_qc.get('nu',False):
                            draw_qc_nu(field,add_qc_nu,options=run_qc)
                            iter_summary["qc_nu_"+c.modality]=add_qc_nu
                        if run_aqc is not None and run_aqc.get('nu',False):
                            make_aqc_nu(field,add_aqc_nu,options=run_aqc)
                            iter_summary["aqc_nu_"+c.modality]=add_aqc_nu
                        # apply field
                        apply_nu(den, field, nuc, parameters=add_nuc_parameters)
                        nuc.mask=den.mask
                    else:
                        nuc=den

                    #
                    iter_summary["add_field"].append(field)
                    iter_summary["add_nuc"].append(nuc)

                    if add_clp_parameters is not None:
                        normalize_intensity(nuc, clp,
                                    parameters=add_clp_parameters,
                                    model=add_model)
                        clp.mask=nuc.mask
                    else:
                        clp=nuc

                    iter_summary["add_clp"].append(clp)

                    # co-registering to T1w
                    if add_stx_parameters.get('independent',True) or (prev_co_xfm is None):
                        # run co-registration unless another one can be used
                        intermodality_co_registration(clp, t1w_clp, co_xfm,
                                        parameters=add_stx_parameters,
                                        corr_xfm=corr_xfm,
                                        corr_ref=corr_t1w,
                                        par=co_par,
                                        log=co_log,
                                        init_xfm=manual_co_xfm )
                        prev_co_xfm=co_xfm
                    else:
                        co_xfm=prev_co_xfm

                    iter_summary["add_xfm"].append(co_xfm)

            if not stx_disable:
                # register to STX space
                lin_registration(t1w_clp, model_t1w, t1w_tal_xfm,
                                parameters=stx_parameters,
                                corr_xfm=corr_t1w,
                                par=t1w_tal_par,
                                log=t1w_tal_log,
                                init_xfm=init_t1w_lin_xfm)

                stx_nuc = stx_parameters.get('nuc',None)
                stx_clp = stx_parameters.get('clp',None)

                if stx_nuc is not None:
                    tmp_t1w=MriScan(prefix=tmp.tempdir,    name='tal_'+dataset_id, modality='t1w')
                    tmp_t1w_n4=MriScan(prefix=tmp.tempdir, name='tal_n4_'+dataset_id, modality='t1w')

                    warp_scan(t1w_clp ,model_t1w, tmp_t1w,
                            transform=t1w_tal_xfm,
                            corr_xfm=corr_t1w,
                            parameters=stx_parameters)
                    tmp_t1w.mask=None
                    tmp_t1w_n4.mask=None

                    estimate_nu(tmp_t1w, t1w_tal_fld,
                                parameters=stx_nuc)

                    apply_nu(tmp_t1w, t1w_tal_fld, tmp_t1w_n4,
                            parameters=stx_nuc)

                    #TODO: maybe apply region-based intensity normalization here?
                    normalize_intensity(tmp_t1w_n4, t1w_tal,
                                    parameters=stx_clp,
                                    model=model_t1w)

                    iter_summary['t1w_tal_fld']=t1w_tal_fld

                else:
                    warp_scan(t1w_clp,model_t1w, t1w_tal,
                            transform=t1w_tal_xfm,
                            corr_xfm=corr_t1w,
                            parameters=options.get('t1w_stx',{}))


                if add_scans is not None:
                    iter_summary["add_stx_xfm"]   = []
                    iter_summary["add_tal_fld"]   = []
                    iter_summary["add_tal"]       = []

                    for i,c in enumerate(add_scans):
                        add_stx_parameters     = add_options.get('stx'    ,stx_parameters)
                        add_model_dir          = add_options.get('model_dir',model_dir)
                        add_model_name         = add_options.get('model'  ,model_name)

                        add_stx_parameters     = add_options.get('{}_stx'    .format(c.modality),add_stx_parameters)
                        add_model_dir          = add_options.get('{}_model_dir'.format(c.modality),add_model_dir)
                        add_model_name         = add_options.get('{}_model'  .format(c.modality),add_model_name)

                        add_model = MriScan(scan=add_model_dir+os.sep+add_model_name+'.mnc',
                                        mask=model_t1w.mask)

                        add_stx_nuc = add_stx_parameters.get('nuc',None)
                        add_stx_clp = add_stx_parameters.get('clp',None)


                        stx_xfm = MriTransform(prefix=tal_dir, name='xfm_'+c.modality+'_'+dataset_id)

                        clp=iter_summary["add_clp"][i]
                        xfm=iter_summary["add_xfm"][i]
                        tal_fld=MriScan(prefix=tal_dir, name='tal_fld_'+dataset_id, modality=c.modality)
                        tal=MriScan(prefix=tal_dir, name='tal_'+dataset_id, modality=c.modality)

                        xfm_concat( [xfm,t1w_tal_xfm], stx_xfm )
                        iter_summary["add_stx_xfm"].append(stx_xfm)

                        corr_xfm=None
                        if corr_add is not None:
                            corr_xfm=corr_add[i]

                        if add_stx_nuc is not None:
                            tmp_=MriScan(prefix=tmp.tempdir,   name='tal_'+dataset_id, modality=c.modality)
                            tmp_n4=MriScan(prefix=tmp.tempdir, name='tal_n4_'+dataset_id, modality=c.modality)

                            warp_scan(clp ,model_t1w, tmp_,
                                    transform=stx_xfm,
                                    corr_xfm=corr_xfm,
                                    parameters=add_stx_parameters)

                            tmp_.mask=None
                            tmp_n4.mask=None

                            estimate_nu(tmp_, tal_fld,
                                        parameters=add_stx_nuc)

                            apply_nu(tmp_, tal_fld, tmp_n4, parameters=add_nuc_parameters)

                            #TODO: maybe apply region-based intensity normalization here?
                            normalize_intensity(tmp_n4, tal,
                                            parameters=add_stx_clp,
                                            model=add_model)

                            iter_summary["add_tal_fld"].append(tal_fld)

                        else:
                            warp_scan(clp,model_t1w, tal,
                                    transform=stx_xfm,
                                    corr_xfm=corr_xfm,
                                    parameters=add_stx_parameters)

                        iter_summary["add_tal"].append(tal)

                if run_qc is not None and run_qc.get('t1w_stx',True):
                    draw_qc_stx(t1w_tal,model_outline,qc_tal,options=run_qc)
                    iter_summary["qc_tal"]=qc_tal

                    if add_scans is not None:
                        iter_summary["qc_add"]=[]
                        for i,c in enumerate(add_scans):
                            qc=MriQCImage(prefix=qc_dir,name='tal_'+c.modality+'_'+dataset_id)
                            if run_qc is not None and run_qc.get('add_stx',True):
                                draw_qc_add(t1w_tal,iter_summary["add_tal"][i],qc,options=run_qc)
                                iter_summary["qc_add"].append(qc)

                if run_aqc is not None and run_aqc.get('t1w_stx',True):
                    make_aqc_stx(t1w_tal,model_outline,aqc_tal,options=run_aqc)
                    iter_summary["aqc_tal"]=aqc_tal

                    if add_scans is not None:
                        iter_summary["aqc_add"]=[]
                        for i,c in enumerate(add_scans):
                            aqc=MriQCImage(prefix=aqc_dir,name='tal_'+c.modality+'_'+dataset_id)
                            if run_aqc is not None and run_aqc.get('add_stx',True):
                                make_aqc_add(t1w_tal,iter_summary["add_tal"][i],aqc,options=run_aqc)
                                iter_summary["aqc_add"].append(aqc)

                # run beast to create brain mask
                beast_parameters=options.get('beast',None)
                if beast_parameters is not None:
                    extract_brain_beast(t1w_tal,parameters=beast_parameters,model=model_t1w)
                    if run_qc is not None and run_qc.get('beast',True):
                        draw_qc_mask(t1w_tal,qc_mask,options=run_qc)
                        iter_summary["qc_mask"]=qc_mask
                    if run_aqc is not None and run_aqc.get('beast',True):
                        make_aqc_mask(t1w_tal,aqc_mask,options=run_aqc)
                        iter_summary["aqc_mask"]=aqc_mask

                else:
                    #extract_brain_nlreg(t1w_tal,parameters=options.get('brain_nl_seg',{}),model=model_t1w)
                    # if we have initial mask, keep using that!
                    if t1w_clp.mask is not None:
                        warp_mask(t1w_clp,model_t1w, t1w_tal,
                            transform=t1w_tal_xfm,
                            corr_xfm=corr_t1w,
                            parameters=options.get('t1w_stx',{}))
                    t1w_tal.mask=None
                    pass

                # create unscaled version
                if create_unscaled:
                    xfm_remove_scale(t1w_tal_xfm, t1w_tal_noscale_xfm, unscale=unscale_xfm)
                    iter_summary["t1w_tal_noscale_xfm"]=t1w_tal_noscale_xfm
                    #warp scan to create unscaled version
                    warp_scan(t1w_clp, model_t1w, t1w_tal_noscale, transform=t1w_tal_noscale_xfm, corr_xfm=corr_t1w)
                    # warping mask from tal space to unscaled tal space
                    warp_mask(t1w_tal, model_t1w, t1w_tal_noscale, transform=unscale_xfm)
                    iter_summary["t1w_tal_noscale"]=t1w_tal_noscale

                    if surfaces_parameters.get('skin',False) \
                        or surfaces_parameters.get('cortex',False) \
                            or surfaces_parameters.get('hippocampus',False):
                        # do skin, cortex, and hippocampus processing here

                        if surfaces_parameters.get('cortex',False) :
                            with mincTools(verbose=2) as minc:
                                #start with t1w_tal_noscale
                                #mincmask t1w_tal_noscale.mnc t1w_tal_noscale_mask.mnc t1w_tal_noscale_masked.mnc
                                minc.command(['mincmask','-clobber',t1w_tal_noscale.scan,t1w_tal_noscale.mask,t1w_tal_noscale_masked.scan])
                                #marching_cubes t1w_tal_noscale_masked.mnc cortex.obj 45
                                minc.command(['marching_cubes',t1w_tal_noscale_masked.scan,t1w_tal_noscale_cortex.fname,'45'])
                                #ascii_binary cortex.obj
                                minc.command(['ascii_binary', t1w_tal_noscale_cortex.fname])
                            iter_summary['cortex_surface'] = t1w_tal_noscale_cortex
                            iter_summary['t1w_tal_noscale_mask'] = t1w_tal_noscale_masked

                        if surfaces_parameters.get('skin',False) :
                            with mincTools(verbose=2) as minc:
                                #start with t1w_tal_noscale, then blur it.
                                tmpname = minc.tmp('tmp_t1_noscaled')
                                minc.command(['mincblur','-clobber','-fwhm','2', t1w_tal_noscale.scan,tmpname])
                                #marching_cubes t1w_tal_noscale_masked.mnc skin.obj 30
                                minc.command(['marching_cubes',tmpname+'_blur.mnc',t1w_tal_noscale_skin.fname,'30'])
                                #ascii_binary cortex.obj
                                minc.command(['ascii_binary', t1w_tal_noscale_skin.fname])
                            iter_summary['skin_surface'] = t1w_tal_noscale_skin

                        if surfaces_parameters.get('hippocampus',False) :
                            with mincTools(verbose=2) as minc:
                                #start with t1w_tal_noscale and then segment hippocampus
                                tmp_work = minc.tmp('tmp_work')
                                tmp_output = minc.tmp('tmp_output')
                                fusion_segment(input_scan= t1w_tal_noscale.scan, 
                                            library_description='/data/ipl/scratch08/vfonov/adni_jens/jens_hc_lib_20170621/library.json',
                                            output_segment=tmp_output,
                                            parameters='/data/ipl/scratch08/vfonov/adni_jens/jens_hc_segment_20170621.json',
                                            work_dir=tmp_work,
                                            cleanup = True)
                                minc.command(['marching_cubes',tmp_output+'_seg.mnc',t1w_tal_noscale_hippocampus.fname,'0'])
                                minc.command(['ascii_binary', t1w_tal_noscale_hippocampus.fname])
                            iter_summary['hippocampus_surface'] = t1w_tal_noscale_hippocampus

                    # perform non-linear registration
                if run_nl:
                    nl_registration(t1w_tal, model_t1w, nl_xfm,
                                parameters=options.get('nl_reg',{}))
                    iter_summary["nl_xfm"]=nl_xfm

                # run tissue classification
                if run_nl and run_cls:
                    classify_tissue(t1w_tal, tal_cls, model_name=model_name,
                                model_dir=model_dir, xfm=nl_xfm,
                                parameters=options.get('tissue_classify',{}))

                    warp_cls_back (t1w_tal, tal_cls, t1w_tal_xfm, t1w_nuc, native_t1w_cls,corr_xfm=corr_t1w)
                    warp_mask_back(t1w_tal, t1w_tal_xfm, t1w_nuc, native_t1w_cls,corr_xfm=corr_t1w)
                    iter_summary["native_t1w_cls"]=native_t1w_cls
                    iter_summary["tal_cls"]=tal_cls
                    if run_qc is not None  and run_qc.get('cls',True):
                        draw_qc_cls(t1w_tal,tal_cls,qc_cls,options=run_qc)
                    if run_aqc is not None  and run_aqc.get('cls',True):
                        make_aqc_cls(t1w_tal,tal_cls,aqc_cls,options=run_aqc)
                else:
                    # just warp mask back
                    if beast_parameters is not None:
                        warp_mask_back(t1w_tal, t1w_tal_xfm, t1w_nuc, native_t1w_cls,corr_xfm=corr_t1w)
                        native_t1w_cls.scan=None
                        iter_summary["tal_cls"]=tal_cls


                # run lobe segmentation
                if run_nl and run_cls and run_lobes:
                    segment_lobes( tal_cls, nl_xfm, tal_lob,
                            model=model_t1w,
                            lobe_atlas_dir=lobe_atlas_dir,
                            parameters=options.get('lobe_segment',{}))
                    iter_summary["tal_lob"]=tal_lob

                    if run_qc is not None  and run_qc.get('lob',True):
                        draw_qc_lobes( t1w_tal, tal_lob,qc_lob,options=run_qc)
                        iter_summary["qc_lob"]=qc_lob
                    if run_aqc is not None  and run_aqc.get('lob',True):
                        make_aqc_lobes( t1w_tal, tal_lob,aqc_lob,options=run_aqc)
                        iter_summary["aqc_lob"]=aqc_lob

                    # calculate volumes
                    extract_volumes(tal_lob, tal_cls, t1w_tal_xfm, lob_volumes,
                                    subject_id=subject_id, timepoint_id=timepoint_id , lobedefs=lobe_atlas_defs)

                    extract_volumes(tal_lob, tal_cls, t1w_tal_xfm, lob_volumes_json,
                                    produce_json=True,subject_id=subject_id, timepoint_id=timepoint_id,lobedefs=lobe_atlas_defs)

                    iter_summary["lob_volumes"]=     lob_volumes
                    iter_summary["lob_volumes_json"]=lob_volumes_json

            # TODO: figure out when this is needed
            if ibis_output:
              save_ibis_summary(iter_summary, ibis_summary_file.fname) # use this 
            else:
              save_summary(iter_summary, summary_file.fname) # to build scene.xml for IBIS
            return iter_summary

    except mincError as e:
        print("Exception in iter_step:{}".format(str(e)))
        traceback.print_exc( file=sys.stdout )
        raise
    except :
        print("Exception in iter_step:{}".format(sys.exc_info()[0]))
        traceback.print_exc( file=sys.stdout)
        raise

#kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on

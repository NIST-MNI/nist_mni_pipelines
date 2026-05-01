#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 2024
#

version = '1.0'

#
# Arbitrary ONNX Segmentation for longitudinal pipeline
# - Similar to run_mindglide.sh but integrated internally
# - Config-driven approach like redskull skull stripping
# - Volumetric measurements output like lobe segmentation

import sys

from .general import *
from optparse import OptionParser
from optparse import OptionGroup

from ipl.minc_tools import mincTools, mincError
from ipl import minc_qc

try:
    from ipl.apply_multi_model_onnx import segment_with_onnx
    _have_segmentation_onnx = True
except:
    _have_segmentation_onnx = False
    import traceback
    traceback.print_exc(file=sys.stdout)
    print("Missing onnxruntime, will not be able to run ONNX segmentation")

import json
import csv
import ray
import copy


def pipeline_onnx_segmentation(patient, tp, config_file, model_prefix=None):
    """
    Apply arbitrary ONNX segmentation model to a timepoint.
    
    Args:
        patient: LngPatient object
        tp: Timepoint name
        config_file: Path to JSON configuration file
        model_prefix: Optional model prefix for ONNX models
    """
    if not _have_segmentation_onnx:
        print("WARNING: ONNX segmentation not available, skipping")
        return False
    
    if not os.path.exists(config_file):
        raise mincError(f'ONNX segmentation config file not found: {config_file}')
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    output_suffix = config.get('output_suffix', 'onnx_seg')
    
    if not os.path.exists(patient[tp].tpdir + 'seg/'):
        os.makedirs(patient[tp].tpdir + 'seg/', exist_ok=True)
    
    onnx_segmentation_v10(patient, tp, config, model_prefix)
    
    return True


def onnx_segmentation_to_json(patient, tp, seg_data, labels_desc, 
                               output_json=None, output_csv=None):
    """
    Save segmentation volumes to JSON and CSV format.
    Similar to lobes_to_json().
    
    Args:
        patient: LngPatient object
        tp: Timepoint name
        seg_data: Dictionary with volume measurements
        labels_desc: List of label descriptions
        output_json: Path to output JSON file
        output_csv: Path to output CSV file
    """
    out = {
        "SubjectID": patient.id,
        "VisitID": tp,
        "ScaleFactor": 1.0,
        "Age": patient[tp].age,
        "Gender": patient.sex
    }
    
    for label in labels_desc:
        out[label] = seg_data.get(label, 0.0)
    
    if output_json is not None:
        with open(output_json, 'w') as f:
            json.dump(out, f, indent=2, sort_keys=True)
    
    if output_csv is not None:
        with open(output_csv, 'w') as f:
            fieldnames = sorted(out.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(out)
    
    return out


@ray.remote(num_cpus=4, memory=10000 * 1024 * 1024)
def run_onnx_segmentation(input_file, output_file, config, threads=1, model_prefix=None):
    """
    Apply ONNX segmentation model using configuration.
    Ray remote function similar to run_redskull_onnx.
    
    Args:
        input_file: Input MINC file
        output_file: Output segmentation MINC file
        config: Configuration dictionary
        threads: Number of threads to use
        model_prefix: Optional model prefix for ONNX models
    """
    assert _have_segmentation_onnx, "Failed to import segment_with_onnx"
    n_threads = int(ray.runtime_context.get_runtime_context().get_assigned_resources()["CPU"])
    
    #settings = dict(config.get('apply_multi_model_onnx_options', {}))
    settings = copy.deepcopy(config)

    models = list(config.get('models', []))
    if model_prefix is not None:
        models = [model_prefix + os.sep + m for m in models]
    settings['models'] = models
    
    if 'n_classes' in config:
        settings['n_classes'] = config['n_classes']
    
    if 'labels_desc' in config:
        settings['labels_desc'] = config['labels_desc']
    
    segment_with_onnx(
        [input_file],
        output_file,
        settings=settings,
        threads=n_threads,
        measure=config.get('measurements_file')
    )


def onnx_segmentation_v10(patient, tp, config, model_prefix=None):
    """
    Core ONNX segmentation implementation.
    
    Args:
        patient: LngPatient object
        tp: Timepoint name
        config: Configuration dictionary
        model_prefix: Optional model prefix for ONNX models
    """
    output_suffix = config.get('output_suffix', 'onnx_seg')
    input_space = config.get('input_space', 'stx2')
    input_sequence = config.get('input_sequence', 't1')
    qc_enabled = config.get('qc_enabled', True)
    
    with mincTools() as minc:
        input_file = None
        
        if input_space == 'native':
            input_file = patient[tp].clp.get(input_sequence)
            stx_scale=1.0
        elif input_space == 'clp':
            input_file = patient[tp].clp.get(input_sequence)
            stx_scale=1.0
        elif input_space == 'nsstx':
            input_file = patient[tp].stx_ns_mnc.get(input_sequence)
            stx_scale=1.0
        elif input_space == 'stx2':
            input_file = patient[tp].stx2_mnc.get(input_sequence)
            # need to calculate scaling factor from the transformation
            with mincTools()  as minc:
                params=minc.xfm2param(patient[tp].stx2_xfm['t1'])
                stx_scale = params['scale'][0] * params['scale'][1] * params['scale'][2]

        else:
            raise mincError(f'Invalid input_space: {input_space}')
        
        if input_file is None or not os.path.exists(input_file):
            raise mincError(f'Input file not found for {input_space}/{input_sequence}: {input_file}')
        
        seg_output = patient[tp].tpdir + f'seg/seg_{output_suffix}_{patient.id}_{tp}.mnc'
        vol_txt = patient[tp].tpdir + f'vol/vol_{output_suffix}_{patient.id}_{tp}.txt'
        vol_json = patient[tp].tpdir + f'vol/vol_{output_suffix}_{patient.id}_{tp}.json'
        vol_csv = patient[tp].tpdir + f'vol/vol_{output_suffix}_{patient.id}_{tp}.csv'
        qc_output = patient.qcdir + f'qc_onnx_seg_{output_suffix}_{patient.id}_{tp}.jpg'
        
        if os.path.exists(seg_output) and os.path.exists(qc_output):
            return True
        
        run_onnx_segmentation_c = run_onnx_segmentation.options(num_cpus=patient.threads)
        ray.get(run_onnx_segmentation_c.remote(
            input_file,
            seg_output,
            config,
            patient.threads,
            model_prefix
        ))
        
        labels_desc = config.get('labels_desc', [])
        
        if labels_desc and os.path.exists(seg_output):
            from ipl.seg_common.postprocess import measure_volumes
            from ipl.seg_common.io import load_volume_np
            
            seg_data, aff = load_volume_np(seg_output, dtype='int16')
            vols = [measure_volumes(seg_data, aff, labels_desc, 
                                   out_seg_f=seg_output, 
                                   in_scan=input_file,
                                   scale=1.0/stx_scale)]
            
            with open(vol_txt, 'w') as f:
                for label in labels_desc:
                    vol = vols[0].get(label, 0.0)
                    f.write(f'{label} {vol}\n')
            
            onnx_segmentation_to_json(patient, tp, vols[0], labels_desc,
                                      output_json=vol_json,
                                      output_csv=vol_csv)
        
        if qc_enabled:
            mask_cmap = config.get('qc_cmap', 'spectral')
            image_range = config.get('qc_image_range', [0, 120])
            
            minc_qc.qc(
                input_file,
                qc_output,
                title=patient[tp].qc_title,
                image_range=image_range,
                mask=seg_output,
                dpi=200,
                use_max=True,
                samples=20,
                bg_color="black",
                fg_color="white",
                mask_cmap=mask_cmap
            )
    
    return True


if __name__ == '__main__':
    usage = """usage: %prog 
    """
    
    parser = OptionParser(usage=usage, version=version)
    
    group = OptionGroup(parser, ' -- Launch options',
                        ' Options to start processing')
    group.add_option('-c', '--config', dest='config',
                     help='ONNX segmentation configuration JSON file')
    group.add_option('-i', '--input', dest='input',
                     help='Input image')
    group.add_option('-o', '--output', dest='output',
                     help='Output segmentation')
    group.add_option('-t', '--threads', dest='threads',
                     help='Number of threads',
                     default='1')
    parser.add_option_group(group)
    
    (opts, args) = parser.parse_args()
    
    if opts.config is None:
        print(' -- ERROR: -c, --config is mandatory')
        sys.exit(-1)
    
    if opts.input is None or opts.output is None:
        print(' -- ERROR: -i, --input and -o, --output are mandatory')
        sys.exit(-1)
    
    with open(opts.config, 'r') as f:
        config = json.load(f)
    
    config['measurements_file'] = opts.output + '.vol.txt'
    
    segment_with_onnx(
        [opts.input],
        opts.output,
        settings=config.get('apply_multi_model_onnx_options', {}),
        threads=int(opts.threads),
        measure=config['measurements_file']
    )

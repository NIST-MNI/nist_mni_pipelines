#! /bin/bash

in_lst=$1
out_pfx=$2

if [[ -z $out_pfx ]];then
    echo "Usage: $0 <input.lst> <output_prefix>"
    exit 1
fi

PRL=4
THREADS=4

python nist_mni_pipelines/iplLongitudinalPipeline.py \
    -3 \
    --denoise \
    -l $in_lst \
    -o $out_pfx \
    --model-dir=/opt/models/icbm152_model_09c \
    --model-name=mni_icbm152_t1_tal_nlin_sym_09c  \
    --large_atrophy \
    --ray_start $PRL \
    --threads $THREADS \
    --nl_ants \
    --nl_step 1.0 \
    --nl_cost_fun CC \
    --bison_pfx /opt/models/ipl_bison_1.3.0 \
    --bison_method  HGB1 \
    --wmh_bison_pfx /opt/models/wmh_bison_1.3.0/t1_3T \
    --wmh_bison_atlas_pfx /opt/models/wmh_bison_1.3.0/3T_2009c_ \
    --wmh_bison_method HGB1 \
    --synthstrip_onnx /opt/models/synthstrip/mri_synthstrip_dyn_native_v2.onnx

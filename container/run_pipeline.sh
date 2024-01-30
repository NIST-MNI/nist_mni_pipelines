#! /bin/bash

#########################################################
# 
# Execute the pipeline using apptainer or docker
# 
# Command line arguments:
# 1. input.lst - list of subjects to process
# 2. output directory prefix
# 
# All paths should be relative to the current directory
#
# Environment variables:
#
# PRL - number of parallel processes (default 4)
# THREADS - number of threads per process (default 4)
# FIELD - 1.5 or 3 (default 3) scanner field strength
# CLEANUP - YES or NO (default NO) remove intermediate files to save disk space
#
#####



in_lst=$1
out_pfx=$2

if [[ -z $out_pfx ]];then
    echo "Usage: $0 <input.lst> <output_prefix>"
    exit 1
fi

PRL=${PRL:-4}
THREADS=${THREADS:-4}
FIELD=${FIELD:-3}
CLEANUP=${CLEANUP:-NO}


########

if [[ $CLEANUP == YES ]];then
    CLEANUP=--cleanup
else
    CLEANUP=
fi

if [[ $FIELD == 3 ]];then
    echo "Running 3T pipeline"

    python /opt/pipeline/iplLongitudinalPipeline.py \
        -3 \
        --denoise \
        -l $in_lst \
        -o $out_pfx \
        --model-dir=/opt/models/icbm152_model_09c \
        --model-name=mni_icbm152_t1_tal_nlin_sym_09c  \
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
        --synthstrip_onnx /opt/models/synthstrip/synthstrip.1.onnx \
        $CLEANUP
else
    echo "Running 1.5T pipeline"

    python /opt/pipeline/iplLongitudinalPipeline.py \
        --denoise \
        -l $in_lst \
        -o $out_pfx \
        --model-dir=/opt/models/icbm152_model_09c \
        --model-name=mni_icbm152_t1_tal_nlin_sym_09c  \
        --ray_start $PRL \
        --threads $THREADS \
        --nl_ants \
        --nl_step 1.0 \
        --nl_cost_fun CC \
        --bison_pfx /opt/models/ipl_bison_1.3.0 \
        --bison_method  HGB1 \
        --wmh_bison_pfx /opt/models/wmh_bison_1.3.0/t1_15T \
        --wmh_bison_atlas_pfx /opt/models/wmh_bison_1.3.0/15T_2009c_ \
        --wmh_bison_method HGB1 \
        --synthstrip_onnx /opt/models/synthstrip/synthstrip.1.onnx \
        $CLEANUP
fi

#    --quiet \
#    --large_atrophy \

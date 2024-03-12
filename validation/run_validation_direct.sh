#!/bin/bash
set -e -x


FIELD=1.5
PRL=4
THREADS=4
out=output
#CLEANUP=YES
## number of threads should be less then PRL (parallel processes)

in_json=subject43.json
out_pfx=test_json

# apptainer run \
#     -B $(pwd):/data --pwd /data \
#     --compat -e --net --network none \
#     --env FIELD=$FIELD,PRL=$PRL,THREADS=$THREADS,RAY_memory_monitor_refresh_ms=0,CLEANUP=$CLEANUP \
#     nist_pipeline_0.1.00.sif subject43.csv ${out}

loc_pfx=../container

python ../ipl_longitudinal_pipeline.py \
    --denoise \
    --json $in_json \
    -o $out_pfx \
    --model-dir=$loc_pfx/models/icbm152_model_09c \
    --model-name=mni_icbm152_t1_tal_nlin_sym_09c  \
    --ray_start $PRL \
    --threads $THREADS \
    --nl_ants \
    --nl_step 1.0 \
    --nl_cost_fun CC \
    --bison_pfx $loc_pfx/models/ipl_bison_1.3.0 \
    --bison_method  HGB1 \
    --wmh_bison_pfx $loc_pfx/models/wmh_bison_1.3.0/t1_15T \
    --wmh_bison_atlas_pfx $loc_pfx/models/wmh_bison_1.3.0/15T_2009c_ \
    --wmh_bison_method HGB1 \
    --synthstrip_onnx $loc_pfx/models/synthstrip/synthstrip.1.onnx \


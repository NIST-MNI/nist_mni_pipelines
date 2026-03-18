#!/bin/bash
set -e -x


FIELD=1.5
PRL=4
THREADS=4
out=output
CLEANUP=NO
## number of threads should be less then PRL (parallel processes)

ver=0.2.05

in_csv=subject43.csv
out_pfx=output_direct_${ver}

# apptainer run \
#     -B $(pwd):/data --pwd /data \
#     --compat -e --net --network none \
#     --env FIELD=$FIELD,PRL=$PRL,THREADS=$THREADS,RAY_memory_monitor_refresh_ms=0,CLEANUP=$CLEANUP \
#     nist_pipeline_0.1.00.sif subject43.csv ${out}

loc_pfx=../container

python ../ipl_longitudinal_pipeline.py \
    --csv $in_csv \
    -o $out_pfx \
    --model-dir=$loc_pfx/models/icbm152_model_09c \
    --model-name=mni_icbm152_t1_tal_nlin_sym_09c  \
    --ray_start $PRL \
    --threads $THREADS \
    --nl_ants \
    --nl_cost_fun CC \
    --bison_pfx $loc_pfx/models/ipl_bison_1.3.0 \
    --bison_method  HGB1 \
    --wmh_bison_pfx $loc_pfx/models/wmh_bison_1.3.0 \
    --wmh_bison_method HGB1 \
    --redskull_onnx $loc_pfx/models/redskull/redskull_fp.onnx \
    --redskull_native

#     --synthstrip_onnx $loc_pfx/models/synthstrip/synthstrip.1.onnx \


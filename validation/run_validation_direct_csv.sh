#!/bin/bash

out=$1

if [[ -z $out ]];then
echo "Usage: $0 <output>" >&2 
exit 1
fi

scikit_version=$(python -c "import sklearn;print(sklearn.__version__)")

FIELD=1.5
PRL=4
THREADS=4
CLEANUP=NO
## number of threads should be less then PRL (parallel processes)

ver=0.2.03

in_csv=subject43.csv
out_pfx=$out # test_csv_${ver}

# apptainer run \
#     -B $(pwd):/data --pwd /data \
#     --compat -e --net --network none \
#     --env FIELD=$FIELD,PRL=$PRL,THREADS=$THREADS,RAY_memory_monitor_refresh_ms=0,CLEANUP=$CLEANUP \
#     nist_pipeline_0.1.00.sif subject43.csv ${out}

loc_pfx=../container

python ../ipl_longitudinal_pipeline.py \
    --denoise \
    --csv $in_csv \
    -o $out_pfx \
    --model-dir=$loc_pfx/models/icbm152_model_09c \
    --model-name=mni_icbm152_t1_tal_nlin_sym_09c  \
    --ray_start $PRL \
    --threads $THREADS \
    --nl_ants \
    --nl_step 1.0 \
    --nl_cost_fun CC \
    --bison_pfx $loc_pfx/models/ipl_bison_${scikit_version} \
    --bison_method  HGB1 \
    --wmh_bison_pfx $loc_pfx/models/wmh_bison_${scikit_version} \
    --wmh_bison_method HGB1 \
    --redskull_onnx $loc_pfx/models/redskull/redskull_fp.onnx \
    --redskull_native

#!/bin/bash
set -e -x

#output_dir=$1


# setup variables

pipeline_dir=$(dirname $0)/../..
data_dir=$(dirname $0)

export PYTHONPATH=$pipeline_dir
export OMP_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

### Run pipeline for NC scan rescan data ###
python3 $pipeline_dir/ipl_preprocess_pipeline.py \
  subject43 1 ${data_dir}/subject43/subject43_1_t1w.mnc \
 --output elx_v1 \
 --options pipeline_options_minimal_elastix.json

python3 $pipeline_dir/ipl_preprocess_pipeline.py \
  subject43 1 ${data_dir}/subject43/subject43_1_t1w.mnc \
 --output elx_v2 \
 --options pipeline_options_v2_elastix.json
 
#!/bin/bash
set -e

output_dir=$1

if [ -z $output_dir ];then
 echo "Usage $0 <output dir> "
 echo "Usefull environment variables:"
 echo "MNI_DATAPATH - location of MNI datasets ( /opt/minc/share )"
 echo "               should include icbm152_model_09c and beast-library-1.1"
 echo "PARALLEL     - number of paralell processes to use"
 exit 1
fi

# setup variables
MNI_DATAPATH=${MNI_DATAPATH:-/opt/minc/share}
PARALLEL=${PARALLEL:-1}


icbm_model_dir=$MNI_DATAPATH/icbm152_model_09c
beast_model_dir=$MNI_DATAPATH/beast-library-1.1


if [ ! -d $icbm_model_dir ];then 
    echo "Missing $icbm_model_dir"
    exit 1
fi

if [ ! -d $beast_model_dir ];then 
    echo "Missing $beast_model_dir"
    exit 1
fi

pipeline_dir=$(dirname $0)/..
data_dir=$(dirname $0)

#export PYTHONPATH=$(realpath $pipeline_dir/python)
export PYTHONPATH=$pipeline_dir/python
#export PATH=$pipeline_dir/bin:$PATH
export OMP_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

### Run pipeline for NC scan rescan data ###
python -m scoop -vvv -n $PARALLEL $pipeline_dir/python/iplLongitudinalPipeline.py \
 -L \
 -l $data_dir/nc_scan_rescan_validation_list.csv \
 -o $output_dir \
 --model-dir=$icbm_model_dir \
 --model-name=mni_icbm152_t1_tal_nlin_sym_09c  \
 --beast-dir=$beast_model_dir

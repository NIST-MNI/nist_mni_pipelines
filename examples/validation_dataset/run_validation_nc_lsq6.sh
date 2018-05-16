#!/bin/bash
set -e -x

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

pipeline_dir=$(dirname $0)/../..
data_dir=$(dirname $0)

export PYTHONPATH=$pipeline_dir
export OMP_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

### Run pipeline for NC scan rescan data ###
python3 -m scoop -vvv -n $PARALLEL $pipeline_dir/iplLongitudinalPipeline.py \
 -L \
 -l $data_dir/subject43_2.csv \
 -o $output_dir \
 --model-dir=$icbm_model_dir \
 --model-name=mni_icbm152_t1_tal_nlin_sym_09c  \
 --beast-dir=$beast_model_dir \
 --VBM --rigid 

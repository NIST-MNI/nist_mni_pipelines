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
progname=$(basename $0)

MAJOR_VERSION=0
MINOR_VERSION=2
MICRO_VERSION=0
# this version relies much more on the priors
ver=${MAJOR_VERSION}.${MINOR_VERSION}.${MICRO_VERSION}

function Usage {
  cat <<EOF

  ${progname} version ${ver}
    Will run longitduinal processing pipeline

  ${progname} parameters:
  -help                          :  show this usage

  --- Input data, choose one ---
    -csv <input.csv>               :  input csv file with columns: subject,visit,t1w[t2w,pdw,flair], with header
    -l <input.csv>                 :  input list file with columns: subject,visit,t1w[t2w,pdw,flair], no header
    -json <input.json>             :  input list file  in json format with columns: subject,visit,t1w[t2w,pdw,flair]

  --- output data ---
    -out <output dir>              :  output directory
 
  --- Optional parameters ---
    -u/-help                        :  show this usage
    -prl <int>                      :  number of parallel processes (default 4)
    -threads <int>                  :  number of maximu threads per process (default 4), should be less then number of processes
    -field <string>                 :  1.5 or 3 (default 3) scanner field strength
    -cleanup                        :  remove intermediate files to save disk space
    -fast                           :  run fast version of the pipeline, mostly for testing (rough nonlinear registration, no denoising)
EOF
}
if [[ $# -eq 0 ]]; then Usage; exit 1; fi

while  [[ $# -gt 0 ]]; do
  if   [[ $1 = -help ]]; then Usage; exit 1
  elif [[ $1 = -u ]]; then Usage; exit 1
  elif [[ $1 = -csv ]]; then in_par="--csv $2"; shift 2
  elif [[ $1 = -json ]]; then in_par="--json $2"; shift 2
  elif [[ $1 = -l ]]; then in_par="-l $2"; shift 2
  elif [[ $1 = -out    ]]; then out_pfx=$2;shift 2;
  elif [[ $1 = -prl    ]]; then PRL=$2;shift 2;
  elif [[ $1 = -threads ]]; then THREADS=$2;shift 2;
  elif [[ $1 = -field    ]]; then FIELD=$2;shift 2;
  elif [[ $1 = -cleanup ]]; then CLEANUP=YES; shift
  elif [[ $1 = -fast ]]; then FAST=YES; shift
  else
    args=( ${args[@]} $1 )
    shift
  fi
done

if [[ -z "$in_par" ]]; then
  echo "Error: missing input file"
  Usage
  exit 1
fi

if [[ -z $out_pfx ]]; then
  echo "Error: missing output directory"
  Usage
  exit 1
fi

PRL=${PRL:-4}
THREADS=${THREADS:-4}
FIELD=${FIELD:-3}
CLEANUP=${CLEANUP:-NO}
FAST=${FAST:-NO}

########

if [[ $CLEANUP == YES ]];then
    CLEANUP=--cleanup
else
    CLEANUP=
fi

if [[ $FAST == YES ]];then
    fast_par="--nl_step 4.0 "
else
    fast_par="--denoise  --nl_step 1.0 "
fi

if [[ $FIELD == 3 ]];then
    echo "Running 3T pipeline"
#        --wmh_bison_atlas_pfx /opt/models/wmh_bison_1.3.0/3T_2009c_ \
    python /opt/pipeline/ipl_longitudinal_pipeline.py \
        -3 \
        $fast_par \
        $in_par \
        -o $out_pfx \
        --model-dir=/opt/models/icbm152_model_09c \
        --model-name=mni_icbm152_t1_tal_nlin_sym_09c  \
        --ray_start $PRL \
        --threads $THREADS \
        --nl_ants \
        --nl_cost_fun CC \
        --bison_pfx /opt/models/ipl_bison_1.3.0 \
        --bison_method  HGB1 \
        --wmh_bison_pfx /opt/models/wmh_bison_1.3.0 \
        --wmh_bison_method HGB1 \
        --synthstrip_onnx /opt/models/synthstrip/synthstrip.1.onnx \
        $CLEANUP
else
    echo "Running 1.5T pipeline"

#        --wmh_bison_atlas_pfx /opt/models/wmh_bison_1.3.0/15T_2009c_ \

    python /opt/pipeline/ipl_longitudinal_pipeline.py \
        $fast_par \
        $in_par \
        -o $out_pfx \
        --model-dir=/opt/models/icbm152_model_09c \
        --model-name=mni_icbm152_t1_tal_nlin_sym_09c  \
        --ray_start $PRL \
        --threads $THREADS \
        --nl_ants \
        --nl_cost_fun CC \
        --bison_pfx /opt/models/ipl_bison_1.3.0 \
        --bison_method  HGB1 \
        --wmh_bison_pfx /opt/models/wmh_bison_1.3.0 \
        --wmh_bison_method HGB1 \
        --synthstrip_onnx /opt/models/synthstrip/synthstrip.1.onnx \
        $CLEANUP
fi

#    --quiet \
#    --large_atrophy \

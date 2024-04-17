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
    --csv <input.csv>               :  input csv file with columns: subject,visit,t1w[t2w,pdw,flair], with header
    --list <input.csv>              :  input list file with columns: subject,visit,t1w[t2w,pdw,flair], no header
    --json <input.json>             :  input list file  in json format with columns: subject,visit,t1w[t2w,pdw,flair]

    --field <string>                 :  1.5 or 3 (default 3) scanner field strength
    --large_atrophy                  :  assume large atrophy

  --- output data ---
    --out <output dir>              :  output directory
    --vbm                           :  create VBM output (more space + time)
    --dbm                           :  create longitudinal DBM output (more space + time)
 
  --- Optional parameters ---
    -u/--help                        :  show this usage
    --prl <int>                      :  number of parallel processes (default 4)
    --threads <int>                  :  number of maximu threads per process (default 4), should be less then number of processes
    --cleanup                        :  remove intermediate files to save disk space
    --fast                           :  run fast version of the pipeline, mostly for testing (rough nonlinear registration, no denoising)
EOF
}
if [[ $# -eq 0 ]]; then Usage; exit 1; fi

while  [[ $# -gt 0 ]]; do
  if   [[ $1 = --help ]]; then Usage; exit 1
  elif [[ $1 = -u ]]; then Usage; exit 1
  elif [[ $1 = --csv ]]; then in_par="--csv $2"; shift 2
  elif [[ $1 = --json ]]; then in_par="--json $2"; shift 2
  elif [[ $1 = --list ]]; then in_par="-l $2"; shift 2
  elif [[ $1 = --out    ]]; then out_pfx=$2;shift 2;
  elif [[ $1 = --prl    ]]; then PRL=$2;shift 2;
  elif [[ $1 = --threads ]]; then THREADS=$2;shift 2;
  elif [[ $1 = --field    ]]; then FIELD=$2;shift 2;
  elif [[ $1 = --cleanup ]]; then CLEANUP=YES; shift
  elif [[ $1 = --vbm ]]; then VBM=YES; shift
  elif [[ $1 = --dbm ]]; then DBM=YES; shift
  elif [[ $1 = --fast ]]; then FAST=YES; shift
  elif [[ $1 = --large_atrophy ]]; then LARGE_ATROPHY=YES; shift
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
VBM=${VBM:-NO}
DBM=${DBM:-NO}
FAST=${FAST:-NO}
LARGE_ATROPHY=${LARGE_ATROPHY:-NO}
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

if [[ $VBM == YES ]];then
    vbm_par="--VBM "
else
    vbm_par=""
fi

if [[ $DBM == YES ]];then
    dbm_par="--DBM "
else
    dbm_par=""
fi

if [[ $LARGE_ATROPHY == YES ]];then
    large_atrophy_par="--large_atrophy "
else
    large_atrophy_par=""
fi

if [[ $FIELD == 3 ]];then
  field_par="--3T "
else
  field_par=""
fi

python /opt/pipeline/ipl_longitudinal_pipeline.py \
    $fast_par $vbm_par $dbm_par $field_par $large_atrophy_par \
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


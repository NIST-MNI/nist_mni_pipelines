#!/bin/bash
set -e -x


FIELD=1.5
PRL=4
THREADS=4
out=output
CLEANUP=YES
## number of threads should be less or equal to PRL (parallel processes)


cat - <<END
######################################################################
# Running validation dataset using apptainer, results will be in $out directory
#
# WARNING: apptainer should have at least 1Gb of available disk space for /tmp
#          check sessiondir max size  parameter in /etc/apptainer/apptainer.conf or ~/.apptainer.conf
#
#
# Using parameters for ${FIELD}T scanner, ${PRL} parallel processes, maximum ${THREADS} threads
#
#######################################################################
END

apptainer run \
    -B $(pwd):/data --pwd /data \
    --compat -e --net --network none \
    --env FIELD=$FIELD,PRL=$PRL,THREADS=$THREADS,RAY_memory_monitor_refresh_ms=0,CLEANUP=$CLEANUP \
    nist_pipeline.sif subject43.csv ${out}



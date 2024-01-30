#!/bin/bash
set -e -x


FIELD=1.5
PRL=4
THREADS=4
out=output_docker
CLEANUP=YES
## number of threads should be less then PRL (parallel processes)


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


docker run --rm --shm-size=10gb -v $(pwd):/data -w /data --user $(id -u):$(id -g) \
	--env PRL,THREADS=$THREADS,RAY_memory_monitor_refresh_ms=0,CLEANUP=$CLEANUP \
         nist_mni_pipeline:0.1.00 subject43.csv output_docker



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
#
# Using parameters for ${FIELD}T scanner, ${PRL} parallel processes, maximum ${THREADS} threads
#
#######################################################################
END

mkdir -p $out

docker run --rm --shm-size=10gb -v $(pwd):/data -w /data --user $(id -u):$(id -g) \
	--env PRL,THREADS=$THREADS,RAY_memory_monitor_refresh_ms=0,CLEANUP=$CLEANUP \
         nist_mni_pipeline:0.2.00 --csv subject43.csv --out output_docker

#!/bin/bash
set -e -x

PRL=4
THREADS=4
loc_pfx=../container

# Convert existing .mnc test data to .nii.gz (skip if already done)
mkdir -p subject43_nifti
for f in subject43/*.mnc; do
    base=$(basename "$f" .mnc)
    out="subject43_nifti/${base}.nii.gz"
    if [ ! -f "$out" ]; then
        mnc2nii -nii "$f" "subject43_nifti/${base}.nii"
        gzip "subject43_nifti/${base}.nii"
    fi
done

python ../ipl_longitudinal_pipeline.py \
    --cleanup \
    --csv subject43_nifti.csv \
    -o test_fast_nifti \
    --model-dir $loc_pfx/models/icbm152_model_09c \
    --model-name mni_icbm152_t1_tal_nlin_sym_09c \
    --ray_start $PRL \
    --threads $THREADS \
    --nl_ants \
    --nl_step 4.0 \
    --nl_cost_fun CC \
    --redskull_onnx $loc_pfx/models/redskull/redskull_fp.onnx \
    --redskull_native \
    --output-nifti

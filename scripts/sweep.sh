#!/bin/bash
# Select the rectification weight beta and the ridge parameter on held-out training images,
# then report the selected configuration (and the baselines with their own selected ridge
# parameter) on the test set.
# Usage: bash scripts/sweep.sh [dinov2|mocov3]
set -e
BACKBONE=${1:-dinov2}
mkdir -p sweeps
for DATASET in cub cifar100 imagenet-r t-imagenet; do
  NAME=sweeps/gacl_gsr_${BACKBONE}_${DATASET}
  python tools/sweep_gsr.py --dataset_name ${DATASET} --backbone ${BACKBONE} --model_name gacl_gsr \
    --seeds 0 1 2 --group_by reg --extra_baselines air \
    --reg 1e2 1e3 1e4 1e5 --beta 1 4 16 64 256 1024 \
    --budget exp --budget_kappa 10 --max_target 200 500 \
    --output ${NAME}.json | tee ${NAME}.log
done

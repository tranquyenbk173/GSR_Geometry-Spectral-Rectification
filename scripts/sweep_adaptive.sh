#!/bin/bash
# Class-agnostic vs count-adaptive rectification budget, each tuned on held-out training
# images and compared on the test set.
# Usage: bash scripts/sweep_adaptive.sh [gacl_gsr|anacp_gsr] [dinov2|mocov3] [long_tail|long_tail_exp]
set -e
MODEL=${1:-gacl_gsr}
BACKBONE=${2:-dinov2}
SCENARIO=${3:-long_tail}
SHARED_COV=True
if [ "${MODEL}" == "anacp_gsr" ] && [ "${BACKBONE}" == "dinov2" ]; then SHARED_COV=False; fi

mkdir -p sweeps
for DATASET in cub cifar100 imagenet-r t-imagenet; do
  NAME=sweeps/budget_${MODEL}_${BACKBONE}_${SCENARIO}_${DATASET}
  python tools/sweep_gsr.py --dataset_name ${DATASET} --backbone ${BACKBONE} --model_name ${MODEL} \
    --shared_cov ${SHARED_COV} --scenario ${SCENARIO} --seeds 0 1 2 --group_by budget \
    --budget proportional uniform balance exp --budget_kappa 1 3 10 \
    --beta 0.5 1 2 4 8 --max_target 200 500 \
    --output ${NAME}.json | tee ${NAME}.log
done

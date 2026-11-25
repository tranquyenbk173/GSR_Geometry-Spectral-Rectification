#!/bin/bash
# AnaCP + GSR with the validation-selected hyper-parameters (models/anacp_gsr.py),
# and the same configuration without the mixup term (beta = 0) as a control.
# Usage: bash scripts/anacp_selected.sh [dinov2|mocov3]
set -e
BACKBONE=${1:-dinov2}
SHARED_COV=$([ "${BACKBONE}" == "dinov2" ] && echo False || echo True)
for DATASET in cifar100 imagenet-r t-imagenet cub; do
  for SEED in 0 1 2; do
    RUN="python run.py --dataset_name ${DATASET} --backbone ${BACKBONE} --seed ${SEED} --model_name anacp_gsr --D 5000 --reg 1e2 --num_heads 3 --shared_cov ${SHARED_COV} --log_dir ./logs_anacp_selected"
    ${RUN}
    ${RUN} --gsr_beta 0
  done
done
python tools/summarize.py --log_dir ./logs_anacp_selected

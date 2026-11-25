#!/bin/bash
# Validation-based selection of the AnaCP + GSR hyper-parameters (reg is kept at 1e2).
# The configuration is chosen on images held out from the training set and then
# evaluated once on the test set; see tools/sweep_gsr.py.
# Usage: bash scripts/sweep_anacp.sh <backbone> <dataset> [gpu]
# Afterwards: python tools/collect_selected.py --sweeps "sweeps/anacp_gsr_*.json" prints the
# entries of SELECTED_HPARAMS in models/anacp_gsr.py.
set -e
BACKBONE=$1
DATASET=$2
GPU=${3:-0}
SHARED_COV=$([ "${BACKBONE}" == "dinov2" ] && echo False || echo True)
mkdir -p sweeps
CUDA_VISIBLE_DEVICES=${GPU} python tools/sweep_gsr.py --dataset_name ${DATASET} --backbone ${BACKBONE} \
  --model_name anacp_gsr --shared_cov ${SHARED_COV} --group_by geometry --reg 1e2 \
  --beta 0.1 0.25 0.5 1 --anacp_logit_power 0 0.25 0.5 --anacp_tail_replay 1 4 \
  --output sweeps/anacp_gsr_${BACKBONE}_${DATASET}.json 2>&1 | tee sweeps/anacp_gsr_${BACKBONE}_${DATASET}.txt

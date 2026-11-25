#!/bin/bash
# Main long-tailed CIL comparison (Tables 1-2): 30% head classes, 5-shot tail classes.
# Usage: bash scripts/main_longtail.sh [dinov2|mocov3]
set -e
BACKBONE=${1:-dinov2}
DATASETS=("cifar100" "imagenet-r" "t-imagenet" "cub")
SEEDS=(0 1 2)
# AnaCP + GSR uses the hyper-parameters selected on a validation split (models/anacp_gsr.py,
# Appendix C.3); add --anacp_hparams default to run it with the un-tuned configuration.
COMMON="--backbone ${BACKBONE} --scenario long_tail --head_ratio 0.3 --tail_shot 5"

for DATASET in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN="python run.py --dataset_name ${DATASET} --seed ${SEED} ${COMMON}"

    # RLS-based methods
    ${RUN} --model_name gacl --D 5000 --reg 1e2
    ${RUN} --model_name gacl_gsr --D 5000 --reg 1e2
    ${RUN} --model_name ranpac --D 5000 --reg 1e2
    ${RUN} --model_name ranpac_gsr --D 5000 --reg 1e2
    ${RUN} --model_name air --D 5000 --reg 1e2
    ${RUN} --model_name anacp --D 5000 --reg 1e2 --num_heads 3 --shared_cov True
    if [ "${BACKBONE}" == "dinov2" ]; then
      ${RUN} --model_name anacp_gsr --D 5000 --reg 1e2 --num_heads 3 --shared_cov False
    else
      ${RUN} --model_name anacp_gsr --D 5000 --reg 1e2 --num_heads 3 --shared_cov True
    fi

    # Covariance- and prototype-based methods
    ${RUN} --model_name ncm
    ${RUN} --model_name fecam
    ${RUN} --model_name slda --reg 1e-4
    if [ "${BACKBONE}" == "dinov2" ]; then
      ${RUN} --model_name klda --D 5000 --gamma 1e-4 --reg 1e-3
    else
      ${RUN} --model_name klda --D 5000 --gamma 1e-2 --reg 1e-4
    fi
  done
done

#!/bin/bash
# Feature-augmentation alternatives (Table 6): Gaussian sampling with a shrunk or
# transferred covariance and inter-class mixup, against GSR, on GACL and AnaCP.
# All use the same per-class budget M_c and the default hyper-parameters, and differ only
# in the sampler.
# Usage: bash scripts/table6.sh [dinov2|mocov3]
set -e
BACKBONE=${1:-dinov2}
SHARED_COV=$([ "${BACKBONE}" == "dinov2" ] && echo False || echo True)
for DATASET in cub imagenet-r t-imagenet; do
  for SEED in 0 1 2; do
    RUN="python run.py --dataset_name ${DATASET} --backbone ${BACKBONE} --seed ${SEED} --D 5000 --reg 1e2 --scenario long_tail --head_ratio 0.3 --tail_shot 5 --log_dir ./logs_table6"
    for SAMPLER in gauss_shrinkage gauss_transfer inter_mixup mixup; do
      ${RUN} --model_name gacl_gsr --gsr_sampler ${SAMPLER}
      ${RUN} --model_name anacp_gsr --gsr_sampler ${SAMPLER} --num_heads 3 --shared_cov ${SHARED_COV} --anacp_hparams default
    done
  done
done
python tools/summarize.py --log_dir ./logs_table6

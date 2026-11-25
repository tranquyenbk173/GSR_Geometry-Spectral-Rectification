#!/bin/bash
# Robustness to the imbalance ratio on Split-CIFAR-100 (Table 4).
# rho = N_max / N_min = 500 / tail_shot, i.e. rho in {10, 20, 50, 100, 150}.
set -e
for TAIL_SHOT in 50 25 10 5 3; do
  for SEED in 0 1 2; do
    RUN="python run.py --dataset_name cifar100 --backbone dinov2 --seed ${SEED} --D 5000 --reg 1e2 --scenario long_tail --head_ratio 0.3 --tail_shot ${TAIL_SHOT}"
    ${RUN} --model_name gacl
    ${RUN} --model_name gacl_gsr
  done
done

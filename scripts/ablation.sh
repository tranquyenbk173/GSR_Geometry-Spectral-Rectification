#!/bin/bash
# Ablations of GSR on top of GACL (D = 5000, reg = 1e2, beta = 1), 3 seeds.
# Usage: bash scripts/ablation.sh [dataset] [dinov2|mocov3]
set -e
DATASET=${1:-cub}
BACKBONE=${2:-dinov2}
for SEED in 0 1 2; do
  RUN="python run.py --dataset_name ${DATASET} --backbone ${BACKBONE} --seed ${SEED} --D 5000 --reg 1e2 --scenario long_tail --head_ratio 0.3 --tail_shot 5"
  ${RUN} --model_name gacl                                   # standard ridge
  # geometry of the mixed features
  ${RUN} --model_name gacl_gsr --gsr_geometry linear         # linear mixup
  ${RUN} --model_name gacl_gsr                               # spherical mixup (GSR)
  # how the mixed samples are allocated across classes (same total number)
  ${RUN} --model_name gacl_gsr --gsr_budget proportional     # proportional to N_c (standard mixup)
  ${RUN} --model_name gacl_gsr --gsr_budget uniform          # same number for every class
  ${RUN} --model_name gacl_gsr --gsr_budget exp --gsr_budget_kappa 10
  # replication instead of mixing, and a count-dependent Beta parameter
  ${RUN} --model_name gacl_gsr --gsr_alpha 0.01              # gamma in {0, 1}: tail replication
  ${RUN} --model_name gacl_gsr --gsr_alpha_mode adaptive     # alpha_c = 0.6 + 0.4 exp(-0.005 N_c)
done
python tools/summarize.py --log_dir ./logs

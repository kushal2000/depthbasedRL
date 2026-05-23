#!/usr/bin/env bash
set -euo pipefail

# Submit the seed-2 ObjectDiversity replicate requested on 2026-05-23.
#
# Layout:
#   - 4 L-peg jobs on juno2 A5000s.
#   - 4 beam_3x_part_0 jobs on move3 A5000s.
#   - 4 FurnitureBench jobs on move5 RTX PRO 6000s.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/panel_a_tyler_finetune.sub"

REPO="/move/u/tylerlum/github_repos/depthbasedRL"
ENV_BASE="/move/u/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311"
ENV_RTX="/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-cu128-py311"
CHECKPOINTS=(1_obj 10_obj 100_obj 1000_obj)

submit_one() {
  local task_tag="$1"
  local experiment_tag="$2"
  local checkpoint_tag="$3"
  local account="$4"
  local partition="$5"
  local node="$6"
  local mem_mb="$7"
  local env_dir="$8"

  echo "submit task=${task_tag} checkpoint=${checkpoint_tag} node=${node} partition=${partition} seed=2"
  sbatch \
    --account="$account" \
    --partition="$partition" \
    --nodelist="$node" \
    --mem="$mem_mb" \
    --time=1-00:00:00 \
    --export=ALL,TASK_TAG="$task_tag",EXPERIMENT_TAG="$experiment_tag",CHECKPOINT_FAMILY=ObjectDiversity,CHECKPOINT_TAG="$checkpoint_tag",SEED=2,REPO_ROOT="$REPO",ISAACSIM_ENV_DIR="$env_dir" \
    "$JOB_SCRIPT"
}

for checkpoint in "${CHECKPOINTS[@]}"; do
  submit_one lpeg_tol0p5mm lpeg_tol0p5mm_finetune_rgf0_dr_seed2 "$checkpoint" juno juno juno2 100000 "$ENV_BASE"
done

for checkpoint in "${CHECKPOINTS[@]}"; do
  submit_one beam_3x_part_0 beam_3x_part_0_finetune_rgf0_dr_seed2 "$checkpoint" move move move3 90000 "$ENV_BASE"
done

for checkpoint in "${CHECKPOINTS[@]}"; do
  submit_one furniture_bench_one_leg furniture_bench_one_leg_finetune_rgf10_dr_seed2 "$checkpoint" move move move5 90000 "$ENV_RTX"
done

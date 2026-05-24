#!/usr/bin/env bash
set -euo pipefail

# Submit the TrainingObjective checkpoint sweep requested on 2026-05-24.
#
# Layout:
#   - 4 FurnitureBench jobs on move5 RTX PRO 6000s.
#   - 4 beam_3x_part_2 jobs on move5 RTX PRO 6000s.
#   - 4 beam_3x_part_0 jobs on move4 L40S GPUs.
#   - 4 L-peg jobs on juno2 A5000 GPUs, split 2 juno / 2 juno-lo.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/panel_a_tyler_finetune.sub"

REPO="/move/u/tylerlum/github_repos/depthbasedRL"
ENV_BASE="/move/u/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311"
ENV_RTX="/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-cu128-py311"
CHECKPOINTS=(Play2Win RotationOnly SingleGoal TranslationOnly)

submit_one() {
  local task_tag="$1"
  local checkpoint_tag="$2"
  local account="$3"
  local partition="$4"
  local node="$5"
  local mem_mb="$6"
  local env_dir="$7"

  echo "submit task=${task_tag} checkpoint=${checkpoint_tag} node=${node} partition=${partition} mem=${mem_mb}"
  sbatch \
    --account="$account" \
    --partition="$partition" \
    --nodelist="$node" \
    --mem="$mem_mb" \
    --time=1-00:00:00 \
    --export=ALL,TASK_TAG="$task_tag",CHECKPOINT_FAMILY=TrainingObjective,CHECKPOINT_TAG="$checkpoint_tag",REPO_ROOT="$REPO",ISAACSIM_ENV_DIR="$env_dir" \
    "$JOB_SCRIPT"
}

for checkpoint in "${CHECKPOINTS[@]}"; do
  submit_one furniture_bench_one_leg "$checkpoint" move move move5 90000 "$ENV_RTX"
done

for checkpoint in "${CHECKPOINTS[@]}"; do
  submit_one beam_3x_part_2 "$checkpoint" move move move5 90000 "$ENV_RTX"
done

for checkpoint in "${CHECKPOINTS[@]}"; do
  submit_one beam_3x_part_0 "$checkpoint" move move move4 80000 "$ENV_BASE"
done

idx=0
for checkpoint in "${CHECKPOINTS[@]}"; do
  partition=juno
  if (( idx >= 2 )); then
    partition=juno-lo
  fi
  submit_one lpeg_tol0p5mm "$checkpoint" juno "$partition" juno2 100000 "$ENV_BASE"
  idx=$((idx + 1))
done

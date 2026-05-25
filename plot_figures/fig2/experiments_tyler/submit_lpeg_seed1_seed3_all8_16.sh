#!/usr/bin/env bash
set -euo pipefail

# Submit the L-peg seed-1/seed-3 sweep requested on 2026-05-25.
#
# Layout:
#   - 8 TrainingObjective jobs on move5 RTX PRO 6000s.
#   - 4 ObjectDiversity seed-1 jobs on move4 L40S GPUs.
#   - 4 ObjectDiversity seed-3 jobs on juno2 A5000 GPUs, split 2 juno / 2 juno-lo.
#
# All jobs use full L-peg scale and include seed in EXPERIMENT_TAG so the seed is
# visible in the W&B run name.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/panel_a_tyler_finetune.sub"

REPO="/move/u/tylerlum/github_repos/depthbasedRL"
ENV_BASE="/move/u/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311"
ENV_RTX="/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-cu128-py311"

OBJECT_DIVERSITY_CHECKPOINTS=(1_obj 10_obj 100_obj 1000_obj)
TRAINING_OBJECTIVE_CHECKPOINTS=(Play2Win RotationOnly SingleGoal TranslationOnly)

submit_one() {
  local checkpoint_family="$1"
  local checkpoint_tag="$2"
  local seed="$3"
  local account="$4"
  local partition="$5"
  local node="$6"
  local mem_mb="$7"
  local env_dir="$8"

  local experiment_tag="lpeg_tol0p5mm_finetune_rgf0_dr_seed${seed}"

  echo "submit family=${checkpoint_family} checkpoint=${checkpoint_tag} seed=${seed} node=${node} partition=${partition} mem=${mem_mb}"
  sbatch \
    --account="$account" \
    --partition="$partition" \
    --nodelist="$node" \
    --mem="$mem_mb" \
    --time=1-00:00:00 \
    --export=ALL,TASK_TAG=lpeg_tol0p5mm,EXPERIMENT_TAG="$experiment_tag",CHECKPOINT_FAMILY="$checkpoint_family",CHECKPOINT_TAG="$checkpoint_tag",SEED="$seed",NUM_ENVS=12288,MINIBATCH_SIZE=98304,EXPL_COEF_BLOCK_SIZE=2048,REPO_ROOT="$REPO",ISAACSIM_ENV_DIR="$env_dir" \
    "$JOB_SCRIPT"
}

for seed in 1 3; do
  for checkpoint in "${TRAINING_OBJECTIVE_CHECKPOINTS[@]}"; do
    submit_one TrainingObjective "$checkpoint" "$seed" move move move5 90000 "$ENV_RTX"
  done
done

for checkpoint in "${OBJECT_DIVERSITY_CHECKPOINTS[@]}"; do
  submit_one ObjectDiversity "$checkpoint" 1 move move move4 80000 "$ENV_BASE"
done

idx=0
for checkpoint in "${OBJECT_DIVERSITY_CHECKPOINTS[@]}"; do
  partition=juno
  if (( idx >= 2 )); then
    partition=juno-lo
  fi
  submit_one ObjectDiversity "$checkpoint" 3 juno "$partition" juno2 100000 "$ENV_BASE"
  idx=$((idx + 1))
done

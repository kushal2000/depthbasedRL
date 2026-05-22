#!/usr/bin/env bash
set -euo pipefail

# Submit the full ObjectDiversity sweep after all sanity checks pass.
# Layout:
#   7 jobs on move5 RTX PRO 6000, leaving 1 GPU free.
#   3 jobs on move4 L40S, leaving several GPUs free.
#   6 jobs on juno2 A5000, with first 2 on juno and remaining 4 on juno-lo.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/panel_a_tyler_finetune.sub"

TASKS=(
  lpeg_tol0p5mm
  beam_3x_part_0
  beam_3x_part_2
  furniture_bench_one_leg
)
CHECKPOINTS=(
  1_obj
  1000_obj
  10_obj
  100_obj
)

submit_one() {
  local idx="$1"
  local task="$2"
  local ckpt="$3"
  local partition nodelist mem repo env

  repo="/move/u/tylerlum/github_repos/depthbasedRL"
  if (( idx < 7 )); then
    partition="move"
    nodelist="move5"
    mem="90000"
    env="/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-cu128-py311"
  elif (( idx < 10 )); then
    partition="move"
    nodelist="move4"
    mem="100000"
    env="/move/u/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311"
  else
    if (( idx < 12 )); then
      partition="juno"
    else
      partition="juno-lo"
    fi
    nodelist="juno2"
    mem="100000"
    env="/move/u/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311"
  fi

  echo "[$idx] task=${task} checkpoint=${ckpt} partition=${partition} node=${nodelist} mem=${mem}"
  TASK_TAG="$task" \
  CHECKPOINT_FAMILY="ObjectDiversity" \
  CHECKPOINT_TAG="$ckpt" \
  REPO_ROOT="$repo" \
  ISAACSIM_ENV_DIR="$env" \
  sbatch --partition="$partition" --nodelist="$nodelist" --mem="$mem" "$JOB_SCRIPT"
}

idx=0
for ckpt in "${CHECKPOINTS[@]}"; do
  for task in "${TASKS[@]}"; do
    submit_one "$idx" "$task" "$ckpt"
    idx=$((idx + 1))
  done
done

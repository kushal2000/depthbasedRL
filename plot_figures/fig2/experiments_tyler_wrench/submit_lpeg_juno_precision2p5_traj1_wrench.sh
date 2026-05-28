#!/usr/bin/env bash
set -euo pipefail

# Six full-scale L-peg wrench runs for juno2 A5000:
#   Precision/2p5cm seeds 0/1/2
#   Trajectory_Count/1 seeds 0/1/2
#
# Dry-run by default. Set DRY_RUN=0 to submit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/panel_a_tyler_wrench_lpeg_finetune.sub"

REPO="${REPO:-/move/u/tylerlum/github_repos/depthbasedRL}"
ENV_BASE="${ENV_BASE:-/move/u/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311}"
DRY_RUN="${DRY_RUN:-1}"

TASK_TAG="${TASK_TAG:-lpeg_tol0p5mm}"
WANDB_GROUP="${WANDB_GROUP:-panel_a_teachers_tyler_traj_precision_wrench_compare}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"

NUM_ENVS="${NUM_ENVS:-12288}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-98304}"
EXPL_COEF_BLOCK_SIZE="${EXPL_COEF_BLOCK_SIZE:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000000}"

FORCE_SCALE="${FORCE_SCALE:-20.0}"
TORQUE_SCALE="${TORQUE_SCALE:-2.0}"
FORCE_ONLY_WHEN_LIFTED="${FORCE_ONLY_WHEN_LIFTED:-False}"
TORQUE_ONLY_WHEN_LIFTED="${TORQUE_ONLY_WHEN_LIFTED:-False}"

FAMILIES=(
  Precision Precision Precision
  Trajectory_Count Trajectory_Count Trajectory_Count
)
TAGS=(
  2p5cm 2p5cm 2p5cm
  1 1 1
)
SEEDS=(
  0 1 2
  0 1 2
)
PARTITIONS=(
  juno juno
  juno-lo juno-lo juno-lo juno-lo
)

checkpoint_path() {
  printf '/juno/u/kedia/depthbasedRL/train_dir/%s/%s/model.pth' "$1" "$2"
}

submit_one() {
  local idx="$1"
  local family="${FAMILIES[$idx]}"
  local tag="${TAGS[$idx]}"
  local seed="${SEEDS[$idx]}"
  local partition="${PARTITIONS[$idx]}"
  local checkpoint
  checkpoint="$(checkpoint_path "$family" "$tag")"

  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi

  local experiment_tag="lpeg_tol0p5mm_finetune_rgf0_dr_wrench_seed${seed}"
  local job_name="lpeg-wrench-${family}-${tag}-s${seed}"
  job_name="${job_name//_/-}"
  job_name="${job_name//\//-}"

  local cmd=(
    sbatch
    --job-name="$job_name"
    --account=juno
    --partition="$partition"
    --nodelist=juno2
    --mem=100000
    --time="$TIME_LIMIT"
    --export=ALL,TASK_TAG="$TASK_TAG",EXPERIMENT_TAG="$experiment_tag",CHECKPOINT_FAMILY="$family",CHECKPOINT_TAG="$tag",CONDITION_TAG=wrench,SEED="$seed",NUM_ENVS="$NUM_ENVS",MINIBATCH_SIZE="$MINIBATCH_SIZE",EXPL_COEF_BLOCK_SIZE="$EXPL_COEF_BLOCK_SIZE",MAX_ITERATIONS="$MAX_ITERATIONS",REPO_ROOT="$REPO",ISAACSIM_ENV_DIR="$ENV_BASE",WANDB_GROUP="$WANDB_GROUP",FORCE_SCALE="$FORCE_SCALE",TORQUE_SCALE="$TORQUE_SCALE",FORCE_ONLY_WHEN_LIFTED="$FORCE_ONLY_WHEN_LIFTED",TORQUE_ONLY_WHEN_LIFTED="$TORQUE_ONLY_WHEN_LIFTED"
    "$JOB_SCRIPT"
  )

  printf '[%02d] %-16s %-8s seed=%s partition=%s node=juno2 env=%s\n' \
    "$idx" "$family" "$tag" "$seed" "$partition" "$ENV_BASE"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "Job script not found: $JOB_SCRIPT" >&2
  exit 1
fi

echo "DRY_RUN=$DRY_RUN time=$TIME_LIMIT group=$WANDB_GROUP"
echo "num_envs=$NUM_ENVS minibatch=$MINIBATCH_SIZE expl_block=$EXPL_COEF_BLOCK_SIZE"
echo "wrench force/torque=$FORCE_SCALE/$TORQUE_SCALE only_lifted=$FORCE_ONLY_WHEN_LIFTED/$TORQUE_ONLY_WHEN_LIFTED"
echo ""

for idx in "${!FAMILIES[@]}"; do
  submit_one "$idx"
done

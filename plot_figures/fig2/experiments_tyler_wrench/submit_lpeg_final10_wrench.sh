#!/usr/bin/env bash
set -euo pipefail

# Final L-peg wrench wave for the currently available trajectory-count and
# precision checkpoints, plus missing TrainingObjective wrench seeds.
#
# Default is DRY_RUN=1. Set DRY_RUN=0 to submit the nine cluster jobs.
# The tenth job is intended to run locally via
# run_lpeg_final10_local_translationonly_seed2.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/panel_a_tyler_wrench_lpeg_finetune.sub"

REPO="${REPO:-/move/u/tylerlum/github_repos/depthbasedRL}"
ENV_BASE="${ENV_BASE:-/move/u/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311}"
ENV_RTX="${ENV_RTX:-/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-cu128-py311}"

DRY_RUN="${DRY_RUN:-1}"
TASK_TAG="${TASK_TAG:-lpeg_tol0p5mm}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"

NUM_ENVS="${NUM_ENVS:-12288}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-98304}"
EXPL_COEF_BLOCK_SIZE="${EXPL_COEF_BLOCK_SIZE:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000000}"

FORCE_SCALE="${FORCE_SCALE:-20.0}"
TORQUE_SCALE="${TORQUE_SCALE:-2.0}"
FORCE_ONLY_WHEN_LIFTED="${FORCE_ONLY_WHEN_LIFTED:-False}"
TORQUE_ONLY_WHEN_LIFTED="${TORQUE_ONLY_WHEN_LIFTED:-False}"

TRAJ_PRECISION_GROUP="${TRAJ_PRECISION_GROUP:-panel_a_teachers_tyler_traj_precision_wrench_compare}"
TRAINING_OBJECTIVE_GROUP="${TRAINING_OBJECTIVE_GROUP:-panel_a_teachers_tyler_wrench_lpeg}"

FAMILIES=(
  Precision Precision Precision
  Trajectory_Count Trajectory_Count Trajectory_Count
  TrainingObjective TrainingObjective TrainingObjective
)
TAGS=(
  10cm 10cm 10cm
  10 10 10
  RotationOnly RotationOnly TranslationOnly
)
SEEDS=(
  0 1 2
  0 1 2
  1 2 1
)
NODES=(
  move5 move5 move5
  move5 move5 move4
  move4 move4 move4
)
ENVS=(
  "$ENV_RTX" "$ENV_RTX" "$ENV_RTX"
  "$ENV_RTX" "$ENV_RTX" "$ENV_BASE"
  "$ENV_BASE" "$ENV_BASE" "$ENV_BASE"
)
MEM_MB=(
  90000 90000 90000
  90000 90000 100000
  100000 100000 100000
)

checkpoint_path() {
  local family="$1"
  local tag="$2"
  printf '/juno/u/kedia/depthbasedRL/train_dir/%s/%s/model.pth' "$family" "$tag"
}

wandb_group_for_family() {
  case "$1" in
    Precision|Trajectory_Count) printf '%s' "$TRAJ_PRECISION_GROUP" ;;
    TrainingObjective) printf '%s' "$TRAINING_OBJECTIVE_GROUP" ;;
    *) echo "Unknown checkpoint family: $1" >&2; exit 2 ;;
  esac
}

validate_job() {
  local family="$1"
  local tag="$2"
  local path
  path="$(checkpoint_path "$family" "$tag")"
  if [[ ! -f "$path" ]]; then
    echo "Missing checkpoint: $path" >&2
    exit 1
  fi
}

submit_one() {
  local idx="$1"
  local family="${FAMILIES[$idx]}"
  local tag="${TAGS[$idx]}"
  local seed="${SEEDS[$idx]}"
  local node="${NODES[$idx]}"
  local env_dir="${ENVS[$idx]}"
  local mem="${MEM_MB[$idx]}"
  local group
  group="$(wandb_group_for_family "$family")"

  validate_job "$family" "$tag"

  local job_name="lpeg-wrench-${family}-${tag}-s${seed}"
  job_name="${job_name//_/-}"
  job_name="${job_name//\//-}"
  local experiment_tag="lpeg_tol0p5mm_finetune_rgf0_dr_wrench_seed${seed}"

  local cmd=(
    sbatch
    --job-name="$job_name"
    --account=move
    --partition=move
    --nodelist="$node"
    --mem="$mem"
    --time="$TIME_LIMIT"
    --export=ALL,TASK_TAG="$TASK_TAG",EXPERIMENT_TAG="$experiment_tag",CHECKPOINT_FAMILY="$family",CHECKPOINT_TAG="$tag",CONDITION_TAG=wrench,SEED="$seed",NUM_ENVS="$NUM_ENVS",MINIBATCH_SIZE="$MINIBATCH_SIZE",EXPL_COEF_BLOCK_SIZE="$EXPL_COEF_BLOCK_SIZE",MAX_ITERATIONS="$MAX_ITERATIONS",REPO_ROOT="$REPO",ISAACSIM_ENV_DIR="$env_dir",WANDB_GROUP="$group",FORCE_SCALE="$FORCE_SCALE",TORQUE_SCALE="$TORQUE_SCALE",FORCE_ONLY_WHEN_LIFTED="$FORCE_ONLY_WHEN_LIFTED",TORQUE_ONLY_WHEN_LIFTED="$TORQUE_ONLY_WHEN_LIFTED"
    "$JOB_SCRIPT"
  )

  printf '[%02d] %-18s %-12s seed=%s node=%s mem=%s group=%s env=%s\n' \
    "$idx" "$family" "$tag" "$seed" "$node" "$mem" "$group" "$env_dir"
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

echo "DRY_RUN=$DRY_RUN time=$TIME_LIMIT"
echo "num_envs=$NUM_ENVS minibatch=$MINIBATCH_SIZE expl_block=$EXPL_COEF_BLOCK_SIZE max_iterations=$MAX_ITERATIONS"
echo "wrench force/torque=$FORCE_SCALE/$TORQUE_SCALE only_lifted=$FORCE_ONLY_WHEN_LIFTED/$TORQUE_ONLY_WHEN_LIFTED"
echo "trajectory/precision group: $TRAJ_PRECISION_GROUP"
echo "training-objective group:   $TRAINING_OBJECTIVE_GROUP"
echo ""

for idx in "${!FAMILIES[@]}"; do
  submit_one "$idx"
done

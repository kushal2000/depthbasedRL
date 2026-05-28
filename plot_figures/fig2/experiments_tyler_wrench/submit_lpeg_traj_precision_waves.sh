#!/usr/bin/env bash
set -euo pipefail

# Prepare L-peg finetunes for trajectory-count / precision checkpoint ablations.
#
# This script is intentionally dry-run by default. Set DRY_RUN=0 only when ready
# to actually submit the jobs.
#
# Current available checkpoints:
#   Trajectory_Count/{1,100}/model.pth
#   Precision/5cm/model.pth
#
# Planned waves:
#   WAVE=first:
#     8 x move5 RTX PRO 6000
#       - Trajectory_Count/100, wrench, seeds 0/1/2
#       - Precision/5cm,       wrench, seeds 0/1/2
#       - Trajectory_Count/1,  wrench, seeds 0/1
#
#   WAVE=later:
#     4 x move L40S
#       - Trajectory_Count/100, no_wrench, seeds 0/1
#       - Precision/5cm,       no_wrench, seeds 0/1
#     6 x juno2 A5000
#       - Trajectory_Count/100, no_wrench, seed 2
#       - Precision/5cm,       no_wrench, seed 2
#       - Trajectory_Count/1,  wrench, seed 2
#       - Trajectory_Count/1,  no_wrench, seeds 0/1/2
#
# Future checkpoints, when available, can be added below:
#   Trajectory_Count/10, Precision/2.5cm, Precision/10cm.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/panel_a_tyler_wrench_lpeg_finetune.sub"

REPO="${REPO:-/move/u/tylerlum/github_repos/depthbasedRL}"
ENV_BASE="${ENV_BASE:-/move/u/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311}"
ENV_RTX="${ENV_RTX:-/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-cu128-py311}"

DRY_RUN="${DRY_RUN:-1}"
WAVE="${WAVE:-all}"  # first, later, or all

TASK_TAG="${TASK_TAG:-lpeg_tol0p5mm}"
WANDB_GROUP="${WANDB_GROUP:-panel_a_teachers_tyler_traj_precision_wrench_compare}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"

# Full L-peg settings, matching the prior RTX-scale L-peg finetunes.
NUM_ENVS="${NUM_ENVS:-12288}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-98304}"
EXPL_COEF_BLOCK_SIZE="${EXPL_COEF_BLOCK_SIZE:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000000}"

WRENCH_FORCE_SCALE="${WRENCH_FORCE_SCALE:-20.0}"
WRENCH_TORQUE_SCALE="${WRENCH_TORQUE_SCALE:-2.0}"
NO_WRENCH_FORCE_SCALE="${NO_WRENCH_FORCE_SCALE:-0.0}"
NO_WRENCH_TORQUE_SCALE="${NO_WRENCH_TORQUE_SCALE:-0.0}"
FORCE_ONLY_WHEN_LIFTED="${FORCE_ONLY_WHEN_LIFTED:-False}"
TORQUE_ONLY_WHEN_LIFTED="${TORQUE_ONLY_WHEN_LIFTED:-False}"

FIRST_FAMILY=()
FIRST_TAG=()
FIRST_SEED=()
FIRST_CONDITION=()

LATER_FAMILY=()
LATER_TAG=()
LATER_SEED=()
LATER_CONDITION=()

add_first() {
  FIRST_FAMILY+=("$1")
  FIRST_TAG+=("$2")
  FIRST_SEED+=("$3")
  FIRST_CONDITION+=("$4")
}

add_later() {
  LATER_FAMILY+=("$1")
  LATER_TAG+=("$2")
  LATER_SEED+=("$3")
  LATER_CONDITION+=("$4")
}

for seed in 0 1 2; do
  add_first Trajectory_Count 100 "$seed" wrench
done
for seed in 0 1 2; do
  add_first Precision 5cm "$seed" wrench
done
for seed in 0 1; do
  add_first Trajectory_Count 1 "$seed" wrench
done

for seed in 0 1; do
  add_later Trajectory_Count 100 "$seed" no_wrench
done
for seed in 0 1; do
  add_later Precision 5cm "$seed" no_wrench
done
add_later Trajectory_Count 100 2 no_wrench
add_later Precision 5cm 2 no_wrench
add_later Trajectory_Count 1 2 wrench
for seed in 0 1 2; do
  add_later Trajectory_Count 1 "$seed" no_wrench
done

checkpoint_path() {
  local family="$1"
  local tag="$2"
  printf '/juno/u/kedia/depthbasedRL/train_dir/%s/%s/model.pth' "$family" "$tag"
}

condition_force_scale() {
  case "$1" in
    wrench) printf '%s' "$WRENCH_FORCE_SCALE" ;;
    no_wrench) printf '%s' "$NO_WRENCH_FORCE_SCALE" ;;
    *) echo "Unknown condition: $1" >&2; exit 2 ;;
  esac
}

condition_torque_scale() {
  case "$1" in
    wrench) printf '%s' "$WRENCH_TORQUE_SCALE" ;;
    no_wrench) printf '%s' "$NO_WRENCH_TORQUE_SCALE" ;;
    *) echo "Unknown condition: $1" >&2; exit 2 ;;
  esac
}

validate_checkpoint() {
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
  local wave_name="$1"
  local job_idx="$2"
  local account="$3"
  local partition="$4"
  local node="$5"
  local mem_mb="$6"
  local env_dir="$7"
  local family="$8"
  local tag="$9"
  local seed="${10}"
  local condition="${11}"

  validate_checkpoint "$family" "$tag"

  local force_scale
  local torque_scale
  force_scale="$(condition_force_scale "$condition")"
  torque_scale="$(condition_torque_scale "$condition")"

  local experiment_tag="lpeg_tol0p5mm_finetune_rgf0_dr_${condition}_seed${seed}"
  local job_name="lpeg-${condition}-${family}-${tag}-s${seed}"
  job_name="${job_name//_/-}"
  job_name="${job_name//\//-}"

  local cmd=(
    sbatch
    --job-name="$job_name"
    --account="$account"
    --partition="$partition"
    --nodelist="$node"
    --mem="$mem_mb"
    --time="$TIME_LIMIT"
    --export=ALL,TASK_TAG="$TASK_TAG",EXPERIMENT_TAG="$experiment_tag",CHECKPOINT_FAMILY="$family",CHECKPOINT_TAG="$tag",CONDITION_TAG="$condition",SEED="$seed",NUM_ENVS="$NUM_ENVS",MINIBATCH_SIZE="$MINIBATCH_SIZE",EXPL_COEF_BLOCK_SIZE="$EXPL_COEF_BLOCK_SIZE",MAX_ITERATIONS="$MAX_ITERATIONS",REPO_ROOT="$REPO",ISAACSIM_ENV_DIR="$env_dir",WANDB_GROUP="$WANDB_GROUP",FORCE_SCALE="$force_scale",TORQUE_SCALE="$torque_scale",FORCE_ONLY_WHEN_LIFTED="$FORCE_ONLY_WHEN_LIFTED",TORQUE_ONLY_WHEN_LIFTED="$TORQUE_ONLY_WHEN_LIFTED"
    "$JOB_SCRIPT"
  )

  printf '[%s %02d] %-16s %-4s seed=%s %-9s node=%s partition=%s account=%s mem=%s env=%s force=%s torque=%s\n' \
    "$wave_name" "$job_idx" "$family" "$tag" "$seed" "$condition" "$node" "$partition" "$account" "$mem_mb" "$env_dir" "$force_scale" "$torque_scale"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

emit_first_wave() {
  for idx in "${!FIRST_TAG[@]}"; do
    submit_one first "$idx" move move move5 90000 "$ENV_RTX" \
      "${FIRST_FAMILY[$idx]}" "${FIRST_TAG[$idx]}" "${FIRST_SEED[$idx]}" "${FIRST_CONDITION[$idx]}"
  done
}

emit_later_wave() {
  local idx
  local l40_nodes=(
    "${L40_NODE_0:-${L40_NODE:-move4}}"
    "${L40_NODE_1:-${L40_NODE:-move4}}"
    "${L40_NODE_2:-${L40_NODE:-move4}}"
    "${L40_NODE_3:-${L40_NODE:-move4}}"
  )

  # First four later jobs on L40S. Use move4 by default; override below if needed.
  for idx in 0 1 2 3; do
    submit_one later "$idx" move move "${l40_nodes[$idx]}" 100000 "$ENV_BASE" \
      "${LATER_FAMILY[$idx]}" "${LATER_TAG[$idx]}" "${LATER_SEED[$idx]}" "${LATER_CONDITION[$idx]}"
  done

  # First two A5000 jobs on juno, next four on juno-lo.
  for idx in 4 5; do
    submit_one later "$idx" juno juno juno2 100000 "$ENV_BASE" \
      "${LATER_FAMILY[$idx]}" "${LATER_TAG[$idx]}" "${LATER_SEED[$idx]}" "${LATER_CONDITION[$idx]}"
  done
  for idx in 6 7 8 9; do
    submit_one later "$idx" juno juno-lo juno2 100000 "$ENV_BASE" \
      "${LATER_FAMILY[$idx]}" "${LATER_TAG[$idx]}" "${LATER_SEED[$idx]}" "${LATER_CONDITION[$idx]}"
  done
}

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "Job script not found: $JOB_SCRIPT" >&2
  exit 1
fi

echo "DRY_RUN=$DRY_RUN WAVE=$WAVE time=$TIME_LIMIT"
echo "W&B group: $WANDB_GROUP"
echo "num_envs=$NUM_ENVS minibatch=$MINIBATCH_SIZE expl_block=$EXPL_COEF_BLOCK_SIZE max_iterations=$MAX_ITERATIONS"
echo "wrench force/torque=$WRENCH_FORCE_SCALE/$WRENCH_TORQUE_SCALE no_wrench force/torque=$NO_WRENCH_FORCE_SCALE/$NO_WRENCH_TORQUE_SCALE"
echo ""

case "$WAVE" in
  first)
    emit_first_wave
    ;;
  later)
    emit_later_wave
    ;;
  all)
    emit_first_wave
    emit_later_wave
    ;;
  *)
    echo "Invalid WAVE=$WAVE. Expected first, later, or all." >&2
    exit 2
    ;;
esac

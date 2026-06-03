#!/usr/bin/env bash
set -euo pipefail

# Submit the 18-run Fabrica beam_3x part-0 wrench finetuning sweep.
#
# Dry-run by default. Set DRY_RUN=0 to submit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/panel_a_tyler_wrench_beam3x_part0_finetune.sub"

REPO="${REPO:-/move/u/tylerlum/github_repos/depthbasedRL}"
ENV_BASE="${ENV_BASE:-/move/u/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311}"
ENV_RTX="${ENV_RTX:-/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-cu128-py311}"

DRY_RUN="${DRY_RUN:-1}"
MOVE_TIME_LIMIT="${MOVE_TIME_LIMIT:-1-00:00:00}"
LONG_TIME_LIMIT="${LONG_TIME_LIMIT:-1-12:00:00}"
WANDB_GROUP="${WANDB_GROUP:-panel_a_teachers_tyler_beam3x_part0_wrench}"
JOB_INDICES="${JOB_INDICES:-}"

NUM_ENVS="${NUM_ENVS:-12288}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-98304}"
EXPL_COEF_BLOCK_SIZE="${EXPL_COEF_BLOCK_SIZE:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000000}"

FORCE_SCALE="${FORCE_SCALE:-20.0}"
TORQUE_SCALE="${TORQUE_SCALE:-2.0}"
FORCE_ONLY_WHEN_LIFTED="${FORCE_ONLY_WHEN_LIFTED:-False}"
TORQUE_ONLY_WHEN_LIFTED="${TORQUE_ONLY_WHEN_LIFTED:-False}"

FAMILIES=(
  TrainingObjective TrainingObjective
  TrainingObjective TrainingObjective
  ObjectDiversity ObjectDiversity
  Trajectory_Count Trajectory_Count
  ObjectDiversity ObjectDiversity
  TrainingObjective TrainingObjective
  Trajectory_Count Trajectory_Count
  Precision Precision
  Precision Precision
)
TAGS=(
  Play2Win Play2Win
  RotationOnly RotationOnly
  100_obj 100_obj
  100 100
  10_obj 10_obj
  TranslationOnly TranslationOnly
  10 10
  10cm 10cm
  5cm 5cm
)
SEEDS=(
  0 1
  0 1
  0 1
  0 1
  0 1
  0 1
  0 1
  0 1
  0 1
)
ACCOUNTS=(
  move move move move move move move move
  juno juno juno juno juno
  move move move move move
)
PARTITIONS=(
  move move move move move move humanoid move
  juno juno juno-lo juno-lo juno-lo
  move move move move move
)
NODES=(
  move4 move4 move4 move4 move4 move4 humanoid1 move5
  juno2 juno2 juno2 juno2 juno2
  move3 move3 move3 move3 move3
)
MEM_MB=(
  90000 90000 90000 90000 90000 90000 200000 90000
  100000 100000 100000 100000 100000
  90000 90000 90000 90000 90000
)
ENVS=(
  "$ENV_BASE" "$ENV_BASE" "$ENV_BASE" "$ENV_BASE" "$ENV_BASE" "$ENV_BASE" "$ENV_BASE" "$ENV_RTX"
  "$ENV_BASE" "$ENV_BASE" "$ENV_BASE" "$ENV_BASE" "$ENV_BASE"
  "$ENV_BASE" "$ENV_BASE" "$ENV_BASE" "$ENV_BASE" "$ENV_BASE"
)

checkpoint_path() {
  local family="$1"
  local tag="$2"
  printf '/juno/u/kedia/depthbasedRL/train_dir/%s/%s/model.pth' "$family" "$tag"
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
  local account="${ACCOUNTS[$idx]}"
  local partition="${PARTITIONS[$idx]}"
  local node="${NODES[$idx]}"
  local mem="${MEM_MB[$idx]}"
  local env_dir="${ENVS[$idx]}"
  local time_limit="$LONG_TIME_LIMIT"
  if [[ "$partition" == "move" ]]; then
    time_limit="$MOVE_TIME_LIMIT"
  fi

  validate_job "$family" "$tag"

  local job_name="b3x0-wrench-${family}-${tag}-s${seed}"
  job_name="${job_name//_/-}"
  job_name="${job_name//\//-}"
  local experiment_tag="beam_3x_part_0_finetune_rgf0_dr_wrench_seed${seed}"

  local cmd=(
    sbatch
    --job-name="$job_name"
    --account="$account"
    --partition="$partition"
    --nodelist="$node"
    --mem="$mem"
    --time="$time_limit"
    --export=ALL,EXPERIMENT_TAG="$experiment_tag",CHECKPOINT_FAMILY="$family",CHECKPOINT_TAG="$tag",CONDITION_TAG=wrench,SEED="$seed",NUM_ENVS="$NUM_ENVS",MINIBATCH_SIZE="$MINIBATCH_SIZE",EXPL_COEF_BLOCK_SIZE="$EXPL_COEF_BLOCK_SIZE",MAX_ITERATIONS="$MAX_ITERATIONS",REPO_ROOT="$REPO",ISAACSIM_ENV_DIR="$env_dir",WANDB_GROUP="$WANDB_GROUP",FORCE_SCALE="$FORCE_SCALE",TORQUE_SCALE="$TORQUE_SCALE",FORCE_ONLY_WHEN_LIFTED="$FORCE_ONLY_WHEN_LIFTED",TORQUE_ONLY_WHEN_LIFTED="$TORQUE_ONLY_WHEN_LIFTED"
    "$JOB_SCRIPT"
  )

  printf '[%02d] %-18s %-16s seed=%s account=%s partition=%s node=%s mem=%s time=%s env=%s\n' \
    "$idx" "$family" "$tag" "$seed" "$account" "$partition" "$node" "$mem" "$time_limit" "$env_dir"
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

echo "DRY_RUN=$DRY_RUN move_time=$MOVE_TIME_LIMIT long_time=$LONG_TIME_LIMIT job_indices=${JOB_INDICES:-all}"
echo "repo=$REPO"
echo "env=$ENV_BASE"
echo "env_rtx=$ENV_RTX"
echo "wandb_group=$WANDB_GROUP"
echo "num_envs=$NUM_ENVS minibatch=$MINIBATCH_SIZE expl_block=$EXPL_COEF_BLOCK_SIZE max_iterations=$MAX_ITERATIONS"
echo "wrench force/torque=$FORCE_SCALE/$TORQUE_SCALE only_lifted=$FORCE_ONLY_WHEN_LIFTED/$TORQUE_ONLY_WHEN_LIFTED"
echo ""

if [[ -n "$JOB_INDICES" ]]; then
  for idx in $JOB_INDICES; do
    submit_one "$idx"
  done
else
  for idx in "${!FAMILIES[@]}"; do
    submit_one "$idx"
  done
fi

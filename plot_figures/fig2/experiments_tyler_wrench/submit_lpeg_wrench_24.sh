#!/usr/bin/env bash
set -euo pipefail

# Prepare/submit 24 L-peg wrench finetunes:
#   8 checkpoints x 3 seeds = 24 jobs.
#
# Job order is intentional:
#   WAVE=1: all 12 ObjectDiversity jobs + TrainingObjective seed 0.
#   WAVE=2: TrainingObjective seeds 1 and 2.
#
# Default is DRY_RUN=1 so this script is safe while current jobs are running.
# Set DRY_RUN=0 when ready to actually sbatch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/panel_a_tyler_wrench_lpeg_finetune.sub"

REPO="${REPO:-/move/u/tylerlum/github_repos/depthbasedRL}"
ENV_BASE="${ENV_BASE:-/move/u/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311}"
ENV_RTX="${ENV_RTX:-/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-cu128-py311}"
DRY_RUN="${DRY_RUN:-1}"
WAVE="${WAVE:-all}"  # all, sanity, 1, or 2

TASK_TAG="lpeg_tol0p5mm"
WANDB_GROUP="${WANDB_GROUP:-panel_a_teachers_tyler_wrench_lpeg}"
NUM_ENVS="${NUM_ENVS:-12288}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-98304}"
EXPL_COEF_BLOCK_SIZE="${EXPL_COEF_BLOCK_SIZE:-2048}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"

FORCE_SCALE="${FORCE_SCALE:-20.0}"
TORQUE_SCALE="${TORQUE_SCALE:-2.0}"
FORCE_ONLY_WHEN_LIFTED="${FORCE_ONLY_WHEN_LIFTED:-False}"
TORQUE_ONLY_WHEN_LIFTED="${TORQUE_ONLY_WHEN_LIFTED:-False}"
FORCE_PROB_RANGE="${FORCE_PROB_RANGE:-[0.001,0.1]}"
TORQUE_PROB_RANGE="${TORQUE_PROB_RANGE:-[0.001,0.1]}"

OBJECT_DIVERSITY_CHECKPOINTS=(1000_obj 100_obj 10_obj 1_obj)
TRAINING_OBJECTIVE_CHECKPOINTS=(Play2Win RotationOnly SingleGoal TranslationOnly)
JOB_FAMILIES=()
JOB_TAGS=()
JOB_SEEDS=()

for seed in 0 1 2; do
  for checkpoint in "${OBJECT_DIVERSITY_CHECKPOINTS[@]}"; do
    JOB_FAMILIES+=(ObjectDiversity)
    JOB_TAGS+=("$checkpoint")
    JOB_SEEDS+=("$seed")
  done
done

for seed in 0 1 2; do
  for checkpoint in "${TRAINING_OBJECTIVE_CHECKPOINTS[@]}"; do
    JOB_FAMILIES+=(TrainingObjective)
    JOB_TAGS+=("$checkpoint")
    JOB_SEEDS+=("$seed")
  done
done

# 16 requested GPU slots:
#   8 x move5 RTX PRO 6000
#   4 x move4 L40S
#   2 x juno2 A5000 on juno
#   2 x juno2 A5000 on juno-lo
SLOT_ACCOUNTS=()
SLOT_PARTITIONS=()
SLOT_NODES=()
SLOT_MEM_MB=()
SLOT_ENVS=()
for _ in {1..8}; do
  SLOT_ACCOUNTS+=(move)
  SLOT_PARTITIONS+=(move)
  SLOT_NODES+=(move5)
  SLOT_MEM_MB+=(90000)
  SLOT_ENVS+=("$ENV_RTX")
done
for _ in {1..4}; do
  SLOT_ACCOUNTS+=(move)
  SLOT_PARTITIONS+=(move)
  SLOT_NODES+=(move4)
  SLOT_MEM_MB+=(100000)
  SLOT_ENVS+=("$ENV_BASE")
done
for _ in {1..2}; do
  SLOT_ACCOUNTS+=(juno)
  SLOT_PARTITIONS+=(juno)
  SLOT_NODES+=(juno2)
  SLOT_MEM_MB+=(100000)
  SLOT_ENVS+=("$ENV_BASE")
done
for _ in {1..2}; do
  SLOT_ACCOUNTS+=(juno)
  SLOT_PARTITIONS+=(juno-lo)
  SLOT_NODES+=(juno2)
  SLOT_MEM_MB+=(100000)
  SLOT_ENVS+=("$ENV_BASE")
done

case "$WAVE" in
  all)
    START_INDEX=0
    END_INDEX=23
    ;;
  sanity)
    # TrainingObjective/Play2Win seed 0 on the first move5 RTX PRO 6000 slot.
    START_INDEX=12
    END_INDEX=12
    ;;
  1)
    START_INDEX=0
    END_INDEX=15
    ;;
  2)
    START_INDEX=16
    END_INDEX=23
    ;;
  *)
    echo "Invalid WAVE=$WAVE. Expected all, sanity, 1, or 2." >&2
    exit 2
    ;;
esac

submit_one() {
  local job_idx="$1"
  local slot_idx="$2"
  local checkpoint_family="$3"
  local checkpoint_tag="$4"
  local seed="$5"

  local account="${SLOT_ACCOUNTS[$slot_idx]}"
  local partition="${SLOT_PARTITIONS[$slot_idx]}"
  local node="${SLOT_NODES[$slot_idx]}"
  local mem_mb="${SLOT_MEM_MB[$slot_idx]}"
  local env_dir="${SLOT_ENVS[$slot_idx]}"
  local experiment_tag="lpeg_tol0p5mm_finetune_rgf0_dr_wrench_seed${seed}"
  local env_prefix=(
    env
    FORCE_PROB_RANGE="$FORCE_PROB_RANGE"
    TORQUE_PROB_RANGE="$TORQUE_PROB_RANGE"
  )

  local cmd=(
    sbatch
    --account="$account"
    --partition="$partition"
    --nodelist="$node"
    --mem="$mem_mb"
    --time="$TIME_LIMIT"
    --export=ALL,TASK_TAG="$TASK_TAG",EXPERIMENT_TAG="$experiment_tag",CHECKPOINT_FAMILY="$checkpoint_family",CHECKPOINT_TAG="$checkpoint_tag",SEED="$seed",NUM_ENVS="$NUM_ENVS",MINIBATCH_SIZE="$MINIBATCH_SIZE",EXPL_COEF_BLOCK_SIZE="$EXPL_COEF_BLOCK_SIZE",REPO_ROOT="$REPO",ISAACSIM_ENV_DIR="$env_dir",WANDB_GROUP="$WANDB_GROUP",FORCE_SCALE="$FORCE_SCALE",TORQUE_SCALE="$TORQUE_SCALE",FORCE_ONLY_WHEN_LIFTED="$FORCE_ONLY_WHEN_LIFTED",TORQUE_ONLY_WHEN_LIFTED="$TORQUE_ONLY_WHEN_LIFTED"
    "$JOB_SCRIPT"
  )

  printf '[%02d slot=%02d] family=%s checkpoint=%s seed=%s node=%s partition=%s mem=%s env=%s\n' \
    "$job_idx" "$slot_idx" "$checkpoint_family" "$checkpoint_tag" "$seed" "$node" "$partition" "$mem_mb" "$env_dir"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  DRY_RUN:'
    printf ' %q' "${env_prefix[@]}"
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${env_prefix[@]}" "${cmd[@]}"
  fi
}

echo "DRY_RUN=$DRY_RUN WAVE=$WAVE jobs=${START_INDEX}-${END_INDEX} time=$TIME_LIMIT"
echo "W&B group: $WANDB_GROUP"
echo "Wrench: force=$FORCE_SCALE torque=$TORQUE_SCALE force_only_lifted=$FORCE_ONLY_WHEN_LIFTED torque_only_lifted=$TORQUE_ONLY_WHEN_LIFTED"

for job_idx in "${!JOB_TAGS[@]}"; do
  if (( job_idx >= START_INDEX && job_idx <= END_INDEX )); then
    slot_idx=$(((job_idx - START_INDEX) % 16))
    submit_one "$job_idx" "$slot_idx" "${JOB_FAMILIES[$job_idx]}" "${JOB_TAGS[$job_idx]}" "${JOB_SEEDS[$job_idx]}"
  fi
done

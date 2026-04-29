#!/usr/bin/env bash
# Generic launcher for Isaac Lab-native PegInHole depth distillation.
#
# Example:
#   RUN_NAME=01_teacher_obs STUDENT_INPUT=teacher_obs NUM_ENVS=4096 \
#     sbatch --nodelist=move4 scripts/cluster/sbatch_kushal_depth_distill_v2.sh

#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/move/u/${USER}/github_repos/depthbasedRL}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv-isaacsim-py311/bin/python}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME for the distillation run.}"
WANDB_GROUP="${WANDB_GROUP:-2026-04-29_KushalEnvDepthDistillV2}"
WANDB_PROJECT="${WANDB_PROJECT:-depthbasedRL-isaacsim-distill}"

STUDENT_INPUT="${STUDENT_INPUT:-camera}"
STUDENT_ARCH="${STUDENT_ARCH:-}"
NUM_ENVS="${NUM_ENVS:-512}"
NUM_ITERS="${NUM_ITERS:-3500000}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
RUN_DIR="${RUN_DIR:-distillation_runs/${RUN_NAME}}"

DEPTH_NOISE_PROFILE="${DEPTH_NOISE_PROFILE:-off}"
DEPTH_NOISE_STRENGTH="${DEPTH_NOISE_STRENGTH:-}"
CAMERA_POSE_PROFILE="${CAMERA_POSE_PROFILE:-off}"
CAMERA_POSE_MODE="${CAMERA_POSE_MODE:-startup}"
CAMERA_POS_NOISE_M="${CAMERA_POS_NOISE_M:-}"
CAMERA_ROT_NOISE_DEG="${CAMERA_ROT_NOISE_DEG:-}"
DEPTH_DEBUG_INTERVAL="${DEPTH_DEBUG_INTERVAL:-1000}"
CAPTURE_VIEWER_LEN="${CAPTURE_VIEWER_LEN:-600}"

cd "$REPO_DIR"
mkdir -p slurm_logs "$(dirname "$RUN_DIR")"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export OMNI_KIT_CACHE_PATH="${OMNI_KIT_CACHE_PATH:-/tmp/${USER}_ov_cache_${SLURM_JOB_ID:-manual}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p "$OMNI_KIT_CACHE_PATH"

if [[ -z "${WANDB_API_KEY:-}" && -f "/juno/u/${USER}/.wandb_api_key" ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '[:space:]' < "/juno/u/${USER}/.wandb_api_key")"
fi

if [[ -z "$STUDENT_ARCH" ]]; then
  if [[ "$STUDENT_INPUT" == "teacher_obs" ]]; then
    STUDENT_ARCH="mlp_recurrent"
  else
    STUDENT_ARCH="mono_transformer_recurrent"
  fi
fi

cmd=(
  "$PYTHON_BIN" isaacsimenvs/distill_depth.py
  --mode train_online
  --student_input "$STUDENT_INPUT"
  --student_arch "$STUDENT_ARCH"
  --num_envs "$NUM_ENVS"
  --num_iters "$NUM_ITERS"
  --log_interval "$LOG_INTERVAL"
  --save_interval "$SAVE_INTERVAL"
  --run_dir "$RUN_DIR"
  --wandb
  --wandb_project "$WANDB_PROJECT"
  --wandb_group "$WANDB_GROUP"
  --wandb_name "$RUN_NAME"
  --capture_viewer
  --capture_viewer_len "$CAPTURE_VIEWER_LEN"
  --headless
)

if [[ "$STUDENT_INPUT" == "camera" ]]; then
  cmd+=(
    --depth_noise_profile "$DEPTH_NOISE_PROFILE"
    --camera_pose_randomization_profile "$CAMERA_POSE_PROFILE"
    --camera_pose_randomization_mode "$CAMERA_POSE_MODE"
    --depth_debug_interval "$DEPTH_DEBUG_INTERVAL"
  )
  if [[ -n "$DEPTH_NOISE_STRENGTH" ]]; then
    cmd+=(--depth_noise_strength "$DEPTH_NOISE_STRENGTH")
  fi
  if [[ -n "$CAMERA_POS_NOISE_M" ]]; then
    read -r -a camera_pos_noise_args <<< "$CAMERA_POS_NOISE_M"
    cmd+=(--camera_pos_noise_m "${camera_pos_noise_args[@]}")
  fi
  if [[ -n "$CAMERA_ROT_NOISE_DEG" ]]; then
    read -r -a camera_rot_noise_args <<< "$CAMERA_ROT_NOISE_DEG"
    cmd+=(--camera_rot_noise_deg "${camera_rot_noise_args[@]}")
  fi
fi

echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "repo=$REPO_DIR"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"
echo "python=$PYTHON_BIN"
echo "run_name=$RUN_NAME"
echo "wandb_group=$WANDB_GROUP"
echo "cache=$OMNI_KIT_CACHE_PATH"
nvidia-smi || true
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"

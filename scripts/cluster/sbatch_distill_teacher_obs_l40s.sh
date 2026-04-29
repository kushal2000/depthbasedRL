#!/usr/bin/env bash
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodelist=move4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --job-name=distill_teacherobs
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/move/u/$USER/github_repos/depthbasedRL}"
RUN_CONFIG="${RUN_CONFIG:-isaacsim_conversion/configs/hammer_distill_teacher_obs_4096_hold_then_decay_long.yaml}"
CAMERA_CONFIG="${CAMERA_CONFIG:-isaacsim_conversion/configs/hammer_camera_depth_160x90.yaml}"
TEACHER_CKPT="${TEACHER_CKPT:-pretrained_policy/model.pth}"
TEACHER_CFG="${TEACHER_CFG:-pretrained_policy/config.yaml}"
OMNI_CACHE="${OMNI_KIT_CACHE_PATH:-/tmp/${USER}_ov_cache}"
STUDENT_ARCH="${STUDENT_ARCH:-mlp_recurrent}"

mkdir -p slurm_logs
cd "$REPO_DIR"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export OMNI_KIT_CACHE_PATH="$OMNI_CACHE"
mkdir -p "$OMNI_KIT_CACHE_PATH" slurm_logs

echo "Hostname: $(hostname)"
nvidia-smi || true
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse HEAD)"
echo "Cache: $OMNI_KIT_CACHE_PATH"
echo "Run config: $RUN_CONFIG"

./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill.py \
  --mode train_online \
  --headless \
  --student_input teacher_obs \
  --student_arch "$STUDENT_ARCH" \
  --distill_config "$RUN_CONFIG" \
  --camera_config "$CAMERA_CONFIG" \
  --teacher_checkpoint "$TEACHER_CKPT" \
  --teacher_config "$TEACHER_CFG"

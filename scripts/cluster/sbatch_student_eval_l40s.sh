#!/usr/bin/env bash
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodelist=move4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=student_eval
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/move/u/$USER/github_repos/depthbasedRL}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=distillation_runs/.../checkpoints/student_*.pt}"
CAMERA_CONFIG="${CAMERA_CONFIG:-isaacsim_conversion/configs/hammer_camera_depth_160x90.yaml}"
TEACHER_CKPT="${TEACHER_CKPT:-pretrained_policy/model.pth}"
TEACHER_CFG="${TEACHER_CFG:-pretrained_policy/config.yaml}"
OMNI_CACHE="${OMNI_KIT_CACHE_PATH:-/tmp/${USER}_ov_cache}"
STUDENT_ARCH="${STUDENT_ARCH:-mlp_recurrent}"
NUM_ENVS="${NUM_ENVS:-512}"
MAX_STEPS="${MAX_STEPS:-700}"

mkdir -p slurm_logs
cd "$REPO_DIR"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export OMNI_KIT_CACHE_PATH="$OMNI_CACHE"
mkdir -p "$OMNI_KIT_CACHE_PATH" slurm_logs

echo "Hostname: $(hostname)"
nvidia-smi || true
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse HEAD)"
echo "Checkpoint: $CHECKPOINT"

./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill_eval.py \
  --mode student_eval \
  --headless \
  --student_input teacher_obs \
  --student_arch "$STUDENT_ARCH" \
  --student_checkpoint "$CHECKPOINT" \
  --max_steps "$MAX_STEPS" \
  --num_envs "$NUM_ENVS" \
  --env_spacing 4.0 \
  --object_start_mode fixed \
  --teacher_checkpoint "$TEACHER_CKPT" \
  --teacher_config "$TEACHER_CFG" \
  --camera_config "$CAMERA_CONFIG"

#!/bin/bash
# Multi-GPU SAPG pretraining via torchrun — Isaac Lab backend (no depth)
#
# Reproduces the config from wandb run:
#   samratsahoo-stanford-university/simpretrain/2k52e9pc
#
# Usage:
#   # Default (auto-detect GPUs):
#   bash pretrain_isaaclab.sh
#
#   # Override GPU count or envs:
#   NGPUS=2 NUM_ENVS=8192 bash pretrain_isaaclab.sh
#
#   # Single GPU:
#   NGPUS=1 NUM_ENVS=8192 bash pretrain_isaaclab.sh
#
#   # Enable WandB logging:
#   WANDB_ACTIVATE=True bash pretrain_isaaclab.sh

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-sapg_il}"
CONDA_PREFIX="$(conda info --base)/envs/${CONDA_ENV}"

# ── GPU config ──
NGPUS="${NGPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

# ── Defaults (override via env vars) ──
NUM_ENVS="${NUM_ENVS:-8192}"
SEED="${SEED:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-1000000}"
WANDB_PROJECT="${WANDB_PROJECT:-simpretrain}"
WANDB_ENTITY="${WANDB_ENTITY:-samratsahoo-stanford-university}"
WANDB_ACTIVATE="${WANDB_ACTIVATE:-false}"
EXPERIMENT="${EXPERIMENT:-00_isaaclab_sapg_$(date +%Y-%m-%d_%H-%M-%S)}"

echo "Launching Isaac Lab training: ${NGPUS} GPUs, ${NUM_ENVS} total envs, conda=${CONDA_ENV}"

# torchrun needs the conda env's lib on LD_LIBRARY_PATH for child processes
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# ── Build Hydra overrides ──
COMMON_ARGS=(
  num_envs="${NUM_ENVS}"
  seed="${SEED}"
  max_epochs="${MAX_EPOCHS}"
  experiment="${EXPERIMENT}"
  # PPO / training
  learning_rate=1e-4
  minibatch_size=32768
  horizon_length=16
  mini_epochs=2
  # SAPG exploration
  expl_type=mixed_expl_learn_param
  expl_reward_type=entropy
  expl_reward_coef_scale=0.005
  expl_coef_block_size=1024
  use_others_experience=lf
  # WandB
  wandb_activate="${WANDB_ACTIVATE}"
  wandb_project="${WANDB_PROJECT}"
  wandb_entity="${WANDB_ENTITY}"
  wandb_name="${EXPERIMENT}"
)

# ── Launch ──
if [[ "${NGPUS}" -gt 1 ]]; then
  echo "Multi-GPU mode: ${NGPUS} GPUs"
  PYTHONUNBUFFERED=1 "${CONDA_PREFIX}/bin/torchrun" \
    --nproc_per_node="${NGPUS}" \
    train_isaaclab.py \
    multi_gpu=true \
    "${COMMON_ARGS[@]}" \
    "$@"
else
  echo "Single-GPU mode"
  PYTHONUNBUFFERED=1 python train_isaaclab.py \
    "${COMMON_ARGS[@]}" \
    "$@"
fi

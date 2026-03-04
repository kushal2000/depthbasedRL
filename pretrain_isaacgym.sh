#!/bin/bash
# Multi-GPU SAPG pretraining via torchrun — IsaacGym backend
#
# Usage:
#   # Default (auto-detect GPUs):
#   bash pretrain_isaacgym.sh
#
#   # Override GPU count or envs:
#   NGPUS=2 NUM_ENVS=8192 bash pretrain_isaacgym.sh
#
#   # Single GPU:
#   NGPUS=1 NUM_ENVS=8192 bash pretrain_isaacgym.sh
#
#   # Enable WandB logging:
#   WANDB_ACTIVATE=True bash pretrain_isaacgym.sh

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-sapg}"
CONDA_PREFIX="$(conda info --base)/envs/${CONDA_ENV}"

# ── GPU config ──
NGPUS="${NGPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

# ── Defaults (override via env vars) ──
NUM_ENVS="${NUM_ENVS:-8192}"
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-1000000}"
WANDB_PROJECT="${WANDB_PROJECT:-simpretrain}"
WANDB_ENTITY="${WANDB_ENTITY:-samratsahoo-stanford-university}"
WANDB_ACTIVATE="${WANDB_ACTIVATE:-False}"
EXPERIMENT="${EXPERIMENT:-00_isaacgym_sapg_$(date +%Y-%m-%d_%H-%M-%S)}"

echo "Launching IsaacGym training: ${NGPUS} GPUs, ${NUM_ENVS} total envs, conda=${CONDA_ENV}"

# torchrun needs the conda env's lib on LD_LIBRARY_PATH for child processes
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# ── Build Hydra overrides ──
COMMON_ARGS=(
  headless=True
  task.env.numEnvs="${NUM_ENVS}"
  seed="${SEED}"
  train.params.config.max_epochs="${MAX_EPOCHS}"
  experiment="${EXPERIMENT}"
  # SAPG exploration
  train.params.config.expl_type=mixed_expl_learn_param
  train.params.config.expl_reward_type=entropy
  train.params.config.expl_reward_coef_scale=0.005
  train.params.config.expl_coef_block_size=1024
  train.params.config.use_others_experience=lf
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
    train_isaacgym.py \
    multi_gpu=True \
    "${COMMON_ARGS[@]}" \
    "$@"
else
  echo "Single-GPU mode"
  PYTHONUNBUFFERED=1 python train_isaacgym.py \
    "${COMMON_ARGS[@]}" \
    "$@"
fi

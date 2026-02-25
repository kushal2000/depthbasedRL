#!/bin/bash
# 5-GPU SAPG pretraining via torchrun
#
# Params tuned for 5x GPUs (matching 3-GPU per-rank setup):
#   NUM_ENVS   = 40960  (8192 per GPU, same as 3-GPU per-rank)
#   block_size = 1024   → 8192/1024 = 8 blocks per rank
#   minibatch  = 163840 (32768 per GPU → 4 minibatches)
#   batch/rank = 16 * 8192 = 131072
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-sapg}"
CONDA_PREFIX="$(conda info --base)/envs/${CONDA_ENV}"

NGPUS=5

# ── Defaults (override via env vars) ──
NUM_ENVS="${NUM_ENVS:-40960}"          # 8192 per GPU → 8 blocks of 1024
SEED="${SEED:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-1000000}"
WANDB_PROJECT="${WANDB_PROJECT:-simpretrain}"
WANDB_ENTITY="${WANDB_ENTITY:-samratsahoo-stanford-university}"
WANDB_ACTIVATE="${WANDB_ACTIVATE:-False}"
WANDB_GROUP="${WANDB_GROUP:-$(date +%Y-%m-%d)}"
EXPERIMENT="${EXPERIMENT:-00_sapg_5gpu_$(date +%Y-%m-%d_%H-%M-%S)}"
TRAIN_DIR="${TRAIN_DIR:-./train_dir/simpretrain/${WANDB_GROUP}/${EXPERIMENT}}"

echo "Launching 5-GPU training: ${NGPUS} GPUs, ${NUM_ENVS} total envs ($(( NUM_ENVS / NGPUS )) per GPU, 8 blocks)"

# torchrun needs the conda env's lib on LD_LIBRARY_PATH for child processes
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

"${CONDA_PREFIX}/bin/torchrun" --nproc_per_node="${NGPUS}" \
  -m isaacgymenvs.train \
  headless=True \
  task=SimToolRealLSTMAsymmetric \
  task.env.numEnvs="${NUM_ENVS}" \
  task.env.goodResetBoundary=0 \
  task.env.objectScaleNoiseMultiplierRange="[0.9,1.1]" \
  task.env.forceConsecutiveNearGoalSteps=True \
  task.env.forceScale=20 \
  task.env.torqueScale=2.0 \
  task.env.objectAngVelPenaltyScale=0.0 \
  train.params.config.minibatch_size=163840 \
  train.params.config.central_value_config.minibatch_size=163840 \
  train.params.config.max_epochs="${MAX_EPOCHS}" \
  train.params.config.good_reset_boundary=0 \
  train.params.config.use_others_experience=lf \
  train.params.config.off_policy_ratio=1.0 \
  train.params.config.expl_type=mixed_expl_learn_param \
  train.params.config.expl_reward_type=entropy \
  train.params.config.expl_coef_block_size=1024 \
  train.params.config.expl_reward_coef_scale=0.005 \
  train.params.network.space.continuous.fixed_sigma=coef_cond \
  multi_gpu=True \
  wandb_activate="${WANDB_ACTIVATE}" \
  wandb_project="${WANDB_PROJECT}" \
  wandb_entity="${WANDB_ENTITY}" \
  wandb_group="${WANDB_GROUP}" \
  seed="${SEED}" \
  experiment="${EXPERIMENT}" \
  hydra.run.dir="${TRAIN_DIR}" \
  ++use_rl=True \
  "$@"

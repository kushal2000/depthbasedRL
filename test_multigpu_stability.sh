#!/bin/bash
# Multi-GPU tuning sweep: match reference 8k-reward config
# Reference run nxysx8rx: dofSpeedScale=1.5, useActionDelay=False, got 8k reward
# Our fix: expl_coef_block_size=1024 (8 blocks/GPU)
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-sapg}"
CONDA_PREFIX="$(conda info --base)/envs/${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

NGPUS=3
NUM_ENVS=36000
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
WANDB_PROJECT="simpretrain"
WANDB_ENTITY="samratsahoo-stanford-university"
WANDB_GROUP="tuning-sweep-$(date +%Y-%m-%d_%H-%M-%S)"

# Configurable params with defaults matching reference 8k run
DOF_SPEED_SCALE="${DOF_SPEED_SCALE:-5.0}"
USE_ACTION_DELAY="${USE_ACTION_DELAY:-False}"
EXPERIMENT="${EXPERIMENT:-00_speed5_12kenv_6blocks}"
TRAIN_DIR="./train_dir/stability_test/${WANDB_GROUP}/${EXPERIMENT}"

echo ""
echo "=========================================="
echo "  Running: ${EXPERIMENT}"
echo "  dofSpeedScale=${DOF_SPEED_SCALE}, useActionDelay=${USE_ACTION_DELAY}"
echo "  Block size: 2000 (6 blocks/GPU)"
echo "  Max epochs: ${MAX_EPOCHS}"
echo "=========================================="
echo ""

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
  task.env.dofSpeedScale="${DOF_SPEED_SCALE}" \
  task.env.useActionDelay="${USE_ACTION_DELAY}" \
  train.params.config.minibatch_size=144000 \
  train.params.config.max_epochs="${MAX_EPOCHS}" \
  train.params.config.good_reset_boundary=0 \
  train.params.config.use_others_experience=lf \
  train.params.config.off_policy_ratio=1.0 \
  train.params.config.expl_type=mixed_expl_learn_param \
  train.params.config.expl_reward_type=entropy \
  train.params.config.expl_coef_block_size=2000 \
  train.params.config.expl_reward_coef_scale=0.005 \
  train.params.network.space.continuous.fixed_sigma=coef_cond \
  multi_gpu=True \
  wandb_activate=True \
  wandb_project="${WANDB_PROJECT}" \
  wandb_entity="${WANDB_ENTITY}" \
  wandb_group="${WANDB_GROUP}" \
  wandb_name="${EXPERIMENT}" \
  seed=0 \
  experiment="${EXPERIMENT}" \
  hydra.run.dir="${TRAIN_DIR}" \
  ++use_rl=True \
  "$@"

echo ""
echo "Experiment complete. Check wandb group: ${WANDB_GROUP}"

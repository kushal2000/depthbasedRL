#!/bin/bash
# Diagnostic multi-GPU run to capture crash errors
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-sapg}"
CONDA_PREFIX="$(conda info --base)/envs/${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

NGPUS=3
NUM_ENVS=24576  # 8192 per GPU
MAX_EPOCHS=500

# ── Diagnostic environment variables ──
# NCCL debug info to capture communication errors
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

# PyTorch distributed debug for more verbose errors
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Show CUDA errors synchronously (slower but gives exact error location)
# Uncomment next line for detailed CUDA debugging (much slower):
# export CUDA_LAUNCH_BLOCKING=1

# Enable Python faulthandler to get tracebacks on segfaults
export PYTHONFAULTHANDLER=1

# Reduce NCCL timeout to 5 minutes so hangs are detected faster
export NCCL_TIMEOUT=300

EXPERIMENT="00_diag_multigpu_$(date +%Y-%m-%d_%H-%M-%S)"
TRAIN_DIR="./train_dir/diagnostics/${EXPERIMENT}"
LOGDIR="./train_dir/diagnostics/${EXPERIMENT}/logs"
mkdir -p "${LOGDIR}"

echo ""
echo "=========================================="
echo "  Diagnostic Multi-GPU Run"
echo "  GPUs: ${NGPUS}, Envs: ${NUM_ENVS}"
echo "  Max epochs: ${MAX_EPOCHS}"
echo "  Logs: ${LOGDIR}"
echo "=========================================="
echo ""

# Run with explicit stderr/stdout capture per rank via torchrun
# torchrun captures stdout/stderr per rank in log files
"${CONDA_PREFIX}/bin/torchrun" \
  --nproc_per_node="${NGPUS}" \
  --redirects=3 \
  --log-dir="${LOGDIR}" \
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
  task.env.capture_video=True \
  task.env.capture_video_freq=1500 \
  train.params.config.minibatch_size=98304 \
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
  wandb_activate=True \
  wandb_project=simpretrain \
  wandb_entity=samratsahoo-stanford-university \
  wandb_group=diag-test \
  seed=0 \
  experiment="${EXPERIMENT}" \
  hydra.run.dir="${TRAIN_DIR}" \
  ++use_rl=True \
  2>&1 | tee "${LOGDIR}/main.log"

echo ""
echo "Run finished. Check logs in: ${LOGDIR}"

#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/ffs_isaacsim_ros_env.sh"

NUM_STEPS="${NUM_STEPS:-80}"
BENCHMARK_WARMUP_STEPS="${BENCHMARK_WARMUP_STEPS:-5}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-2}"
DEPTH_EVERY_N="${DEPTH_EVERY_N:-1}"
INITIAL_ROBOT_POSE="${INITIAL_ROBOT_POSE:-deployment_home}"
OBJECT_INIT_MODE="${OBJECT_INIT_MODE:-default}"
PEG_URDF="${PEG_URDF:-assets/urdf/peg_in_hole/peg_L/peg_L.urdf}"
FFS_BASELINE_M="${FFS_BASELINE_M:-0.12}"
FFS_VALID_ITERS="${FFS_VALID_ITERS:-4}"
FFS_MAX_DISP="${FFS_MAX_DISP:-192}"
FFS_DEBUG_DIR="${FFS_DEBUG_DIR:-local_logs/ffs_isaac_stereo_debug}"
FFS_DEBUG_EVERY_N="${FFS_DEBUG_EVERY_N:-10}"

"${ISAACSIM_PYTHON}" deployment/isaac/isaac_depth_env_node.py \
  --benchmark \
  --enable_depth \
  --depth_render_backend ffs_stereo \
  --depth_publish_every_n "${DEPTH_EVERY_N}" \
  --num_steps "${NUM_STEPS}" \
  --initial_robot_pose "${INITIAL_ROBOT_POSE}" \
  --object_init_mode "${OBJECT_INIT_MODE}" \
  --peg_urdf "${PEG_URDF}" \
  --benchmark_warmup_steps "${BENCHMARK_WARMUP_STEPS}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --ffs_repo_root "${FAST_FOUNDATIONSTEREO_ROOT}" \
  --ffs_model_path "${FFS_MODEL_PATH}" \
  --ffs_baseline_m "${FFS_BASELINE_M}" \
  --ffs_valid_iters "${FFS_VALID_ITERS}" \
  --ffs_max_disp "${FFS_MAX_DISP}" \
  --ffs_debug_dir "${FFS_DEBUG_DIR}" \
  --ffs_debug_every_n "${FFS_DEBUG_EVERY_N}" \
  --headless

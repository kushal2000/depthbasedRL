#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/isaacsim_ros_env.sh"

NUM_STEPS="${NUM_STEPS:-300}"
BENCHMARK_WARMUP_STEPS="${BENCHMARK_WARMUP_STEPS:-10}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-2}"
INITIAL_ROBOT_POSE="${INITIAL_ROBOT_POSE:-deployment_home}"
OBJECT_INIT_MODE="${OBJECT_INIT_MODE:-default}"
PEG_URDF="${PEG_URDF:-assets/urdf/peg_in_hole/peg_L/peg_L.urdf}"
EXTRA_ARGS=()
if [[ "${DEPLOYMENT_MODE:-1}" == "0" ]]; then EXTRA_ARGS+=(--no-deployment_mode); fi
if [[ "${DISABLE_ENV_RESETS:-1}" == "0" ]]; then EXTRA_ARGS+=(--no-disable_env_resets); fi
if [[ "${ZERO_TRAINING_RANDOMIZATION:-1}" == "0" ]]; then EXTRA_ARGS+=(--no-zero_training_randomization); fi
if [[ -n "${OBJECT_INIT_POSE_WXYZ:-}" ]]; then EXTRA_ARGS+=(--object_init_pose_wxyz ${OBJECT_INIT_POSE_WXYZ}); fi
if [[ -n "${OBJECT_INIT_POSITION_NOISE_M:-}" ]]; then EXTRA_ARGS+=(--object_init_position_noise_m ${OBJECT_INIT_POSITION_NOISE_M}); fi
if [[ -n "${OBJECT_INIT_YAW_NOISE_DEG:-}" ]]; then EXTRA_ARGS+=(--object_init_yaw_noise_deg "${OBJECT_INIT_YAW_NOISE_DEG}"); fi
if [[ -n "${OBJECT_INIT_ORIENTATION_MODE:-}" ]]; then EXTRA_ARGS+=(--object_init_orientation_mode "${OBJECT_INIT_ORIENTATION_MODE}"); fi

"${ISAACSIM_PYTHON}" deployment/isaac/isaac_depth_env_node.py \
  --benchmark \
  --enable_depth \
  --depth_publish_every_n 1 \
  --num_steps "${NUM_STEPS}" \
  --initial_robot_pose "${INITIAL_ROBOT_POSE}" \
  --object_init_mode "${OBJECT_INIT_MODE}" \
  --peg_urdf "${PEG_URDF}" \
  --benchmark_warmup_steps "${BENCHMARK_WARMUP_STEPS}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --headless \
  "${EXTRA_ARGS[@]}"

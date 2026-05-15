#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/isaacsim_ros_env.sh"

RUN_DURATION_S="${RUN_DURATION_S:--1}"
DEPTH_EVERY_N="${DEPTH_EVERY_N:-4}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-2}"
INITIAL_ROBOT_POSE="${INITIAL_ROBOT_POSE:-deployment_home}"
OBJECT_INIT_MODE="${OBJECT_INIT_MODE:-default}"
EXTRA_ARGS=()
if [[ "${DISABLE_ENV_RESETS:-1}" == "0" ]]; then EXTRA_ARGS+=(--no-disable_env_resets); fi
if [[ "${ZERO_TRAINING_RANDOMIZATION:-1}" == "0" ]]; then EXTRA_ARGS+=(--no-zero_training_randomization); fi
if [[ -n "${OBJECT_INIT_POSE_WXYZ:-}" ]]; then EXTRA_ARGS+=(--object_init_pose_wxyz ${OBJECT_INIT_POSE_WXYZ}); fi
if [[ -n "${OBJECT_INIT_POSITION_NOISE_M:-}" ]]; then EXTRA_ARGS+=(--object_init_position_noise_m ${OBJECT_INIT_POSITION_NOISE_M}); fi
if [[ -n "${OBJECT_INIT_YAW_NOISE_DEG:-}" ]]; then EXTRA_ARGS+=(--object_init_yaw_noise_deg "${OBJECT_INIT_YAW_NOISE_DEG}"); fi
if [[ -n "${OBJECT_INIT_ORIENTATION_MODE:-}" ]]; then EXTRA_ARGS+=(--object_init_orientation_mode "${OBJECT_INIT_ORIENTATION_MODE}"); fi

"${ISAACSIM_PYTHON}" deployment/isaac/isaac_depth_env_node.py \
  --enable_depth \
  --publish_depth \
  --depth_publish_every_n "${DEPTH_EVERY_N}" \
  --initial_robot_pose "${INITIAL_ROBOT_POSE}" \
  --object_init_mode "${OBJECT_INIT_MODE}" \
  --run_duration_s "${RUN_DURATION_S}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --realtime \
  --headless \
  "${EXTRA_ARGS[@]}"

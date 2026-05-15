#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/isaacsim_ros_env.sh"

RUN_DURATION_S="${RUN_DURATION_S:--1}"
DEPTH_EVERY_N="${DEPTH_EVERY_N:-4}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-2}"
INITIAL_ROBOT_POSE="${INITIAL_ROBOT_POSE:-deployment_home}"
OBJECT_INIT_MODE="${OBJECT_INIT_MODE:-default}"
PEG_URDF="${PEG_URDF:-assets/urdf/peg_in_hole/peg_L/peg_L.urdf}"
DEPTH_NOISE_PROFILE="${DEPTH_NOISE_PROFILE:-off}"
PUBLISHED_DEPTH_SOURCE="${PUBLISHED_DEPTH_SOURCE:-raw}"
CAMERA_POSE_RANDOMIZATION_PROFILE="${CAMERA_POSE_RANDOMIZATION_PROFILE:-off}"
CAMERA_POSE_RANDOMIZATION_MODE="${CAMERA_POSE_RANDOMIZATION_MODE:-startup}"
EXTRA_ARGS=()
if [[ "${DEPLOYMENT_MODE:-1}" == "0" ]]; then EXTRA_ARGS+=(--no-deployment_mode); fi
if [[ "${DISABLE_ENV_RESETS:-1}" == "0" ]]; then EXTRA_ARGS+=(--no-disable_env_resets); fi
if [[ "${ZERO_TRAINING_RANDOMIZATION:-1}" == "0" ]]; then EXTRA_ARGS+=(--no-zero_training_randomization); fi
if [[ -n "${OBJECT_INIT_POSE_WXYZ:-}" ]]; then EXTRA_ARGS+=(--object_init_pose_wxyz ${OBJECT_INIT_POSE_WXYZ}); fi
if [[ -n "${OBJECT_INIT_POSITION_NOISE_M:-}" ]]; then EXTRA_ARGS+=(--object_init_position_noise_m ${OBJECT_INIT_POSITION_NOISE_M}); fi
if [[ -n "${OBJECT_INIT_YAW_NOISE_DEG:-}" ]]; then EXTRA_ARGS+=(--object_init_yaw_noise_deg "${OBJECT_INIT_YAW_NOISE_DEG}"); fi
if [[ -n "${OBJECT_INIT_ORIENTATION_MODE:-}" ]]; then EXTRA_ARGS+=(--object_init_orientation_mode "${OBJECT_INIT_ORIENTATION_MODE}"); fi
if [[ -n "${DEPTH_NOISE_STRENGTH:-}" ]]; then EXTRA_ARGS+=(--depth_noise_strength "${DEPTH_NOISE_STRENGTH}"); fi
if [[ -n "${CAMERA_POS_NOISE_M:-}" ]]; then EXTRA_ARGS+=(--camera_pos_noise_m ${CAMERA_POS_NOISE_M}); fi
if [[ -n "${CAMERA_ROT_NOISE_DEG:-}" ]]; then EXTRA_ARGS+=(--camera_rot_noise_deg ${CAMERA_ROT_NOISE_DEG}); fi

"${ISAACSIM_PYTHON}" deployment/isaac/isaac_depth_env_node.py \
  --enable_depth \
  --publish_depth \
  --depth_publish_every_n "${DEPTH_EVERY_N}" \
  --depth_noise_profile "${DEPTH_NOISE_PROFILE}" \
  --published_depth_source "${PUBLISHED_DEPTH_SOURCE}" \
  --camera_pose_randomization_profile "${CAMERA_POSE_RANDOMIZATION_PROFILE}" \
  --camera_pose_randomization_mode "${CAMERA_POSE_RANDOMIZATION_MODE}" \
  --initial_robot_pose "${INITIAL_ROBOT_POSE}" \
  --object_init_mode "${OBJECT_INIT_MODE}" \
  --peg_urdf "${PEG_URDF}" \
  --run_duration_s "${RUN_DURATION_S}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --realtime \
  --headless \
  "${EXTRA_ARGS[@]}"

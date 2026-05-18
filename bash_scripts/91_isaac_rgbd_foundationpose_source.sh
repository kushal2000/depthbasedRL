#!/usr/bin/env bash
set -euo pipefail

# IsaacSim/IsaacLab RGB-D source for FoundationPose tests.
# Publishes RGB, metric depth, CameraInfo, joint states, and an Isaac GT debug
# pose. It deliberately does not publish /robot_frame/current_object_pose so
# FoundationPose can own that topic.

source "$(dirname "${BASH_SOURCE[0]}")/isaacsim_ros_env.sh"

RUN_DURATION_S="${RUN_DURATION_S:--1}"
DEPTH_EVERY_N="${DEPTH_EVERY_N:-1}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-2}"
INITIAL_ROBOT_POSE="${INITIAL_ROBOT_POSE:-deployment_home}"
OBJECT_INIT_MODE="${OBJECT_INIT_MODE:-default}"
PEG_URDF="${PEG_URDF:-assets/urdf/peg_in_hole/peg_L/peg_L.urdf}"
TASK="${TASK:-Isaacsimenvs-PegInHoleDepthStudent-Direct-v0}"
TEACHER_CONFIG="${TEACHER_CONFIG:-/juno/u/kedia/depthbasedRL/train_dir/Apr28/isaacSim_PegInHole/config.yaml}"
CAMERA_WIDTH="${CAMERA_WIDTH:-960}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-540}"
CAMERA_K_FILE="${CAMERA_K_FILE:-/juno/u/kedia/FoundationPose/human_videos/Jan_17/brush/red_brush/sweep_forward/cam_K.txt}"
CAMERA_INFO_MODE="${CAMERA_INFO_MODE:-centered}"
DEPTH_NOISE_PROFILE="${DEPTH_NOISE_PROFILE:-off}"
PUBLISHED_DEPTH_SOURCE="${PUBLISHED_DEPTH_SOURCE:-raw}"
CAMERA_POSE_RANDOMIZATION_PROFILE="${CAMERA_POSE_RANDOMIZATION_PROFILE:-off}"
CAMERA_POSE_RANDOMIZATION_MODE="${CAMERA_POSE_RANDOMIZATION_MODE:-startup}"

EXTRA_ARGS=()
if [[ -n "${PEG_URDF}" ]]; then EXTRA_ARGS+=(--peg_urdf "${PEG_URDF}"); fi
if [[ -n "${CAMERA_POS_NOISE_M:-}" ]]; then EXTRA_ARGS+=(--camera_pos_noise_m ${CAMERA_POS_NOISE_M}); fi
if [[ -n "${CAMERA_ROT_NOISE_DEG:-}" ]]; then EXTRA_ARGS+=(--camera_rot_noise_deg ${CAMERA_ROT_NOISE_DEG}); fi

"${ISAACSIM_PYTHON}" deployment/isaac/isaac_depth_env_node.py \
  --task "${TASK}" \
  --teacher_config "${TEACHER_CONFIG}" \
  --enable_depth \
  --publish_depth \
  --enable_rgb \
  --publish_rgb \
  --depth_publish_every_n "${DEPTH_EVERY_N}" \
  --depth_noise_profile "${DEPTH_NOISE_PROFILE}" \
  --published_depth_source "${PUBLISHED_DEPTH_SOURCE}" \
  --camera_pose_randomization_profile "${CAMERA_POSE_RANDOMIZATION_PROFILE}" \
  --camera_pose_randomization_mode "${CAMERA_POSE_RANDOMIZATION_MODE}" \
  --camera_image_width "${CAMERA_WIDTH}" \
  --camera_image_height "${CAMERA_HEIGHT}" \
  --camera_k_file "${CAMERA_K_FILE}" \
  --camera_info_mode "${CAMERA_INFO_MODE}" \
  --no-publish_object_pose \
  --publish_gt_object_pose_debug \
  --initial_robot_pose "${INITIAL_ROBOT_POSE}" \
  --object_init_mode "${OBJECT_INIT_MODE}" \
  --run_duration_s "${RUN_DURATION_S}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --realtime \
  --headless \
  "${EXTRA_ARGS[@]}"

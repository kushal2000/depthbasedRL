#!/usr/bin/env bash
set -euo pipefail

# Clean metric depth with stronger startup camera-pose randomization.
export DEPTH_EVERY_N="${DEPTH_EVERY_N:-2}"
export DEPTH_NOISE_PROFILE="${DEPTH_NOISE_PROFILE:-off}"
export PUBLISHED_DEPTH_SOURCE="${PUBLISHED_DEPTH_SOURCE:-raw}"
export CAMERA_POSE_RANDOMIZATION_PROFILE="${CAMERA_POSE_RANDOMIZATION_PROFILE:-custom}"
export CAMERA_POSE_RANDOMIZATION_MODE="${CAMERA_POSE_RANDOMIZATION_MODE:-startup}"
export CAMERA_POS_NOISE_M="${CAMERA_POS_NOISE_M:-0.04 0.04 0.04}"
export CAMERA_ROT_NOISE_DEG="${CAMERA_ROT_NOISE_DEG:-4 4 4}"

exec "$(dirname "${BASH_SOURCE[0]}")/40_isaac_depth_ros_node_render_every_4.sh"

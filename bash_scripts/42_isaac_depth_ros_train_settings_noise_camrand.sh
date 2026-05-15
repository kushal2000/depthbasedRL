#!/usr/bin/env bash
set -euo pipefail

# Deployment sim source matching the main training robustness setting:
# medium metric depth noise, 20 mm / 2 deg startup camera-pose randomization.
export DEPTH_EVERY_N="${DEPTH_EVERY_N:-2}"
export DEPTH_NOISE_PROFILE="${DEPTH_NOISE_PROFILE:-medium}"
export PUBLISHED_DEPTH_SOURCE="${PUBLISHED_DEPTH_SOURCE:-noisy}"
export CAMERA_POSE_RANDOMIZATION_PROFILE="${CAMERA_POSE_RANDOMIZATION_PROFILE:-custom}"
export CAMERA_POSE_RANDOMIZATION_MODE="${CAMERA_POSE_RANDOMIZATION_MODE:-startup}"
export CAMERA_POS_NOISE_M="${CAMERA_POS_NOISE_M:-0.02 0.02 0.02}"
export CAMERA_ROT_NOISE_DEG="${CAMERA_ROT_NOISE_DEG:-2 2 2}"

exec "$(dirname "${BASH_SOURCE[0]}")/40_isaac_depth_ros_node_render_every_4.sh"

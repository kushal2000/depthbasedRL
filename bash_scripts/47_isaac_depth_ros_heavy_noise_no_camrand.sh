#!/usr/bin/env bash
set -euo pipefail

# Heavy/obvious metric depth noise with fixed nominal camera pose.
export DEPTH_EVERY_N="${DEPTH_EVERY_N:-2}"
export DEPTH_NOISE_PROFILE="${DEPTH_NOISE_PROFILE:-strong}"
export PUBLISHED_DEPTH_SOURCE="${PUBLISHED_DEPTH_SOURCE:-noisy}"
export CAMERA_POSE_RANDOMIZATION_PROFILE="${CAMERA_POSE_RANDOMIZATION_PROFILE:-off}"
export CAMERA_POSE_RANDOMIZATION_MODE="${CAMERA_POSE_RANDOMIZATION_MODE:-startup}"

exec "$(dirname "${BASH_SOURCE[0]}")/40_isaac_depth_ros_node_render_every_4.sh"

#!/usr/bin/env bash
set -euo pipefail

# Clean raw metric depth, fixed nominal camera, and no visible/collidable hole
# fixture. The wooden table and peg remain in the scene. Use this to test what
# a depth student does when the target fixture is absent from the camera image.

export DEPTH_EVERY_N="${DEPTH_EVERY_N:-2}"
export DEPTH_NOISE_PROFILE="${DEPTH_NOISE_PROFILE:-off}"
export PUBLISHED_DEPTH_SOURCE="${PUBLISHED_DEPTH_SOURCE:-raw}"
export CAMERA_POSE_RANDOMIZATION_PROFILE="${CAMERA_POSE_RANDOMIZATION_PROFILE:-off}"
export CAMERA_POSE_RANDOMIZATION_MODE="${CAMERA_POSE_RANDOMIZATION_MODE:-startup}"
export HIDE_HOLE_FIXTURE=1

exec "$(dirname "${BASH_SOURCE[0]}")/40_isaac_depth_ros_node_render_every_4.sh"

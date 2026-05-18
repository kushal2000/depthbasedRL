#!/usr/bin/env bash
set -euo pipefail

# IsaacSim/IsaacLab source for teacher-policy baseline tests.
# Publishes joint states and Isaac ground-truth object pose on
# /robot_frame/current_object_pose, but does not render RGB or depth.

source "$(dirname "${BASH_SOURCE[0]}")/isaacsim_ros_env.sh"

RUN_DURATION_S="${RUN_DURATION_S:--1}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-2}"
INITIAL_ROBOT_POSE="${INITIAL_ROBOT_POSE:-deployment_home}"
OBJECT_INIT_MODE="${OBJECT_INIT_MODE:-default}"
PEG_URDF="${PEG_URDF:-assets/urdf/peg_in_hole/peg_L/peg_L.urdf}"
TASK="${TASK:-Isaacsimenvs-PegInHoleDepthStudent-Direct-v0}"
TEACHER_CONFIG="${TEACHER_CONFIG:-/juno/u/kedia/depthbasedRL/train_dir/Apr28/isaacSim_PegInHole/config.yaml}"

EXTRA_ARGS=()
if [[ -n "${PEG_URDF}" ]]; then EXTRA_ARGS+=(--peg_urdf "${PEG_URDF}"); fi

"${ISAACSIM_PYTHON}" deployment/isaac/isaac_depth_env_node.py \
  --task "${TASK}" \
  --teacher_config "${TEACHER_CONFIG}" \
  --no-enable_depth \
  --no-publish_depth \
  --no-enable_rgb \
  --no-publish_rgb \
  --publish_object_pose \
  --no-publish_gt_object_pose_debug \
  --initial_robot_pose "${INITIAL_ROBOT_POSE}" \
  --object_init_mode "${OBJECT_INIT_MODE}" \
  --run_duration_s "${RUN_DURATION_S}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --realtime \
  --headless \
  "${EXTRA_ARGS[@]}"

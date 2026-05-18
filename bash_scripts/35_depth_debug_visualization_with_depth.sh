#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/depth_deploy_debug_env.sh"

OBJECT_NAME="${OBJECT_NAME:-peg_L}"
LOAD_POINT_CLOUD="${LOAD_POINT_CLOUD:-0}"
LOAD_POLICY_DEPTH_IMAGE="${LOAD_POLICY_DEPTH_IMAGE:-1}"
PREDICTED_OBJECT_POSE_TOPIC="${PREDICTED_OBJECT_POSE_TOPIC:-/robot_frame/predicted_object_pose}"
EXTRA_ARGS=()
if [[ "${LOAD_POINT_CLOUD}" == "1" ]]; then
  EXTRA_ARGS+=(--load-point-cloud)
fi
if [[ "${LOAD_POLICY_DEPTH_IMAGE}" == "1" ]]; then
  EXTRA_ARGS+=(--load-policy-depth-image)
fi

python deployment/visualization_node.py \
  --object-name "${OBJECT_NAME}" \
  --predicted-object-pose-topic "${PREDICTED_OBJECT_POSE_TOPIC}" \
  --load-depth-image \
  "${EXTRA_ARGS[@]}"

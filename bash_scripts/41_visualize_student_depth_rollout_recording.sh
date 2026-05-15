#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/depth_deploy_debug_env.sh"

DEFAULT_RECORDING="student_depth_ros_topic_publish_recording/2026-05-15_00-16-24_student_depth_rollout.npz"
RECORDING="${1:-${RECORDING:-${DEFAULT_RECORDING}}}"
OBJECT_NAME="${OBJECT_NAME:-peg_L}"
EXTRA_ARGS=()
if [[ "${FLIP_DEPTH_IMAGE_Y:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--flip-depth-image-y)
fi

python deployment/visualize_student_depth_rollout.py \
  --recording "${RECORDING}" \
  --object-name "${OBJECT_NAME}" \
  "${EXTRA_ARGS[@]}"

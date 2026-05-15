#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/depth_deploy_debug_env.sh"

OBJECT_NAME="${OBJECT_NAME:-peg_L}"
LOAD_POINT_CLOUD="${LOAD_POINT_CLOUD:-0}"
EXTRA_ARGS=()
if [[ "${LOAD_POINT_CLOUD}" == "1" ]]; then
  EXTRA_ARGS+=(--load-point-cloud)
fi

python deployment/visualization_node.py \
  --object-name "${OBJECT_NAME}" \
  --load-depth-image \
  "${EXTRA_ARGS[@]}"

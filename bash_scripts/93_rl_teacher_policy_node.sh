#!/usr/bin/env bash
set -euo pipefail

# Generic teacher policy node entrypoint. This consumes /robot_frame/current_object_pose
# from either Isaac GT pose-only mode or FoundationPose.

source "$(dirname "${BASH_SOURCE[0]}")/depth_deploy_debug_env.sh"

POLICY_PATH="${POLICY_PATH:-pretrained_policy}"
OBJECT_NAME="${OBJECT_NAME:-claw_hammer}"

python deployment/rl_policy_node.py \
  --policy-path "${POLICY_PATH}" \
  --object-name "${OBJECT_NAME}"

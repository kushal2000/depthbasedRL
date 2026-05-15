#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/depth_deploy_debug_env.sh"

RATE_HZ="${RATE_HZ:-60}"
RUN_DURATION_S="${RUN_DURATION_S:--1}"

python deployment/fake/fake_robot_node.py \
  --rate_hz "${RATE_HZ}" \
  --run_duration_s "${RUN_DURATION_S}" \
  --initial_pose student_default \
  --no-wait_for_commands

#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/isaacsim_ros_env.sh"

RUN_DURATION_S="${RUN_DURATION_S:--1}"
DEPTH_EVERY_N="${DEPTH_EVERY_N:-4}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-2}"

"${ISAACSIM_PYTHON}" deployment/isaac/isaac_depth_env_node.py \
  --enable_depth \
  --publish_depth \
  --depth_publish_every_n "${DEPTH_EVERY_N}" \
  --run_duration_s "${RUN_DURATION_S}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --realtime \
  --headless

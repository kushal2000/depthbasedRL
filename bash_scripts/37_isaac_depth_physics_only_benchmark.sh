#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/isaacsim_ros_env.sh"

NUM_STEPS="${NUM_STEPS:-600}"
BENCHMARK_WARMUP_STEPS="${BENCHMARK_WARMUP_STEPS:-10}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-2}"

"${ISAACSIM_PYTHON}" deployment/isaac/isaac_depth_env_node.py \
  --benchmark \
  --no-enable_depth \
  --num_steps "${NUM_STEPS}" \
  --benchmark_warmup_steps "${BENCHMARK_WARMUP_STEPS}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --headless

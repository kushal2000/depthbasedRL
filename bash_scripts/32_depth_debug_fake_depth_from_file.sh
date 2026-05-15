#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/depth_deploy_debug_env.sh"

DEPTH_PATH="${1:-${DEPTH_PATH:-}}"
RATE_HZ="${RATE_HZ:-30}"
RUN_DURATION_S="${RUN_DURATION_S:--1}"

if [[ -z "${DEPTH_PATH}" || ! -f "${DEPTH_PATH}" ]]; then
  echo "Usage: $0 /path/to/depth.npz|npy|png|mp4" >&2
  echo "Or set DEPTH_PATH=/path/to/depth_file." >&2
  exit 1
fi

python deployment/fake/fake_depth_image_node.py \
  --depth_path "${DEPTH_PATH}" \
  --rate_hz "${RATE_HZ}" \
  --run_duration_s "${RUN_DURATION_S}" \
  --publish_camera_info

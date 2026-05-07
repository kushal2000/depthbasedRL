#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SAVE_DIR="${SAVE_DIR:-./zed_nonblocking_debug}"

python deployment/test_zed_nonblocking.py \
  --duration_s 10 \
  --consumer_hz 60 \
  --producer_preprocess policy \
  --save_dir "${SAVE_DIR}"

echo "ZED debug files: ${SAVE_DIR}"

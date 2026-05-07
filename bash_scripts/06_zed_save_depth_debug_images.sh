#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python deployment/test_zed_nonblocking.py \
  --duration_s 10 \
  --consumer_hz 60 \
  --producer_preprocess policy \
  --save_dir /tmp/zed_nonblocking_debug

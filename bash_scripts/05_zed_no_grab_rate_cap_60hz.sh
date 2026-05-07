#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python deployment/test_zed_nonblocking.py \
  --duration_s 30 \
  --consumer_hz 60 \
  --zed_grab_hz 0 \
  --producer_preprocess none \
  --consumer_preprocess none

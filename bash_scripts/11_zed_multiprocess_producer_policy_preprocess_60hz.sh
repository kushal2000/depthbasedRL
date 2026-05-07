#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python deployment/test_zed_multiprocess.py \
  --duration_s 30 \
  --consumer_hz 60 \
  --producer_preprocess policy

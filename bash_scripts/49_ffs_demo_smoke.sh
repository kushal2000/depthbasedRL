#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/ffs_isaacsim_ros_env.sh"

OUT_DIR="${OUT_DIR:-local_logs/ffs_demo_smoke}"
VALID_ITERS="${VALID_ITERS:-4}"
MAX_DISP="${MAX_DISP:-192}"

"${ISAACSIM_PYTHON}" "${FAST_FOUNDATIONSTEREO_ROOT}/scripts/run_demo.py" \
  --model_dir "${FFS_MODEL_PATH}" \
  --left_file "${FAST_FOUNDATIONSTEREO_ROOT}/demo_data/left.png" \
  --right_file "${FAST_FOUNDATIONSTEREO_ROOT}/demo_data/right.png" \
  --intrinsic_file "${FAST_FOUNDATIONSTEREO_ROOT}/demo_data/K.txt" \
  --out_dir "${OUT_DIR}" \
  --remove_invisible 0 \
  --denoise_cloud 0 \
  --scale 1 \
  --get_pc 0 \
  --vis 0 \
  --valid_iters "${VALID_ITERS}" \
  --max_disp "${MAX_DISP}" \
  --zfar 10

echo "[49_ffs_demo_smoke] wrote ${OUT_DIR}"

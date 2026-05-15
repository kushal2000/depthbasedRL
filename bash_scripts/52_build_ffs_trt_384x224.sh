#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/ffs_isaacsim_ros_env.sh"

MODEL_NAME="${MODEL_NAME:-23-36-37}"
FFS_VALID_ITERS="${FFS_VALID_ITERS:-4}"
FFS_MAX_DISP="${FFS_MAX_DISP:-192}"
FFS_STEREO_WIDTH="${FFS_STEREO_WIDTH:-384}"
FFS_STEREO_HEIGHT="${FFS_STEREO_HEIGHT:-224}"
ONNX_DIR="${ONNX_DIR:-${FAST_FOUNDATIONSTEREO_ROOT}/weights/${MODEL_NAME}/onnx_${FFS_STEREO_WIDTH}x${FFS_STEREO_HEIGHT}_iters${FFS_VALID_ITERS}}"
MODEL_PATH="${MODEL_PATH:-${FAST_FOUNDATIONSTEREO_ROOT}/weights/${MODEL_NAME}/model_best_bp2_serialize.pth}"

mkdir -p "${ONNX_DIR}"

# Preload cv2 before Fast-FoundationStereo imports timm/torchvision/PIL. In this
# mixed IsaacSim env, importing those first can load an incompatible libjpeg
# symbol path and make cv2 fail during ONNX export.
python - <<PY
import cv2  # noqa: F401
import runpy
import sys

sys.argv = [
    "${FAST_FOUNDATIONSTEREO_ROOT}/scripts/make_onnx.py",
    "--model_dir", "${MODEL_PATH}",
    "--save_path", "${ONNX_DIR}",
    "--height", "${FFS_STEREO_HEIGHT}",
    "--width", "${FFS_STEREO_WIDTH}",
    "--valid_iters", "${FFS_VALID_ITERS}",
    "--max_disp", "${FFS_MAX_DISP}",
]
runpy.run_path(sys.argv[0], run_name="__main__")
PY

python deployment/build_trt_engine.py --onnx_dir "${ONNX_DIR}"

echo "FFS_ENGINE_DIR=${ONNX_DIR}"

#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/ffs_isaacsim_ros_env.sh"

"${ISAACSIM_PYTHON}" - <<'PY'
import sys

print("python", sys.executable)
for name in ("torch", "rospy", "cv2", "numpy", "isaaclab", "isaacsim", "timm", "skimage", "open3d"):
    mod = __import__(name)
    print(name, getattr(mod, "__version__", "ok"))

from deployment.isaac.fast_foundation_stereo_backend import FastFoundationStereoDepth

print("FFS wrapper import ok:", FastFoundationStereoDepth.__name__)
PY

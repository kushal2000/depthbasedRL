#!/usr/bin/env bash
set -euo pipefail

# FoundationPose tracker that consumes RGB-D ROS topics and publishes
# /robot_frame/current_object_pose for rl_policy_node.py.

DEPTHBASED_RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${DEPTHBASED_RL_ROOT}/bash_scripts/foundationpose_ros_env.sh"

MESH_PATH="${MESH_PATH:-${DEPTHBASED_RL_ROOT}/assets/urdf/peg_in_hole/peg_L/peg_L.obj}"
CALIBRATION_FILE="${CALIBRATION_FILE:-${FOUNDATIONPOSE_ROOT}/calibration/T_RC_example.txt}"
CAM_K_FILE="${CAM_K_FILE:-/juno/u/kedia/FoundationPose/human_videos/Jan_17/brush/red_brush/sweep_forward/cam_K.txt}"
RGB_TOPIC="${RGB_TOPIC:-/zed/zed_node/rgb/image_rect_color}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/zed/zed_node/depth/depth_registered}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/zed/zed_node/rgb/camera_info}"
FPS="${FPS:-30}"
# Keep debug visualization off by default for closed-loop control latency.
# Override with DEBUG=1 when you explicitly want the OpenCV overlay window.
DEBUG="${DEBUG:-0}"
EST_REFINE_ITER="${EST_REFINE_ITER:-5}"
TRACK_REFINE_ITER="${TRACK_REFINE_ITER:-2}"
SAVE_DIR="${SAVE_DIR:-${DEPTHBASED_RL_ROOT}/local_logs/foundationpose_ros_topic}"

python live_tracking_from_ros_topics.py \
  --mesh_path "${MESH_PATH}" \
  --calibration "${CALIBRATION_FILE}" \
  --cam_K "${CAM_K_FILE}" \
  --rgb_topic "${RGB_TOPIC}" \
  --depth_topic "${DEPTH_TOPIC}" \
  --camera_info_topic "${CAMERA_INFO_TOPIC}" \
  --fps "${FPS}" \
  --debug "${DEBUG}" \
  --est_refine_iter "${EST_REFINE_ITER}" \
  --track_refine_iter "${TRACK_REFINE_ITER}" \
  --save_dir "${SAVE_DIR}"

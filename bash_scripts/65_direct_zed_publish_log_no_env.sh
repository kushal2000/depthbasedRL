#!/usr/bin/env bash
set -euo pipefail

# Same policy run as 64_ws16_direct_zed_publish_log.sh, but deliberately does
# not activate conda or set ROS_MASTER_URI/ROS_IP/LD_LIBRARY_PATH. Use this
# when the caller's shell already has the correct ROS + ZED + Python env.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEFAULT_CHECKPOINT="/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
if [[ ! -f "${DEFAULT_CHECKPOINT}" ]]; then
  DEFAULT_CHECKPOINT="distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
fi
CHECKPOINT="${1:-${DEFAULT_STUDENT_CHECKPOINT:-${DEFAULT_CHECKPOINT}}}"

RUN_DURATION_S="${RUN_DURATION_S:-3}"
PUBLISH_DURATION_S="${PUBLISH_DURATION_S:-${RUN_DURATION_S}}"
RECORD_DIR="${RECORD_DIR:-./student_depth_zed_direct_publish_recording}"
RECORD_DEPTH_FORMAT="${RECORD_DEPTH_FORMAT:-uint8}"
RECORD_DEPTH_VIDEO_FPS="${RECORD_DEPTH_VIDEO_FPS:-30}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-0.5}"
CONTROL_HZ="${CONTROL_HZ:-60}"
WARMUP_STEPS="${WARMUP_STEPS:-30}"
ZED_CAMERA_FPS="${ZED_CAMERA_FPS:-30}"
ZED_RETRIEVE_WIDTH="${ZED_RETRIEVE_WIDTH:-160}"
ZED_RETRIEVE_HEIGHT="${ZED_RETRIEVE_HEIGHT:-90}"
ZED_MAX_CACHED_DEPTH_AGE_S="${ZED_MAX_CACHED_DEPTH_AGE_S:-0.20}"
MAX_ARM_TARGET_DELTA_DEG="${MAX_ARM_TARGET_DELTA_DEG:-0}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

cat <<EOF
[direct_zed_publish_log_no_env]
repo=$(pwd)
python=$(command -v python)
ROS_MASTER_URI=${ROS_MASTER_URI:-<unset>}
ROS_IP=${ROS_IP:-<unset>}
ROS_HOSTNAME=${ROS_HOSTNAME:-<unset>}
checkpoint=${CHECKPOINT}
run_duration_s=${RUN_DURATION_S}
publish_duration_s=${PUBLISH_DURATION_S}
record_dir=${RECORD_DIR}
record_depth_video_fps=${RECORD_DEPTH_VIDEO_FPS}

This script does not activate conda or set ROS networking variables.
Depth source is direct ZED SDK capture, not a ROS depth topic.
Joint command publishing is ENABLED for publish_duration_s.
EOF

python deployment/student_depth_policy_node_nonblocking.py \
  --checkpoint_path "${CHECKPOINT}" \
  --depth_source zed_sdk \
  --zed_nonblocking \
  --zed_camera_fps "${ZED_CAMERA_FPS}" \
  --zed_retrieve_width "${ZED_RETRIEVE_WIDTH}" \
  --zed_retrieve_height "${ZED_RETRIEVE_HEIGHT}" \
  --zed_max_cached_depth_age_s "${ZED_MAX_CACHED_DEPTH_AGE_S}" \
  --control_hz "${CONTROL_HZ}" \
  --run_duration_s "${RUN_DURATION_S}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --publish_object_pose \
  --publish_joint_commands \
  --publish_joint_commands_duration_s "${PUBLISH_DURATION_S}" \
  --max_arm_target_delta_deg "${MAX_ARM_TARGET_DELTA_DEG}" \
  --record_rollout_dir "${RECORD_DIR}" \
  --record_depth_format "${RECORD_DEPTH_FORMAT}" \
  --record_depth_video_fps "${RECORD_DEPTH_VIDEO_FPS}"

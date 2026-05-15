#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/ffs_isaacsim_ros_env.sh"

RUN_DURATION_S="${RUN_DURATION_S:--1}"
DEPTH_EVERY_N="${DEPTH_EVERY_N:-2}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-2}"
INITIAL_ROBOT_POSE="${INITIAL_ROBOT_POSE:-deployment_home}"
OBJECT_INIT_MODE="${OBJECT_INIT_MODE:-default}"
PEG_URDF="${PEG_URDF:-assets/urdf/peg_in_hole/peg_L/peg_L.urdf}"
FFS_BASELINE_M="${FFS_BASELINE_M:-0.12}"
FFS_VALID_ITERS="${FFS_VALID_ITERS:-4}"
FFS_MAX_DISP="${FFS_MAX_DISP:-192}"
FFS_ENGINE_DIR="${FFS_ENGINE_DIR:-}"
FFS_STEREO_WIDTH="${FFS_STEREO_WIDTH:-384}"
FFS_STEREO_HEIGHT="${FFS_STEREO_HEIGHT:-224}"
FFS_PUBLISH_WIDTH="${FFS_PUBLISH_WIDTH:-160}"
FFS_PUBLISH_HEIGHT="${FFS_PUBLISH_HEIGHT:-90}"
FFS_DOWNSAMPLE_TO_POLICY_RES="${FFS_DOWNSAMPLE_TO_POLICY_RES:-1}"
FFS_RENDER_QUALITY="${FFS_RENDER_QUALITY:-1}"
FFS_SAMPLES_PER_PIXEL="${FFS_SAMPLES_PER_PIXEL:-16}"
FFS_DEBUG_DIR="${FFS_DEBUG_DIR:-}"
FFS_DEBUG_EVERY_N="${FFS_DEBUG_EVERY_N:-30}"
DEPTH_COMPARE_DIR="${DEPTH_COMPARE_DIR:-}"
DEPTH_COMPARE_EVERY_N="${DEPTH_COMPARE_EVERY_N:-30}"
CAMERA_POSE_RANDOMIZATION_PROFILE="${CAMERA_POSE_RANDOMIZATION_PROFILE:-off}"
CAMERA_POSE_RANDOMIZATION_MODE="${CAMERA_POSE_RANDOMIZATION_MODE:-startup}"

EXTRA_ARGS=()
if [[ "${DEPLOYMENT_MODE:-1}" == "0" ]]; then EXTRA_ARGS+=(--no-deployment_mode); fi
if [[ "${DISABLE_ENV_RESETS:-1}" == "0" ]]; then EXTRA_ARGS+=(--no-disable_env_resets); fi
if [[ "${ZERO_TRAINING_RANDOMIZATION:-1}" == "0" ]]; then EXTRA_ARGS+=(--no-zero_training_randomization); fi
if [[ -n "${OBJECT_INIT_POSE_WXYZ:-}" ]]; then EXTRA_ARGS+=(--object_init_pose_wxyz ${OBJECT_INIT_POSE_WXYZ}); fi
if [[ -n "${OBJECT_INIT_POSITION_NOISE_M:-}" ]]; then EXTRA_ARGS+=(--object_init_position_noise_m ${OBJECT_INIT_POSITION_NOISE_M}); fi
if [[ -n "${OBJECT_INIT_YAW_NOISE_DEG:-}" ]]; then EXTRA_ARGS+=(--object_init_yaw_noise_deg "${OBJECT_INIT_YAW_NOISE_DEG}"); fi
if [[ -n "${OBJECT_INIT_ORIENTATION_MODE:-}" ]]; then EXTRA_ARGS+=(--object_init_orientation_mode "${OBJECT_INIT_ORIENTATION_MODE}"); fi
if [[ -n "${CAMERA_POS_NOISE_M:-}" ]]; then EXTRA_ARGS+=(--camera_pos_noise_m ${CAMERA_POS_NOISE_M}); fi
if [[ -n "${CAMERA_ROT_NOISE_DEG:-}" ]]; then EXTRA_ARGS+=(--camera_rot_noise_deg ${CAMERA_ROT_NOISE_DEG}); fi
if [[ -n "${FFS_DEBUG_DIR}" ]]; then EXTRA_ARGS+=(--ffs_debug_dir "${FFS_DEBUG_DIR}"); fi
if [[ -n "${FFS_ENGINE_DIR}" ]]; then EXTRA_ARGS+=(--ffs_engine_dir "${FFS_ENGINE_DIR}"); fi
if [[ "${FFS_DOWNSAMPLE_TO_POLICY_RES}" == "0" ]]; then EXTRA_ARGS+=(--no-ffs_downsample_to_policy_res); fi
if [[ "${FFS_RENDER_QUALITY}" == "0" ]]; then EXTRA_ARGS+=(--no-ffs_render_quality); fi
if [[ -n "${DEPTH_COMPARE_DIR}" ]]; then EXTRA_ARGS+=(--depth_compare_dir "${DEPTH_COMPARE_DIR}" --depth_compare_every_n "${DEPTH_COMPARE_EVERY_N}"); fi

"${ISAACSIM_PYTHON}" deployment/isaac/isaac_depth_env_node.py \
  --enable_depth \
  --publish_depth \
  --depth_render_backend ffs_stereo \
  --depth_publish_every_n "${DEPTH_EVERY_N}" \
  --camera_pose_randomization_profile "${CAMERA_POSE_RANDOMIZATION_PROFILE}" \
  --camera_pose_randomization_mode "${CAMERA_POSE_RANDOMIZATION_MODE}" \
  --initial_robot_pose "${INITIAL_ROBOT_POSE}" \
  --object_init_mode "${OBJECT_INIT_MODE}" \
  --peg_urdf "${PEG_URDF}" \
  --run_duration_s "${RUN_DURATION_S}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --ffs_repo_root "${FAST_FOUNDATIONSTEREO_ROOT}" \
  --ffs_model_path "${FFS_MODEL_PATH}" \
  --ffs_stereo_width "${FFS_STEREO_WIDTH}" \
  --ffs_stereo_height "${FFS_STEREO_HEIGHT}" \
  --ffs_publish_width "${FFS_PUBLISH_WIDTH}" \
  --ffs_publish_height "${FFS_PUBLISH_HEIGHT}" \
  --ffs_samples_per_pixel "${FFS_SAMPLES_PER_PIXEL}" \
  --ffs_baseline_m "${FFS_BASELINE_M}" \
  --ffs_valid_iters "${FFS_VALID_ITERS}" \
  --ffs_max_disp "${FFS_MAX_DISP}" \
  --ffs_debug_every_n "${FFS_DEBUG_EVERY_N}" \
  --realtime \
  --headless \
  "${EXTRA_ARGS[@]}"

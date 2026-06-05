#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv-isaacsim-py311/bin/python}"

CHECKPOINT="${CHECKPOINT:-/juno/u/kedia/depthbasedRL/train_dir/May26/screwing_newer/model.pth}"
ROBOT_URDF="${ROBOT_URDF:-/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted_pretty.urdf}"
OUT_DIR="${OUT_DIR:-local_logs/furniturebench_200mm_finetuned}"

NUM_ENVS="${NUM_ENVS:-1}"
STEPS="${STEPS:-1200}"
SEED="${SEED:-0}"
NO_RENDER="${NO_RENDER:-0}"
MAKE_VIDEO="${MAKE_VIDEO:-1}"
CAPTURE_PNG_STEPS="${CAPTURE_PNG_STEPS:-0,300,600,900,1200}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"
VIDEO_FPS="${VIDEO_FPS:-30}"
HEADLESS="${HEADLESS:-1}"
HOLD_OPEN_S="${HOLD_OPEN_S:-0}"
RENDER_QUALITY_PRESET="${RENDER_QUALITY_PRESET:-beauty}"
RENDER_MODE="${RENDER_MODE:-rt}"
RENDER_SPP="${RENDER_SPP:-64}"
CAMERA_EYE="${CAMERA_EYE:-0.12 -1.55 0.98}"
CAMERA_TARGET="${CAMERA_TARGET:-0.02 0.02 0.58}"
CAMERA_XYZ="${CAMERA_XYZ:-0.0 -0.4115866854641337 0.7392877590177354}"
CAMERA_WXYZ="${CAMERA_WXYZ:-0.6051540840361335 -0.7945251569598455 -0.014533776794422243 0.04803206079704674}"
CAMERA_FORWARD_WORLD="${CAMERA_FORWARD_WORLD:-0.0 0.9644869986422913 -0.26413032663816677}"
CAMERA_FORWARD_AXIS="${CAMERA_FORWARD_AXIS:-0 0 1}"
CAMERA_TARGET_DISTANCE_M="${CAMERA_TARGET_DISTANCE_M:-1.4}"
CAMERA_FOCAL_LENGTH_CM="${CAMERA_FOCAL_LENGTH_CM:-10.0}"
CAMERA_FOCUS_DISTANCE_M="${CAMERA_FOCUS_DISTANCE_M:-0.8}"
DEFAULT_LIGHT_INTENSITY="${DEFAULT_LIGHT_INTENSITY:-360}"
SKY_DOME_INTENSITY="${SKY_DOME_INTENSITY:-1200}"
SINGLE_SUN_EXPOSURE="${SINGLE_SUN_EXPOSURE:-9.35}"
SINGLE_SUN_ANGLE="${SINGLE_SUN_ANGLE:-0.12}"
IMAGE_EXPOSURE="${IMAGE_EXPOSURE:-0.0}"
IMAGE_CONTRAST="${IMAGE_CONTRAST:-1.0}"
IMAGE_SATURATION="${IMAGE_SATURATION:-1.0}"
TABLE_COLOR="${TABLE_COLOR:-0.42 0.27 0.15}"
FLOOR_COLOR="${FLOOR_COLOR:-0.44 0.45 0.43}"
BACKDROP_STYLE="${BACKDROP_STYLE:-gradient_sky}"
BACKDROP_COLOR_R="${BACKDROP_COLOR_R:-0.25}"
BACKDROP_COLOR_G="${BACKDROP_COLOR_G:-0.48}"
BACKDROP_COLOR_B="${BACKDROP_COLOR_B:-0.76}"
BACKDROP_HORIZON_COLOR_R="${BACKDROP_HORIZON_COLOR_R:-0.68}"
BACKDROP_HORIZON_COLOR_G="${BACKDROP_HORIZON_COLOR_G:-0.78}"
BACKDROP_HORIZON_COLOR_B="${BACKDROP_HORIZON_COLOR_B:-0.88}"
BACKDROP_DISTANCE="${BACKDROP_DISTANCE:-5.0}"
BACKDROP_EXTENT_MARGIN="${BACKDROP_EXTENT_MARGIN:-80.0}"
RESET_POSITION_CENTER_X="${RESET_POSITION_CENTER_X:-0.075}"
RESET_POSITION_CENTER_Y="${RESET_POSITION_CENTER_Y:-0.07}"
RESET_POSITION_NOISE_X="${RESET_POSITION_NOISE_X:-0.025}"
RESET_POSITION_NOISE_Y="${RESET_POSITION_NOISE_Y:-0.015}"
RESET_POSITION_NOISE_Z="${RESET_POSITION_NOISE_Z:-0.005}"
RESET_ORIENTATION_MODE="${RESET_ORIENTATION_MODE:-yaw}"
RESET_ORIENTATION_YAW_RANGE_DEG="${RESET_ORIENTATION_YAW_RANGE_DEG:-20.0}"
RESET_ORIENTATION_AXIS_ANGLE_RANGE_DEG="${RESET_ORIENTATION_AXIS_ANGLE_RANGE_DEG:-0.0}"
HOLE_X_RANGE="${HOLE_X_RANGE:--0.085 -0.055}"
HOLE_Y_RANGE="${HOLE_Y_RANGE:--0.075 -0.055}"
HOLE_YAW_RANGE_DEG="${HOLE_YAW_RANGE_DEG:-3.0}"
GOAL_XY_OBS_NOISE="${GOAL_XY_OBS_NOISE:-0.002}"
GOAL_YAW_OBS_NOISE_DEG="${GOAL_YAW_OBS_NOISE_DEG:-1.0}"

# Default to screwing-only episodes for a single-env video. Set to 0.1 to match
# the exact finetune launcher distribution including random goal envs.
RANDOM_GOAL_FRACTION="${RANDOM_GOAL_FRACTION:-0.0}"
TRAIN_DR="${TRAIN_DR:-1}"

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --robot_urdf "${ROBOT_URDF}"
  --out_dir "${OUT_DIR}"
  --num_envs "${NUM_ENVS}"
  --steps "${STEPS}"
  --seed "${SEED}"
  --capture_png_steps "${CAPTURE_PNG_STEPS}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --video_fps "${VIDEO_FPS}"
  --render_quality_preset "${RENDER_QUALITY_PRESET}"
  --render_mode "${RENDER_MODE}"
  --render_samples_per_pixel "${RENDER_SPP}"
  --random_goal_fraction "${RANDOM_GOAL_FRACTION}"
  --hold_open_s "${HOLD_OPEN_S}"
  --camera_eye ${CAMERA_EYE}
  --camera_target ${CAMERA_TARGET}
  --camera_xyz ${CAMERA_XYZ}
  --camera_wxyz ${CAMERA_WXYZ}
  --camera_forward_world ${CAMERA_FORWARD_WORLD}
  --camera_forward_axis ${CAMERA_FORWARD_AXIS}
  --camera_target_distance_m "${CAMERA_TARGET_DISTANCE_M}"
  --camera_focal_length_cm "${CAMERA_FOCAL_LENGTH_CM}"
  --camera_focus_distance_m "${CAMERA_FOCUS_DISTANCE_M}"
  --default_light_intensity "${DEFAULT_LIGHT_INTENSITY}"
  --sky_dome_intensity "${SKY_DOME_INTENSITY}"
  --single_sun_exposure "${SINGLE_SUN_EXPOSURE}"
  --single_sun_angle "${SINGLE_SUN_ANGLE}"
  --image_exposure "${IMAGE_EXPOSURE}"
  --image_contrast "${IMAGE_CONTRAST}"
  --image_saturation "${IMAGE_SATURATION}"
  --table_color ${TABLE_COLOR}
  --floor_color ${FLOOR_COLOR}
  --backdrop_style "${BACKDROP_STYLE}"
  --backdrop_color "${BACKDROP_COLOR_R}" "${BACKDROP_COLOR_G}" "${BACKDROP_COLOR_B}"
  --backdrop_horizon_color "${BACKDROP_HORIZON_COLOR_R}" "${BACKDROP_HORIZON_COLOR_G}" "${BACKDROP_HORIZON_COLOR_B}"
  --backdrop_distance "${BACKDROP_DISTANCE}"
  --backdrop_extent_margin "${BACKDROP_EXTENT_MARGIN}"
  --reset_position_center_x "${RESET_POSITION_CENTER_X}"
  --reset_position_center_y "${RESET_POSITION_CENTER_Y}"
  --reset_position_noise_x "${RESET_POSITION_NOISE_X}"
  --reset_position_noise_y "${RESET_POSITION_NOISE_Y}"
  --reset_position_noise_z "${RESET_POSITION_NOISE_Z}"
  --reset_orientation_mode "${RESET_ORIENTATION_MODE}"
  --reset_orientation_yaw_range_deg "${RESET_ORIENTATION_YAW_RANGE_DEG}"
  --reset_orientation_axis_angle_range_deg "${RESET_ORIENTATION_AXIS_ANGLE_RANGE_DEG}"
  --hole_x_range ${HOLE_X_RANGE}
  --hole_y_range ${HOLE_Y_RANGE}
  --hole_yaw_range_deg "${HOLE_YAW_RANGE_DEG}"
  --goal_xy_obs_noise "${GOAL_XY_OBS_NOISE}"
  --goal_yaw_obs_noise_deg "${GOAL_YAW_OBS_NOISE_DEG}"
)

if [[ "${NO_RENDER}" == "1" || "${NO_RENDER}" == "true" || "${NO_RENDER}" == "True" ]]; then
  ARGS+=(--no_render)
fi
if [[ "${MAKE_VIDEO}" == "1" || "${MAKE_VIDEO}" == "true" || "${MAKE_VIDEO}" == "True" ]]; then
  ARGS+=(--make_video)
fi
if [[ "${HEADLESS}" == "0" || "${HEADLESS}" == "false" || "${HEADLESS}" == "False" ]]; then
  ARGS+=(--no-headless)
else
  ARGS+=(--headless)
fi
if [[ "${TRAIN_DR}" == "0" || "${TRAIN_DR}" == "false" || "${TRAIN_DR}" == "False" ]]; then
  ARGS+=(--no-train_dr)
else
  ARGS+=(--train_dr)
fi

# Optional raw argument escape hatch, e.g.
# EXTRA_ARGS="--render_mode pt --render_samples_per_pixel 128".
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
  ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

"${PYTHON_BIN}" isaacsimenvs/render_peg_in_hole_finetuned.py "${ARGS[@]}"

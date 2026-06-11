#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv-isaacsim-py311/bin/python}"

CHECKPOINT="${CHECKPOINT:-/juno/u/kedia/depthbasedRL/train_dir/TrainingObjective/Play2Win/model.pth}"
ROBOT_URDF="${ROBOT_URDF:-assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted_pretty.urdf}"
SEED="${SEED:-42}"

# Defaults reproduce the camera path from:
# local_logs/2026-06-04_17-21-59_simtoolreal_cinematic_warm_gray_high_quality_less_light_20s
NUM_ENVS="${NUM_ENVS:-100}"
GRID_COLS="${GRID_COLS:-10}"
ENV_SPACING_X="${ENV_SPACING_X:-0.8}"
ENV_SPACING_Y="${ENV_SPACING_Y:-2.45}"
STEPS="${STEPS:-1200}"
VIDEO_FPS="${VIDEO_FPS:-30}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"
QUALITY="${QUALITY:-high}"
RENDER_QUALITY_PRESET="${RENDER_QUALITY_PRESET:-beauty}"
RENDER_MODE="${RENDER_MODE:-rt}"
RENDER_SPP="${RENDER_SPP:-64}"
CAPTURE_SOURCE="${CAPTURE_SOURCE:-camera_sensor}"
DOME_LIGHT_UPPER_LOWER_STRATEGY="${DOME_LIGHT_UPPER_LOWER_STRATEGY:-}"
HEADLESS="${HEADLESS:-1}"
HOLD_OPEN_S="${HOLD_OPEN_S:-0}"

MAKE_VIDEO="${MAKE_VIDEO:-0}"
CAPTURE_PNG_STEPS="${CAPTURE_PNG_STEPS:-0,60,180,360,600,900,1200}"
RUN_TIMESTAMP="$(date +%F_%H-%M-%S)"
OUT_DIR="${OUT_DIR:-local_logs/${RUN_TIMESTAMP}_simtoolreal_ref_pan_easy_fixed_init_visual_probe}"

OBJECT_DISTRIBUTION_MODE="${OBJECT_DISTRIBUTION_MODE:-mixed_training_easy_video_25_25_50}"
RESET_POSITION_NOISE_M="${RESET_POSITION_NOISE_M:-0 0 0}"
RESET_ORIENTATION_MODE="${RESET_ORIENTATION_MODE:-identity}"
TABLE_RESET_Z_RANGE_M="${TABLE_RESET_Z_RANGE_M:-0}"
RESET_DOF_POS_NOISE_ARM="${RESET_DOF_POS_NOISE_ARM:-0}"
RESET_DOF_POS_NOISE_FINGERS="${RESET_DOF_POS_NOISE_FINGERS:-0}"
RESET_DOF_VEL_NOISE="${RESET_DOF_VEL_NOISE:-0}"
FORCE_SCALE="${FORCE_SCALE:-0}"
TORQUE_SCALE="${TORQUE_SCALE:-0}"

FLOOR_STYLE="${FLOOR_STYLE:-nvidia_precast_concrete_dark_gray}"
TABLE_COLOR_R="${TABLE_COLOR_R:-0.50}"
TABLE_COLOR_G="${TABLE_COLOR_G:-0.32}"
TABLE_COLOR_B="${TABLE_COLOR_B:-0.18}"
SKY_STYLE="${SKY_STYLE:-blue_dome}"
SKY_COLOR_R="${SKY_COLOR_R:-0.50}"
SKY_COLOR_G="${SKY_COLOR_G:-0.66}"
SKY_COLOR_B="${SKY_COLOR_B:-0.86}"
SKY_DOME_INTENSITY="${SKY_DOME_INTENSITY:-1200}"
SKY_HDRI_PRESET="${SKY_HDRI_PRESET:-stinson_beach}"
SKY_HDRI_PATH="${SKY_HDRI_PATH:-}"
DYNAMIC_SKY_PRESET="${DYNAMIC_SKY_PRESET:-remote_clear}"
DYNAMIC_SKY_PATH="${DYNAMIC_SKY_PATH:-}"
BACKDROP_STYLE="${BACKDROP_STYLE:-gradient_sky}"
BACKDROP_COLOR_R="${BACKDROP_COLOR_R:-0.25}"
BACKDROP_COLOR_G="${BACKDROP_COLOR_G:-0.48}"
BACKDROP_COLOR_B="${BACKDROP_COLOR_B:-0.76}"
BACKDROP_HORIZON_COLOR_R="${BACKDROP_HORIZON_COLOR_R:-0.68}"
BACKDROP_HORIZON_COLOR_G="${BACKDROP_HORIZON_COLOR_G:-0.78}"
BACKDROP_HORIZON_COLOR_B="${BACKDROP_HORIZON_COLOR_B:-0.88}"
BACKDROP_DISTANCE="${BACKDROP_DISTANCE:-5.0}"
BACKDROP_HEIGHT="${BACKDROP_HEIGHT:-18.0}"
BACKDROP_EXTENT_MARGIN="${BACKDROP_EXTENT_MARGIN:-80.0}"
BACKDROP_GRADIENT_BANDS="${BACKDROP_GRADIENT_BANDS:-4}"
LIGHTING_STYLE="${LIGHTING_STYLE:-single_sun}"
DEFAULT_LIGHT_INTENSITY="${DEFAULT_LIGHT_INTENSITY:-}"

VIDEO_ARGS=()
if [[ "${MAKE_VIDEO}" == "1" ]]; then
  VIDEO_ARGS+=(--make_video)
fi

LIGHTING_ARGS=(--lighting_style "${LIGHTING_STYLE}")
if [[ "${LIGHTING_STYLE}" == "single_sun" ]]; then
  DEFAULT_LIGHT_INTENSITY="${DEFAULT_LIGHT_INTENSITY:-360}"
  LIGHTING_ARGS+=(
    --single_sun_exposure "${SINGLE_SUN_EXPOSURE:-9.35}"
    --single_sun_angle "${SINGLE_SUN_ANGLE:-0.12}"
    --single_sun_color_temperature "${SINGLE_SUN_COLOR_TEMPERATURE:-5250}"
    --single_sun_color "${SINGLE_SUN_COLOR_R:-1.0}" "${SINGLE_SUN_COLOR_G:-0.97}" "${SINGLE_SUN_COLOR_B:-0.90}"
    --single_sun_yaw_offset_deg "${SINGLE_SUN_YAW_OFFSET_DEG:-70}"
  )
  if [[ -n "${SINGLE_SUN_ELEVATION_DEG:-}" ]]; then
    LIGHTING_ARGS+=(--single_sun_elevation_deg "${SINGLE_SUN_ELEVATION_DEG}")
  fi
fi
if [[ -n "${DEFAULT_LIGHT_INTENSITY}" ]]; then
  LIGHTING_ARGS+=(--default_light_intensity "${DEFAULT_LIGHT_INTENSITY}")
fi

SKY_ARGS=(--sky_hdri_preset "${SKY_HDRI_PRESET}")
if [[ -n "${SKY_HDRI_PATH}" ]]; then
  SKY_ARGS+=(--sky_hdri_path "${SKY_HDRI_PATH}")
fi
SKY_ARGS+=(--dynamic_sky_preset "${DYNAMIC_SKY_PRESET}")
if [[ -n "${DYNAMIC_SKY_PATH}" ]]; then
  SKY_ARGS+=(--dynamic_sky_path "${DYNAMIC_SKY_PATH}")
fi

FLOOR_ARGS=(
  --floor_style "${FLOOR_STYLE}"
  --floor_color "${FLOOR_COLOR_R:-0.60}" "${FLOOR_COLOR_G:-0.60}" "${FLOOR_COLOR_B:-0.56}"
  --floor_tile_count "${FLOOR_TILE_COUNT:-96}"
  --floor_tile_size "${FLOOR_TILE_SIZE:-0.9}"
  --floor_tile_gap "${FLOOR_TILE_GAP:-0.006}"
  --floor_texture_scale "${FLOOR_TEXTURE_SCALE:-4.0}"
  --floor_roughness "${FLOOR_ROUGHNESS:-0.42}"
  --floor_normal_strength "${FLOOR_NORMAL_STRENGTH:-0.35}"
  --floor_specular_level "${FLOOR_SPECULAR_LEVEL:-0.5}"
)
if [[ -n "${FLOOR_TEXTURE_PATH:-}" ]]; then
  FLOOR_ARGS+=(--floor_texture_path "${FLOOR_TEXTURE_PATH}")
fi
if [[ -n "${FLOOR_NORMAL_PATH:-}" ]]; then
  FLOOR_ARGS+=(--floor_normal_path "${FLOOR_NORMAL_PATH}")
fi

RENDER_ARGS=(
  --render_quality_preset "${RENDER_QUALITY_PRESET}"
  --render_mode "${RENDER_MODE}"
  --render_samples_per_pixel "${RENDER_SPP}"
  --capture_source "${CAPTURE_SOURCE}"
  --hold_open_s "${HOLD_OPEN_S}"
)
if [[ -n "${DOME_LIGHT_UPPER_LOWER_STRATEGY}" ]]; then
  RENDER_ARGS+=(--dome_light_upper_lower_strategy "${DOME_LIGHT_UPPER_LOWER_STRATEGY}")
fi
if [[ "${HEADLESS}" == "0" || "${HEADLESS}" == "false" || "${HEADLESS}" == "False" ]]; then
  RENDER_ARGS+=(--no-headless)
else
  RENDER_ARGS+=(--headless)
fi

"${PYTHON_BIN}" isaacsimenvs/render_simtoolreal_pretrained.py \
  --num_envs "${NUM_ENVS}" \
  --grid_cols "${GRID_COLS}" \
  --env_spacing_xy "${ENV_SPACING_X}" "${ENV_SPACING_Y}" \
  --steps "${STEPS}" \
  --capture_png_steps "${CAPTURE_PNG_STEPS}" \
  "${VIDEO_ARGS[@]}" \
  --video_fps "${VIDEO_FPS}" \
  --quality "${QUALITY}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --seed "${SEED}" \
  "${RENDER_ARGS[@]}" \
  --deterministic \
  --checkpoint "${CHECKPOINT}" \
  --robot_urdf "${ROBOT_URDF}" \
  --object_distribution_mode "${OBJECT_DISTRIBUTION_MODE}" \
  --num_assets_per_type "${NUM_ASSETS_PER_TYPE:-8}" \
  --reset_position_noise_m ${RESET_POSITION_NOISE_M} \
  --reset_orientation_mode "${RESET_ORIENTATION_MODE}" \
  --table_reset_z_range_m "${TABLE_RESET_Z_RANGE_M}" \
  --reset_dof_pos_noise_arm "${RESET_DOF_POS_NOISE_ARM}" \
  --reset_dof_pos_noise_fingers "${RESET_DOF_POS_NOISE_FINGERS}" \
  --reset_dof_vel_noise "${RESET_DOF_VEL_NOISE}" \
  --force_scale "${FORCE_SCALE}" \
  --torque_scale "${TORQUE_SCALE}" \
  --hide_goal_viz \
  --style_table \
  --table_style display_color \
  --table_color "${TABLE_COLOR_R}" "${TABLE_COLOR_G}" "${TABLE_COLOR_B}" \
  --style_floor \
  "${FLOOR_ARGS[@]}" \
  --sky_style "${SKY_STYLE}" \
  --sky_color "${SKY_COLOR_R}" "${SKY_COLOR_G}" "${SKY_COLOR_B}" \
  --sky_dome_intensity "${SKY_DOME_INTENSITY}" \
  "${SKY_ARGS[@]}" \
  --backdrop_style "${BACKDROP_STYLE}" \
  --backdrop_color "${BACKDROP_COLOR_R}" "${BACKDROP_COLOR_G}" "${BACKDROP_COLOR_B}" \
  --backdrop_horizon_color "${BACKDROP_HORIZON_COLOR_R}" "${BACKDROP_HORIZON_COLOR_G}" "${BACKDROP_HORIZON_COLOR_B}" \
  --backdrop_distance "${BACKDROP_DISTANCE}" \
  --backdrop_height "${BACKDROP_HEIGHT}" \
  --backdrop_extent_margin "${BACKDROP_EXTENT_MARGIN}" \
  --backdrop_gradient_bands "${BACKDROP_GRADIENT_BANDS}" \
  "${LIGHTING_ARGS[@]}" \
  --camera_motion sapg_ref_pan \
  --sapg_ref_anchor_env -1 \
  --sapg_ref_start_target 0.0 0.0 0.63 \
  --sapg_ref_start_eye_offset -0.75 -0.85 0.55 \
  --sapg_ref_end_target_grid_scale -0.5 -0.5 0.0 \
  --sapg_ref_end_eye_grid_scale -0.8 -1.2 0.5 \
  --camera_render_warmup_frames "${CAMERA_RENDER_WARMUP_FRAMES:-1}" \
  --image_exposure "${IMAGE_EXPOSURE:-0.0}" \
  --image_contrast "${IMAGE_CONTRAST:-1.0}" \
  --image_saturation "${IMAGE_SATURATION:-1.0}" \
  --image_gamma "${IMAGE_GAMMA:-1.0}" \
  --object_color_saturation "${OBJECT_COLOR_SATURATION:-1.6}" \
  --object_color_value_scale "${OBJECT_COLOR_VALUE_SCALE:-0.82}" \
  --no_timestamp_out_dir \
  --out_dir "${OUT_DIR}"

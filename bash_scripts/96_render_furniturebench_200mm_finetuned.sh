#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv-isaacsim-py311/bin/python}"

CHECKPOINT="${CHECKPOINT:-/juno/u/kedia/depthbasedRL/train_dir/May26/screwing_newer/model.pth}"
OUT_DIR="${OUT_DIR:-local_logs/furniturebench_200mm_finetuned}"

NUM_ENVS="${NUM_ENVS:-1}"
STEPS="${STEPS:-1200}"
SEED="${SEED:-0}"
NO_RENDER="${NO_RENDER:-0}"
MAKE_VIDEO="${MAKE_VIDEO:-0}"
CAPTURE_PNG_STEPS="${CAPTURE_PNG_STEPS:-0}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-540}"
VIDEO_FPS="${VIDEO_FPS:-30}"
HEADLESS="${HEADLESS:-1}"
HOLD_OPEN_S="${HOLD_OPEN_S:-0}"
CAMERA_EYE="${CAMERA_EYE:-0.12 -1.55 0.98}"
CAMERA_TARGET="${CAMERA_TARGET:-0.02 0.02 0.58}"
DEFAULT_LIGHT_INTENSITY="${DEFAULT_LIGHT_INTENSITY:-120}"
SKY_DOME_INTENSITY="${SKY_DOME_INTENSITY:-650}"
SINGLE_SUN_EXPOSURE="${SINGLE_SUN_EXPOSURE:-6.8}"
SINGLE_SUN_ANGLE="${SINGLE_SUN_ANGLE:-0.45}"
IMAGE_EXPOSURE="${IMAGE_EXPOSURE:-0.72}"
IMAGE_CONTRAST="${IMAGE_CONTRAST:-1.08}"
IMAGE_SATURATION="${IMAGE_SATURATION:-1.03}"
TABLE_COLOR="${TABLE_COLOR:-0.42 0.27 0.15}"
FLOOR_COLOR="${FLOOR_COLOR:-0.44 0.45 0.43}"

# Default to screwing-only episodes for a single-env video. Set to 0.1 to match
# the exact finetune launcher distribution including random goal envs.
RANDOM_GOAL_FRACTION="${RANDOM_GOAL_FRACTION:-0.0}"
TRAIN_DR="${TRAIN_DR:-1}"

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --out_dir "${OUT_DIR}"
  --num_envs "${NUM_ENVS}"
  --steps "${STEPS}"
  --seed "${SEED}"
  --capture_png_steps "${CAPTURE_PNG_STEPS}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --video_fps "${VIDEO_FPS}"
  --random_goal_fraction "${RANDOM_GOAL_FRACTION}"
  --hold_open_s "${HOLD_OPEN_S}"
  --camera_eye ${CAMERA_EYE}
  --camera_target ${CAMERA_TARGET}
  --default_light_intensity "${DEFAULT_LIGHT_INTENSITY}"
  --sky_dome_intensity "${SKY_DOME_INTENSITY}"
  --single_sun_exposure "${SINGLE_SUN_EXPOSURE}"
  --single_sun_angle "${SINGLE_SUN_ANGLE}"
  --image_exposure "${IMAGE_EXPOSURE}"
  --image_contrast "${IMAGE_CONTRAST}"
  --image_saturation "${IMAGE_SATURATION}"
  --table_color ${TABLE_COLOR}
  --floor_color ${FLOOR_COLOR}
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

#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_TIMESTAMP="$(date +%F_%H-%M-%S)"
ROOT_DIR="${ROOT_DIR:-local_logs/${RUN_TIMESTAMP}_simtoolreal_visual_stack_probe}"
mkdir -p "${ROOT_DIR}"

NUM_ENVS="${NUM_ENVS:-100}"
GRID_COLS="${GRID_COLS:-10}"
STEPS="${STEPS:-1200}"
CAPTURE_PNG_STEPS="${CAPTURE_PNG_STEPS:-0,600,1200}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"
QUALITY="${QUALITY:-high}"
RENDER_SPP="${RENDER_SPP:-64}"

run_probe() {
  local label="$1"
  shift
  echo "======================================================================"
  echo "[visual_stack_probe] ${label}"
  echo "======================================================================"
  env \
    NUM_ENVS="${NUM_ENVS}" \
    GRID_COLS="${GRID_COLS}" \
    STEPS="${STEPS}" \
    CAPTURE_PNG_STEPS="${CAPTURE_PNG_STEPS}" \
    WIDTH="${WIDTH}" \
    HEIGHT="${HEIGHT}" \
    QUALITY="${QUALITY}" \
    RENDER_SPP="${RENDER_SPP}" \
    MAKE_VIDEO=0 \
    OUT_DIR="${ROOT_DIR}/${label}" \
    "$@" \
    bash_scripts/95_render_simtoolreal_ref_pan_cinematic.sh
}

# A small but targeted matrix:
# - Keep accepted rollout/camera defaults.
# - Vary only floor/background/sky/light capture hypotheses.
run_probe "00_baseline_current"

run_probe "01_smooth_gradient_current_concrete" \
  BACKDROP_GRADIENT_BANDS=16 \
  FLOOR_TILE_GAP=0.002 \
  FLOOR_TEXTURE_SCALE=2.5

run_probe "02_slate_blue_gray_floor_smooth_gradient" \
  FLOOR_STYLE=display_color \
  FLOOR_COLOR_R=0.30 \
  FLOOR_COLOR_G=0.36 \
  FLOOR_COLOR_B=0.42 \
  BACKDROP_GRADIENT_BANDS=16 \
  BACKDROP_COLOR_R=0.22 \
  BACKDROP_COLOR_G=0.42 \
  BACKDROP_COLOR_B=0.68 \
  BACKDROP_HORIZON_COLOR_R=0.62 \
  BACKDROP_HORIZON_COLOR_G=0.73 \
  BACKDROP_HORIZON_COLOR_B=0.82

run_probe "03_warm_gray_floor_smooth_gradient" \
  FLOOR_STYLE=display_color \
  FLOOR_COLOR_R=0.47 \
  FLOOR_COLOR_G=0.46 \
  FLOOR_COLOR_B=0.42 \
  BACKDROP_GRADIENT_BANDS=16 \
  BACKDROP_COLOR_R=0.24 \
  BACKDROP_COLOR_G=0.46 \
  BACKDROP_COLOR_B=0.72 \
  BACKDROP_HORIZON_COLOR_R=0.66 \
  BACKDROP_HORIZON_COLOR_G=0.76 \
  BACKDROP_HORIZON_COLOR_B=0.86

run_probe "04_dark_desaturated_blue_floor" \
  FLOOR_STYLE=display_color \
  FLOOR_COLOR_R=0.18 \
  FLOOR_COLOR_G=0.23 \
  FLOOR_COLOR_B=0.29 \
  BACKDROP_GRADIENT_BANDS=16 \
  BACKDROP_COLOR_R=0.23 \
  BACKDROP_COLOR_G=0.44 \
  BACKDROP_COLOR_B=0.72 \
  BACKDROP_HORIZON_COLOR_R=0.62 \
  BACKDROP_HORIZON_COLOR_G=0.73 \
  BACKDROP_HORIZON_COLOR_B=0.84

run_probe "05_dynamic_local_simple_no_backdrop" \
  SKY_STYLE=dynamic_clear_sky \
  DYNAMIC_SKY_PRESET=local_simple \
  BACKDROP_STYLE=none \
  LIGHTING_STYLE=none \
  DEFAULT_LIGHT_INTENSITY=360

run_probe "06_dynamic_local_sunstudy_no_backdrop" \
  SKY_STYLE=dynamic_clear_sky \
  DYNAMIC_SKY_PRESET=local_sunstudy \
  BACKDROP_STYLE=none \
  LIGHTING_STYLE=none \
  DEFAULT_LIGHT_INTENSITY=360

run_probe "07_hdri_carlight_no_backdrop_low_intensity" \
  SKY_STYLE=hdri \
  SKY_HDRI_PATH="${REPO_DIR}/.venv-isaacsim-py311/lib/python3.11/site-packages/isaacsim/extscache/omni.kit.environment.core-1.3.24/data/tests/Skies/Hdr/CarLight_512x256.hdr" \
  SKY_DOME_INTENSITY=250 \
  DOME_LIGHT_UPPER_LOWER_STRATEGY=0 \
  BACKDROP_STYLE=none \
  LIGHTING_STYLE=none \
  DEFAULT_LIGHT_INTENSITY=0

CONTACT_SHEET_PYTHON="${CONTACT_SHEET_PYTHON:-python3}"
"${CONTACT_SHEET_PYTHON}" - "${ROOT_DIR}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


root = Path(sys.argv[1])
labels = sorted(path.name for path in root.iterdir() if path.is_dir())
steps = ["step_0000.png", "step_0600.png", "step_1200.png"]
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

for step_name in steps:
    frames = []
    for label in labels:
        path = root / label / step_name
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB").resize((480, 270), resample_filter)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 480, 36), fill=(0, 0, 0))
        draw.text((8, 5), label, font=small_font, fill=(255, 255, 255))
        frames.append(img)
    if not frames:
        continue
    cols = 2
    rows = (len(frames) + cols - 1) // cols
    title_h = 48
    sheet = Image.new("RGB", (cols * 480, title_h + rows * 270), (20, 20, 20))
    for idx, frame in enumerate(frames):
        sheet.paste(frame, ((idx % cols) * 480, title_h + (idx // cols) * 270))
    draw = ImageDraw.Draw(sheet)
    title = f"SimToolReal visual stack probe: {step_name}"
    draw.rectangle((0, 0, sheet.width, title_h), fill=(0, 0, 0))
    draw.text((12, 7), title, font=font, fill=(255, 255, 255))
    out = root / f"contact_sheet_{step_name}"
    sheet.save(out)
    print(f"[visual_stack_probe] wrote {out}")
PY

echo "======================================================================"
echo "[visual_stack_probe] done"
echo "ROOT_DIR=${ROOT_DIR}"
echo "Contact sheets:"
find "${ROOT_DIR}" -maxdepth 1 -type f -name 'contact_sheet_*.png' -print | sort

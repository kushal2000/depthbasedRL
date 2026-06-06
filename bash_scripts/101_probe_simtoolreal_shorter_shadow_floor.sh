#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_TIMESTAMP="$(date +%F_%H-%M-%S)"
ROOT_DIR="${ROOT_DIR:-local_logs/${RUN_TIMESTAMP}_simtoolreal_shorter_shadow_floor_probe}"
mkdir -p "${ROOT_DIR}"

NUM_ENVS="${NUM_ENVS:-16}"
GRID_COLS="${GRID_COLS:-4}"
STEPS="${STEPS:-300}"
CAPTURE_PNG_STEPS="${CAPTURE_PNG_STEPS:-0,150,300}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
QUALITY="${QUALITY:-medium}"
RENDER_SPP="${RENDER_SPP:-48}"

run_probe() {
  local label="$1"
  shift
  echo "======================================================================"
  echo "[shorter_shadow_floor_probe] ${label}"
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
    BACKDROP_GRADIENT_BANDS="${BACKDROP_GRADIENT_BANDS:-16}" \
    OUT_DIR="${ROOT_DIR}/${label}" \
    "$@" \
    bash_scripts/98_render_simtoolreal_ref_pan_cinematic_greige_floor.sh
}

# Current favorite for direct comparison.
run_probe "00_side_sun_softened_old_elev" \
  SINGLE_SUN_EXPOSURE=9.65 \
  SINGLE_SUN_ANGLE=0.28 \
  SINGLE_SUN_COLOR_TEMPERATURE=5050 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.96 SINGLE_SUN_COLOR_B=0.84 \
  SINGLE_SUN_YAW_OFFSET_DEG=110 \
  DEFAULT_LIGHT_INTENSITY=520 \
  SKY_DOME_INTENSITY=1700 \
  IMAGE_EXPOSURE=-0.10

# Same look, but with sun raised to shorten shadows.
run_probe "01_greige_elev30" \
  SINGLE_SUN_ELEVATION_DEG=30 \
  SINGLE_SUN_EXPOSURE=9.65 \
  SINGLE_SUN_ANGLE=0.28 \
  SINGLE_SUN_COLOR_TEMPERATURE=5050 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.96 SINGLE_SUN_COLOR_B=0.84 \
  SINGLE_SUN_YAW_OFFSET_DEG=110 \
  DEFAULT_LIGHT_INTENSITY=520 \
  SKY_DOME_INTENSITY=1700 \
  IMAGE_EXPOSURE=-0.10

run_probe "02_greige_elev40" \
  SINGLE_SUN_ELEVATION_DEG=40 \
  SINGLE_SUN_EXPOSURE=9.65 \
  SINGLE_SUN_ANGLE=0.32 \
  SINGLE_SUN_COLOR_TEMPERATURE=5050 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.96 SINGLE_SUN_COLOR_B=0.84 \
  SINGLE_SUN_YAW_OFFSET_DEG=110 \
  DEFAULT_LIGHT_INTENSITY=500 \
  SKY_DOME_INTENSITY=1650 \
  IMAGE_EXPOSURE=-0.08

# Lighter floor options closer to the reference image, using the raised sun to
# avoid the old white-floor overexposure failure mode.
run_probe "03_soft_concrete_elev35" \
  FLOOR_STYLE=soft_concrete_pbr_tiles \
  FLOOR_COLOR_R=0.62 FLOOR_COLOR_G=0.61 FLOOR_COLOR_B=0.57 \
  FLOOR_TILE_COUNT=1 FLOOR_TILE_SIZE=120 FLOOR_TILE_GAP=0 \
  FLOOR_TEXTURE_SCALE=9.0 FLOOR_ROUGHNESS=0.88 FLOOR_NORMAL_STRENGTH=0.04 FLOOR_SPECULAR_LEVEL=0.05 \
  SINGLE_SUN_ELEVATION_DEG=35 \
  SINGLE_SUN_EXPOSURE=9.55 \
  SINGLE_SUN_ANGLE=0.30 \
  SINGLE_SUN_COLOR_TEMPERATURE=5100 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.96 SINGLE_SUN_COLOR_B=0.86 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=460 \
  SKY_DOME_INTENSITY=1550 \
  IMAGE_EXPOSURE=-0.16

run_probe "04_white_stone_elev35" \
  FLOOR_STYLE=white_stone_slabs \
  FLOOR_COLOR_R=0.70 FLOOR_COLOR_G=0.69 FLOOR_COLOR_B=0.65 \
  FLOOR_TILE_COUNT=1 FLOOR_TILE_SIZE=120 FLOOR_TILE_GAP=0 \
  FLOOR_TEXTURE_SCALE=8.0 FLOOR_ROUGHNESS=0.92 FLOOR_NORMAL_STRENGTH=0.03 FLOOR_SPECULAR_LEVEL=0.04 \
  SINGLE_SUN_ELEVATION_DEG=35 \
  SINGLE_SUN_EXPOSURE=9.40 \
  SINGLE_SUN_ANGLE=0.30 \
  SINGLE_SUN_COLOR_TEMPERATURE=5150 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.97 SINGLE_SUN_COLOR_B=0.88 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=420 \
  SKY_DOME_INTENSITY=1450 \
  IMAGE_EXPOSURE=-0.22

run_probe "05_light_greige_elev35" \
  FLOOR_COLOR_R=0.58 FLOOR_COLOR_G=0.56 FLOOR_COLOR_B=0.51 \
  FLOOR_TEXTURE_SCALE=10.0 FLOOR_ROUGHNESS=0.92 FLOOR_NORMAL_STRENGTH=0.04 FLOOR_SPECULAR_LEVEL=0.05 \
  SINGLE_SUN_ELEVATION_DEG=35 \
  SINGLE_SUN_EXPOSURE=9.55 \
  SINGLE_SUN_ANGLE=0.30 \
  SINGLE_SUN_COLOR_TEMPERATURE=5100 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.96 SINGLE_SUN_COLOR_B=0.86 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=460 \
  SKY_DOME_INTENSITY=1550 \
  IMAGE_EXPOSURE=-0.14

CONTACT_SHEET_PYTHON="${CONTACT_SHEET_PYTHON:-python3}"
"${CONTACT_SHEET_PYTHON}" - "${ROOT_DIR}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

root = Path(sys.argv[1])
labels = sorted(path.name for path in root.iterdir() if path.is_dir())
steps = ["step_0000.png", "step_0150.png", "step_0300.png"]
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

for step_name in steps:
    frames = []
    for label in labels:
        path = root / label / step_name
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB").resize((480, 270), resample_filter)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 480, 34), fill=(0, 0, 0))
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
    draw.rectangle((0, 0, sheet.width, title_h), fill=(0, 0, 0))
    draw.text((12, 7), f"SimToolReal shorter shadow / floor probe: {step_name}", font=font, fill=(255, 255, 255))
    out = root / f"contact_sheet_{step_name}"
    sheet.save(out)
    print(f"[shorter_shadow_floor_probe] wrote {out}")
PY

echo "======================================================================"
echo "[shorter_shadow_floor_probe] done"
echo "ROOT_DIR=${ROOT_DIR}"
echo "Contact sheets:"
find "${ROOT_DIR}" -maxdepth 1 -type f -name 'contact_sheet_*.png' -print | sort

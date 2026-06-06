#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_TIMESTAMP="$(date +%F_%H-%M-%S)"
ROOT_DIR="${ROOT_DIR:-local_logs/${RUN_TIMESTAMP}_simtoolreal_lighting_polish_probe}"
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
  echo "[lighting_polish_probe] ${label}"
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
    BACKDROP_GRADIENT_BANDS="${BACKDROP_GRADIENT_BANDS:-32}" \
    FLOOR_STYLE=soft_concrete_pbr_tiles \
    FLOOR_TILE_COUNT=1 FLOOR_TILE_SIZE=120 FLOOR_TILE_GAP=0 \
    FLOOR_TEXTURE_SCALE=8.0 FLOOR_ROUGHNESS=0.92 FLOOR_NORMAL_STRENGTH=0.05 FLOOR_SPECULAR_LEVEL=0.04 \
    OUT_DIR="${ROOT_DIR}/${label}" \
    "$@" \
    bash_scripts/98_render_simtoolreal_ref_pan_cinematic_greige_floor.sh
}

# Accepted short-shadow baseline from 2026-06-06.
run_probe "00_baseline_soft_concrete_elev55" \
  FLOOR_COLOR_R=0.58 FLOOR_COLOR_G=0.58 FLOOR_COLOR_B=0.55 \
  TABLE_COLOR_R=0.50 TABLE_COLOR_G=0.32 TABLE_COLOR_B=0.18 \
  BACKDROP_COLOR_R=0.25 BACKDROP_COLOR_G=0.48 BACKDROP_COLOR_B=0.76 \
  BACKDROP_HORIZON_COLOR_R=0.68 BACKDROP_HORIZON_COLOR_G=0.78 BACKDROP_HORIZON_COLOR_B=0.88 \
  SINGLE_SUN_ELEVATION_DEG=55 \
  SINGLE_SUN_EXPOSURE=9.35 \
  SINGLE_SUN_ANGLE=0.34 \
  SINGLE_SUN_COLOR_TEMPERATURE=5250 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.97 SINGLE_SUN_COLOR_B=0.90 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=440 \
  SKY_DOME_INTENSITY=1400 \
  IMAGE_EXPOSURE=-0.18

# More depth cue than 55 deg, but still much shorter shadows than the old side sun.
run_probe "01_elev50_darker_table_bluer_horizon" \
  FLOOR_COLOR_R=0.58 FLOOR_COLOR_G=0.58 FLOOR_COLOR_B=0.55 \
  TABLE_COLOR_R=0.42 TABLE_COLOR_G=0.28 TABLE_COLOR_B=0.17 \
  BACKDROP_COLOR_R=0.22 BACKDROP_COLOR_G=0.44 BACKDROP_COLOR_B=0.74 \
  BACKDROP_HORIZON_COLOR_R=0.58 BACKDROP_HORIZON_COLOR_G=0.70 BACKDROP_HORIZON_COLOR_B=0.84 \
  SINGLE_SUN_ELEVATION_DEG=50 \
  SINGLE_SUN_EXPOSURE=9.45 \
  SINGLE_SUN_ANGLE=0.32 \
  SINGLE_SUN_COLOR_TEMPERATURE=5200 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.97 SINGLE_SUN_COLOR_B=0.89 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=430 \
  SKY_DOME_INTENSITY=1350 \
  IMAGE_EXPOSURE=-0.16

# Slightly stronger shadow/readability option.
run_probe "02_elev45_darker_table_bluer_horizon" \
  FLOOR_COLOR_R=0.58 FLOOR_COLOR_G=0.58 FLOOR_COLOR_B=0.55 \
  TABLE_COLOR_R=0.42 TABLE_COLOR_G=0.28 TABLE_COLOR_B=0.17 \
  BACKDROP_COLOR_R=0.22 BACKDROP_COLOR_G=0.44 BACKDROP_COLOR_B=0.74 \
  BACKDROP_HORIZON_COLOR_R=0.58 BACKDROP_HORIZON_COLOR_G=0.70 BACKDROP_HORIZON_COLOR_B=0.84 \
  SINGLE_SUN_ELEVATION_DEG=45 \
  SINGLE_SUN_EXPOSURE=9.50 \
  SINGLE_SUN_ANGLE=0.30 \
  SINGLE_SUN_COLOR_TEMPERATURE=5200 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.97 SINGLE_SUN_COLOR_B=0.88 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=420 \
  SKY_DOME_INTENSITY=1320 \
  IMAGE_EXPOSURE=-0.16

# Darker floor to reduce white-floor/white-sky washout while staying non-brown.
run_probe "03_elev50_slightly_darker_floor" \
  FLOOR_COLOR_R=0.53 FLOOR_COLOR_G=0.53 FLOOR_COLOR_B=0.51 \
  TABLE_COLOR_R=0.42 TABLE_COLOR_G=0.28 TABLE_COLOR_B=0.17 \
  BACKDROP_COLOR_R=0.22 BACKDROP_COLOR_G=0.44 BACKDROP_COLOR_B=0.74 \
  BACKDROP_HORIZON_COLOR_R=0.58 BACKDROP_HORIZON_COLOR_G=0.70 BACKDROP_HORIZON_COLOR_B=0.84 \
  SINGLE_SUN_ELEVATION_DEG=50 \
  SINGLE_SUN_EXPOSURE=9.45 \
  SINGLE_SUN_ANGLE=0.32 \
  SINGLE_SUN_COLOR_TEMPERATURE=5200 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.97 SINGLE_SUN_COLOR_B=0.89 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=430 \
  SKY_DOME_INTENSITY=1350 \
  IMAGE_EXPOSURE=-0.14

# Less saturated table: separates from floor without becoming a dark block.
run_probe "04_elev50_desaturated_table" \
  FLOOR_COLOR_R=0.57 FLOOR_COLOR_G=0.57 FLOOR_COLOR_B=0.54 \
  TABLE_COLOR_R=0.45 TABLE_COLOR_G=0.34 TABLE_COLOR_B=0.24 \
  BACKDROP_COLOR_R=0.24 BACKDROP_COLOR_G=0.46 BACKDROP_COLOR_B=0.75 \
  BACKDROP_HORIZON_COLOR_R=0.62 BACKDROP_HORIZON_COLOR_G=0.73 BACKDROP_HORIZON_COLOR_B=0.84 \
  SINGLE_SUN_ELEVATION_DEG=50 \
  SINGLE_SUN_EXPOSURE=9.45 \
  SINGLE_SUN_ANGLE=0.32 \
  SINGLE_SUN_COLOR_TEMPERATURE=5200 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.97 SINGLE_SUN_COLOR_B=0.89 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=430 \
  SKY_DOME_INTENSITY=1350 \
  IMAGE_EXPOSURE=-0.15

# Brighter daylight with the same separation changes; useful if the above reads too flat.
run_probe "05_elev50_brighter_daylight" \
  FLOOR_COLOR_R=0.56 FLOOR_COLOR_G=0.56 FLOOR_COLOR_B=0.53 \
  TABLE_COLOR_R=0.42 TABLE_COLOR_G=0.28 TABLE_COLOR_B=0.17 \
  BACKDROP_COLOR_R=0.22 BACKDROP_COLOR_G=0.44 BACKDROP_COLOR_B=0.74 \
  BACKDROP_HORIZON_COLOR_R=0.58 BACKDROP_HORIZON_COLOR_G=0.70 BACKDROP_HORIZON_COLOR_B=0.84 \
  SINGLE_SUN_ELEVATION_DEG=50 \
  SINGLE_SUN_EXPOSURE=9.65 \
  SINGLE_SUN_ANGLE=0.30 \
  SINGLE_SUN_COLOR_TEMPERATURE=5100 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.965 SINGLE_SUN_COLOR_B=0.86 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=390 \
  SKY_DOME_INTENSITY=1250 \
  IMAGE_EXPOSURE=-0.22

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
    draw.text((12, 7), f"SimToolReal lighting polish probe: {step_name}", font=font, fill=(255, 255, 255))
    out = root / f"contact_sheet_{step_name}"
    sheet.save(out)
    print(f"[lighting_polish_probe] wrote {out}")
PY

echo "======================================================================"
echo "[lighting_polish_probe] done"
echo "ROOT_DIR=${ROOT_DIR}"
echo "Contact sheets:"
find "${ROOT_DIR}" -maxdepth 1 -type f -name 'contact_sheet_*.png' -print | sort

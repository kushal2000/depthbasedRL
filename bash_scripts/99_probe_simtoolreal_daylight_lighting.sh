#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_TIMESTAMP="$(date +%F_%H-%M-%S)"
ROOT_DIR="${ROOT_DIR:-local_logs/${RUN_TIMESTAMP}_simtoolreal_daylight_lighting_probe}"
mkdir -p "${ROOT_DIR}"

# Keep this cheaper than a full final render while preserving the accepted
# camera, floor, object distribution, backdrop, and pretty robot.
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
  echo "[daylight_lighting_probe] ${label}"
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

# Baseline: current accepted greige-floor daylight stack.
run_probe "00_current_greige_single_sun" \
  SINGLE_SUN_EXPOSURE=9.35 \
  SINGLE_SUN_ANGLE=0.12 \
  SINGLE_SUN_COLOR_TEMPERATURE=5250 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.97 SINGLE_SUN_COLOR_B=0.90 \
  SINGLE_SUN_YAW_OFFSET_DEG=70 \
  DEFAULT_LIGHT_INTENSITY=360 \
  SKY_DOME_INTENSITY=1200 \
  IMAGE_EXPOSURE=0.0

# More like the reference: low golden sun, stronger cool sky fill, slight
# negative output exposure to avoid table/floor washout.
run_probe "01_golden_key_cool_fill" \
  SINGLE_SUN_EXPOSURE=9.85 \
  SINGLE_SUN_ANGLE=0.16 \
  SINGLE_SUN_COLOR_TEMPERATURE=5050 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.95 SINGLE_SUN_COLOR_B=0.82 \
  SINGLE_SUN_YAW_OFFSET_DEG=80 \
  DEFAULT_LIGHT_INTENSITY=420 \
  SKY_DOME_INTENSITY=1550 \
  IMAGE_EXPOSURE=-0.08

# Softer shadows and more readable robots.
run_probe "02_soft_daylight_more_fill" \
  SINGLE_SUN_EXPOSURE=9.55 \
  SINGLE_SUN_ANGLE=0.45 \
  SINGLE_SUN_COLOR_TEMPERATURE=5200 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.96 SINGLE_SUN_COLOR_B=0.86 \
  SINGLE_SUN_YAW_OFFSET_DEG=75 \
  DEFAULT_LIGHT_INTENSITY=560 \
  SKY_DOME_INTENSITY=1700 \
  IMAGE_EXPOSURE=-0.04

# Crisper long shadows, closer to the hero-shot sun direction.
run_probe "03_crisp_low_golden_sun" \
  SINGLE_SUN_EXPOSURE=10.10 \
  SINGLE_SUN_ANGLE=0.08 \
  SINGLE_SUN_COLOR_TEMPERATURE=4850 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.93 SINGLE_SUN_COLOR_B=0.78 \
  SINGLE_SUN_YAW_OFFSET_DEG=62 \
  DEFAULT_LIGHT_INTENSITY=330 \
  SKY_DOME_INTENSITY=1450 \
  IMAGE_EXPOSURE=-0.16

# Same warmth but sun rotated; useful if foreground shadows block object colors.
run_probe "04_side_sun_less_front_shadow" \
  SINGLE_SUN_EXPOSURE=9.85 \
  SINGLE_SUN_ANGLE=0.14 \
  SINGLE_SUN_COLOR_TEMPERATURE=5000 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.95 SINGLE_SUN_COLOR_B=0.82 \
  SINGLE_SUN_YAW_OFFSET_DEG=115 \
  DEFAULT_LIGHT_INTENSITY=460 \
  SKY_DOME_INTENSITY=1600 \
  IMAGE_EXPOSURE=-0.10

# Warmer horizon/fill, but still kept below overexposure.
run_probe "05_warm_horizon_gold_fill" \
  SINGLE_SUN_EXPOSURE=9.70 \
  SINGLE_SUN_ANGLE=0.20 \
  SINGLE_SUN_COLOR_TEMPERATURE=4950 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.94 SINGLE_SUN_COLOR_B=0.80 \
  SINGLE_SUN_YAW_OFFSET_DEG=80 \
  DEFAULT_LIGHT_INTENSITY=500 \
  SKY_DOME_INTENSITY=1500 \
  BACKDROP_HORIZON_COLOR_R=0.78 BACKDROP_HORIZON_COLOR_G=0.80 BACKDROP_HORIZON_COLOR_B=0.76 \
  BACKDROP_COLOR_R=0.36 BACKDROP_COLOR_G=0.55 BACKDROP_COLOR_B=0.78 \
  IMAGE_EXPOSURE=-0.12

# Older multi-light beauty stack, included as a control; expected to be flatter
# and more studio-like than the single-sun options.
run_probe "06_beauty_stack_control" \
  LIGHTING_STYLE=beauty \
  DEFAULT_LIGHT_INTENSITY=180 \
  SKY_DOME_INTENSITY=850 \
  IMAGE_EXPOSURE=-0.12

# Slightly cooler daylight, useful if golden variants make the table/robot too
# yellow or the objects too pastel.
run_probe "07_cooler_clear_daylight" \
  SINGLE_SUN_EXPOSURE=9.75 \
  SINGLE_SUN_ANGLE=0.18 \
  SINGLE_SUN_COLOR_TEMPERATURE=5700 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.98 SINGLE_SUN_COLOR_B=0.92 \
  SINGLE_SUN_YAW_OFFSET_DEG=80 \
  DEFAULT_LIGHT_INTENSITY=420 \
  SKY_DOME_INTENSITY=1500 \
  IMAGE_EXPOSURE=-0.08

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
    draw.text((12, 7), f"SimToolReal daylight lighting probe: {step_name}", font=font, fill=(255, 255, 255))
    out = root / f"contact_sheet_{step_name}"
    sheet.save(out)
    print(f"[daylight_lighting_probe] wrote {out}")
PY

echo "======================================================================"
echo "[daylight_lighting_probe] done"
echo "ROOT_DIR=${ROOT_DIR}"
echo "Contact sheets:"
find "${ROOT_DIR}" -maxdepth 1 -type f -name 'contact_sheet_*.png' -print | sort

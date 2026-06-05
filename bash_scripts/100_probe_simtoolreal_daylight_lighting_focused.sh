#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_TIMESTAMP="$(date +%F_%H-%M-%S)"
ROOT_DIR="${ROOT_DIR:-local_logs/${RUN_TIMESTAMP}_simtoolreal_daylight_lighting_focused_probe}"
mkdir -p "${ROOT_DIR}"

# Focused follow-up to 99_probe_simtoolreal_daylight_lighting.sh.  These keep
# the accepted greige floor/backdrop/camera and only explore sun direction,
# softness, warmth, fill, and output exposure.
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
  echo "[daylight_lighting_focused_probe] ${label}"
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

# Current accepted single-sun look, included as a direct reference.
run_probe "00_current" \
  SINGLE_SUN_EXPOSURE=9.35 \
  SINGLE_SUN_ANGLE=0.12 \
  SINGLE_SUN_COLOR_TEMPERATURE=5250 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.97 SINGLE_SUN_COLOR_B=0.90 \
  SINGLE_SUN_YAW_OFFSET_DEG=70 \
  DEFAULT_LIGHT_INTENSITY=360 \
  SKY_DOME_INTENSITY=1200 \
  IMAGE_EXPOSURE=0.0

# The most reference-like broad-sweep direction, but softened enough that the
# table/robot stay readable.
run_probe "01_side_sun_softened" \
  SINGLE_SUN_EXPOSURE=9.65 \
  SINGLE_SUN_ANGLE=0.28 \
  SINGLE_SUN_COLOR_TEMPERATURE=5050 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.96 SINGLE_SUN_COLOR_B=0.84 \
  SINGLE_SUN_YAW_OFFSET_DEG=110 \
  DEFAULT_LIGHT_INTENSITY=520 \
  SKY_DOME_INTENSITY=1700 \
  IMAGE_EXPOSURE=-0.10

# Slightly stronger key light and stronger negative exposure for more contrast.
run_probe "02_golden_contrast_balanced" \
  SINGLE_SUN_EXPOSURE=9.90 \
  SINGLE_SUN_ANGLE=0.22 \
  SINGLE_SUN_COLOR_TEMPERATURE=5000 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.95 SINGLE_SUN_COLOR_B=0.82 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=470 \
  SKY_DOME_INTENSITY=1600 \
  IMAGE_EXPOSURE=-0.16

# Warmer golden-hour feel with enough sky fill to avoid black robot backs.
run_probe "03_warm_sun_high_fill" \
  SINGLE_SUN_EXPOSURE=9.75 \
  SINGLE_SUN_ANGLE=0.24 \
  SINGLE_SUN_COLOR_TEMPERATURE=4850 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.93 SINGLE_SUN_COLOR_B=0.78 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=560 \
  SKY_DOME_INTENSITY=1850 \
  IMAGE_EXPOSURE=-0.14

# Cooler clear daylight; useful if the warm variants make tables/robots too tan.
run_probe "04_clear_sun_side_fill" \
  SINGLE_SUN_EXPOSURE=9.80 \
  SINGLE_SUN_ANGLE=0.24 \
  SINGLE_SUN_COLOR_TEMPERATURE=5700 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.98 SINGLE_SUN_COLOR_B=0.92 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=460 \
  SKY_DOME_INTENSITY=1650 \
  IMAGE_EXPOSURE=-0.12

# Same direction but lower fill for stronger shadows, still less harsh than
# broad-sweep 04.
run_probe "05_crisp_sun_controlled_fill" \
  SINGLE_SUN_EXPOSURE=9.95 \
  SINGLE_SUN_ANGLE=0.16 \
  SINGLE_SUN_COLOR_TEMPERATURE=5000 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.95 SINGLE_SUN_COLOR_B=0.82 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=380 \
  SKY_DOME_INTENSITY=1400 \
  IMAGE_EXPOSURE=-0.18

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
    draw.text((12, 7), f"SimToolReal focused daylight probe: {step_name}", font=font, fill=(255, 255, 255))
    out = root / f"contact_sheet_{step_name}"
    sheet.save(out)
    print(f"[daylight_lighting_focused_probe] wrote {out}")
PY

echo "======================================================================"
echo "[daylight_lighting_focused_probe] done"
echo "ROOT_DIR=${ROOT_DIR}"
echo "Contact sheets:"
find "${ROOT_DIR}" -maxdepth 1 -type f -name 'contact_sheet_*.png' -print | sort

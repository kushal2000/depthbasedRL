#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_TIMESTAMP="$(date +%F_%H-%M-%S)"
ROOT_DIR="${ROOT_DIR:-local_logs/${RUN_TIMESTAMP}_simtoolreal_floor_material_probe}"
mkdir -p "${ROOT_DIR}"

# Keep this probe cheaper than the final 100-env video while preserving the
# accepted camera/lighting/backdrop stack.
NUM_ENVS="${NUM_ENVS:-16}"
GRID_COLS="${GRID_COLS:-4}"
STEPS="${STEPS:-600}"
CAPTURE_PNG_STEPS="${CAPTURE_PNG_STEPS:-0,300,600}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
QUALITY="${QUALITY:-medium}"
RENDER_SPP="${RENDER_SPP:-48}"

run_probe() {
  local label="$1"
  shift
  echo "======================================================================"
  echo "[floor_material_probe] ${label}"
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
    FLOOR_TILE_GAP="${FLOOR_TILE_GAP:-0.002}" \
    OUT_DIR="${ROOT_DIR}/${label}" \
    "$@" \
    bash_scripts/95_render_simtoolreal_ref_pan_cinematic.sh
}

run_probe "00_current_procedural_dark_gray" \
  FLOOR_STYLE=nvidia_precast_concrete_dark_gray \
  FLOOR_TEXTURE_SCALE=2.5

run_probe "01_display_warm_gray_old_escape_hatch" \
  FLOOR_STYLE=display_color \
  FLOOR_COLOR_R=0.47 \
  FLOOR_COLOR_G=0.46 \
  FLOOR_COLOR_B=0.42

run_probe "02_soft_concrete_pbr_scale2p5" \
  FLOOR_STYLE=soft_concrete_pbr_tiles \
  FLOOR_TEXTURE_SCALE=2.5 \
  FLOOR_ROUGHNESS=0.72 \
  FLOOR_NORMAL_STRENGTH=0.16 \
  FLOOR_SPECULAR_LEVEL=0.25

run_probe "03_soft_concrete_pbr_scale4p0" \
  FLOOR_STYLE=soft_concrete_pbr_tiles \
  FLOOR_TEXTURE_SCALE=4.0 \
  FLOOR_ROUGHNESS=0.74 \
  FLOOR_NORMAL_STRENGTH=0.12 \
  FLOOR_SPECULAR_LEVEL=0.22

run_probe "04_matte_warm_gray_pbr_scale2p4" \
  FLOOR_STYLE=matte_warm_gray_pbr_tiles \
  FLOOR_TEXTURE_SCALE=2.4 \
  FLOOR_ROUGHNESS=0.88 \
  FLOOR_NORMAL_STRENGTH=0.10 \
  FLOOR_SPECULAR_LEVEL=0.12

run_probe "05_matte_warm_gray_pbr_scale4p0" \
  FLOOR_STYLE=matte_warm_gray_pbr_tiles \
  FLOOR_TEXTURE_SCALE=4.0 \
  FLOOR_ROUGHNESS=0.90 \
  FLOOR_NORMAL_STRENGTH=0.08 \
  FLOOR_SPECULAR_LEVEL=0.10

run_probe "06_matte_slate_pbr_scale2p4" \
  FLOOR_STYLE=matte_slate_pbr_tiles \
  FLOOR_TEXTURE_SCALE=2.4 \
  FLOOR_ROUGHNESS=0.90 \
  FLOOR_NORMAL_STRENGTH=0.09 \
  FLOOR_SPECULAR_LEVEL=0.10

run_probe "07_warm_limestone_pbr_scale2p0" \
  FLOOR_STYLE=warm_limestone_pbr_tiles \
  FLOOR_TEXTURE_SCALE=2.0 \
  FLOOR_ROUGHNESS=0.68 \
  FLOOR_NORMAL_STRENGTH=0.22 \
  FLOOR_SPECULAR_LEVEL=0.22

run_probe "08_warm_limestone_pbr_scale3p5" \
  FLOOR_STYLE=warm_limestone_pbr_tiles \
  FLOOR_TEXTURE_SCALE=3.5 \
  FLOOR_ROUGHNESS=0.72 \
  FLOOR_NORMAL_STRENGTH=0.18 \
  FLOOR_SPECULAR_LEVEL=0.20

run_probe "09_cool_concrete_pbr_scale2p0" \
  FLOOR_STYLE=cool_concrete_pbr_tiles \
  FLOOR_TEXTURE_SCALE=2.0 \
  FLOOR_ROUGHNESS=0.70 \
  FLOOR_NORMAL_STRENGTH=0.20 \
  FLOOR_SPECULAR_LEVEL=0.22

run_probe "10_white_stone_slabs_scale3p0" \
  FLOOR_STYLE=white_stone_slabs \
  FLOOR_TEXTURE_SCALE=3.0 \
  FLOOR_ROUGHNESS=0.70 \
  FLOOR_NORMAL_STRENGTH=0.10 \
  FLOOR_SPECULAR_LEVEL=0.20

run_probe "11_precast_texture_pbr_scale2p0" \
  FLOOR_STYLE=nvidia_precast_concrete_pbr_tiles \
  FLOOR_TEXTURE_SCALE=2.0 \
  FLOOR_ROUGHNESS=0.74 \
  FLOOR_NORMAL_STRENGTH=0.18 \
  FLOOR_SPECULAR_LEVEL=0.20

run_probe "12_fieldstone_pbr_tinted" \
  FLOOR_STYLE=fieldstone_pbr_tiles \
  FLOOR_COLOR_R=0.60 \
  FLOOR_COLOR_G=0.58 \
  FLOOR_COLOR_B=0.52 \
  FLOOR_TEXTURE_SCALE=2.4 \
  FLOOR_ROUGHNESS=0.78 \
  FLOOR_NORMAL_STRENGTH=0.15 \
  FLOOR_SPECULAR_LEVEL=0.18

CONTACT_SHEET_PYTHON="${CONTACT_SHEET_PYTHON:-python3}"
"${CONTACT_SHEET_PYTHON}" - "${ROOT_DIR}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


root = Path(sys.argv[1])
labels = sorted(path.name for path in root.iterdir() if path.is_dir())
steps = ["step_0000.png", "step_0300.png", "step_0600.png"]
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
    title = f"SimToolReal floor material probe: {step_name}"
    draw.rectangle((0, 0, sheet.width, title_h), fill=(0, 0, 0))
    draw.text((12, 7), title, font=font, fill=(255, 255, 255))
    out = root / f"contact_sheet_{step_name}"
    sheet.save(out)
    print(f"[floor_material_probe] wrote {out}")
PY

echo "======================================================================"
echo "[floor_material_probe] done"
echo "ROOT_DIR=${ROOT_DIR}"
echo "Contact sheets:"
find "${ROOT_DIR}" -maxdepth 1 -type f -name 'contact_sheet_*.png' -print | sort

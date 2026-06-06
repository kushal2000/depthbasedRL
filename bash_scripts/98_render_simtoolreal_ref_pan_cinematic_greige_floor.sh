#!/usr/bin/env bash
set -euo pipefail

# Same cinematic SimToolReal pretraining video as script 95, but with the
# current best floor candidate: one continuous matte greige PBR surface.
# The old floor remains available by running script 95 directly.

export FLOOR_STYLE="${FLOOR_STYLE:-matte_greige_pbr_tiles}"
export FLOOR_COLOR_R="${FLOOR_COLOR_R:-0.50}"
export FLOOR_COLOR_G="${FLOOR_COLOR_G:-0.48}"
export FLOOR_COLOR_B="${FLOOR_COLOR_B:-0.43}"
export FLOOR_TILE_COUNT="${FLOOR_TILE_COUNT:-1}"
export FLOOR_TILE_SIZE="${FLOOR_TILE_SIZE:-120}"
export FLOOR_TILE_GAP="${FLOOR_TILE_GAP:-0}"
export FLOOR_TEXTURE_SCALE="${FLOOR_TEXTURE_SCALE:-12.0}"
export FLOOR_ROUGHNESS="${FLOOR_ROUGHNESS:-0.92}"
export FLOOR_NORMAL_STRENGTH="${FLOOR_NORMAL_STRENGTH:-0.05}"
export FLOOR_SPECULAR_LEVEL="${FLOOR_SPECULAR_LEVEL:-0.06}"
export BACKDROP_GRADIENT_BANDS="${BACKDROP_GRADIENT_BANDS:-16}"

exec bash_scripts/95_render_simtoolreal_ref_pan_cinematic.sh "$@"

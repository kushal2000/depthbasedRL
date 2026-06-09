#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_TIMESTAMP="$(date +%F_%H-%M-%S)"
RUN_ROOT="${RUN_ROOT:-local_logs/${RUN_TIMESTAMP}_simtoolreal_refined_crisper_objectpop_9seeds}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50}"
VIDEO_DIR="${RUN_ROOT}/all_videos"
GRID_VIDEO="${RUN_ROOT}/grid_3x3_refined_crisper_objectpop.mp4"

mkdir -p "${RUN_ROOT}" "${VIDEO_DIR}"

seed_array=(${SEEDS})
if [[ "${#seed_array[@]}" -ne 9 ]]; then
  echo "[error] Expected exactly 9 seeds for a 3x3 grid, got ${#seed_array[@]}: ${SEEDS}" >&2
  exit 1
fi

echo "======================================================================"
echo "[simtoolreal_9seeds] RUN_ROOT=${RUN_ROOT}"
echo "[simtoolreal_9seeds] SEEDS=${SEEDS}"
echo "======================================================================"

for seed in "${seed_array[@]}"; do
  render_dir="${RUN_ROOT}/seed_${seed}_render"
  copied_video="${VIDEO_DIR}/seed_${seed}.mp4"
  echo "======================================================================"
  echo "[simtoolreal_9seeds] rendering seed=${seed}"
  echo "======================================================================"
  SEED="${seed}" \
    OUT_DIR="${render_dir}" \
    MAKE_VIDEO=1 \
    STEPS="${STEPS:-1200}" \
    CAPTURE_PNG_STEPS="${CAPTURE_PNG_STEPS:-0,600,1200}" \
    bash_scripts/105_render_simtoolreal_ref_pan_refined_crisper_objectpop.sh

  if [[ ! -f "${render_dir}/rollout.mp4" ]]; then
    echo "[error] missing rollout: ${render_dir}/rollout.mp4" >&2
    exit 1
  fi
  cp -f "${render_dir}/rollout.mp4" "${copied_video}"
  echo "[simtoolreal_9seeds] copied ${copied_video}"
done

ffmpeg_inputs=()
filter_parts=()
stack_inputs=""
for idx in "${!seed_array[@]}"; do
  seed="${seed_array[${idx}]}"
  video="${VIDEO_DIR}/seed_${seed}.mp4"
  ffmpeg_inputs+=(-i "${video}")
  filter_parts+=(
    "[${idx}:v]scale=640:360,setsar=1,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='seed ${seed}':x=16:y=14:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.60[v${idx}]"
  )
  stack_inputs+="[v${idx}]"
done

filter_complex="$(IFS=';'; echo "${filter_parts[*]}");${stack_inputs}xstack=inputs=9:layout=0_0|640_0|1280_0|0_360|640_360|1280_360|0_720|640_720|1280_720:fill=black[out]"

echo "======================================================================"
echo "[simtoolreal_9seeds] writing grid video"
echo "======================================================================"
ffmpeg -hide_banner -loglevel error -y \
  "${ffmpeg_inputs[@]}" \
  -filter_complex "${filter_complex}" \
  -map "[out]" \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  "${GRID_VIDEO}"

echo "======================================================================"
echo "[simtoolreal_9seeds] done"
echo "RUN_ROOT=${RUN_ROOT}"
echo "VIDEO_DIR=${VIDEO_DIR}"
echo "GRID_VIDEO=${GRID_VIDEO}"
echo "======================================================================"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEFAULT_CHECKPOINT="${DEFAULT_STUDENT_CHECKPOINT:-/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/10_juno_rot6d_medium_noise_camrand20mm2deg_256env_48h/checkpoints/student_latest.pt}"
if [[ ! -f "${DEFAULT_CHECKPOINT}" && -f "distillation_runs/10_local_rot6d_medium_noise_camrand20mm2deg_256env/checkpoints/student_latest.pt" ]]; then
  DEFAULT_CHECKPOINT="distillation_runs/10_local_rot6d_medium_noise_camrand20mm2deg_256env/checkpoints/student_latest.pt"
fi
CHECKPOINT="${1:-${DEFAULT_CHECKPOINT}}"
DEBUG_DIR="${DEBUG_DIR:-./depth_student_debug}"
DEBUG_VIDEO="${DEBUG_VIDEO:-${DEBUG_DIR}/depth_debug.mp4}"
RUN_DURATION_S="${RUN_DURATION_S:-10}"
DEBUG_EVERY_N="${DEBUG_EVERY_N:-2}"
DEBUG_VIDEO_FPS="${DEBUG_VIDEO_FPS:-30}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-0.5}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Pass a checkpoint as arg 1, or set DEFAULT_STUDENT_CHECKPOINT." >&2
  exit 1
fi

mkdir -p "${DEBUG_DIR}"

python deployment/student_depth_policy_node_nonblocking.py \
  --checkpoint_path "${CHECKPOINT}" \
  --run_duration_s "${RUN_DURATION_S}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --debug_depth_dir "${DEBUG_DIR}" \
  --debug_depth_every_n "${DEBUG_EVERY_N}" \
  --debug_depth_video_path "${DEBUG_VIDEO}" \
  --debug_depth_video_fps "${DEBUG_VIDEO_FPS}" \
  --no-publish_joint_commands

echo "Depth debug files: ${DEBUG_DIR}"
echo "Depth debug video: ${DEBUG_VIDEO}"

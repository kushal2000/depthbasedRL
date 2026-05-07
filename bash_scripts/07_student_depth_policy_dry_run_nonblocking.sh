#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEFAULT_CHECKPOINT="${DEFAULT_STUDENT_CHECKPOINT:-/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/10_juno_rot6d_medium_noise_camrand20mm2deg_256env_48h/checkpoints/student_latest.pt}"
if [[ ! -f "${DEFAULT_CHECKPOINT}" && -f "distillation_runs/10_local_rot6d_medium_noise_camrand20mm2deg_256env/checkpoints/student_latest.pt" ]]; then
  DEFAULT_CHECKPOINT="distillation_runs/10_local_rot6d_medium_noise_camrand20mm2deg_256env/checkpoints/student_latest.pt"
fi
CHECKPOINT="${1:-${DEFAULT_CHECKPOINT}}"
RUN_DURATION_S="${RUN_DURATION_S:-30}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-1.0}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Pass a checkpoint as arg 1, or set DEFAULT_STUDENT_CHECKPOINT." >&2
  exit 1
fi

python deployment/student_depth_policy_node_nonblocking.py \
  --checkpoint_path "${CHECKPOINT}" \
  --run_duration_s "${RUN_DURATION_S}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --no-publish_joint_commands

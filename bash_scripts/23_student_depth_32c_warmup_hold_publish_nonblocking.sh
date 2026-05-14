#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEFAULT_CHECKPOINT="/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
if [[ ! -f "${DEFAULT_CHECKPOINT}" ]]; then
  DEFAULT_CHECKPOINT="distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
fi
CHECKPOINT="${1:-${DEFAULT_STUDENT_CHECKPOINT:-${DEFAULT_CHECKPOINT}}}"
RUN_DURATION_S="${RUN_DURATION_S:-5}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-0.5}"
WARMUP_STEPS="${WARMUP_STEPS:-60}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Pass a checkpoint as arg 1, or set DEFAULT_STUDENT_CHECKPOINT." >&2
  exit 1
fi

echo "Publishing current sensed joint positions as hold targets during warmup only."
echo "Policy joint commands remain disabled after warmup."

python deployment/student_depth_policy_node_nonblocking.py \
  --checkpoint_path "${CHECKPOINT}" \
  --run_duration_s "${RUN_DURATION_S}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --warmup_publish_current_targets \
  --publish_object_pose \
  --no-publish_joint_commands

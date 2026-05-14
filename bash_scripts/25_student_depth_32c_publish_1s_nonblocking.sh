#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEFAULT_CHECKPOINT="/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
if [[ ! -f "${DEFAULT_CHECKPOINT}" ]]; then
  DEFAULT_CHECKPOINT="distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
fi
CHECKPOINT="${1:-${DEFAULT_STUDENT_CHECKPOINT:-${DEFAULT_CHECKPOINT}}}"
RUN_DURATION_S="${RUN_DURATION_S:--1}"
PUBLISH_DURATION_S="${PUBLISH_DURATION_S:-1.0}"
STATUS_INTERVAL_S="${STATUS_INTERVAL_S:-0.5}"
MAX_ARM_TARGET_DELTA_DEG="${MAX_ARM_TARGET_DELTA_DEG:-15}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Pass a checkpoint as arg 1, or set DEFAULT_STUDENT_CHECKPOINT." >&2
  exit 1
fi

python deployment/student_depth_policy_node_nonblocking.py \
  --checkpoint_path "${CHECKPOINT}" \
  --run_duration_s "${RUN_DURATION_S}" \
  --status_interval_s "${STATUS_INTERVAL_S}" \
  --publish_joint_commands \
  --publish_joint_commands_duration_s "${PUBLISH_DURATION_S}" \
  --max_arm_target_delta_deg "${MAX_ARM_TARGET_DELTA_DEG}" \
  --publish_object_pose

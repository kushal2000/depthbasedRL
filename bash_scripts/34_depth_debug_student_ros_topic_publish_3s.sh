#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/depth_deploy_debug_env.sh"

DEFAULT_CHECKPOINT="/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
if [[ ! -f "${DEFAULT_CHECKPOINT}" ]]; then
  DEFAULT_CHECKPOINT="distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
fi
CHECKPOINT="${1:-${DEFAULT_STUDENT_CHECKPOINT:-${DEFAULT_CHECKPOINT}}}"
RUN_DURATION_S="${RUN_DURATION_S:-10}"
PUBLISH_DURATION_S="${PUBLISH_DURATION_S:-3}"
RECORD_DIR="${RECORD_DIR:-./student_depth_ros_topic_publish_recording}"
PUBLISH_DEBUG_POLICY_DEPTH="${PUBLISH_DEBUG_POLICY_DEPTH:-1}"
EXTRA_ARGS=()
if [[ "${PUBLISH_DEBUG_POLICY_DEPTH}" == "1" ]]; then
  EXTRA_ARGS+=(--publish_debug_policy_depth)
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

python deployment/student_depth_policy_node_nonblocking.py \
  --checkpoint_path "${CHECKPOINT}" \
  --depth_source ros_topic \
  --run_duration_s "${RUN_DURATION_S}" \
  --publish_object_pose \
  --publish_joint_commands \
  --publish_joint_commands_duration_s "${PUBLISH_DURATION_S}" \
  --record_rollout_dir "${RECORD_DIR}" \
  --record_depth_format uint8 \
  "${EXTRA_ARGS[@]}"

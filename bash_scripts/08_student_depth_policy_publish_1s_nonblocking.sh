#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHECKPOINT="${1:-distillation_runs/09ctd_rot6d_medium_noise_camrand50mm5deg/checkpoints/student_latest.pt}"

python deployment/student_depth_policy_node_nonblocking.py \
  --checkpoint_path "${CHECKPOINT}" \
  --publish_joint_commands \
  --publish_joint_commands_duration_s 1.0

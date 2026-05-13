#!/usr/bin/env bash
# Evaluate the 32c L-peg depth policy with the same camera/noise settings used in training.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DEFAULT_CHECKPOINT="/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
if [[ ! -f "$DEFAULT_CHECKPOINT" ]]; then
  DEFAULT_CHECKPOINT="distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
fi
CHECKPOINT="${1:-${DEFAULT_STUDENT_CHECKPOINT:-$DEFAULT_CHECKPOINT}}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-eval_runs/32c_L_defaultcam_q1_medium_noise_camrand20mm2deg_${RUN_STAMP}}"
NUM_ENVS="${NUM_ENVS:-16}"
NUM_STEPS="${NUM_STEPS:-2400}"
NUM_COMPLETED_EPISODES="${NUM_COMPLETED_EPISODES:-64}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x ".venv-isaacsim-py311/bin/python" ]]; then
  PYTHON_BIN=".venv-isaacsim-py311/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

cmd=(
  "$PYTHON_BIN" isaacsimenvs/eval_depth_policy.py
  --student_checkpoint "$CHECKPOINT"
  --student_input camera
  --student_arch mono_transformer_recurrent
  --policy_source "${POLICY_SOURCE:-student}"
  --num_envs "$NUM_ENVS"
  --num_steps "$NUM_STEPS"
  --num_completed_episodes "$NUM_COMPLETED_EPISODES"
  --log_interval "$LOG_INTERVAL"
  --run_dir "$RUN_DIR"
  --aux_pose_mode rot6d_keypoints
  --aux_object_pos_weight 1.0
  --aux_object_keypoint_weight 1.0
  --peg_urdf assets/urdf/peg_in_hole/peg_L/peg_L.urdf
  --peg_goal_mode "${PEG_GOAL_MODE:-preInsertAndFinal}"
  --student_image_delay_queue_size 1
  --depth_noise_profile medium
  --student_camera_preset default
  --camera_pose_randomization_profile custom
  --camera_pose_randomization_mode startup
  --camera_pos_noise_m 0.02 0.02 0.02
  --camera_rot_noise_deg 2 2 2
  --capture_viewer
  --capture_viewer_len "${CAPTURE_VIEWER_LEN:-600}"
  --depth_debug_interval "${DEPTH_DEBUG_INTERVAL:-600}"
  --depth_rollout_video
  --depth_rollout_video_len "${DEPTH_ROLLOUT_VIDEO_LEN:-600}"
  --depth_rollout_video_fps "${DEPTH_ROLLOUT_VIDEO_FPS:-60}"
  --depth_rollout_video_interval "${DEPTH_ROLLOUT_VIDEO_INTERVAL:-600}"
  --headless
)

if [[ -n "${SEED:-}" ]]; then
  cmd+=(--seed "$SEED")
fi

if [[ -n "${PEG_ENABLE_RETRACT:-}" ]]; then
  if [[ "$PEG_ENABLE_RETRACT" == "1" || "$PEG_ENABLE_RETRACT" == "true" ]]; then
    cmd+=(--peg_enable_retract)
  elif [[ "$PEG_ENABLE_RETRACT" == "0" || "$PEG_ENABLE_RETRACT" == "false" ]]; then
    cmd+=(--no-peg_enable_retract)
  else
    echo "PEG_ENABLE_RETRACT must be one of 1/0/true/false, got: $PEG_ENABLE_RETRACT" >&2
    exit 2
  fi
fi

if [[ "${SERVE_VISER:-0}" == "1" ]]; then
  cmd+=(
    --serve_viser
    --viser_port "${VISER_PORT:-8080}"
    --viser_env_id "${VISER_ENV_ID:-0}"
    --viser_update_interval "${VISER_UPDATE_INTERVAL:-1}"
    --viser_point_stride "${VISER_POINT_STRIDE:-4}"
  )
  if [[ "${VISER_START_PAUSED:-1}" == "0" || "${VISER_START_PAUSED:-1}" == "false" ]]; then
    cmd+=(--no-viser_start_paused)
  fi
fi

if [[ "${WANDB:-0}" == "1" ]]; then
  cmd+=(
    --wandb
    --wandb_group "${WANDB_GROUP:-2026-05-13_DepthPolicyEval}"
    --wandb_name "${WANDB_NAME:-32c_L_defaultcam_eval}"
  )
fi

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"

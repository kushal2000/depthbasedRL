#!/usr/bin/env bash
set -euo pipefail

# Launch a localhost-only IsaacSim source node + student dry-run stack.
# The student consumes the IsaacSim-published ROS depth topic and records a
# rollout NPZ. Joint commands are disabled by default.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SESSION="${SESSION:-depth_isaac_debug}"
DEFAULT_CHECKPOINT="/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
if [[ ! -f "${DEFAULT_CHECKPOINT}" ]]; then
  DEFAULT_CHECKPOINT="distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
fi
CHECKPOINT="${1:-${DEFAULT_STUDENT_CHECKPOINT:-${DEFAULT_CHECKPOINT}}}"
ISAAC_DEPTH_SCRIPT="${ISAAC_DEPTH_SCRIPT:-bash_scripts/45_isaac_depth_ros_clean_no_camrand.sh}"
STUDENT_SCRIPT="${STUDENT_SCRIPT:-bash_scripts/33_depth_debug_student_ros_topic_dry_run.sh}"
DEPTH_EVERY_N="${DEPTH_EVERY_N:-1}"
RUN_DURATION_S="${RUN_DURATION_S:-30}"
RECORD_DIR="${RECORD_DIR:-./student_depth_ros_topic_recording}"
PUBLISH_DEBUG_POLICY_DEPTH="${PUBLISH_DEBUG_POLICY_DEPTH:-1}"
OBJECT_NAME="${OBJECT_NAME:-peg_L}"
LOAD_POINT_CLOUD="${LOAD_POINT_CLOUD:-0}"
LOAD_POLICY_DEPTH_IMAGE="${LOAD_POLICY_DEPTH_IMAGE:-1}"
START_VISER_AFTER_S="${START_VISER_AFTER_S:-8}"
START_STUDENT_AFTER_S="${START_STUDENT_AFTER_S:-20}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${ISAAC_DEPTH_SCRIPT}" ]]; then
  echo "Isaac depth script not found: ${ISAAC_DEPTH_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${STUDENT_SCRIPT}" ]]; then
  echo "Student script not found: ${STUDENT_SCRIPT}" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for this multi-window launcher." >&2
  exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

q() {
  printf "%q" "$1"
}

REPO_ROOT="$(pwd)"
CHECKPOINT_Q="$(q "${CHECKPOINT}")"
RECORD_DIR_Q="$(q "${RECORD_DIR}")"
REPO_ROOT_Q="$(q "${REPO_ROOT}")"
ISAAC_DEPTH_SCRIPT_Q="$(q "${ISAAC_DEPTH_SCRIPT}")"
STUDENT_SCRIPT_Q="$(q "${STUDENT_SCRIPT}")"

tmux new-session -d -s "${SESSION}" -n roscore \
  "cd ${REPO_ROOT_Q}; bash bash_scripts/30_depth_debug_roscore_local.sh"
tmux set-option -t "${SESSION}" remain-on-exit on >/dev/null

tmux new-window -t "${SESSION}" -n isaac_depth \
  "cd ${REPO_ROOT_Q}; sleep 2; DEPTH_EVERY_N=${DEPTH_EVERY_N} bash ${ISAAC_DEPTH_SCRIPT_Q}"
tmux new-window -t "${SESSION}" -n viser \
  "cd ${REPO_ROOT_Q}; sleep ${START_VISER_AFTER_S}; OBJECT_NAME=${OBJECT_NAME} LOAD_POINT_CLOUD=${LOAD_POINT_CLOUD} LOAD_POLICY_DEPTH_IMAGE=${LOAD_POLICY_DEPTH_IMAGE} bash bash_scripts/35_depth_debug_visualization_with_depth.sh"
tmux new-window -t "${SESSION}" -n student \
  "cd ${REPO_ROOT_Q}; sleep ${START_STUDENT_AFTER_S}; RUN_DURATION_S=${RUN_DURATION_S} RECORD_DIR=${RECORD_DIR_Q} PUBLISH_DEBUG_POLICY_DEPTH=${PUBLISH_DEBUG_POLICY_DEPTH} bash ${STUDENT_SCRIPT_Q} ${CHECKPOINT_Q}"

cat <<EOF
Started tmux session: ${SESSION}

Windows:
  roscore      localhost-only ROS master
  isaac_depth  ${ISAAC_DEPTH_SCRIPT} with DEPTH_EVERY_N=${DEPTH_EVERY_N}
  viser        live Viser listener
  student      ${STUDENT_SCRIPT}, records to ${RECORD_DIR}

Attach:
  tmux attach -t ${SESSION}

Common variants:
  ISAAC_DEPTH_SCRIPT=bash_scripts/42_isaac_depth_ros_train_settings_noise_camrand.sh bash_scripts/61_depth_debug_local_isaac_pipeline_tmux.sh
  ISAAC_DEPTH_SCRIPT=bash_scripts/43_isaac_depth_ros_no_noise_camrand.sh bash_scripts/61_depth_debug_local_isaac_pipeline_tmux.sh
  ISAAC_DEPTH_SCRIPT=bash_scripts/45_isaac_depth_ros_clean_no_camrand.sh bash_scripts/61_depth_debug_local_isaac_pipeline_tmux.sh

Visualize newest recording after student exits:
  OBJECT_NAME=peg_L bash bash_scripts/63_visualize_latest_student_depth_recording.sh ${RECORD_DIR}
EOF

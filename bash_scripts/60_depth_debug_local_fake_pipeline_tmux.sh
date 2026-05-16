#!/usr/bin/env bash
set -euo pipefail

# Launch the localhost-only fake robot + fake depth + student dry-run stack.
# This never talks to a real robot because all child scripts force
# ROS_MASTER_URI=http://127.0.0.1:11311.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SESSION="${SESSION:-depth_fake_debug}"
DEFAULT_CHECKPOINT="/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
if [[ ! -f "${DEFAULT_CHECKPOINT}" ]]; then
  DEFAULT_CHECKPOINT="distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt"
fi
CHECKPOINT="${1:-${DEFAULT_STUDENT_CHECKPOINT:-${DEFAULT_CHECKPOINT}}}"
DEPTH_FILE="${DEPTH_FILE:-fake_depth_0p85m_160x90.npz}"
RUN_DURATION_S="${RUN_DURATION_S:-10}"
RECORD_DIR="${RECORD_DIR:-./student_depth_ros_topic_recording}"
PUBLISH_DEBUG_POLICY_DEPTH="${PUBLISH_DEBUG_POLICY_DEPTH:-1}"
RATE_HZ="${RATE_HZ:-30}"
OBJECT_NAME="${OBJECT_NAME:-peg_L}"
LOAD_POINT_CLOUD="${LOAD_POINT_CLOUD:-0}"
LOAD_POLICY_DEPTH_IMAGE="${LOAD_POLICY_DEPTH_IMAGE:-1}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${DEPTH_FILE}" && "${DEPTH_FILE}" == "fake_depth_0p85m_160x90.npz" ]]; then
  python - <<'PY'
import numpy as np

depth = np.full((90, 160), 0.85, dtype=np.float32)
np.savez_compressed("fake_depth_0p85m_160x90.npz", raw_depth_m=depth)
PY
fi
if [[ ! -f "${DEPTH_FILE}" ]]; then
  echo "Depth file not found: ${DEPTH_FILE}" >&2
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
DEPTH_FILE_Q="$(q "${DEPTH_FILE}")"
RECORD_DIR_Q="$(q "${RECORD_DIR}")"
REPO_ROOT_Q="$(q "${REPO_ROOT}")"

tmux new-session -d -s "${SESSION}" -n roscore \
  "cd ${REPO_ROOT_Q}; bash bash_scripts/30_depth_debug_roscore_local.sh"
tmux set-option -t "${SESSION}" remain-on-exit on >/dev/null

tmux new-window -t "${SESSION}" -n fake_robot \
  "cd ${REPO_ROOT_Q}; sleep 2; bash bash_scripts/31_depth_debug_fake_robot.sh"
tmux new-window -t "${SESSION}" -n fake_depth \
  "cd ${REPO_ROOT_Q}; sleep 2; RATE_HZ=${RATE_HZ} bash bash_scripts/32_depth_debug_fake_depth_from_file.sh ${DEPTH_FILE_Q}"
tmux new-window -t "${SESSION}" -n viser \
  "cd ${REPO_ROOT_Q}; sleep 3; OBJECT_NAME=${OBJECT_NAME} LOAD_POINT_CLOUD=${LOAD_POINT_CLOUD} LOAD_POLICY_DEPTH_IMAGE=${LOAD_POLICY_DEPTH_IMAGE} bash bash_scripts/35_depth_debug_visualization_with_depth.sh"
tmux new-window -t "${SESSION}" -n student \
  "cd ${REPO_ROOT_Q}; sleep 4; RUN_DURATION_S=${RUN_DURATION_S} RECORD_DIR=${RECORD_DIR_Q} PUBLISH_DEBUG_POLICY_DEPTH=${PUBLISH_DEBUG_POLICY_DEPTH} bash bash_scripts/33_depth_debug_student_ros_topic_dry_run.sh ${CHECKPOINT_Q}"

cat <<EOF
Started tmux session: ${SESSION}

Windows:
  roscore     localhost-only ROS master
  fake_robot  fake iiwa/sharpa joint states
  fake_depth  fixed/replayed depth from ${DEPTH_FILE}
  viser       live Viser listener
  student     dry-run policy, no joint commands, records to ${RECORD_DIR}

Attach:
  tmux attach -t ${SESSION}

Visualize newest recording after student exits:
  OBJECT_NAME=peg_L bash bash_scripts/63_visualize_latest_student_depth_recording.sh ${RECORD_DIR}
EOF

#!/usr/bin/env bash
set -euo pipefail

# Visualize a student-depth rollout recording. Pass either a specific .npz file
# or a directory; if omitted, this picks the newest recording from common debug
# output directories.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT="${1:-}"
OBJECT_NAME="${OBJECT_NAME:-peg_L}"

find_latest_recording() {
  local search_path="$1"
  find "${search_path}" -maxdepth 1 -type f -name "*student_depth_rollout.npz" 2>/dev/null \
    -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2-
}

if [[ -n "${INPUT}" && -f "${INPUT}" ]]; then
  RECORDING="${INPUT}"
elif [[ -n "${INPUT}" && -d "${INPUT}" ]]; then
  RECORDING="$(find_latest_recording "${INPUT}")"
else
  RECORDING="$(
    {
      find_latest_recording student_depth_zed_direct_recording
      find_latest_recording student_depth_zed_direct_publish_recording
      find_latest_recording student_depth_ros_topic_recording
      find_latest_recording student_depth_ros_topic_publish_recording
    } | sed '/^$/d' | while IFS= read -r path; do
      printf "%s %s\n" "$(stat -c %Y "${path}")" "${path}"
    done | sort -nr | head -1 | cut -d' ' -f2-
  )"
fi

if [[ -z "${RECORDING:-}" || ! -f "${RECORDING}" ]]; then
  echo "No student depth rollout recording found." >&2
  echo "Pass a .npz file or a directory containing *student_depth_rollout.npz." >&2
  exit 1
fi

echo "Visualizing recording: ${RECORDING}"
OBJECT_NAME="${OBJECT_NAME}" bash bash_scripts/41_visualize_student_depth_rollout_recording.sh "${RECORDING}"

#!/usr/bin/env bash
# Source this before running FoundationPose ROS-topic debug scripts.
# It intentionally uses localhost ROS and the existing FoundationPose conda env.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOUNDATIONPOSE_ROOT="${FOUNDATIONPOSE_ROOT:-/home/tylerlum/github_repos/FoundationPose}"

_FP_NOUNSET_WAS_ON=0
case "$-" in
  *u*)
    _FP_NOUNSET_WAS_ON=1
    set +u
    ;;
esac
eval "$(/home/tylerlum/miniforge3/bin/conda shell.bash hook)"
conda activate "${FOUNDATIONPOSE_CONDA_ENV:-foundationpose}"
if [[ "${_FP_NOUNSET_WAS_ON}" == "1" ]]; then
  set -u
fi

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
export ROS_IP="${ROS_IP:-127.0.0.1}"
unset ROS_HOSTNAME
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CONDA_PREFIX}/lib"
export PYTHONPATH="${FOUNDATIONPOSE_ROOT}:${PYTHONPATH:-}"

cd "${FOUNDATIONPOSE_ROOT}"
echo "[foundationpose_ros_env] repo=${FOUNDATIONPOSE_ROOT}"
echo "[foundationpose_ros_env] conda=${CONDA_DEFAULT_ENV} python=$(command -v python)"
echo "[foundationpose_ros_env] ROS_MASTER_URI=${ROS_MASTER_URI} ROS_IP=${ROS_IP}"

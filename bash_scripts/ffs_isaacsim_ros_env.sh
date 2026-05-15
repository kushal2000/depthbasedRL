#!/usr/bin/env bash
# Source this for the experimental IsaacSim + ROS + Fast-FoundationStereo env.
# It uses an isolated conda env and forces localhost ROS so it cannot talk to
# the real robot ROS master by accident.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

_FFS_NOUNSET_WAS_ON=0
case "$-" in
  *u*)
    _FFS_NOUNSET_WAS_ON=1
    set +u
    ;;
esac
eval "$(/home/tylerlum/miniforge3/bin/conda shell.bash hook)"
conda activate "${FFS_ISAACSIM_ROS_CONDA_ENV:-ffs_isaacsim_ros_py311}"
if [[ "${_FFS_NOUNSET_WAS_ON}" == "1" ]]; then
  set -u
fi

export ROS_MASTER_URI="http://127.0.0.1:11311"
export ROS_IP="127.0.0.1"
unset ROS_HOSTNAME
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ISAACSIM_PYTHON="${ISAACSIM_PYTHON:-${CONDA_PREFIX}/bin/python}"
export FAST_FOUNDATIONSTEREO_ROOT="${FAST_FOUNDATIONSTEREO_ROOT:-/home/tylerlum/github_repos/Fast-FoundationStereo}"
export FFS_MODEL_PATH="${FFS_MODEL_PATH:-${FAST_FOUNDATIONSTEREO_ROOT}/weights/20-30-48/model_best_bp2_serialize.pth}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export ISAACSIM_CACHE_ROOT="${ISAACSIM_CACHE_ROOT:-/tmp/${USER}/ffs_isaacsim_ros_cache}"
export OMNI_USER_DIR="${OMNI_USER_DIR:-${ISAACSIM_CACHE_ROOT}/omni_user}"
export OMNI_CACHE_DIR="${OMNI_CACHE_DIR:-${ISAACSIM_CACHE_ROOT}/omni_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ISAACSIM_CACHE_ROOT}/xdg_cache}"
mkdir -p "${ISAACSIM_CACHE_ROOT}" "${OMNI_USER_DIR}" "${OMNI_CACHE_DIR}" "${XDG_CACHE_HOME}"

echo "[ffs_isaacsim_ros_env] repo=${REPO_ROOT}"
echo "[ffs_isaacsim_ros_env] conda=${CONDA_DEFAULT_ENV} python=$(command -v python)"
echo "[ffs_isaacsim_ros_env] ISAACSIM_PYTHON=${ISAACSIM_PYTHON}"
echo "[ffs_isaacsim_ros_env] ROS_MASTER_URI=${ROS_MASTER_URI} ROS_IP=${ROS_IP}"

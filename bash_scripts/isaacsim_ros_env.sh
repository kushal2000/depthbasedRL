#!/usr/bin/env bash
# Source this before running IsaacSim scripts that also need ROS Python modules.
# It activates the ROS conda env for rospy/sensor_msgs, then runs IsaacSim with
# the IsaacSim Python while adding the ROS site-packages to PYTHONPATH.

source "$(dirname "${BASH_SOURCE[0]}")/depth_deploy_debug_env.sh"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ISAACSIM_PYTHON="${ISAACSIM_PYTHON:-${PWD}/.venv-isaacsim-py311/bin/python}"
export PYTHONPATH="${CONDA_PREFIX}/lib/python3.11/site-packages:${PYTHONPATH:-}"
export ISAACSIM_CACHE_ROOT="${ISAACSIM_CACHE_ROOT:-/tmp/${USER}/isaacsim_depth_deploy_cache}"
export OMNI_USER_DIR="${OMNI_USER_DIR:-${ISAACSIM_CACHE_ROOT}/omni_user}"
export OMNI_CACHE_DIR="${OMNI_CACHE_DIR:-${ISAACSIM_CACHE_ROOT}/omni_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ISAACSIM_CACHE_ROOT}/xdg_cache}"
mkdir -p "${ISAACSIM_CACHE_ROOT}" "${OMNI_USER_DIR}" "${OMNI_CACHE_DIR}" "${XDG_CACHE_HOME}"

echo "[isaacsim_ros_env] ISAACSIM_PYTHON=${ISAACSIM_PYTHON}"
echo "[isaacsim_ros_env] PYTHONPATH prepends ROS site-packages from ${CONDA_PREFIX}"

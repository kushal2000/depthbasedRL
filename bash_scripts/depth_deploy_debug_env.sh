#!/usr/bin/env bash
# Source this from ROS deployment-debug scripts. It intentionally forces a
# localhost ROS master so these fake/sim nodes cannot talk to the real robot.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

eval "$(/home/tylerlum/miniforge3/bin/conda shell.bash hook)"
conda activate "${DEPTH_DEPLOY_CONDA_ENV:-simtoolreal_ros_env}"

export ROS_MASTER_URI="http://127.0.0.1:11311"
export ROS_IP="127.0.0.1"
unset ROS_HOSTNAME
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CONDA_PREFIX}/lib"

echo "[depth_deploy_debug_env] repo=${REPO_ROOT}"
echo "[depth_deploy_debug_env] conda=${CONDA_DEFAULT_ENV} python=$(command -v python)"
echo "[depth_deploy_debug_env] ROS_MASTER_URI=${ROS_MASTER_URI} ROS_IP=${ROS_IP}"

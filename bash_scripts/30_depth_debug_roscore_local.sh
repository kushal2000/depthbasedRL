#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/depth_deploy_debug_env.sh"
exec roscore

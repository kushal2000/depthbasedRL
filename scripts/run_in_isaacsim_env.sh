#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ISAACSIM_ENV_DIR:-$ROOT_DIR/.venv-isaacsim-py311}"
PYTHON_VERSION_DIR="$ENV_DIR/lib/python3.11"
ISAACLAB_SOURCE_DIR="$PYTHON_VERSION_DIR/site-packages/isaaclab/source/isaaclab"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "Isaac Sim env not found at $ENV_DIR" >&2
    echo "Create it first with: $ROOT_DIR/scripts/setup_isaacsim_uv_env.sh" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 1
fi

cd "$ROOT_DIR"
if [[ -d "$ISAACLAB_SOURCE_DIR" ]]; then
    export PYTHONPATH="$ROOT_DIR:$ISAACLAB_SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
COMMAND="$1"
shift
exec "$ENV_DIR/bin/$COMMAND" "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-$ROOT_DIR/.venv-isaacsim-py311}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "Repo root: $ROOT_DIR"
echo "Isaac Sim env: $ENV_DIR"
echo "Python version: $PYTHON_VERSION"

cd "$ROOT_DIR"

uv python install "$PYTHON_VERSION"
uv venv "$ENV_DIR" --python "$PYTHON_VERSION"

uv pip install --python "$ENV_DIR/bin/python" torch \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$ENV_DIR/bin/python" -e ./rl_games/
uv pip install --python "$ENV_DIR/bin/python" \
    omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy requests tqdm tyro
uv pip install --python "$ENV_DIR/bin/python" \
    "isaaclab[isaacsim,all]==2.3.2.post1" \
    --extra-index-url https://pypi.nvidia.com

cat <<EOF

Isaac Sim environment created successfully.

Activate it with:
  source "$ENV_DIR/bin/activate"

For non-interactive Isaac Sim launches, set:
  export OMNI_KIT_ACCEPT_EULA=YES

Or run commands without activating:
  "$ROOT_DIR/scripts/run_in_isaacsim_env.sh" python isaacsim_conversion/test_inference.py

Do not run:
  uv pip install -e .

That installs the root package pins used by the Python 3.8 Isaac Gym workflow
and will conflict with the Python 3.11 Isaac Sim stack.
EOF

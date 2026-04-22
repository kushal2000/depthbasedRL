#!/usr/bin/env bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-move}"
PARTITION="${PARTITION:-move-interactive}"
NODELIST="${NODELIST:-}"
REPO_URL="${REPO_URL:-https://github.com/kushal2000/depthbasedRL.git}"
BRANCH="${BRANCH:-2026-04-15_distillation}"
SHARED_ROOT="${SHARED_ROOT:-/move/u/$USER/github_repos}"
REPO_DIR="${REPO_DIR:-$SHARED_ROOT/depthbasedRL}"
OMNI_CACHE="${OMNI_KIT_CACHE_PATH:-/tmp/${USER}_ov_cache}"

echo "Account: $ACCOUNT"
echo "Partition: $PARTITION"
echo "Branch: $BRANCH"
echo "Shared root: $SHARED_ROOT"
echo "Repo dir: $REPO_DIR"
echo "Omniverse cache: $OMNI_CACHE"

mkdir -p "$SHARED_ROOT"
if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
fi

cd "$REPO_DIR"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export OMNI_KIT_CACHE_PATH="$OMNI_CACHE"
mkdir -p "$OMNI_KIT_CACHE_PATH"

hostname
nvidia-smi || true
df -h / /move /tmp 2>/dev/null || true

./scripts/setup_isaacsim_uv_env.sh
./scripts/run_in_isaacsim_env.sh python download_pretrained_policy.py
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/test_inference.py

cat <<EOF

Cluster bootstrap complete.

Suggested teacher smoke test:
  ./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill_eval.py \\
    --mode teacher_eval \\
    --headless \\
    --max_steps 50 \\
    --num_envs 1 \\
    --teacher_checkpoint pretrained_policy/model.pth \\
    --teacher_config pretrained_policy/config.yaml \\
    --camera_config isaacsim_conversion/configs/hammer_camera_depth_160x90.yaml
EOF

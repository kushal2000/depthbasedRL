#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="$1"
REPO="$2"
ENV_DIR="$3"
DISTILL_CONFIG="$4"
CAMERA_CONFIG="$5"
STUDENT_INPUT="$6"
STUDENT_ARCH="$7"
STUDENT_MODALITY="$8"
ONLINE_ITERS="$9"
ONLINE_LOG_INTERVAL="${10}"
ONLINE_UPDATE_INTERVAL="${11}"
WANDB_PROJECT="${12}"
shift 12
ENV_COUNTS=("$@")

cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACSIM_ENV_DIR="$ENV_DIR"
export WANDB_DATA_DIR=/move/u/tylerlum/wandb_data
export OMNI_KIT_CACHE_PATH=/tmp/${USER}_ov_cache_${SLURM_JOB_ID}
export UV_CACHE_DIR=/tmp/${USER}_uv_cache_${SLURM_JOB_ID}
export PIP_CACHE_DIR=/tmp/${USER}_pip_cache_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/${USER}_xdg_cache_${SLURM_JOB_ID}
export XDG_CONFIG_HOME=/tmp/${USER}_xdg_config_${SLURM_JOB_ID}
export XDG_DATA_HOME=/tmp/${USER}_xdg_data_${SLURM_JOB_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WANDB_DATA_DIR" "$OMNI_KIT_CACHE_PATH" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" slurm_logs distillation_runs

if [[ -s /juno/u/tylerlum/.wandb_api_key ]]; then
  export WANDB_API_KEY="$(tr -d "\n\r" < /juno/u/tylerlum/.wandb_api_key)"
  echo "WANDB_API_KEY loaded from file (not printed)"
else
  echo "WARNING: /juno/u/tylerlum/.wandb_api_key missing or empty; wandb may be disabled"
fi

memlog="slurm_logs/${JOB_NAME}_${SLURM_JOB_ID}_nvidia_smi.csv"
( while true; do date +%s; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; sleep 10; done ) > "$memlog" 2>&1 &
monpid=$!
trap "kill $monpid 2>/dev/null || true" EXIT

MODALITY_ARGS=()
if [[ -n "$STUDENT_MODALITY" && "$STUDENT_MODALITY" != "none" ]]; then
  MODALITY_ARGS=(--student_modality "$STUDENT_MODALITY")
fi

EXTRA_ARGS=()
if [[ -n "${DISTILL_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "$DISTILL_EXTRA_ARGS"
fi
CAMERA_BACKEND="${CAMERA_BACKEND:-tiled}"
IMAGE_TILED_MAX_ENVS="${IMAGE_TILED_MAX_ENVS:-4096}"
IMAGE_PREFLIGHT_BRIGHT_FRAC_MAX="${IMAGE_PREFLIGHT_BRIGHT_FRAC_MAX:-0.4}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-pretrained_policy/model.pth}"
TEACHER_CONFIG="${TEACHER_CONFIG:-pretrained_policy/config.yaml}"
WANDB_GROUP="${WANDB_GROUP:-}"

echo "Hostname: $(hostname)"
nvidia-smi || true
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse HEAD)"
echo "Run config: $DISTILL_CONFIG"
echo "Camera config: $CAMERA_CONFIG"
echo "Camera backend: $CAMERA_BACKEND"
echo "Image tiled max envs: $IMAGE_TILED_MAX_ENVS"
echo "Image preflight bright frac max: $IMAGE_PREFLIGHT_BRIGHT_FRAC_MAX"
echo "Teacher checkpoint: $TEACHER_CHECKPOINT"
echo "Teacher config: $TEACHER_CONFIG"
echo "Wandb group: ${WANDB_GROUP:-<none>}"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"

SELECTED_ENVS=""
for N in "${ENV_COUNTS[@]}"; do
  if [[ "$STUDENT_INPUT" == "camera" && "$CAMERA_BACKEND" == "tiled" && "$N" -gt "$IMAGE_TILED_MAX_ENVS" ]]; then
    echo "SKIP_PREFLIGHT job=$JOB_NAME num_envs=$N: camera+tiled capped at IMAGE_TILED_MAX_ENVS=$IMAGE_TILED_MAX_ENVS due to high-env image corruption"
    continue
  fi
  PREFLIGHT_DIR="distillation_runs/${JOB_NAME}_preflight_${N}env"
  rm -rf "$PREFLIGHT_DIR"
  echo "=== ONLINE PREFLIGHT job=$JOB_NAME num_envs=$N ==="
  PREFLIGHT_IMAGE_ARGS=()
  if [[ "$STUDENT_INPUT" == "camera" ]]; then
    MID=$((N / 2))
    LAST=$((N - 1))
    PREFLIGHT_IMAGE_ARGS=(
      --debug_policy_image_stats
      --debug_policy_image_stats_stride 1
      --debug_policy_image_stats_env_ids "0,1,${MID},${LAST}"
    )
  fi
  if timeout 900 ./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill.py \
    --mode train_online \
    --headless \
    --student_input "$STUDENT_INPUT" \
    --student_arch "$STUDENT_ARCH" \
    "${MODALITY_ARGS[@]}" \
    --num_envs "$N" \
    --env_spacing 4.0 \
    --camera_backend "$CAMERA_BACKEND" \
    --online_num_iters 2 \
    --online_log_interval 1 \
    --online_update_interval "$ONLINE_UPDATE_INTERVAL" \
    --distill_config "$DISTILL_CONFIG" \
    --camera_config "$CAMERA_CONFIG" \
    --teacher_checkpoint "$TEACHER_CHECKPOINT" \
    --teacher_config "$TEACHER_CONFIG" \
    --run_dir "$PREFLIGHT_DIR" \
    "${PREFLIGHT_IMAGE_ARGS[@]}"; then
    if [[ "$STUDENT_INPUT" == "camera" && "${STUDENT_MODALITY:-}" != "depth" && "${STUDENT_MODALITY:-}" != "rgbd" ]]; then
      if ! python - "$PREFLIGHT_DIR/policy_image_stats.csv" "$IMAGE_PREFLIGHT_BRIGHT_FRAC_MAX" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
threshold = float(sys.argv[2])
rows = list(csv.DictReader(path.open()))
keys = [key for key in rows[-1] if key.startswith("policy_image/env") and key.endswith("/bright_frac")]
bright = sum(float(rows[-1][key]) for key in keys) / max(len(keys), 1)
print(f"IMAGE_PREFLIGHT bright_frac={bright:.3f} threshold={threshold:.3f}")
raise SystemExit(0 if bright <= threshold else 1)
PY
      then
        echo "PREFLIGHT_FAILED job=$JOB_NAME num_envs=$N: policy image bright fraction too high"
        continue
      fi
    fi
    SELECTED_ENVS="$N"
    echo "PREFLIGHT_OK job=$JOB_NAME selected_envs=$SELECTED_ENVS"
    break
  fi
  echo "PREFLIGHT_FAILED job=$JOB_NAME num_envs=$N"
done

if [[ -z "$SELECTED_ENVS" ]]; then
  echo "ERROR: all preflight env counts failed for $JOB_NAME: ${ENV_COUNTS[*]}" >&2
  exit 1
fi

RUN_DIR="distillation_runs/${JOB_NAME}_${SELECTED_ENVS}env"
rm -rf "$RUN_DIR"
echo "=== ONLINE TRAIN job=$JOB_NAME num_envs=$SELECTED_ENVS iters=$ONLINE_ITERS update_interval=$ONLINE_UPDATE_INTERVAL run_dir=$RUN_DIR ==="
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill.py \
  --mode train_online \
  --headless \
  --student_input "$STUDENT_INPUT" \
  --student_arch "$STUDENT_ARCH" \
  "${MODALITY_ARGS[@]}" \
  --num_envs "$SELECTED_ENVS" \
  --env_spacing 4.0 \
  --camera_backend "$CAMERA_BACKEND" \
  --online_num_iters "$ONLINE_ITERS" \
  --online_log_interval "$ONLINE_LOG_INTERVAL" \
  --online_update_interval "$ONLINE_UPDATE_INTERVAL" \
  --distill_config "$DISTILL_CONFIG" \
  --camera_config "$CAMERA_CONFIG" \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  --teacher_config "$TEACHER_CONFIG" \
  --run_dir "$RUN_DIR" \
  --wandb \
  --wandb_project "$WANDB_PROJECT" \
  ${WANDB_GROUP:+--wandb_group "$WANDB_GROUP"} \
  --wandb_name "$JOB_NAME" \
  "${EXTRA_ARGS[@]}"

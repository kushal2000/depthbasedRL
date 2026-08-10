#!/bin/bash

# Shared runner for four unbolted-fixture, 1 mm unified-policy evaluations.

set -uo pipefail

: "${EVAL_TAG:?set EVAL_TAG}"
: "${EVAL_PROBLEM:?set EVAL_PROBLEM}"
: "${EVAL_MAX_STEPS:?set EVAL_MAX_STEPS}"

REPO_ROOT=/share/portal/kk837/depthbasedRL
RUN_DIR="$REPO_ROOT/train_dir/multiproblem/cotrain/multiproblem_4way_peg1mm_unbolted_2026-08-09_16-24-37"
CHECKPOINT="$RUN_DIR/eval_snapshot_unbolted1mm_20260809.pth"
TRAIN_OVERRIDES="$RUN_DIR/.hydra/overrides.yaml"
OUT_DIR="$REPO_ROOT/plot_figures/rebuttal/experiments/outputs/unified_unbolted1mm_checkpoints"
OUTPUT_JSON="$OUT_DIR/${EVAL_TAG}_unbolted_job${SLURM_JOB_ID}.json"
DETAIL_LOG="$OUT_DIR/${EVAL_TAG}_unbolted_job${SLURM_JOB_ID}.log"

mkdir -p "$OUT_DIR" "$REPO_ROOT/plot_figures/rebuttal/experiments/logs"
cd "$REPO_ROOT"
source .venv_isaacsim/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_CACHE_PATH="/tmp/${USER}_ov_cache_uni_unbolted_${EVAL_TAG}_${SLURM_JOB_ID}"
mkdir -p "$OMNI_KIT_CACHE_PATH"

python -u peg_in_hole_dynamic/offline_eval_teacher_robustness.py \
  --teacher-checkpoint "$CHECKPOINT" \
  --train-overrides "$TRAIN_OVERRIDES" \
  --exact-train --num-envs 512 --seed 42 \
  --max-steps-per-episode "$EVAL_MAX_STEPS" \
  --override "env.peg_in_hole.problems=[$EVAL_PROBLEM]" \
  --override env.peg_in_hole.fixture_bolted=false \
  --override env.peg_in_hole.random_goal_fraction=0.0 \
  --override env.domain_randomization.force_scale=0.0 \
  --override env.domain_randomization.torque_scale=0.0 \
  --output-json "$OUTPUT_JSON" > "$DETAIL_LOG" 2>&1 &
EVAL_PID=$!

POLL_LIMIT=180
[ "$EVAL_MAX_STEPS" -gt 600 ] && POLL_LIMIT=360
COMPLETED=0
for _ in $(seq 1 "$POLL_LIMIT"); do
  if grep -q "=> wrote" "$DETAIL_LOG" 2>/dev/null; then COMPLETED=1; break; fi
  kill -0 "$EVAL_PID" 2>/dev/null || break
  sleep 10
done
kill -9 "$EVAL_PID" 2>/dev/null || true
wait "$EVAL_PID" 2>/dev/null || true

grep -E "fixture_bolted|problem=|all envs|no early drop|terminations:|Traceback|Error|=> wrote" "$DETAIL_LOG" || true
if [ "$COMPLETED" -ne 1 ] || [ ! -s "$OUTPUT_JSON" ]; then
  echo "FAILED: evaluation did not produce $OUTPUT_JSON" >&2
  exit 1
fi
jq -r '.results[0] | "RESULT all_envs insertion=\(.all_envs.insertion_rate) retract=\(.all_envs.retract_rate); filtered insertion=\(.early_drop_filtered.insertion_rate) retract=\(.early_drop_filtered.retract_rate); n=\(.early_drop_filtered.n); unfinished=\(.unfinished_envs)"' "$OUTPUT_JSON"

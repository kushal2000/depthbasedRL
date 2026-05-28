#!/usr/bin/env bash
set -euo pipefail

# Launch the tenth final L-peg wrench run locally in tmux.

REPO="${REPO:-/home/tylerlum/github_repos/depthbasedRL}"
ISAACSIM_ENV_DIR="${ISAACSIM_ENV_DIR:-${REPO}/.venv-isaacsim-py311}"
JOB_SCRIPT="${JOB_SCRIPT:-${REPO}/plot_figures/fig2/experiments_tyler_wrench/panel_a_tyler_wrench_lpeg_finetune.sub}"
SESSION="${SESSION:-lpeg_wrench_translationonly_s2_local}"

if [[ ! -d "$REPO/.git" ]]; then
  echo "Repo not found: $REPO" >&2
  exit 1
fi
if [[ ! -x "$ISAACSIM_ENV_DIR/bin/python" ]]; then
  echo "Isaac Sim env not found: $ISAACSIM_ENV_DIR" >&2
  exit 1
fi
if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "Job script not found: $JOB_SCRIPT" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd '$REPO' && \
   REPO_ROOT='$REPO' \
   ISAACSIM_ENV_DIR='$ISAACSIM_ENV_DIR' \
   RUN_ROOT='$REPO/train_dir' \
   WANDB_GROUP='panel_a_teachers_tyler_wrench_lpeg' \
   TASK_TAG='lpeg_tol0p5mm' \
   CHECKPOINT_FAMILY='TrainingObjective' \
   CHECKPOINT_TAG='TranslationOnly' \
   CONDITION_TAG='wrench' \
   SEED='2' \
   NUM_ENVS='12288' \
   MINIBATCH_SIZE='98304' \
   EXPL_COEF_BLOCK_SIZE='2048' \
   MAX_ITERATIONS='10000000' \
   FORCE_SCALE='20.0' \
   TORQUE_SCALE='2.0' \
   FORCE_ONLY_WHEN_LIFTED='False' \
   TORQUE_ONLY_WHEN_LIFTED='False' \
   bash '$JOB_SCRIPT'"

echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Stop:   tmux kill-session -t $SESSION"

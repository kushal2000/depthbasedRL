#!/bin/bash
# Preliminary speed benchmark: depth camera overhead vs env count
# Runs 100 epochs on 2 GPUs for each (env_count, depth_on/off) combination.
set -uo pipefail

CONDA_ENV="${CONDA_ENV:-sapg}"
CONDA_PREFIX="$(conda info --base)/envs/${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
TORCHRUN="${CONDA_PREFIX}/bin/torchrun"

NGPUS=2
MAX_EPOCHS=100
RESULTS_FILE="bench_depth_results.csv"

echo "num_envs,depth,step_fps,wallclock_fps,play_time_s,update_time_s,epoch_time_s,total_min" > "${RESULTS_FILE}"

# Env counts in the thousands (target ~24k for real training)
ENV_COUNTS=(2048 4096 8192 16384 24576)

for NUM_ENVS in "${ENV_COUNTS[@]}"; do
  for DEPTH in False True; do
    TAG="envs${NUM_ENVS}_depth${DEPTH}"
    echo "========================================"
    echo "  Benchmark: ${TAG}"
    echo "========================================"

    LOG_FILE="bench_logs/${TAG}.log"
    mkdir -p bench_logs

    "${TORCHRUN}" --nproc_per_node="${NGPUS}" --master_port=29501 \
      -m isaacgymenvs.train \
      headless=True \
      task=SimToolRealLSTMAsymmetric \
      "task.env.numEnvs=${NUM_ENVS}" \
      "task.env.depthCamera.enabled=${DEPTH}" \
      task.env.capture_video=False \
      task.env.goodResetBoundary=0 \
      "task.env.objectScaleNoiseMultiplierRange=[0.9,1.1]" \
      task.env.forceConsecutiveNearGoalSteps=True \
      task.env.forceScale=20 \
      task.env.torqueScale=2.0 \
      task.env.objectAngVelPenaltyScale=0.0 \
      train.params.config.minibatch_size=4096 \
      "train.params.config.max_epochs=${MAX_EPOCHS}" \
      train.params.config.good_reset_boundary=0 \
      train.params.config.use_others_experience=lf \
      train.params.config.off_policy_ratio=1.0 \
      train.params.config.expl_type=mixed_expl_learn_param \
      train.params.config.expl_reward_type=entropy \
      train.params.config.expl_coef_block_size=64 \
      train.params.config.expl_reward_coef_scale=0.005 \
      train.params.network.space.continuous.fixed_sigma=coef_cond \
      multi_gpu=True \
      wandb_activate=False \
      seed=0 \
      "experiment=00_bench_${TAG}" \
      ++use_rl=True \
      2>&1 | tee "${LOG_FILE}"

    RC=$?
    if [ $RC -ne 0 ]; then
      echo "  *** RUN FAILED (exit code ${RC}) ***"
      echo "${NUM_ENVS},${DEPTH},FAIL,FAIL,FAIL,FAIL,FAIL,FAIL" >> "${RESULTS_FILE}"
      continue
    fi

    # Parse the last 10 fps lines and average them (skip early warmup epochs)
    STEP_FPS=$(grep "fps step" "${LOG_FILE}" | tail -10 | sed 's/.*: //' | tr -d ',' | awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "N/A"}')
    WALL_FPS=$(grep "fps wallclock" "${LOG_FILE}" | tail -10 | sed 's/.*: //' | tr -d ',' | awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "N/A"}')
    PLAY_TIME=$(grep "Play time" "${LOG_FILE}" | tail -10 | sed 's/.*: //' | tr -d ' s' | awk '{s+=$1; n++} END {if(n>0) printf "%.3f", s/n; else print "N/A"}')
    UPDATE_TIME=$(grep "Update time" "${LOG_FILE}" | tail -10 | sed 's/.*: //' | tr -d ' s' | awk '{s+=$1; n++} END {if(n>0) printf "%.3f", s/n; else print "N/A"}')
    EPOCH_TIME=$(grep "Epoch wallclock" "${LOG_FILE}" | tail -10 | sed 's/.*: //' | tr -d ' s' | awk '{s+=$1; n++} END {if(n>0) printf "%.3f", s/n; else print "N/A"}')
    TOTAL_MIN=$(grep "elapsed_minutes" "${LOG_FILE}" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    TOTAL_MIN="${TOTAL_MIN:-N/A}"

    echo "${NUM_ENVS},${DEPTH},${STEP_FPS},${WALL_FPS},${PLAY_TIME},${UPDATE_TIME},${EPOCH_TIME},${TOTAL_MIN}" >> "${RESULTS_FILE}"
    echo ""
    echo "  → step_fps=${STEP_FPS}  wallclock_fps=${WALL_FPS}  play=${PLAY_TIME}s  update=${UPDATE_TIME}s  epoch=${EPOCH_TIME}s  total=${TOTAL_MIN}min"
    echo ""
  done
done

echo ""
echo "========================================"
echo "  RESULTS SUMMARY"
echo "========================================"
column -t -s',' "${RESULTS_FILE}"

echo ""
echo "========================================"
echo "  SLOWDOWN ANALYSIS"
echo "========================================"
python3 -c "
import csv
rows = list(csv.DictReader(open('${RESULTS_FILE}')))
print(f\"{'Envs':>8} {'FPS (no depth)':>15} {'FPS (depth)':>15} {'Slowdown':>10} {'Overhead %':>10}\")
print('-' * 65)
for i in range(0, len(rows), 2):
    if i+1 >= len(rows):
        break
    no_depth = rows[i]
    depth = rows[i+1]
    envs = no_depth['num_envs']
    fps_nd_s = no_depth['step_fps']
    fps_d_s = depth['step_fps']
    if fps_nd_s not in ('N/A','FAIL') and fps_d_s not in ('N/A','FAIL'):
        fps_nd = float(fps_nd_s)
        fps_d = float(fps_d_s)
        if fps_nd > 0 and fps_d > 0:
            slowdown = fps_nd / fps_d
            overhead_pct = (1 - fps_d/fps_nd) * 100
            print(f'{envs:>8} {fps_nd:>15,.0f} {fps_d:>15,.0f} {slowdown:>9.2f}x {overhead_pct:>9.1f}%')
            continue
    print(f'{envs:>8} {fps_nd_s:>15} {fps_d_s:>15} {\"N/A\":>10} {\"N/A\":>10}')
"

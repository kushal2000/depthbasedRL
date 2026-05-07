# Isaac Sim Cluster Runbook

This runbook is the practical guide for using the Stanford `move`/`humanoid`
Slurm cluster for Isaac Sim / Isaac Lab distillation jobs in this repo.

Last updated: 2026-05-06.

## Current Default Setup

- Current active branch for PegInHole depth distillation:
  `2026-04-29_KushalEnvDepthDistillationV2`
- Main shared clone:
  `/move/u/$USER/github_repos/depthbasedRL`
- RTX PRO 6000 Blackwell clone, only when needed:
  `/move/u/$USER/github_repos/depthbasedRL_rtx6000`
- Main Isaac Sim venv:
  `/move/u/$USER/github_repos/depthbasedRL/.venv-isaacsim-py311`
- RTX PRO 6000 venv, only when needed:
  `/move/u/$USER/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-cu128-py311`
- W&B key path:
  `/juno/u/$USER/.wandb_api_key`
- Slurm logs:
  `/move/u/$USER/github_repos/depthbasedRL/slurm_logs`

Use the main clone/venv unless you specifically need the RTX PRO 6000
Blackwell setup.

## Sync Code

Push local changes first, then fast-forward the cluster clone:

```bash
ssh $USER@sc
cd /move/u/$USER/github_repos/depthbasedRL
git fetch origin
git checkout 2026-04-29_KushalEnvDepthDistillationV2
git pull --ff-only origin 2026-04-29_KushalEnvDepthDistillationV2
git status --short
```

If using the RTX PRO 6000 clone:

```bash
cd /move/u/$USER/github_repos/depthbasedRL_rtx6000
git fetch origin
git checkout 2026-04-29_KushalEnvDepthDistillationV2
git pull --ff-only origin 2026-04-29_KushalEnvDepthDistillationV2
git status --short
```

Do not edit code independently inside multiple cluster clones unless you intend
to commit/push from that clone. Prefer one source of truth and fast-forward the
others.

## Check Cluster State

Queue:

```bash
squeue -u $USER -o "%.18i %.16P %.30j %.8u %.2t %.12M %.12l %.20R"
```

All relevant partitions:

```bash
squeue -p move,move-interactive,humanoid,humanoid-interactive
```

GPU availability helper:

```bash
/juno/u/$USER/sgpu_tyler.py -p move,move-interactive,humanoid,humanoid-interactive
```

Useful interpretation:

- `R`: running.
- `PD (Resources)`: waiting for free GPU/memory/node resources.
- `PD (ReqNodeNotAvail, May be reserved for other job)`: requested node is not
  currently schedulable. Pick another node/GPU type or remove the node pin.
- Jobs on `move` are usually limited to 24 hours. Use `juno-long`/`juno-lo`
  only when explicitly appropriate for a different machine/partition setup.

## Interactive Allocation

Use this for smoke tests, debugging, or setting up environments:

```bash
ssh $USER@sc
srun --account move -p move-interactive \
  --time=6:00:00 \
  --gres=gpu:1 \
  --mem=96G \
  --cpus-per-task=8 \
  --pty bash -i
```

To pin a known node:

```bash
srun --account move -p move-interactive \
  --nodelist=move4 \
  --time=6:00:00 \
  --gres=gpu:1 \
  --mem=96G \
  --cpus-per-task=8 \
  --pty bash -i
```

Inside the allocation:

```bash
cd /move/u/$USER/github_repos/depthbasedRL
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_CACHE_PATH=/tmp/${USER}_ov_cache_${SLURM_JOB_ID:-interactive}
export PYTHONPATH=$PWD:${PYTHONPATH:-}
mkdir -p "$OMNI_KIT_CACHE_PATH" slurm_logs distillation_runs
nvidia-smi
```

## Environment Setup

First-time main clone setup:

```bash
cd /move/u/$USER/github_repos
git clone -b 2026-04-29_KushalEnvDepthDistillationV2 \
  https://github.com/kushal2000/depthbasedRL.git
cd depthbasedRL
./scripts/setup_isaacsim_uv_env.sh .venv-isaacsim-py311
```

If the venv already exists, use it directly:

```bash
cd /move/u/$USER/github_repos/depthbasedRL
.venv-isaacsim-py311/bin/python -c "import torch; print(torch.__version__)"
```

Use repo wrappers when available:

```bash
./scripts/run_in_isaacsim_env.sh python isaacsimenvs/distill_depth.py --help
```

For direct Slurm scripts, prefer setting `PYTHON_BIN` explicitly to the desired
venv Python.

## Recommended GPU Choices

- `L40S`: reliable for Isaac Sim / Isaac Lab image training.
- `RTX PRO 6000 Blackwell`: fast and useful, but use the separate cu128 RTX
  clone/venv if the normal venv has CUDA kernel-image errors.
- `RTX A5000`: works but slower and lower VRAM; useful for throughput
  comparison or lower-priority jobs.
- `Titan RTX`: can run simpler jobs; treat as a fallback.

Typical GPU nodes seen during this project:

- `move4`: often L40S.
- `humanoid1`: L40S partition/node.
- `move5`: RTX PRO 6000 Blackwell.
- `move3`: A5000.
- `move1`, `move2`: Titan RTX.

Always check current state with `sgpu_tyler.py`; node/GPU availability changes.

## Generic PegInHole Depth Distillation Job

The current generic launcher is:

```text
scripts/cluster/sbatch_kushal_depth_distill_v2.sh
```

It wraps:

```text
isaacsimenvs/distill_depth.py --mode train_online
```

and accepts configuration through environment variables.

Example: clean depth-image student, no depth noise or camera pose randomization:

```bash
cd /move/u/$USER/github_repos/depthbasedRL

RUN_NAME=example_depth_clean \
STUDENT_INPUT=camera \
NUM_ENVS=256 \
NUM_ITERS=7000000 \
WANDB_GROUP=2026-05-XX_PegInHoleDepth \
DEPTH_NOISE_PROFILE=off \
CAMERA_POSE_PROFILE=off \
sbatch --job-name=example_depth_clean \
  --nodelist=move4 \
  scripts/cluster/sbatch_kushal_depth_distill_v2.sh
```

Example: medium depth noise and 20 mm / 2 deg camera pose randomization:

```bash
cd /move/u/$USER/github_repos/depthbasedRL

RUN_NAME=depth_medium_noise_camrand20mm2deg \
STUDENT_INPUT=camera \
NUM_ENVS=256 \
NUM_ITERS=7000000 \
WANDB_GROUP=2026-05-XX_PegInHoleDepth \
DEPTH_NOISE_PROFILE=medium \
CAMERA_POSE_PROFILE=custom \
CAMERA_POSE_MODE=startup \
CAMERA_POS_NOISE_M="0.02 0.02 0.02" \
CAMERA_ROT_NOISE_DEG="2 2 2" \
DEPTH_DEBUG_INTERVAL=1000 \
DEPTH_ROLLOUT_VIDEO_LEN=600 \
DEPTH_ROLLOUT_VIDEO_FPS=60 \
DEPTH_ROLLOUT_VIDEO_INTERVAL=600 \
DEPTH_ROLLOUT_VIDEO_ENV_IDS=0 \
AUX_POSE_MODE=position \
AUX_OBJECT_POS_WEIGHT=1.0 \
AUX_OBJECT_KEYPOINT_WEIGHT=1.0 \
sbatch --job-name=depth_camrand20mm2deg \
  --nodelist=move4 \
  scripts/cluster/sbatch_kushal_depth_distill_v2.sh
```

Example: continue from an existing checkpoint:

```bash
cd /move/u/$USER/github_repos/depthbasedRL

RUN_NAME=06ctd_pos_medium_noise_camrand20mm2deg \
STUDENT_INPUT=camera \
NUM_ENVS=256 \
NUM_ITERS=7000000 \
WANDB_GROUP=2026-05-01_KushalDepthPoseCamRandContinued \
STUDENT_CHECKPOINT=distillation_runs/06_pos_medium_noise_camrand20mm2deg/checkpoints/student_latest.pt \
DEPTH_NOISE_PROFILE=medium \
CAMERA_POSE_PROFILE=custom \
CAMERA_POSE_MODE=startup \
CAMERA_POS_NOISE_M="0.02 0.02 0.02" \
CAMERA_ROT_NOISE_DEG="2 2 2" \
AUX_POSE_MODE=position \
AUX_OBJECT_POS_WEIGHT=1.0 \
AUX_OBJECT_KEYPOINT_WEIGHT=1.0 \
sbatch --job-name=06ctd_pos_camrand20mm2deg \
  scripts/cluster/sbatch_kushal_depth_distill_v2.sh
```

For a full 24-hour run, set `NUM_ITERS` high enough that the wall-time limit,
not the iteration count, stops the run. `7,000,000` has been used for this.

## Teacher-Observation Distillation

Teacher-observation jobs are useful sanity checks because they avoid camera
rendering and should converge quickly.

```bash
cd /move/u/$USER/github_repos/depthbasedRL

RUN_NAME=teacher_obs_sanity \
STUDENT_INPUT=teacher_obs \
NUM_ENVS=4096 \
NUM_ITERS=1000000 \
WANDB_GROUP=2026-05-XX_PegInHoleDepth \
sbatch --job-name=teacher_obs_sanity \
  --nodelist=move4 \
  scripts/cluster/sbatch_kushal_depth_distill_v2.sh
```

## Direct Command Equivalent

For debugging in an interactive allocation, this is the equivalent direct form:

```bash
cd /move/u/$USER/github_repos/depthbasedRL
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_CACHE_PATH=/tmp/${USER}_ov_cache_${SLURM_JOB_ID:-manual}
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export WANDB_API_KEY="$(tr -d '[:space:]' < /juno/u/$USER/.wandb_api_key)"

.venv-isaacsim-py311/bin/python isaacsimenvs/distill_depth.py \
  --mode train_online \
  --student_input camera \
  --student_arch mono_transformer_recurrent \
  --num_envs 256 \
  --num_iters 7000000 \
  --log_interval 1000 \
  --save_interval 1000 \
  --run_dir distillation_runs/manual_depth_medium_noise_camrand20mm2deg \
  --aux_pose_mode position \
  --aux_object_pos_weight 1.0 \
  --aux_object_keypoint_weight 1.0 \
  --wandb \
  --wandb_project depthbasedRL-isaacsim-distill \
  --wandb_group 2026-05-XX_PegInHoleDepth \
  --wandb_name manual_depth_medium_noise_camrand20mm2deg \
  --capture_viewer \
  --capture_viewer_len 600 \
  --headless \
  --depth_noise_profile medium \
  --camera_pose_randomization_profile custom \
  --camera_pose_randomization_mode startup \
  --camera_pos_noise_m 0.02 0.02 0.02 \
  --camera_rot_noise_deg 2 2 2 \
  --depth_debug_interval 1000 \
  --wandb_depth_rollout_video_len 600 \
  --wandb_depth_rollout_video_fps 60 \
  --wandb_depth_rollout_video_interval 600 \
  --wandb_depth_rollout_video_env_ids 0
```

## Logs And Health Checks

Find jobs:

```bash
squeue -u $USER -o "%.18i %.16P %.30j %.8u %.2t %.12M %.12l %.20R"
```

Tail log:

```bash
tail -f /move/u/$USER/github_repos/depthbasedRL/slurm_logs/<job_name>_<JOBID>.out
```

Search for useful events:

```bash
grep -E "View run|wandb: Syncing run|online iter|recent_reset|Traceback|RuntimeError|CUDA out|ERROR|Killed" \
  /move/u/$USER/github_repos/depthbasedRL/slurm_logs/<job_name>_<JOBID>.out
```

Monitor GPU usage if the script logs it:

```bash
tail -f /move/u/$USER/github_repos/depthbasedRL/slurm_logs/<job_name>_<JOBID>_nvidia_smi.csv
```

If a run starts but W&B does not appear:

- check that `/juno/u/$USER/.wandb_api_key` exists and is non-empty
- inspect the Slurm log for import errors before `wandb.init`
- ensure the run uses `--wandb`
- check that the job did not die during Isaac Sim startup/shader compilation

## Continuing Jobs After Time Limit

When a job hits the 24-hour limit, continue from latest checkpoint with a new
run name and run dir:

```bash
RUN_NAME=<old_name>_ctd2 \
STUDENT_CHECKPOINT=distillation_runs/<old_name>/checkpoints/student_latest.pt \
RUN_DIR=distillation_runs/<old_name>_ctd2 \
NUM_ITERS=7000000 \
sbatch --job-name=<old_name>_ctd2 scripts/cluster/sbatch_kushal_depth_distill_v2.sh
```

Prefer `student_latest.pt` for continuation. Use `student_best.pt` only when
you intentionally want to branch from the best historical validation metric.

## Kill Jobs

Cancel one job:

```bash
scancel <JOBID>
```

Cancel all of your jobs on a partition only after checking carefully:

```bash
squeue -u $USER -p move
scancel <JOBID1> <JOBID2>
```

Do not kill other users' jobs.

## Common Failure Modes

- `ReqNodeNotAvail`: the pinned node is reserved/down/otherwise unavailable.
  Remove `--nodelist`, pick another node, or use a different GPU type.
- CUDA kernel image errors on RTX PRO 6000: use the separate RTX clone and
  cu128 venv.
- Very slow first startup: Isaac Sim shader/cache compilation. This can take
  minutes; check the log before assuming the job is dead.
- Image/rendering problems: run a short local/interactive debug with depth media
  logging enabled and inspect W&B `student_depth_debug` plus realtime rollout.
- OOM: reduce `NUM_ENVS`, use a larger GPU, or reduce media/debug logging.
- W&B missing: load `WANDB_API_KEY`, verify network, and inspect the Slurm log.

## Legacy Hammer / IsaacSim Conversion Commands

These commands are retained for the older hammer distillation path under
`isaacsim_conversion/`. They are not the primary PegInHole depth-student path.

Viewer baseline:

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/rollout.py \
  --task_source dextoolbench \
  --object_category hammer \
  --object_name claw_hammer \
  --task_name swing_down \
  --max_steps 700 \
  --checkpoint pretrained_policy/model.pth \
  --config pretrained_policy/config.yaml
```

Teacher eval:

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill_eval.py \
  --mode teacher_eval \
  --headless \
  --max_steps 700 \
  --num_envs 4096 \
  --env_spacing 4.0 \
  --object_start_mode fixed \
  --teacher_checkpoint pretrained_policy/model.pth \
  --teacher_config pretrained_policy/config.yaml \
  --camera_config isaacsim_conversion/configs/hammer_camera_depth_160x90.yaml
```

Batch train:

```bash
sbatch scripts/cluster/sbatch_distill_teacher_obs_l40s.sh
```

Batch eval:

```bash
sbatch --export=CHECKPOINT=distillation_runs/<run>/checkpoints/student_best.pt \
  scripts/cluster/sbatch_student_eval_l40s.sh
```


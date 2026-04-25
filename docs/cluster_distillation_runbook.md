# Cluster Distillation Runbook

This note summarizes the Slurm/cluster workflow used for Isaac Sim distillation runs.

## Repos And Environments

RTX6000 clone:

```bash
/move/u/tylerlum/github_repos/depthbasedRL_rtx6000
```

RTX6000 Isaac Sim environment:

```bash
/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-cu128-py311
```

L40S clone:

```bash
/move/u/tylerlum/github_repos/depthbasedRL
```

L40S Isaac Sim environment:

```bash
/move/u/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311
```

Before submitting jobs, sync the target clone:

```bash
ssh move5 'cd /move/u/tylerlum/github_repos/depthbasedRL_rtx6000 && git fetch origin 2206-04-24_MoreRandomizations && git checkout 2206-04-24_MoreRandomizations && git pull --ff-only'
ssh move4 'cd /move/u/tylerlum/github_repos/depthbasedRL && git fetch origin 2206-04-24_MoreRandomizations && git checkout 2206-04-24_MoreRandomizations && git pull --ff-only'
```

## Main Wrapper

Use:

```bash
scripts/cluster/codex_distill_online_with_preflight.sh
```

The wrapper:

- Loads W&B credentials from `/juno/u/tylerlum/.wandb_api_key`.
- Sets per-job cache directories under `/tmp`.
- Runs a short preflight for candidate env counts.
- Selects the first env count that passes.
- Launches `isaacsim_conversion/distill.py --mode train_online`.
- Writes Slurm logs to `slurm_logs/%x_%j.out`.
- Writes run outputs to `distillation_runs/<job_name>_<num_envs>env`.

## Typical RTX6000 Submit

Example peg-in-hole depth image job:

```bash
ssh move5 'cd /move/u/tylerlum/github_repos/depthbasedRL_rtx6000 && \
export TEACHER_CHECKPOINT=/juno/u/kedia/depthbasedRL/train_dir/Apr20/multiInit/model.pth \
TEACHER_CONFIG=/juno/u/kedia/depthbasedRL/train_dir/Apr20/multiInit/config.yaml \
WANDB_GROUP=2026-04-24_PegInHoleStrongInit \
DISTILL_EXTRA_ARGS="--task_source peg_in_hole --object_category peg_in_hole --object_name peg --task_name insert --peg_scene_idx 82 --peg_idx 5 --peg_tol_slot_idx 5 --peg_goal_mode preInsertAndFinal --capture_viewer --capture_viewer_video --capture_viewer_len 600 --capture_viewer_env_id 0" && \
sbatch --account=move --partition=move --nodelist=move5 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
--job-name=13_peg_depth_image_80x45_strong_randomized_a --output=slurm_logs/%x_%j.out \
scripts/cluster/codex_distill_online_with_preflight.sh \
13_peg_depth_image_80x45_strong_randomized_a \
/move/u/tylerlum/github_repos/depthbasedRL_rtx6000 \
/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-cu128-py311 \
isaacsim_conversion/configs/peg_distill_depth_80x45_online_dagger_512_strong_randomized.yaml \
isaacsim_conversion/configs/peg_camera_depth_80x45.yaml \
camera mono_transformer_recurrent depth \
350000 1000 1 depthbasedRL-isaacsim-distill \
512 256 128 64'
```

Arguments after the project name are candidate `num_envs` values. The wrapper tries them in order.

## Teacher-Obs Jobs

Use `teacher_obs mlp_recurrent none` instead of camera settings:

```bash
... peg_distill_teacher_obs_4096_strong_yawonly.yaml \
isaacsim_conversion/configs/peg_camera_depth_80x45.yaml \
teacher_obs mlp_recurrent none \
350000 1000 1 depthbasedRL-isaacsim-distill \
4096 2048 1024 512
```

The wrapper intentionally drops `--capture_viewer_video` for `teacher_obs` jobs. Otherwise, Isaac Lab instantiates camera sensors across all 4096 envs, making startup very slow. HTML viewer logging remains enabled through `--capture_viewer`.

## Monitoring

Queue:

```bash
ssh move5 'squeue -u tylerlum -o "%18i %48j %10T %10M %12R"'
```

Recent logs:

```bash
ssh move5 'cd /move/u/tylerlum/github_repos/depthbasedRL_rtx6000 && tail -n 100 slurm_logs/JOB_NAME_JOBID.out'
```

Useful log filter:

```bash
ssh move5 'cd /move/u/tylerlum/github_repos/depthbasedRL_rtx6000 && egrep "Branch:|Commit:|Extra args:|PREFLIGHT_OK|ONLINE TRAIN|View run|online iter|Traceback|Exception|Object/claw|Object/peg" slurm_logs/JOB_NAME_JOBID.out | tail -n 120'
```

Metrics:

```bash
ssh move5 'cd /move/u/tylerlum/github_repos/depthbasedRL_rtx6000 && tail -n 10 distillation_runs/RUN_DIR/metrics.csv'
```

Cancel:

```bash
ssh move5 'scancel JOBID'
```

Inspect job details:

```bash
ssh move5 'scontrol show job JOBID'
```

Inspect node allocation:

```bash
ssh move5 'scontrol show node move5 | egrep "NodeName|State|CfgTRES|AllocTRES|Gres"'
```

## Artifacts

Per-run outputs are under:

```bash
distillation_runs/<run_name>_<num_envs>env
```

Common artifact paths:

```bash
distillation_runs/<run>/metrics.csv
distillation_runs/<run>/resolved_distill_config.yaml
distillation_runs/<run>/interactive_viewer/train_online_chunk_0001_rollout.html
distillation_runs/<run>/interactive_viewer/train_online_chunk_0001_rollout.mp4
```

## Important Gotchas

- Always include peg task args in `DISTILL_EXTRA_ARGS`. Without them, `distill.py` falls back to the default claw-hammer task.
- Preflight must receive `DISTILL_EXTRA_ARGS`; otherwise preflight can test claw hammer while training uses peg-in-hole.
- Check logs for `Object/peg`, not `Object/claw_hammer`, before trusting a run.
- Sync the cluster clone before submitting. Some runs were launched from older branches and missed bug fixes.
- For RGBD camera jobs, the wrapper skips the RGB bright-image preflight heuristic. Depth and RGBD should not be rejected by that RGB-only heuristic.
- For image jobs with `TiledCamera`, candidate env counts should be conservative. The wrapper supports env-count fallback, e.g. `512 256 128 64`.
- `ReqNodeNotA...` pending usually means the requested node is temporarily unavailable or not accepting new allocations; check `scontrol show job JOBID` and `scontrol show node <node>`.
- SSH directly into some compute nodes may fail with `pam_slurm_adopt` unless you have an active job there. Submitting through an accessible node with `sbatch --nodelist=<target>` works.


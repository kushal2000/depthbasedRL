# Depth Policy Evaluation

Use `isaacsimenvs/eval_depth_policy.py` for evaluation-only rollouts of the
Isaac Lab peg-in-hole depth students. It uses the same env config overlays,
student checkpoint loading, depth preprocessing, and goal accounting as
`isaacsimenvs/distill_depth.py`, but writes evaluation artifacts:

- `metrics.csv`: interval metrics, comparable to W&B training logs.
- `episodes.csv`: one row per completed episode.
- `summary.json`: aggregate completed-episode metrics.
- `interactive_viewer/*.html`: optional pose viewer.
- `depth_debug/*` and `depth_rollout_videos/*`: optional policy-input media.

## 32c L-Peg Eval

```bash
bash_scripts/14_eval_depth_policy_32c_L_defaultcam.sh
```

With live viser, camera frustums, and point cloud:

```bash
SERVE_VISER=1 bash_scripts/14_eval_depth_policy_32c_L_defaultcam.sh
```

The script defaults to:

```text
student_checkpoint=/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt
run_dir=eval_runs/32c_L_defaultcam_q1_medium_noise_camrand20mm2deg_<timestamp>
peg_urdf=assets/urdf/peg_in_hole/peg_L/peg_L.urdf
peg_goal_mode=preInsertAndFinal
depth_noise_profile=medium
student_camera_preset=default
student_image_delay_queue_size=1
camera_pose_randomization=custom startup, +/-0.02 m xyz, +/-2 deg rpy
aux_pose_mode=rot6d_keypoints
```

## Metric Semantics

`current_goal_idx_avg` is the mean of `env._successes` across currently active
envs at log time. This is current in-progress episode state, not final episode
performance.

`current_goal_completion_ratio_avg` is:

```text
mean(env._successes / env.env_max_goals)
```

`recent_reset_goal_idx_avg` is the mean completed goal count only for envs that
reset during the current logging interval. It uses:

```text
env._prev_episode_successes[done_envs]
```

`recent_reset_goal_completion_ratio_avg` is:

```text
mean(env._prev_episode_successes[done_envs] / env.prev_episode_env_max_goals[done_envs])
```

If no envs reset during an interval, the training script logs `0` for the
recent-reset averages. This means `recent_reset_*` can look noisy or misleading
at short intervals: early successful episodes can make an interval look very
good while timeout/failure episodes have not reset yet.

For evaluation, prefer `summary.json` and `episodes.csv`:

- `completed_goal_idx_avg`: mean final goal count over completed episodes.
- `completed_goal_completion_ratio_avg`: mean final goal fraction.
- `completed_full_success_rate`: fraction of episodes with `goal_idx >= max_goals`.
- `completed_goal_idx_eq_*_count`: exact final-goal histogram.
- `done_timeout_count`, `done_fall_count`, etc.: termination reason counts.
- `by_scene_idx`, `by_tol_slot_idx`, `by_peg_idx`, `by_object_asset_idx`:
  completed-episode breakdowns for checking whether failures cluster around a
  particular scene, tolerance slot, peg/object variant, or start distribution.

Set `SEED=...` when using the helper script, or pass `--seed ...` directly, for
repeatable local comparisons when sweeping one variable at a time.

For the old T-shaped peg ambiguity question, use `episodes.csv` plus the
`summary.json` grouped breakdowns rather than a single W&B scalar. The eval CSV
records the starting object quaternion and peg/tolerance/scene indices for every
completed episode, so failures can be checked against object orientation and
scene/tolerance distribution directly.

Peg-in-hole goal modes:

- `preInsertAndFinal`: `goal_idx=0` means pre-insert was never reached,
  `goal_idx=1` means pre-insert was reached but final insertion was not, and
  `goal_idx=2` means both were reached.
- `finalGoalOnly`: `goal_idx=0` means final insertion was not reached and
  `goal_idx=1` means final insertion was reached.

For a final-insertion-only eval that resets immediately after insertion, run the
helper with:

```bash
PEG_GOAL_MODE=finalGoalOnly PEG_ENABLE_RETRACT=0 bash_scripts/14_eval_depth_policy_32c_L_defaultcam.sh
```

## Validation Notes

Smoke-tested locally on May 13, 2026:

```text
eval_runs/smoke_eval_depth_policy_32c_force_exit
eval_runs/smoke_eval_depth_policy_viser
eval_runs/quick_eval_32c_L_defaultcam_medium_noise_camrand20mm2deg
eval_runs/smoke_eval_depth_policy_seeded_groups
```

The quick eval used 16 envs and stopped after 16 completed episodes. It produced
10 full-success episodes and 6 zero-goal episodes:

```text
completed_goal_idx_avg=1.25
completed_full_success_rate=0.625
completed_goal_idx_eq_0_count=6
completed_goal_idx_eq_2_count=10
```

This is a small randomized sample, so use a larger `NUM_COMPLETED_EPISODES`
before treating it as the final policy estimate.

# SimToolReal isaacgym vs isaacsim debug

Side-by-side experiments to characterize physics-response differences
between the legacy `isaacgymenvs/tasks/simtoolreal/env.py` and the ported
`isaacsimenvs/tasks/simtoolreal/`.

## Experiment 1 — sine wave PD targets on the SHARPA hand

A 2-second open-loop sine on the 22 hand DOFs (arm targets pinned to default
reset pose, no reset noise, num_envs=1, action pipeline bypassed). Each
driver dumps `(joint_names, joint_pos, joint_vel, target)` per policy step
to `data/`. The diff script aligns by joint name and plots per-joint
overlays.

```bash
# isaacgym side (uses .venv)
source .venv/bin/activate
python debug_differences/sine_hand_isaacgym.py

# isaacsim side (uses .venv_isaacsim) — first run regenerates handle_head USDs
source .venv_isaacsim/bin/activate
python debug_differences/sine_hand_isaacsim.py

# diff + plot (any env with numpy + matplotlib)
python debug_differences/plot_sine_hand_diff.py
```

Outputs land in `data/` (npz) and `plots/` (per-joint PNGs + summary CSV).
Both subdirs are gitignored.

## Experiment 2 — pretrained policy rollout, fixed object + fixed goal

Closed-loop rollout of `pretrained_policy/model.pth` on both backends with
the procedural pool collapsed to a single hammer URDF and the goal pinned
to one env-local pose (`(0, 0, 0.78, identity)` — center of the trained
target volume). All DR/reset noise off, deterministic actions. Each driver
dumps `(obs, action, joint_pos, joint_vel, joint_targets, object_state,
goal_pose, reward)` per policy step plus an mp4. The diff script aligns
the two npz traces by joint name (canonical order on both sides) and
plots overlays.

```bash
# isaacgym side
source .venv/bin/activate
python debug_differences/policy_rollout_isaacgym.py

# isaacsim side
source .venv_isaacsim/bin/activate
python debug_differences/policy_rollout_isaacsim.py

# diff + plot (any env with numpy + matplotlib)
python debug_differences/plot_policy_rollout_diff.py
```

Outputs:
- `data/{isaacgym,isaacsim}_policy_rollout.npz`
- `plots/{isaacgym,isaacsim}_policy_rollout.mp4`
- `plots/policy_joint_overlay.png` — 5×6 q grid (gym vs sim vs target)
- `plots/policy_action_overlay.png` — 5×6 action grid
- `plots/policy_object_traj.png` — object/goal xyz + ‖Δobj_pos‖, Δobj_rot
- `plots/policy_reward_curve.png`
- `plots/policy_summary.csv` — per-channel mean/max abs error, sorted

## Experiment 3 — pretrained policy distributional eval

Multi-env rollout of `pretrained_policy/model.pth` against the *full*
procedural asset pool (12 ObjectSizeDistribution entries × `num_assets_per_type`
shape variants) under the *natural* per-reset goal-sampling distribution.
All DR / reset noise still off — the only sources of variance are
asset shape and goal pose, both seeded identically across backends. Per-step
trajectory overlay is meaningless across backends here (each side samples its
own goal sequence), so the diff script reports aggregate distributional
stats only: mean reward / step, cumulative success rate, lift fraction, mean
obj→goal distance, first-reset CDF, per-asset-type breakdown bars, and a
top-level summary CSV.

```bash
# isaacgym side  (defaults: 64 envs × 600 steps, 10 assets/type)
source .venv/bin/activate
python debug_differences/policy_eval_isaacgym.py

# isaacsim side  (default num_assets_per_type=10 keeps URDF→USD conversion
# bounded; raise it once the conversion is cached)
source .venv_isaacsim/bin/activate
python debug_differences/policy_eval_isaacsim.py

# aggregate-stats diff (numpy + matplotlib)
python debug_differences/plot_policy_eval_diff.py
```

Defaults capture an env-0 mp4 on each backend (`--video_envs 0`); pass
`--video_envs ""` to disable, `--video_envs 0,32,128` to capture multiple.
Per-env video adds a camera sensor only for the selected envs, so cost
scales with the size of the subset, not with `num_envs`.

Outputs (all under `debug_differences/`, `data/policy_eval/` and
`plots/policy_eval/`):
- `data/policy_eval/{isaacgym,isaacsim}_policy_eval.npz` — per-step `(T, N)`
  traces (reward / is_success / reset / progress / successes / lifted_object /
  obj_pos_world / goal_pos_world / obj_to_goal_dist) + per-env asset_type tag
- `plots/policy_eval/reward.png`       — mean reward / step (mean ± std band)
- `plots/policy_eval/success_rate.png` — cumulative + per-step hit rate
- `plots/policy_eval/obj_to_goal.png`  — mean ‖obj − goal‖ over time
- `plots/policy_eval/lifted.png`       — fraction with lifted_object=True
- `plots/policy_eval/episodes.png`     — first-reset-step CDF
- `plots/policy_eval/per_asset.png`    — per-asset-type bars
- `plots/policy_eval/per_env.png`      — gym-vs-sim scatter (1 pt per env)
- `plots/policy_eval/per_env.csv`      — per-env aggregates side-by-side
- `plots/policy_eval/summary.csv`      — top-level metrics, gym vs sim, |diff|
- `plots/policy_eval/videos/`          — per-env mp4s (env-0 by default)

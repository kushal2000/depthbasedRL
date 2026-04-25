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

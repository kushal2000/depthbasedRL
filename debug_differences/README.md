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

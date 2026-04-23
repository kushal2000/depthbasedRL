# SimToolReal overnight port — status

Port landed 2026-04-23 (overnight, kk837 asleep). Goal from the evening:
> *Port over the SimToolReal env into isaacsimenvs. Your goal is to run the
>  pretrained policy in isaacsim by the time I wake up.*

## What works

- **Pretrained policy runs in Isaac Sim via isaacsimenvs.** Driver:
  `isaacsimenvs/tasks/simtoolreal/play_simtoolreal.py`. The rollout loop
  (140-dim obs → rl_games LSTM → 29-dim action → PD targets → 2-substep
  120 Hz physics step) completes without crashing and produces an mp4 plus a
  `trajectory.npz`. Verified at both 240-step and 6000-step scales.
- **Task is registered.** `gym.spec("Isaacsimenvs-SimToolReal-Direct-v0")`
  resolves; `load_cfg_from_registry(task, "env_cfg_entry_point")` returns
  `SimToolRealEnvCfg()`, `rl_games_cfg_entry_point` returns the rl_games dict.
- **Configs exist.** `cfg/task/SimToolReal.yaml` (overlay) and
  `cfg/train/SimToolRealPPO.yaml` (future-training) are in the same shape as
  Cartpole's, with the `num_actors: ${....env.scene.num_envs}` Phase 2
  convention.
- **Same prefix policy as Cartpole.** Task id uses the `Isaacsimenvs-` prefix
  (from the Phase 2 commit note) so it doesn't collide with Isaac Lab's stock
  `Isaac-*-Direct-v0` registrations.

## Rollout behavior: 0/12 goals reached

The 6000-step beam-2 rollout finished with **0/12 goals hit** — the object
stayed parked at `z=0.536` (roughly table height) for the entire run, and
the kp-dist metric was frozen at `0.0871` from step ~60 onward. The policy
is clearly running (observations are being computed, actions are being
emitted, PD targets are applied), but the object never moves.

**The port is faithful.** Ran the same 6000-step rollout through legacy
`isaacsim_conversion/rollout.py` for comparison. Trajectories are
**bit-identical** (q[5999] and object_pose[5999] match to float32
precision; kp_dist min/mean/max identical). So the 0/12 result is
entirely upstream — it's not caused by anything in the isaacsimenvs port.

Upstream root cause to investigate on wake — the FGT comparison run finished
and also hit 0/12. Both checkpoints **run** but **fail differently**:

| Metric                 | Generic `pretrained_policy/` | Beam-FGT `Apr16_experiments/beam_final_goal_only_dr/` |
|------------------------|------------------------------|-------------------------------------------------------|
| kp_dist min            | 0.024                        | 0.048                                                 |
| kp_dist mean           | 0.087                        | 0.656                                                 |
| object XY delta        | 7 cm                         | 33 cm                                                 |
| object z max reached   | **0.624** (10 cm above table)| 0.537 (never lifted)                                  |
| object z end           | 0.536 (on table)             | **0.006** (fell to floor)                             |

Takeaways:
- Generic briefly lifts the beam 10 cm then loses grip; FGT never lifts but
  pushes the beam ~33 cm and knocks it off the table.
- Neither reproduces the README's "all 12 goals" claim.
- Rollout videos for both are in `isaacsimenvs/rollout_videos/` (generic) and
  `isaacsimenvs/rollout_videos_fgt/` (FGT) for side-by-side inspection.

Since the port is bit-exact faithful to legacy, the fix is **not** in
isaacsimenvs. Prioritized upstream root causes:

1. **Asset drift — strong lead.** `git log assets/urdf/fabrica/beam/`
   shows commit `04948ef Re-run beam CoACD with seed 2 and generalize
   sweep script`. The coacd collision meshes were regenerated. A policy
   trained on the old meshes has to grasp a geometrically-different peg
   now. This is a very plausible explanation for the contact failure
   observed (generic policy briefly lifts to z=0.62 then drops it).
   **Action:** checkout the repo at the commit BEFORE `04948ef`, regenerate
   USD cache (delete `/tmp/isaaclab_usd_cache`), rerun the rollout. If
   goals are reached, confirmed.
2. **Isaac Sim version / PhysX.** The `.venv_isaacsim/` snapshot may have
   updated since the README claim. `pip show isaaclab` vs the Isaac Lab
   version referenced in the README (2.3.2.post1).
3. **Different beam / part_id.** The README doesn't name the exact
   assembly; maybe the "all 12 goals" run was for a different fabrica part.
4. **DexToolBench path** (`--task_source dextoolbench`) may behave
   differently — that code path is less exercised by this port but might
   be what the README's claim actually referred to.

Artifacts of the 6000-step run:
- `isaacsimenvs/rollout_videos/rollout_beam_2.mp4` (1.7 MB, 100 s @ 30fps)
- `isaacsimenvs/rollout_videos/trajectory.npz` (985 KB, 6000 steps of q,
  object_pose, kp_dist, goal_idx — useful for the behavioral diagnosis)

## What doesn't work / is stub

- **`SimToolRealEnv.__init__` raises `NotImplementedError`.** The training
  path — scene setup, reward, reset, obs construction — is NOT ported. This
  is a 6000+-line env; rewriting it for DirectRLEnv is not overnight work and
  wasn't the user's ask.
- **`gym.make("Isaacsimenvs-SimToolReal-Direct-v0")` will raise.** Use
  `play_simtoolreal.py` for now; the raw env class is for future work.
- **The config YAMLs are lightly ported.** `SimToolRealPPO.yaml` drops the
  isaacgymenvs-specific interpolations (`${....task.env.numEnvs}`,
  `${...seed}`) in favor of hardcoded defaults + the Phase 2 convention.
  Field names on the configclass are a thin subset of what the full env
  needs — add fields as the training port fills out.

## The architectural bet

`play_simtoolreal.py` delegates to `isaacsim_conversion.IsaacSimEnv`, the
standalone single-env scene builder that has already been validated (see
`isaacsim_conversion/README.md`: *"All 12 goals of the beam assembly task
completed successfully in Isaac Sim using the FGT curriculum policy trained
in Isaac Gym."*). Rather than duplicate that ~600-line scene setup into
isaacsimenvs, the play script imports and reuses it.

To make that import reliable, `isaacsim_conversion*` was added to
`pyproject.toml`'s `[tool.setuptools.packages.find]` and the isaacsim venv
was reinstalled via `uv pip install -e . --no-deps`.

## How to run

```bash
source .venv_isaacsim/bin/activate
python isaacsimenvs/tasks/simtoolreal/play_simtoolreal.py \
    --assembly beam --part_id 2 --collision_method coacd \
    --max_steps 6000 --enable_cameras
```

Output:
- `rollout_videos/rollout_beam_2.mp4` — the rollout video
- `rollout_videos/trajectory.npz` — per-step joint/object/kp-dist log

For a different object, swap `--assembly/--part_id` or use
`--task_source dextoolbench --object_category ... --object_name ...`.

## Next steps (waking-hours work)

1. **DirectRLEnv training path.** Port
   `isaacgymenvs/tasks/simtoolreal/env.py` `_create_envs`, `post_physics_step`,
   `compute_reward`, `reset_idx` into `SimToolRealEnv`. This needs
   InteractiveScene + per-env cloning (vs. the current single-env `IsaacSimEnv`
   setup). Expect the obs calculation in `observation_action_utils_sharpa.py`
   to be reusable — it's pure numpy on q/qd/object_pose tensors.
2. **Policy evaluation.** Once `gym.make` works, wire `SimToolReal` into
   `train.py --test --checkpoint pretrained_policy/model.pth` so the uniform
   Cartpole CLI also drives SimToolReal inference.
3. **Fold `isaacsim_conversion/` into `isaacsimenvs/tasks/simtoolreal/`.**
   Once the DirectRLEnv path works, the legacy dir can be deleted (or kept
   as a reference).
4. **Smoke test for the play script.** Mirror
   `tests/test_cartpole_play.py` — a 60-step rollout against a fixed seed,
   assert exit code + mp4 produced + some `near_goal` progress recorded.

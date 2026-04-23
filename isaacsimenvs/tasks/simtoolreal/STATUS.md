# SimToolReal overnight port — status

Port landed 2026-04-23 (overnight, kk837 asleep). Goal from the evening:
> *Port over the SimToolReal env into isaacsimenvs. Your goal is to run the
>  pretrained policy in isaacsim by the time I wake up.*

## What works

- **Pretrained policy runs in Isaac Sim via isaacsimenvs.** Driver:
  `isaacsimenvs/tasks/simtoolreal/play_simtoolreal.py`. The rollout loop
  (140-dim obs → rl_games LSTM → 29-dim action → PD targets → 2-substep
  120 Hz physics step) completes without crashing and produces an mp4 plus a
  `trajectory.npz`.
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

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

Two things to check on wake:
1. Does `isaacsim_conversion/rollout.py` (the legacy driver, pre-port) see
   the same 0/12 behavior? A parallel comparison is queued; check
   `rollout_videos_legacy/trajectory.npz` vs `isaacsimenvs/rollout_videos/
   trajectory.npz`. If legacy also sees 0/12, the port is faithful and the
   problem is upstream — probably checkpoint mismatch (the generic
   `pretrained_policy/model.pth` wasn't trained on fabrica/beam, see
   `hardware_rollouts/Apr16_experiments/beam_*` for beam-specific variants).
2. Try a beam-specific checkpoint: `hardware_rollouts/Apr16_experiments/
   beam_multi_init_table_rand_dr/model.pth` or `beam_final_goal_only_dr/`.
   The `isaacsim_conversion/README.md` status line claims "all 12 goals …
   using the FGT curriculum policy" — "FGT" = `final_goal_only` variant.

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

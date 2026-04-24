# SimToolReal — port status

Updated 2026-04-23.

The full training path (not just inference) is now ported from `isaacgymenvs`
to `isaacsimenvs`. `env.reset()` + `env.step()` return the expected rl_games
asymmetric-obs tuple end-to-end, and both PPO + SAPG smoke training runs exit
cleanly with checkpoints.

## What works

- **All seven DirectRLEnv hooks implemented.** `_setup_scene`, `_pre_physics_step`,
  `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`.
- **Asymmetric actor-critic obs** `{"policy": (N, 140), "critic": (N, 162)}`.
  Dimensions auto-computed in `__init__` from `cfg.obs.obs_list` /
  `cfg.obs.state_list`. SAPG critic input expands to 194 when the 32-dim
  exploration-coef embedding is concatenated.
- **Full domain-randomization set** folded in where each knob belongs:
  obs delay + object-state delay + xyz noise + rotation noise +
  object-scale multiplier + joint-vel obs noise live in `_get_observations`;
  force/torque impulses live in `_pre_physics_step`; action delay queue
  lives at the top of `_pre_physics_step`.
- **Success-tolerance curriculum** advances
  `_current_success_tolerance` from 0.075 → 0.01 (multiplicative 0.9) once
  mean(`_prev_episode_successes`) ≥ 3 after `tolerance_curriculum_interval`
  policy steps.
- **PPO + SAPG smoke training** (8 envs × 3 epochs × horizon 16) both
  complete cleanly, write tensorboard events + a checkpoint under
  `outputs/<date>/<time>/0_simtoolreal{,_sapg}/`. Second-epoch fps ≈ 475
  total on a single RTX 6000 Ada.
- **Pretrained-policy inference** (`play_simtoolreal.py`) still works.
  It's an independent script over `isaacsim_conversion.IsaacSimEnv`, not
  touched by the training port, and its legacy-faithful 0/12 behavior on
  beam-2 is unchanged (see "Legacy 0/12" below).

## Port layout

```
isaacsimenvs/tasks/simtoolreal/
├── simtoolreal_env.py          # 90 lines — pure DirectRLEnv dispatcher
├── simtoolreal_env_cfg.py      # 8 sectioned @configclass dataclasses
├── __init__.py                 # gym.register(Isaacsimenvs-SimToolReal-Direct-v0)
├── play_simtoolreal.py         # pretrained inference (unchanged from overnight port)
├── STATUS.md                   # this file
└── utils/                      # all math + orchestration
    ├── action_utils.py         # apply_action_pipeline + apply_wrench_dr
    ├── obs_utils.py            # build_observations + compute_intermediate_values
    ├── reward_utils.py         # 4 reward-term helpers + compute_rewards
    ├── termination_utils.py    # compute_terminations + update_tolerance_curriculum
    ├── reset_utils.py          # allocate_state_buffers + reset_env_state
    ├── scene_utils.py          # setup_scene + cfg builders + PD tables + materials
    ├── generate_objects.py     # procedural URDF emitter
    ├── object_size_distributions.py
    └── goal_sampling.py        # absolute + delta goal-pose samplers

isaacsimenvs/cfg/
├── task/SimToolReal.yaml       # 8-section overlay (YAML ↔ configclass 1:1)
├── train/SimToolRealPPO.yaml   # LSTM Asymmetric PPO baseline
└── train/SimToolRealSAPG.yaml  # LSTM Asymmetric SAPG (default)
```

`simtoolreal_env.py` is intentionally ~90 lines — every hook dispatches
into `utils/`:

```python
def _setup_scene(self):          setup_scene(self)
def _pre_physics_step(self, a):  apply_action_pipeline(self, a); apply_wrench_dr(self)
def _apply_action(self):         self.robot.set_joint_position_target(self._cur_targets)
def _get_observations(self):     return build_observations(self)
def _get_rewards(self):          return compute_rewards(self)
def _get_dones(self):
    update_tolerance_curriculum(self)
    compute_intermediate_values(self)
    return compute_terminations(self)
```

## How to run

```bash
# PPO smoke (3 epochs, 8 envs)
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_cfg_entry_point \
  --headless \
  env.scene.num_envs=8 \
  env.assets.num_assets_per_type=2 \
  agent.params.config.max_epochs=3 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=128 \
  agent.params.config.mini_epochs=1 \
  agent.params.config.central_value_config.minibatch_size=128 \
  agent.params.config.central_value_config.mini_epochs=1

# SAPG smoke — swap --agent and add expl_coef_block_size divisor
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  env.scene.num_envs=8 \
  env.assets.num_assets_per_type=2 \
  agent.params.config.max_epochs=3 \
  agent.params.config.horizon_length=16 \
  agent.params.config.minibatch_size=128 \
  agent.params.config.mini_epochs=1 \
  agent.params.config.expl_coef_block_size=8 \
  agent.params.config.central_value_config.minibatch_size=128 \
  agent.params.config.central_value_config.mini_epochs=1
```

Constraints to remember when picking smoke sizes:

- `num_envs * horizon_length >= minibatch_size`
- `horizon_length >= seq_length` (YAML default 16 — keep horizon ≥ 16 or override both)
- SAPG: `num_envs % expl_coef_block_size == 0`

For full runs at the legacy training scale (num_envs=8192 or 24576 with
horizon_length=16, minibatch_size=98304), the shipped YAML defaults are
correct and no overrides are needed.

## Deliberate simplifications vs legacy

- **Object geometry is homogeneous within a launch.** All envs share one
  handle-head USD; per-env variability is injected via the
  `_object_scale_multiplier` DR applied to keypoint offsets and
  `object_scales` obs. This trades away per-env mesh diversity for scene
  stability — see "Known issues" below.
- **Dropped legacy config fields** (parsed but never read in legacy code,
  or tied to code paths we don't port): `fallDistance`, `fallPenalty`,
  `observationType`, `stiffnessScale`, `forceLimitScale`,
  `useRelativeControl`, entire `task.randomization_params` tree,
  `linVelImpulse*` / `angVelImpulse*` (always 0), Tyler curriculum,
  final-goal tolerance curriculum.
- **Physics rate**: `sim.dt=1/120`, `decimation=2` → 60 Hz policy / 120 Hz
  physics (matches legacy `dt=1/60 + substeps=2`). `controlFrequencyInv` is
  not exposed — Isaac Lab's `decimation` fills the role.
- **Joint limits**: port uses `robot.data.joint_pos_limits` (raw URDF),
  not `soft_joint_pos_limits` (Isaac Lab shrinks by a safety factor that
  legacy doesn't apply).
- **Action penalty** is `L1 norm of joint velocities × scale × -1` per
  legacy `env.py:2481-2494`, not `||action||²` (the legacy name is
  misleading; the math is velocity-based).

## Known issues

### Homogeneous object geometry — workaround for an Isaac Lab cloner bug

At `num_envs ≥ 8`, spawning per-env prims manually after
`scene.clone_environments(copy_from_source=True)` silently drops instances:
8 envs produce 7 Object prims in the PhysX view, 16 envs produce 14. The
`write_root_pose_to_sim` call then crashes inside PhysX. We worked around
it by spawning one Object + GoalViz in env_0 before
`clone_environments(copy_from_source=False)`; the cloner replicates those
prims to every env via instance proxies (no count mismatch at any size).

The procedural URDF pool is still generated per launch — we just only use
`usd_paths[0]`. If per-env mesh heterogeneity turns out to matter for
transfer, the fix space is:
- (a) investigate the `copy_from_source=True` drop root-cause;
- (b) switch to a `RigidObjectCollection`-based pattern with N distinct cfg entries;
- (c) handle heterogeneity via scale DR rather than mesh DR.

### Legacy 0/12 on beam-2 (inference path, upstream)

`play_simtoolreal.py` runs the pretrained checkpoint but the object never
reaches any of the 12 fabrica beam goals. The rollout is bit-exact
faithful to legacy `isaacsim_conversion/rollout.py` (trajectories match to
float32), so the 0/12 is upstream — current best lead is CoACD
collision-mesh drift vs. what the pretrained policy was trained against
(regenerated in commit `04948ef`). Not blocked by the training port;
revisit after retraining against the new port.

## Verification checkpoints

- `isaacsimenvs/tests/test_gym_register.py` — registry smoke (Cartpole-level).
- `isaacsimenvs/tests/test_simtoolreal_play.py` — pretrained-policy rollout
  smoke (240 steps on beam-2; asserts mp4 + trajectory.npz written).
- Inline verification scripts per phase in the plan at
  `.claude/plans/we-are-currently-in-twinkling-bengio.md` walk through
  every hook individually.

## Deferred / next steps

- **Full training convergence check**: smoke verified the pipeline; actual
  reward curves + learned-policy behavior on goal-pose reaching still need
  a long run.
- **Per-env mesh heterogeneity**: see "Known issues". Worth revisiting if
  sim-to-real tool-variation transfer is a bottleneck.
- **Fabrica / DexToolBench / Peg-in-Hole extensions**: explicitly out of
  scope for this port. Hooks for a `useFixedGoalStates`-style fixed-goal
  sequence will need to be re-added when those tasks land.

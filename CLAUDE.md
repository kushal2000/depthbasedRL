# CLAUDE.md

## Project Overview
Depth-based RL for robotic assembly. The `fabrica/` module handles beam/part assembly tasks with visualization and trajectory generation.

## Key Directories
- `assets/urdf/fabrica/<assembly>/` — per-assembly mesh parts (OBJ) and config (assembly_order.json)
- `assets/urdf/fabrica/environments/` — pre-built environment URDFs
- `fabrica/` — Python scripts for visualization, trajectory generation, evaluation
- `assets/urdf/kuka_sharpa_description/` — robot URDF

## Assemblies
beam, car, cooling_manifold, duct, gamepad, plumbers_block, stool_circular

Each assembly has numbered part subdirs (e.g., `beam/0/0.obj`) and an `assembly_order.json` with:
```json
{"steps": ["6","3","2","1","0"], "start_rotations": {"2": [1,0,0,90]}}
```

## Scene Constants (eval.py)
- TABLE_Z = 0.38, table surface = 0.53
- Robot at (0, 0.8, 0)
- Fixture at (0.12, -0.152, 0.15) relative to table

## Visualization
- Uses **viser** for web-based 3D viewers
- `viser_utils.py` has shared loaders: `load_all_assemblies`, `load_assembly_order`, `SceneManager`
- Viser `add_line_segments` does NOT reliably render — use mesh geometry for visual markers
- Default viser port: 8082

## Workflow
- **Never kill ports blindly** — other processes (training jobs, user sessions) may be using them. Always use a free port instead of killing existing ones.
- Use `lsof -i:<port>` to check if a port is occupied before binding to it
- Only run one eval server at a time, always on port 8080

## Conventions
- Assembly task specs (order, rotations) go in JSON config files, not hardcoded in Python
- OBJ meshes are in assembled reference frame — no rotation needed at final position
- Part IDs are strings ("0", "1", ...) not ints
- Branch names start with the creation date in `YYYY_MM_DD` format (e.g., `2025_03_18_my_feature`)

---

# CLAUDE.md — simtoolreal_private

This file captures the current implementation plan so context can be restored across sessions.

## Branch
`2026-04-15_WandbLog_SimpleRL` (pushed to `origin`)

## Environment Setup
```bash
cd /home/tylerlum/github_repos/simtoolreal_private
uv venv --python 3.8
echo 'export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))"):$LD_LIBRARY_PATH' >> .venv/bin/activate
source .venv/bin/activate
uv pip install -e .
cd /home/tylerlum/isaacgym/python && uv pip install -e . && cd -
cd rl_games && uv pip install -e . && cd -
```

---

## Project 1: WandbLog — Interactive Viewer (DONE ✓)

**Goal:** Replace RGB frame → MP4 → `wandb.Video` with pose-state → `create_html()` → `wandb.Html`.
Both video and HTML are logged simultaneously so they can be compared side-by-side in wandb.

### Files added/modified
- `isaacgymenvs/tasks/simtoolreal/interactive_viewer/` — copied from `wandb_interactive_robot_viewer` branch `experiment/goal-object-brush-recolor-fix`
  - `index.template.html` — browser runtime (Three.js + urdf-loader)
  - `viewer_api.py` — `create_html()`, `make_url_robot()`, `make_embedded_robot()`. Fixed: `TypeAlias` → `Tuple` for Python 3.8 compat; `from viewer_common` → `from .viewer_common`
  - `viewer_common.py` — `render_template()` and helpers. Fixed: `DEFAULT_TEMPLATE_PATH` → `Path(__file__).parent / "index.template.html"`
  - `__init__.py` — re-exports the above
- `isaacgymenvs/tasks/simtoolreal/env.py` — added:
  - `ViewerFrame` dataclass (typed container for one timestep of pose state)
  - `self.viewer_state_frames` state variable (3-state machine mirroring `self.video_frames`)
  - `_capture_viewer_if_needed()` — called alongside `_capture_video_if_needed()`
  - `_capture_viewer_frame()` — collects robot/object/goal/table poses into `ViewerFrame`
  - `_finalize_viewer_capture()` — builds HTML and logs `wandb.Html`
- `isaacgymenvs/cfg/task/SimToolReal.yaml` — added `capture_viewer`, `capture_viewer_freq`, `capture_viewer_len`

### Robot URDF URL
Hardcoded for `iiwa14_left_sharpa_adjusted_restricted` (the only robot currently used):
```
https://raw.githubusercontent.com/tylerlum/simtoolreal/main/isaacgymenvs/assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf
```
A `ValueError` is raised at init if a different robot asset is configured.

### Quaternion convention
Isaac Gym root state tensor: `[x, y, z, qx, qy, qz, qw]` (xyzw).
`viewer_api.py` expects the same. Three.js `quaternion.set(x, y, z, w)` expects xyzw. All consistent.

### Smoke test commands
```bash
cd /home/tylerlum/github_repos/simtoolreal_private
source .venv/bin/activate

# Viewer smoke test (freq=50 so it arms immediately, captures 30 frames)
python isaacgymenvs/train.py task=SimToolReal headless=True num_envs=64 \
    task.env.capture_viewer=True task.env.capture_viewer_freq=50 task.env.capture_viewer_len=30 \
    wandb_activate=True wandb_project=simtoolreal_test \
    train.params.config.minibatch_size=512

# DexToolBench eval with pretrained policy (confirmed: 100% success on claw_hammer swing_down)
python dextoolbench/eval.py \
    --object-category hammer \
    --object-name claw_hammer \
    --task-name swing_down \
    --config-path pretrained_policy/config.yaml \
    --checkpoint-path pretrained_policy/model.pth \
    --num-episodes 1 \
    --output-dir /tmp/eval_results
```

---

## Project 2: SimpleRL — SAPG/EPO Integration (DONE ✓)

**Goal:** Add `simple_rl` as a clean training path alongside rl_games (don't delete rl_games).
Source: `/home/tylerlum/github_repos/simple_rl`, branch `2025-06-13_SAPG_EPO`.

### Files added
- `simple_rl/` — copied from source repo: `agent.py`, `player.py`, `utils/`
- `isaacgymenvs/train_simple_rl.py` — new entry point, modeled on `human2sim2robot/sim_training/run.py`
- `isaacgymenvs/utils/simple_rl_env_wrapper.py` — thin wrapper to add `get_env_info()` to isaacgym env
- `isaacgymenvs/cfg/train/SimToolRealSimpleRLPPO.yaml`
- `isaacgymenvs/cfg/train/SimToolRealSimpleRLSAPG.yaml` (M=6, conditioning_dim=32 — matches rl_games pretrained)
- `isaacgymenvs/cfg/train/SimToolRealSimpleRLEPO.yaml` (M=64, conditioning_dim=32)
- `deployment/rl_player_simple_rl.py` — drop-in replacement for `rl_player.py` using `simple_rl.Player`
- `test_simple_rl_load.py` — confirms pretrained rl_games checkpoint is NOT directly compatible with simple_rl
- `test_rl_player_simple_rl.py` — E2E test for `RlPlayerSimpleRL` (instantiation, forward pass, LSTM, reset)
- `convert_rlgames_to_simple_rl.py` — one-rename converter: `a2c_network.extra_params` → `a2c_network.conditioning`
- `pretrained_policy/model_simple_rl.pth` — converted pretrained policy (simple_rl format)
- `pretrained_policy/config_simple_rl.yaml` — minimal simple_rl config (M=6, dim=32, LSTM=1024, MLP=[1024,1024,512,512])
- `isaacgymenvs/launch_simple_rl.py` — tyro CLI launcher for PPO/SAPG/EPO (see `docs/rl.md`)
- `docs/rl.md` — reference doc: training commands, eval, wandb viewer, checkpoint resumption
- `rl_games/rl_games/common/a2c_common.py` — fixed policy_idx parsing to fall back to 0 gracefully

### Key architecture notes
- simple_rl checkpoint format: `{0: {"model": ..., "running_mean_std": ...}}` (rank-wrapped)
- SAPG at inference: append `conditioning_idx=0` (leader, integer 0) to obs — replaces the rl_games `50.0` hack
- **Sigma architecture**: `simple_rl/utils/network.py` uses per-block sigma `(num_conditionings, actions)` when `num_conditionings is not None` (SAPG/EPO), indexed by `conditioning_idxs`. PPO uses shared `(actions,)`. Matches rl_games `coef_cond` exactly.
- **Weight transfer from rl_games to simple_rl**: only ONE key rename needed — `a2c_network.extra_params` → `a2c_network.conditioning`. All 24 other keys (MLP, LSTM, mu, value, layer norm, running stats) are identical in name and shape when `conditioning_dim=32, num_conditionings=6`. Verified: both paths achieve 100% on claw_hammer swing_down (3 episodes).
- `train_simple_rl.py` handles `wandb_name` interpolation error and now auto-derives a unique name per algorithm (PPO/SAPG/EPO) — each gets its own `runs/SimToolReal_SimpleRL_{ALGO}/` dir
- `train_simple_rl.py` uses timestamped wandb run IDs/names (same convention as `wandb_utils.py`)
- `deployment/rl_player_simple_rl.py`: must set `player.batch_size = num_envs` before `player.init_rnn()`; `_DeploymentEnv` needs a no-op `set_env_state()` method
- `deployment/rl_policy_node.py`: pass `--use_simple_rl` to load simple_rl checkpoints; reads `obsList` from config directly (bypasses `player.cfg`)
- `dextoolbench/eval.py`: pass `--use-simple-rl` to use `RlPlayerSimpleRL` instead of `RlPlayer`
- EPO smoke test requires `use_others_experience=False` at small env counts (M=64 off-policy augmentation creates non-divisible batch sizes with LSTM seq_length)

### Training commands
```bash
source .venv/bin/activate

# PPO
python isaacgymenvs/train_simple_rl.py \
    task=SimToolReal train=SimToolRealSimpleRLPPO \
    wandb_activate=True wandb_project=simtoolreal_test

# SAPG
python isaacgymenvs/train_simple_rl.py \
    task=SimToolReal train=SimToolRealSimpleRLSAPG \
    wandb_activate=True wandb_project=simtoolreal_test

# EPO
python isaacgymenvs/train_simple_rl.py \
    task=SimToolReal train=SimToolRealSimpleRLEPO \
    wandb_activate=True wandb_project=simtoolreal_test
```

---

## DexToolBench Results
- Pretrained policy (rl_games path): 100% success on claw_hammer swing_down (3 episodes, ~9.5s each)
- Pretrained policy (simple_rl path, converted): 100% success on claw_hammer swing_down (3 episodes, ~10.5s each)
- `python download_pretrained_policy.py` → `pretrained_policy/config.yaml` + `pretrained_policy/model.pth`
- `python download_dextoolbench_data.py --object-name claw_hammer --task-name swing_down`

### Eval commands
```bash
# rl_games path (original pretrained)
python dextoolbench/eval.py \
    --object-category hammer --object-name claw_hammer --task-name swing_down \
    --config-path pretrained_policy/config.yaml \
    --checkpoint-path pretrained_policy/model.pth \
    --num-episodes 3 --output-dir /tmp/eval_rlgames

# simple_rl path (converted pretrained — same weights, same results)
python dextoolbench/eval.py \
    --object-category hammer --object-name claw_hammer --task-name swing_down \
    --config-path pretrained_policy/config_simple_rl.yaml \
    --checkpoint-path pretrained_policy/model_simple_rl.pth \
    --num-episodes 3 --output-dir /tmp/eval_simple_rl_pretrained \
    --use-simple-rl
```

---

## Full plan
See `/afs/cs.stanford.edu/u/tylerlum/.claude/plans/lively-mapping-teapot.md`

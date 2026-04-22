# Isaac Sim Policy Rollout

Roll out pretrained Isaac Gym policies in Isaac Sim (via Isaac Lab) for visualization/evaluation.

**Status**: All 12 goals of the beam assembly task completed successfully in Isaac Sim using the FGT curriculum policy trained in Isaac Gym.

## Prerequisites

- Python 3.11 (Isaac Sim 5.x / Isaac Lab 2.3.x requirement)
- NVIDIA GPU with driver >= 525.60 (we have 570.124.06)
- CUDA 12+ (we have 12.8)
- `uv` for package management

## Installation

All commands assume you're in the repo root: `/share/portal/kk837/depthbasedRL`

### 1. Create Python 3.11 venv

```bash
uv venv .venv_isaacsim --python 3.11
source .venv_isaacsim/bin/activate
```

If Python 3.11 is not available: `uv python install 3.11`

### 2. Install PyTorch (CUDA 12.1)

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

### 3. Install vendored rl_games and inference deps

```bash
uv pip install -e ./rl_games/
uv pip install omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy
```

**Do NOT** run `uv pip install -e .` — the project's pyproject.toml pins `numpy==1.23.0`, `isaacgym-stubs`, and `warp-lang==0.10.1` which conflict with Python 3.11 / Isaac Sim.

### 4. Install Isaac Lab (bundles Isaac Sim)

```bash
uv pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com
```

This is a ~15GB download. The `[isaacsim,all]` extra bundles Isaac Sim + all extensions + Isaac Lab modules.

**First launch notes**:
- Isaac Sim will prompt to accept the NVIDIA EULA on first run.
- First launch takes ~2-5 minutes (shader compilation). Subsequent launches are ~16-30s.
- Shader cache lives in `~/.cache/ov/`. On NFS this is slow — see step 5 for speedup.

### 5. (Optional) Speed up startup with local cache

Isaac Sim startup is slow (~2 min) when caches are on NFS (`~/.cache/ov/` is on `portal-nfs-01`). On compute nodes with local SSD (`/scratch`):

```bash
export OMNI_KIT_CACHE_PATH=/scratch/$USER/ov_cache
mkdir -p $OMNI_KIT_CACHE_PATH
```

Note: `/scratch` is node-local and won't persist across different SLURM nodes.

### 6. Verify installation

```bash
# 1. Policy inference smoke test (no simulator needed)
PYTHONPATH=. python isaacsim_conversion/test_inference.py

# 2. Isaac Sim + Isaac Lab launch test (~2 min first time)
python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
import isaaclab.sim as sim_utils
print('Isaac Sim + Isaac Lab OK')
app.close()
"
```

## Quick reference: full install from scratch

```bash
cd /share/portal/kk837/depthbasedRL
uv venv .venv_isaacsim --python 3.11
source .venv_isaacsim/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e ./rl_games/
uv pip install omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy
uv pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com
```

## Usage

### Smoke test (inference only, no simulator)

```bash
source .venv_isaacsim/bin/activate
PYTHONPATH=. python isaacsim_conversion/test_inference.py
```

### Full rollout (without video)

```bash
source .venv_isaacsim/bin/activate
PYTHONPATH=. python isaacsim_conversion/rollout.py --headless --max_steps 700 \
    --checkpoint path/to/last/model.pth \
    --config path/to/config.yaml
```

### Full rollout with video recording

```bash
source .venv_isaacsim/bin/activate
# Kill any lingering Isaac Sim processes first!
ps aux | grep "python.*isaacsim\|python.*rollout" | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null

PYTHONPATH=. python isaacsim_conversion/rollout.py --headless --enable_cameras --max_steps 700 \
    --checkpoint path/to/last/model.pth \
    --config path/to/config.yaml \
    --video_dir rollout_videos/my_experiment
```

### Example: beam assembly with FGT curriculum policy

```bash
FGT_BASE="train_dir/FABRICA_TRAINING/fabrica/no_dr_fgt_curriculum_beam_2_coacd_2026-03-18_01-34-01/runs/00_no_dr_fgt_curriculum_beam_2_coacd_2026-03-18_01-34-01"

PYTHONPATH=. python isaacsim_conversion/rollout.py --headless --enable_cameras --max_steps 700 \
    --checkpoint "$FGT_BASE/last/model.pth" \
    --config "$FGT_BASE/config.yaml" \
    --video_dir rollout_videos/fgt_beam
```

## Critical transfer details: Isaac Gym → Isaac Sim

These are the key findings that made the policy transfer work. Getting any of these wrong causes the policy to fail silently (robot moves but doesn't complete the task).

### 1. Drive gain compensation (180/pi)

Isaac Lab's `UrdfConverter` multiplies revolute joint stiffness/damping by `pi/180` internally (rad→deg conversion for USD). But our values from Isaac Gym are already in PhysX units. We pre-multiply by `180/pi` to cancel this out:

```python
JOINT_STIFFNESSES_COMPENSATED = {k: v * 180.0 / math.pi for k, v in JOINT_STIFFNESSES.items()}
```

Without this, drives are ~57x too weak and the robot barely moves.

### 2. contact_offset = 0.002

Training uses `contact_offset=0.002` (in `SimToolReal.yaml`). Isaac Sim defaults to `0.02` (10x larger). Must be set explicitly via `CollisionPropertiesCfg`:

```python
collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
```

### 3. Robot gravity disabled

Isaac Gym disables gravity on the robot (`asset_options.disable_gravity = True`). Must match in Isaac Sim:

```python
rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
```

### 4. Self-collision off

```python
articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)
```

### 5. Robot default arm pose

The robot arm starts at a specific configuration, NOT all zeros:

```python
default_arm_pos = [-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308]  # SharPa mount
```

Hand joints start at all zeros.

### 6. Object scales = fixedSize from training config

The `object_scales` observation uses `fixedSize` from the training config (actual bounding box in meters), NOT `Object.scale` from the registry (which is the mesh scale factor):

```python
# CORRECT: from training config
fixedSize: [0.141, 0.03025, 0.0271]

# WRONG: from Object.scale (mesh scale factor, ~25x larger)
Object.scale: [1.905, 0.300, 0.288]
```

### 7. Joint ordering permutation

Isaac Gym uses depth-first ordering (all thumb joints, then all index joints...), Isaac Sim uses breadth-first (all first-level joints across fingers, then second-level...). A permutation array maps between them automatically.

### 8. Quaternion convention

Isaac Sim uses wxyz, Isaac Gym/observation code uses xyzw. Convert at the sim boundary:

```python
quat_xyzw = quat_wxyz[[1, 2, 3, 0]]  # wxyz → xyzw
```

### 9. Use `last/model.pth`, not early checkpoints

The `nn/` checkpoint is saved early in training and barely trained. Always use `last/model.pth` or `best/model.pth`.

### 10. merge_fixed_joints = True

Match Isaac Gym's `collapse_fixed_joints=True`. Without this, extra fixed-joint bodies cause collision issues.

## Video recording details

Video recording uses the Isaac Lab `Camera` sensor with `--enable_cameras` flag.

**Camera setup:**
- Camera sensor must be created AFTER `IsaacSimEnv.__init__()`, followed by `sim.reset()` to initialize internal buffers (`_timestamp`).
- Object poses set before `sim.reset()` get wiped — always place objects AFTER camera init.
- Use `convention="opengl"` for the `CameraCfg.OffsetCfg` (standard camera: -Z forward, +Y up).
- Camera position matching Isaac Gym viewer: `pos=(0, -1, 1.03)` looking at `target=(0, 0, 0.53)`.
- First run with `--enable_cameras` takes ~5-15 min for RTX shader compilation. Subsequent runs are cached.
- **Kill all old Isaac Sim processes before starting** — multiple processes on one GPU will hang.

## Architecture

The inference pipeline is simulator-agnostic:

```
Isaac Sim → q, qd, object_pose → compute_observation() → RlPlayer → compute_joint_pos_targets() → Isaac Sim
                                    (numpy, yourdfpy FK)     (rl_games)      (numpy)
```

Only 3 values come from the simulator per step:
- Joint positions (29)
- Joint velocities (29)
- Object world pose (7: xyz + quaternion xyzw)

Everything else (palm pose, fingertips, keypoints) is computed via forward kinematics in `observation_action_utils_sharpa.py`.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `test_inference.py` | Smoke test — full obs→policy→action pipeline without simulator |
| `test_camera.py` | Standalone camera sensor test |
| `isaacsim_env.py` | Isaac Lab scene setup, URDF import, state extraction, physics config |
| `rollout.py` | Main entry point — policy rollout with goal switching and video recording |
| `README.md` | This file |

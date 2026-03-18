# Isaac Sim Policy Rollout

Roll out the pretrained Isaac Gym policy in Isaac Sim (via Isaac Lab) for visualization/evaluation.

## Prerequisites

- Python 3.10 (Isaac Sim 4.x requirement)
- NVIDIA GPU with driver >= 525.60 (we have 570.124.06)
- CUDA 12+ (we have 12.8)
- `uv` for package management

## Installation

All commands assume you're in the repo root: `/share/portal/kk837/depthbasedRL`

### 1. Create Python 3.10 venv

```bash
uv venv .venv_isaacsim --python 3.10
source .venv_isaacsim/bin/activate
```

If Python 3.10 is not available: `uv python install 3.10`

### 2. Install PyTorch (CUDA 12.1)

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

### 3. Install vendored rl_games

```bash
uv pip install -e ./rl_games/
```

### 4. Install inference dependencies

```bash
uv pip install omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy
```

**Do NOT** run `uv pip install -e .` — the project's pyproject.toml pins `numpy==1.23.0`, `isaacgym-stubs`, and `warp-lang==0.10.1` which conflict with Python 3.10 / Isaac Sim.

### 5. Install Isaac Sim 4.5.0

```bash
uv pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

This is a ~15GB+ download. The `[all,extscache]` extra installs all extensions including `isaacsim.simulation_app`, physics, and the kit SDK.

**First launch notes**:
- Isaac Sim will prompt to accept the NVIDIA EULA on first run.
- First launch takes ~2-5 minutes (shader compilation). Subsequent launches use cached shaders (~30s).
- Shader cache lives in `~/.cache/ov/`. On NFS this is slow — see step 7 for speedup.

### 6. Install Isaac Lab 2.1.0

```bash
# flatdict (isaaclab dependency) has a broken build config — pre-install it
uv pip install flatdict==4.0.1 --no-build-isolation
# Then install Isaac Lab
uv pip install "isaaclab[all]==2.1.0" --extra-index-url https://pypi.nvidia.com
```

Isaac Lab provides higher-level APIs on top of Isaac Sim for articulations, scene setup, and URDF import. We use it instead of raw `pxr`/`omni.isaac.core` APIs.

### 7. (Optional) Speed up startup with local cache

Isaac Sim startup is slow (~2 min) when caches are on NFS (`~/.cache/ov/` is on `portal-nfs-01`). On compute nodes with local SSD (`/scratch`):

```bash
export OMNI_KIT_CACHE_PATH=/scratch/$USER/ov_cache
mkdir -p $OMNI_KIT_CACHE_PATH
```

Note: `/scratch` is node-local and won't persist across different SLURM nodes.

### 8. Verify installation

```bash
# 1. Policy inference smoke test (no simulator needed)
PYTHONPATH=. python isaacsim_conversion/test_inference.py

# 2. Isaac Sim launch test (~2 min first time)
python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
print('Isaac Sim OK')
app.close()
"
```

## Quick reference: full install from scratch

```bash
cd /share/portal/kk837/depthbasedRL
uv venv .venv_isaacsim --python 3.10
source .venv_isaacsim/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e ./rl_games/
uv pip install omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy
uv pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
uv pip install flatdict==4.0.1 --no-build-isolation
uv pip install "isaaclab[all]==2.1.0" --extra-index-url https://pypi.nvidia.com
```

## Usage

### Smoke test (inference only, no simulator)

```bash
source .venv_isaacsim/bin/activate
PYTHONPATH=. python isaacsim_conversion/test_inference.py
```

### Full rollout

```bash
source .venv_isaacsim/bin/activate
PYTHONPATH=. python isaacsim_conversion/rollout.py \
    --assembly beam --part_id 2 --collision_method coacd
```

## Key gotchas

- **Quaternion convention**: Isaac Sim uses wxyz, our observation code expects xyzw. Convert at the sim boundary.
- **Joint ordering**: Must match `JOINT_NAMES_ISAACGYM` from `observation_action_utils_sharpa.py`. Validated at startup.
- **Headless mode**: On the SLURM cluster (no display), always use `SimulationApp({"headless": True})`. For visual feedback, use Isaac Sim's WebRTC livestream.
- **contact_offset=0.002**: The training uses a very small contact offset (default is 0.02). Must set explicitly or contacts behave differently.
- **NFS vs local cache**: First Isaac Sim launch is slow on NFS. Use `/scratch` for faster subsequent launches.
- **flatdict build issue**: `uv` can't build `flatdict==4.0.1` due to missing `setuptools` in build isolation. Pre-install it with `--no-build-isolation`.

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
| `test_inference.py` | Phase 2 smoke test — full obs→policy→action pipeline without simulator |
| `isaacsim_env.py` | Isaac Lab scene setup, URDF import, state extraction, physics config |
| `rollout.py` | Main entry point — policy rollout with goal switching |
| `README.md` | This file |

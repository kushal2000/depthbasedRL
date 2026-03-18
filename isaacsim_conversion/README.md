# Isaac Sim Policy Rollout

Roll out the pretrained Isaac Gym policy in Isaac Sim (via Isaac Lab) for visualization/evaluation.

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

This is a ~15GB download. The `[isaacsim,all]` extra bundles Isaac Sim + all extensions + Isaac Lab modules (`isaaclab.sim`, `isaaclab.assets`, etc.).

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
- **SimulationApp must be created first**: Before any `isaaclab.*` or `omni.*` imports. Our code handles this.

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

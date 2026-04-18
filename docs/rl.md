# RL Training with simple_rl (PPO / SAPG / EPO)

This document covers how to train, evaluate, and monitor policies using the `simple_rl` training path.

---

## Algorithms

| Algorithm | Config | Description |
|-----------|--------|-------------|
| **PPO** | `SimToolRealSimpleRLPPO` | Vanilla PPO baseline |
| **SAPG** | `SimToolRealSimpleRLSAPG` | M=6 entropy-conditioned blocks; each block has a different exploration coefficient |
| **EPO** | `SimToolRealSimpleRLEPO` | SAPG with M=64 blocks + evolutionary update every 50 epochs (weak blocks replaced by copies of strong blocks) |

Checkpoints and configs are saved to:
- `runs/SimToolReal_SimpleRL_PPO/`
- `runs/SimToolReal_SimpleRL_SAPG/`
- `runs/SimToolReal_SimpleRL_EPO/`

---

## Quick Start: Launcher Script (recommended)

`isaacgymenvs/launch_simple_rl.py` is a self-documenting CLI that assembles and runs the training command for you.

```bash
# Show all options
python isaacgymenvs/launch_simple_rl.py --help

# SAPG (default)
python isaacgymenvs/launch_simple_rl.py

# PPO
python isaacgymenvs/launch_simple_rl.py --algo ppo

# EPO
python isaacgymenvs/launch_simple_rl.py --algo epo

# Custom project / entity
python isaacgymenvs/launch_simple_rl.py --algo sapg \
    --wandb_project simtoolreal --wandb_entity tylerlum

# Small debug run (no wandb)
python isaacgymenvs/launch_simple_rl.py --algo ppo \
    --num_envs 64 --wandb_activate False

# Resume from checkpoint
python isaacgymenvs/launch_simple_rl.py --algo sapg \
    --checkpoint runs/SimToolReal_SimpleRL_SAPG/nn/best.pth
```

---

## Manual Commands (direct hydra interface)

If you need fine-grained overrides, call `train_simple_rl.py` directly.

### PPO

```bash
# Minimal
python isaacgymenvs/train_simple_rl.py \
    task=SimToolReal train=SimToolRealSimpleRLPPO

# Full scale
python isaacgymenvs/train_simple_rl.py \
    task=SimToolReal train=SimToolRealSimpleRLPPO \
    num_envs=8192 \
    wandb_activate=True wandb_project=simtoolreal wandb_entity=tylerlum \
    headless=True
```

### SAPG (M=6 blocks)

```bash
python isaacgymenvs/train_simple_rl.py \
    task=SimToolReal train=SimToolRealSimpleRLSAPG \
    num_envs=8192 \
    wandb_activate=True wandb_project=simtoolreal wandb_entity=tylerlum \
    headless=True
```

`num_envs` must be divisible by 6.

### EPO (M=64 blocks + evolution)

```bash
python isaacgymenvs/train_simple_rl.py \
    task=SimToolReal train=SimToolRealSimpleRLEPO \
    num_envs=8192 \
    wandb_activate=True wandb_project=simtoolreal wandb_entity=tylerlum \
    headless=True
```

`num_envs` must be divisible by 64.

> **Small-scale EPO note:** When `num_envs < 8192`, the SAPG off-policy batch augmentation can make batch sizes non-divisible by the LSTM sequence length. Add `"train.ppo.sapg.use_others_experience=False"` to work around this.

### Resuming from a checkpoint

Append `checkpoint=<path>` to any of the above commands:

```bash
python isaacgymenvs/train_simple_rl.py \
    task=SimToolReal train=SimToolRealSimpleRLSAPG \
    checkpoint=runs/SimToolReal_SimpleRL_SAPG/nn/best.pth
```

---

## Evaluation

Run the DexToolBench evaluator against a trained checkpoint:

```bash
python dextoolbench/eval.py \
    --object-category hammer \
    --object-name claw_hammer \
    --task-name swing_down \
    --config-path runs/SimToolReal_SimpleRL_SAPG/config.yaml \
    --checkpoint-path runs/SimToolReal_SimpleRL_SAPG/nn/best.pth \
    --num-episodes 5 \
    --output-dir /tmp/eval_results
```

Add `--wandb-project simtoolreal` to log videos and the interactive viewer to WandB during eval.

---

## WandB Interactive Viewer

Every `capture_viewer_freq` environment steps, the training loop captures `capture_viewer_len` frames of robot + object + goal **poses** (no pixel rendering) and generates a self-contained 3D HTML file. It is logged to WandB as the `interactive_viewer` panel, alongside the `video` panel, so both can be compared side-by-side.

### What you see in the viewer

- Animated robot arm (URDF loaded from GitHub)
- Object (tool mesh or procedurally-generated primitive shape)
- Goal object (same shape, shown in green)
- Table
- Playback scrubber and play/pause controls

### Configuration

In `isaacgymenvs/cfg/task/SimToolReal.yaml`:

```yaml
capture_viewer: True          # enable/disable
capture_viewer_freq: 6000     # steps between captures (matches video cadence)
capture_viewer_len: 600       # frames per clip = 1 full episode at 60 Hz
```

Override on the command line:
```bash
task.env.capture_viewer=True
task.env.capture_viewer_freq=6000
task.env.capture_viewer_len=600
```

### Performance

- **No rendering cost.** Only joint positions and rigid-body poses are extracted — no camera sensors, no GPU frame capture. Runs at full training FPS.
- `capture_video` (MP4) does require `enableCameraSensors=True` and has a small overhead. Both can run together.

### Local output

HTML files are written to `videos/` alongside the MP4:

```
videos/
  2026-04-16_11-02-48_viewer_6000.html   ← interactive viewer
  2026-04-16_11-02-48_video_6000.mp4     ← side-by-side video
```

Open the `.html` in any modern browser — no server needed.

### Debugging: fast capture cycle

To get a viewer clip within the first minute of training:

```bash
python isaacgymenvs/launch_simple_rl.py --algo ppo --num_envs 64 \
    --wandb_activate False \
    # then add these as extra hydra overrides via train_simple_rl.py directly:

python isaacgymenvs/train_simple_rl.py \
    task=SimToolReal train=SimToolRealSimpleRLPPO \
    num_envs=64 wandb_activate=False headless=True \
    task.env.capture_viewer=True \
    task.env.capture_viewer_freq=50 \
    task.env.capture_viewer_len=30
```

---

## Key Files

| File | Purpose |
|------|---------|
| `isaacgymenvs/launch_simple_rl.py` | Launcher CLI (PPO/SAPG/EPO) |
| `isaacgymenvs/train_simple_rl.py` | Training entry point (hydra) |
| `isaacgymenvs/cfg/train/SimToolRealSimpleRLPPO.yaml` | PPO hyperparameters |
| `isaacgymenvs/cfg/train/SimToolRealSimpleRLSAPG.yaml` | SAPG hyperparameters (M=6) |
| `isaacgymenvs/cfg/train/SimToolRealSimpleRLEPO.yaml` | EPO hyperparameters (M=64) |
| `isaacgymenvs/cfg/task/SimToolReal.yaml` | Env config (viewer settings live here) |
| `isaacgymenvs/tasks/simtoolreal/env.py` | Env implementation + viewer capture logic |
| `deployment/rl_player_simple_rl.py` | Inference player for simple_rl checkpoints |
| `deployment/rl_policy_node.py` | ROS policy node (`--use_simple_rl` flag) |

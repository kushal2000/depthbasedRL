"""Launch simple_rl training (PPO / SAPG / EPO) with a clean CLI.

Follows the same pattern as isaacgymenvs/launch_training.py:
  - tyro dataclass for self-documenting CLI
  - assembles hydra override string
  - prints the full command then runs it via subprocess

Usage examples
--------------
# SAPG (default)
python isaacgymenvs/launch_simple_rl.py

# PPO, small run for debugging
python isaacgymenvs/launch_simple_rl.py --algo ppo --num_envs 64 --wandb_activate False

# EPO, full scale
python isaacgymenvs/launch_simple_rl.py --algo epo --num_envs 8192 \\
    --wandb_project simtoolreal --wandb_entity tylerlum

# Resume SAPG from checkpoint
python isaacgymenvs/launch_simple_rl.py --algo sapg \\
    --checkpoint runs/SimToolReal_SimpleRL_SAPG/nn/best.pth
"""

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import tyro

# Maps algo name → Hydra train config file (in isaacgymenvs/cfg/train/)
ALGO_TO_TRAIN_CFG = {
    "ppo":  "SimToolRealSimpleRLPPO",
    "sapg": "SimToolRealSimpleRLSAPG",
    "epo":  "SimToolRealSimpleRLEPO",
}

# Number of SAPG conditioning blocks per algorithm (used for divisibility check)
ALGO_NUM_CONDITIONINGS = {
    "ppo":  1,
    "sapg": 6,
    "epo":  64,
}


@dataclass
class LaunchSimpleRLArgs:
    """Launch simple_rl training with configurable parameters."""

    # === Algorithm ===
    algo: Literal["ppo", "sapg", "epo"] = "sapg"
    """Which algorithm to train with.
      ppo  – vanilla PPO
      sapg – SAPG with M=6 entropy-conditioned blocks
      epo  – EPO: SAPG with M=64 blocks + evolutionary policy update every 50 epochs
    """

    # === Experiment ===
    experiment: str = ""
    """Optional custom experiment name suffix appended to the auto-generated name.
    If empty, the experiment dir is derived automatically by train_simple_rl.py."""

    seed: int = 42
    """Random seed."""

    checkpoint: Optional[Path] = None
    """Path to a .pth checkpoint to resume training from. If None, trains from scratch."""

    # === Environment ===
    num_envs: int = 8192
    """Number of parallel Isaac Gym environments.
    Reduce to 4096 if you run out of GPU memory.
    For EPO (M=64), must be divisible by 64. For SAPG (M=6), must be divisible by 6."""

    headless: bool = True
    """Run headless (no viewer window). Should be True for training."""

    # === WandB ===
    wandb_activate: bool = True
    """Enable WandB logging."""

    wandb_project: str = "simtoolreal"
    """WandB project name."""

    wandb_entity: str = "tylerlum"
    """WandB entity (user or team)."""

    wandb_group: str = ""
    """WandB run group. Defaults to today's date (YYYY-MM-DD) if empty."""

    wandb_tags: List[str] = field(default_factory=list)
    """WandB tags for this run."""

    wandb_notes: str = ""
    """Free-text notes attached to the WandB run."""

    # === Viewer / Video capture ===
    capture_viewer: bool = True
    """Log interactive 3D HTML viewer to WandB every capture_viewer_freq steps.
    Pure pose extraction — no rendering cost at training scale."""

    capture_video: bool = True
    """Log MP4 video to WandB every capture_video_freq steps.
    Requires enableCameraSensors=True (slight overhead); runs alongside the viewer."""

    def __post_init__(self) -> None:
        M = ALGO_NUM_CONDITIONINGS[self.algo]
        if M > 1:
            assert self.num_envs % M == 0, (
                f"--num_envs ({self.num_envs}) must be divisible by the number of "
                f"conditioning blocks for {self.algo.upper()} (M={M}). "
                f"Try --num_envs {(self.num_envs // M) * M}."
            )
        if self.checkpoint is not None:
            assert self.checkpoint.exists(), (
                f"--checkpoint not found: {self.checkpoint}"
            )


def launch(args: LaunchSimpleRLArgs) -> None:
    wandb_group = args.wandb_group or datetime.now().strftime("%Y-%m-%d")
    wandb_tags_str = "[" + ",".join(args.wandb_tags) + "]"

    cmd_parts = [
        "python", "isaacgymenvs/train_simple_rl.py",
        "task=SimToolReal",
        f"train={ALGO_TO_TRAIN_CFG[args.algo]}",
        f"num_envs={args.num_envs}",
        f"seed={args.seed}",
        f"headless={args.headless}",
        # WandB
        f"wandb_activate={args.wandb_activate}",
        f"wandb_project={args.wandb_project}",
        f"wandb_entity={args.wandb_entity}",
        f"wandb_group={wandb_group}",
        f"wandb_tags={wandb_tags_str}",
        f"++wandb_notes='{args.wandb_notes}'",
        # Capture
        f"task.env.capture_viewer={args.capture_viewer}",
        f"task.env.capture_video={args.capture_video}",
    ]

    if args.experiment:
        cmd_parts.append(f"experiment={args.experiment}")

    if args.checkpoint is not None:
        cmd_parts.append(f"checkpoint={args.checkpoint}")

    cmd = " \\\n    ".join(cmd_parts)
    print(f"Running command:\n{cmd}\n")
    subprocess.run(" ".join(cmd_parts), shell=True, check=True)


def main() -> None:
    args: LaunchSimpleRLArgs = tyro.cli(LaunchSimpleRLArgs)
    launch(args)


if __name__ == "__main__":
    main()

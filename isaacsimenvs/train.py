"""Train an isaacsimenvs task with our vendored rl_games.

Pipeline:
    Hydra config (cfg/config.yaml + cfg/task/<Name>.yaml + cfg/train/<Name>PPO.yaml)
        ↓
    AppLauncher (boots Kit)
        ↓
    DirectRLEnv from isaacsim_task_map
        ↓
    isaaclab_rl.RlGamesVecEnvWrapper (clipping, device bridging, obs-group routing)
        ↓
    rl_games.torch_runner.Runner (PPO / SAPG — both live in ./rl_games/)
"""

from __future__ import annotations

import argparse
import os
import sys

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def main(cfg: DictConfig) -> None:
    # 1. AppLauncher FIRST — must run before any isaaclab.* import (see CLAUDE.md).
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    launcher_args, _ = parser.parse_known_args([])
    launcher_args.headless = bool(cfg.headless)
    launcher_args.enable_cameras = bool(cfg.capture_video)
    app_launcher = AppLauncher(launcher_args)
    app = app_launcher.app

    # 2. Safe to import isaaclab-backed modules now.
    from hydra.core.hydra_config import HydraConfig
    from rl_games.torch_runner import Runner

    from isaacsimenvs.tasks import isaacsim_task_map
    from isaacsimenvs.utils.reformat import omegaconf_to_dict
    from isaacsimenvs.utils.rlgames_utils import (
        MultiObserver,
        apply_task_overrides,
        register_rlgames_env,
    )

    # Resolve the Hydra-managed output dir once — used for rl_games' train_dir
    # and for the video observer, so all artifacts co-locate under it.
    hydra_run_dir = HydraConfig.get().runtime.output_dir

    # 3. Build env.
    env_cls, cfg_cls = isaacsim_task_map[cfg.task_name]
    env_cfg = cfg_cls()
    # Top-level `num_envs` CLI override is mirrored into the task YAML so the
    # train YAML's `num_actors: ${....task.scene.num_envs}` interpolation sees it.
    if cfg.num_envs not in ("", None):
        cfg.task.scene.num_envs = int(cfg.num_envs)
    apply_task_overrides(env_cfg, cfg.task, num_envs=cfg.num_envs)
    env_cfg.sim.device = cfg.sim_device
    env = env_cls(cfg=env_cfg)

    # 4. Optional recording camera for wandb video logging. Attach BEFORE
    # RlGamesVecEnvWrapper so we're still talking to the raw DirectRLEnv's scene.
    if cfg.capture_video:
        from isaacsimenvs.utils.video_capture import attach_record_camera

        attach_record_camera(env)

    # 5. Wrap + register under rl_games name "rlgpu". Clip bounds live at the
    # top level of the task YAML — not on the configclass, but consumed here.
    register_rlgames_env(
        env,
        rl_device=str(cfg.rl_device),
        clip_obs=float(cfg.task.clip_observations),
        clip_actions=float(cfg.task.clip_actions),
    )

    # 6. Observers.
    observers = []
    if cfg.wandb_activate:
        from isaacsimenvs.utils.wandb_utils import WandbAlgoObserver

        observers.append(WandbAlgoObserver(cfg))
    if cfg.capture_video:
        from pathlib import Path
        from isaacsimenvs.utils.video_capture import WandbVideoObserver

        exp_name = cfg.train.params.config.name
        video_dir = Path(hydra_run_dir) / exp_name / "videos"
        observers.append(
            WandbVideoObserver(
                env,
                video_interval=int(cfg.get("video_interval", 10)),
                capture_frames=int(cfg.get("video_capture_frames", 120)),
                video_fps=int(cfg.get("video_fps", 30)),
                video_dir=video_dir,
            )
        )

    # 7. rl_games Runner.
    runner = Runner(MultiObserver(observers)) if observers else Runner()

    rlg_cfg = omegaconf_to_dict(cfg.train)
    # Point rl_games at the Hydra run dir so checkpoints (`<name>/nn/`) and
    # tensorboard summaries (`<name>/summaries/`) co-locate with videos + slurm
    # logs + the resolved Hydra config.
    rlg_cfg["params"]["config"]["train_dir"] = hydra_run_dir
    rlg_cfg["params"]["config"]["device"] = str(cfg.rl_device)
    rlg_cfg["params"]["config"]["device_name"] = str(cfg.rl_device)
    if cfg.max_iterations not in ("", None):
        rlg_cfg["params"]["config"]["max_epochs"] = int(cfg.max_iterations)
    if cfg.seed not in ("", None):
        rlg_cfg["params"]["seed"] = int(cfg.seed)

    runner.load(rlg_cfg)
    runner.reset()
    runner.run(
        {
            "train": not bool(cfg.test),
            "play": bool(cfg.test),
            "checkpoint": str(cfg.checkpoint) if cfg.checkpoint else None,
        }
    )

    # 8. Kit shutdown hangs (per CLAUDE.md + isaacsim_conversion/distill.py).
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

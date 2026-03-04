#!/usr/bin/env python3
"""IsaacGym training entry point for SimToolReal.

Clean top-level script using Hydra config -> IsaacVecEnv -> SAPGAgent.

Usage:
    # Single GPU:
    python train_isaacgym.py headless=True

    # Multi-GPU (2 GPUs):
    torchrun --nproc_per_node=2 train_isaacgym.py multi_gpu=True headless=True

    # With depth camera:
    python train_isaacgym.py headless=True task.env.depthCamera.enabled=True

    # Override experiment name:
    python train_isaacgym.py headless=True wandb_activate=True wandb_name=my_run
"""

import os
import sys

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf, open_dict

from envs.isaacgym.reformat import omegaconf_to_dict

# Register OmegaConf resolvers used by Hydra configs (must happen before @hydra.main)
import envs.isaacgym  # noqa: F401


def preprocess_train_config(cfg, config_dict):
    """Add common configuration parameters to the train config."""
    train_cfg = config_dict["params"]["config"]
    train_cfg["device"] = cfg.rl_device
    train_cfg["population_based_training"] = False
    train_cfg["pbt_idx"] = None
    train_cfg["full_experiment_name"] = cfg.get("full_experiment_name")

    print(f"Using rl_device: {cfg.rl_device}")
    print(f"Using sim_device: {cfg.sim_device}")

    try:
        model_size_multiplier = config_dict["params"]["network"]["mlp"][
            "model_size_multiplier"
        ]
        if model_size_multiplier != 1:
            units = config_dict["params"]["network"]["mlp"]["units"]
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(
                f"Modified MLP units by x{model_size_multiplier} to {config_dict['params']['network']['mlp']['units']}"
            )
    except KeyError:
        pass

    return config_dict


@hydra.main(version_base="1.1", config_name="config", config_path="./envs/isaacgym/cfg")
def main(cfg: DictConfig):
    # isort: off
    from isaacgym import gymapi, gymtorch, gymutil
    # isort: on

    import torch

    # ── Global CUDA performance settings ──
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from hydra.utils import to_absolute_path

    from envs.isaacgym.reformat import omegaconf_to_dict, print_dict
    from envs.isaacgym.utils import set_np_formatting, set_seed
    from rl.agent import SAPGAgent
    from rl.vec_env import IsaacVecEnv
    from tensorboardX import SummaryWriter as TBWriter

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    print_dict(omegaconf_to_dict(cfg))
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    cfg.seed = set_seed(
        cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank
    )

    # dump config dict
    if not cfg.test:
        experiment_dir = os.path.join("runs", cfg.train.params.config.name)
        os.makedirs(experiment_dir, exist_ok=True)
        if not os.path.exists(os.path.join(experiment_dir, "config.yaml")):
            with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg))
            with open(os.path.join(experiment_dir, "cmd.txt"), "w") as f:
                f.write(" ".join(sys.argv))

    # ── Multi-GPU setup ──
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    multi_gpu = cfg.get('multi_gpu', False) and world_size > 1

    if multi_gpu:
        num_envs_total = cfg.task.env.numEnvs
        assert num_envs_total % world_size == 0, \
            f"num_envs ({num_envs_total}) must be divisible by world_size ({world_size})"
        with open_dict(cfg):
            cfg.task.env.numEnvs = num_envs_total // world_size
            cfg.sim_device = f'cuda:{local_rank}'
            cfg.rl_device = f'cuda:{local_rank}'
            depth_enabled = cfg.task.env.get('depthCamera', {}).get('enabled', False)
            cfg.graphics_device_id = 0 if depth_enabled else local_rank
            cfg.train.params.config.minibatch_size = (
                cfg.train.params.config.minibatch_size // world_size
            )
            if hasattr(cfg.train.params.config, 'central_value_config'):
                cfg.train.params.config.central_value_config.minibatch_size = (
                    cfg.train.params.config.central_value_config.minibatch_size // world_size
                )
        print(f"[Rank {global_rank}] Running {cfg.task.env.numEnvs} envs "
              f"(total {num_envs_total}) on cuda:{local_rank}")

    train_cfg = omegaconf_to_dict(cfg.train)
    train_cfg = preprocess_train_config(cfg, train_cfg)
    config = train_cfg['params']['config']
    config['multi_gpu'] = multi_gpu

    # ── Create env ──
    from envs.isaacgym import isaacgym_task_map
    task_config = omegaconf_to_dict(cfg.task)
    raw_env = isaacgym_task_map[cfg.task_name](
        cfg=task_config,
        rl_device=cfg.rl_device,
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=cfg.capture_video,
        force_render=cfg.force_render,
    )
    vec_env = IsaacVecEnv(raw_env)

    # ── WandB ──
    if cfg.wandb_activate and global_rank == 0:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.get('wandb_group', None),
            name=cfg.wandb_name,
            config=omegaconf_to_dict(cfg),
            sync_tensorboard=True,
        )
        wandb.define_metric('info/epochs', hidden=False)
        wandb.define_metric('*', step_metric='info/epochs')
        wandb.define_metric('performance/elapsed_minutes')
        wandb.define_metric('performance/frames_vs_minutes',
                            step_metric='performance/elapsed_minutes')

    # ── TensorBoard writer (rank 0 only) ──
    writer = None
    if global_rank == 0:
        summaries_dir = os.path.join(config.get('train_dir', 'runs'), config['name'], 'summaries')
        os.makedirs(summaries_dir, exist_ok=True)
        writer = TBWriter(summaries_dir)

    agent = SAPGAgent(vec_env, config, writer)

    if cfg.checkpoint:
        agent.restore(cfg.checkpoint)

    # ── Viser trajectory visualizer + diagnostics (rank 0 only) ──
    if global_rank == 0:
        from misc.viser_visualizer import setup_isaacgym_viser, make_isaacgym_diag_callback

        if cfg.get('viser_port', 0) > 0:
            viser_port = cfg.get('viser_port', 8013)
            vs_cb, ve_cb = setup_isaacgym_viser(raw_env, interval=10, port=viser_port)
            agent.on_step_callbacks.append(vs_cb)
            agent.on_epoch_end_callbacks.append(ve_cb)

        agent.on_epoch_end_callbacks.append(make_isaacgym_diag_callback(raw_env, interval=10))

    agent.train()


if __name__ == "__main__":
    main()

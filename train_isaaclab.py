#!/usr/bin/env python3
"""Isaac Lab training entry point for SimToolReal.

Replaces isaacgymenvs/train.py for the Isaac Lab backend while reusing
the exact same SAPGAgent from rl/.

Usage:
    # Single GPU:
    python train_isaaclab.py --num_envs=8192

    # With depth camera:
    python train_isaaclab.py --num_envs=8192 --enable_cameras --use_depth

    # Multi-GPU (2 GPUs):
    torchrun --nproc_per_node=2 train_isaaclab.py --num_envs=4096 --distributed --multi_gpu

    # Multi-GPU with depth:
    torchrun --nproc_per_node=2 train_isaaclab.py --num_envs=4096 --distributed --multi_gpu --enable_cameras --use_depth
"""

import argparse
import os
import sys

# ── Per-rank renderer pinning ──
# Must happen BEFORE AppLauncher / Kit initialises so the Vulkan renderer
# lands on the correct physical GPU.
#
# torchrun sets LOCAL_RANK → we inject the Kit arg that pins the renderer
# to that GPU.  AppLauncher will additionally set physics_gpu and
# active_gpu from LOCAL_RANK when distributed=True, but the Kit-level
# --/renderer/activeGpu is the authoritative setting for the RTX / Vulkan
# backend and must be present before SimulationApp is created.
_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
sys.argv.append(f"--/renderer/activeGpu={_local_rank}")

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimToolReal with Isaac Lab")

    # Environment
    parser.add_argument("--num_envs", type=int, default=8192, help="Total number of environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_depth", action="store_true", help="Enable depth camera")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
    parser.add_argument("--enable_cameras", action="store_true", help="Enable camera sensors")

    # Training
    parser.add_argument("--max_epochs", type=int, default=1_000_000, help="Max training epochs")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint to restore")
    parser.add_argument("--experiment", type=str, default="isaaclab_simtoolreal", help="Experiment name")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU training")
    parser.add_argument("--distributed", action="store_true", help="Run with multiple GPUs (required for torchrun)")

    # WandB
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="isaaclab_simtoolreal")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="")

    # Config overrides
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--minibatch_size", type=int, default=32768)
    parser.add_argument("--horizon_length", type=int, default=16)
    parser.add_argument("--mini_epochs", type=int, default=4)
    parser.add_argument("--randomize", action="store_true", help="Enable domain randomization")

    # Isaac Lab CLI args (handled by AppLauncher)
    parser.add_argument("--video", action="store_true", help="Record video")
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--video_interval", type=int, default=2000)

    args, unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    # ── Isaac Lab app launcher ──
    # AppLauncher reads LOCAL_RANK and sets active_gpu / physics_gpu.
    # The --/renderer/activeGpu Kit arg (injected above) pins the Vulkan
    # renderer to the same physical GPU.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=args.enable_cameras or args.use_depth,
        distributed=args.distributed,
    )
    simulation_app = app_launcher.app

    # ── Now safe to import Isaac Lab and our modules ──
    import gymnasium

    # Enable TF32 for performance
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Import our env (registers gymnasium env)
    import isaaclab_envs  # noqa: F401
    from isaaclab_envs.sim_tool_real_cfg import SimToolRealEnvCfg
    from rl.agent import SAPGAgent
    from rl.vec_env import IsaacLabVecEnv

    # ── Multi-GPU setup ──
    local_rank = app_launcher.local_rank
    global_rank = app_launcher.global_rank
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    multi_gpu = args.multi_gpu and world_size > 1

    # ── Startup verification ──
    device_str = f"cuda:{local_rank}"
    print(f"[Rank {global_rank}] LOCAL_RANK={local_rank}  "
          f"torch.cuda.current_device()={torch.cuda.current_device()}  "
          f"GPU={torch.cuda.get_device_name(local_rank)}  "
          f"renderer/activeGpu={_local_rank}", flush=True)

    # ── Build env config ──
    env_cfg = SimToolRealEnvCfg()
    env_cfg.sim.device = device_str

    if multi_gpu:
        assert args.num_envs % world_size == 0, (
            f"num_envs ({args.num_envs}) must be divisible by world_size ({world_size})"
        )
        env_cfg.scene.num_envs = args.num_envs // world_size
        print(f"[Rank {global_rank}] Running {env_cfg.scene.num_envs} envs "
              f"(total {args.num_envs}) on {device_str}", flush=True)
    else:
        env_cfg.scene.num_envs = args.num_envs

    env_cfg.use_depth_camera = args.use_depth
    env_cfg.randomize = args.randomize

    # ── Create Isaac Lab env ──
    env = gymnasium.make("SimToolReal-Direct-v0", cfg=env_cfg)

    # ── Wrap for our RL pipeline ──
    vec_env = IsaacLabVecEnv(env)

    # ── Build training config ──
    # SAPGAgent reads LOCAL_RANK from env to set ppo_device = cuda:{local_rank}.
    # In distributed mode it also calls dist.init_process_group("nccl").
    config = {
        'name': args.experiment,
        'device': device_str,
        'multi_gpu': multi_gpu,

        # PPO hyperparameters (matching SimToolRealPPO.yaml)
        'ppo': True,
        'mixed_precision': False,
        'normalize_input': True,
        'normalize_value': True,
        'normalize_advantage': True,
        'reward_shaper': {'scale_value': 0.01},

        'num_actors': env_cfg.scene.num_envs,
        'gamma': 0.99,
        'tau': 0.95,
        'learning_rate': args.learning_rate,
        'lr_schedule': 'adaptive',
        'schedule_type': 'standard',
        'kl_threshold': 0.016,
        'score_to_win': 1_000_000,
        'max_epochs': args.max_epochs,
        'max_frames': 100_000_000_000_000,
        'save_best_after': 100,
        'save_frequency': 3000,
        'print_stats': True,
        'grad_norm': 1.0,
        'entropy_coef': 0.0,
        'truncate_grads': True,
        'e_clip': 0.1,
        'minibatch_size': args.minibatch_size // world_size if multi_gpu else args.minibatch_size,
        'mini_epochs': args.mini_epochs,
        'critic_coef': 4.0,
        'clip_value': True,
        'horizon_length': args.horizon_length,
        'seq_length': 16,
        'bounds_loss_coef': 0.0001,

        # SAPG exploration (disabled by default)
        'expl_type': 'none',
        'expl_reward_coef_embd_size': 32,
        'expl_reward_coef_scale': 1.0,
        'expl_reward_type': 'rnd',
        'expl_coef_block_size': 4096,

        # Depth camera
        'use_depth_camera': args.use_depth,
        'depth_encoder_type': env_cfg.depth_encoder_type,
        'depth_feature_dim': env_cfg.depth_feature_dim,
        'freeze_depth_encoder': env_cfg.depth_freeze_encoder,
        'unfreeze_depth_after_epochs': env_cfg.depth_unfreeze_after_epochs,
        'depth_encoder_lr': env_cfg.depth_encoder_lr,
        'depth_image_height': env_cfg.depth_height,
        'depth_image_width': env_cfg.depth_width,

        # Misc
        'population_based_training': False,
        'pbt_idx': None,
        'full_experiment_name': args.experiment,
        'use_others_experience': 'none',
        'off_policy_ratio': 1.0,
        'good_reset_boundary': 0,
        'value_bootstrap': True,
    }

    # ── WandB ──
    if args.wandb and global_rank == 0:
        import wandb
        wandb_name = args.wandb_name or args.experiment
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=wandb_name,
            config={
                'num_envs': args.num_envs,
                'use_depth': args.use_depth,
                'multi_gpu': multi_gpu,
                'world_size': world_size,
                **config,
            },
            sync_tensorboard=True,
        )
        wandb.define_metric('performance/elapsed_minutes')
        wandb.define_metric('performance/frames_vs_minutes',
                            step_metric='performance/elapsed_minutes')

    # ── TensorBoard writer ──
    writer = None
    if global_rank == 0:
        from tensorboardX import SummaryWriter as TBWriter
        summaries_dir = os.path.join('runs', config['name'], 'summaries')
        os.makedirs(summaries_dir, exist_ok=True)
        writer = TBWriter(summaries_dir)

    # ── Create agent and train ──
    agent = SAPGAgent(vec_env, config, writer)

    if args.checkpoint:
        agent.restore(args.checkpoint)

    agent.train()

    # ── Cleanup ──
    simulation_app.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Isaac Lab training entry point for SimToolReal.

Replaces isaacgymenvs/train.py for the Isaac Lab backend while reusing
the exact same SAPGAgent from rl/.

Usage:
    # Single GPU:
    python train_isaaclab.py num_envs=8192

    # With depth camera:
    python train_isaaclab.py num_envs=8192 enable_cameras=true use_depth=true

    # Multi-GPU (2 GPUs):
    torchrun --nproc_per_node=2 train_isaaclab.py num_envs=4096 multi_gpu=true

    # Multi-GPU with depth:
    torchrun --nproc_per_node=2 train_isaaclab.py num_envs=4096 multi_gpu=true enable_cameras=true use_depth=true
"""

import os
import sys

# Only force headless EGL when headless is explicitly requested.
# Quick pre-parse so we can decide before Kit/CUDA init.
# Support both Hydra (headless=true) and legacy (--headless) syntax.
_pre_headless = any(a in ("--headless", "headless=true", "headless=True") for a in sys.argv)
if _pre_headless:
    os.environ.pop("DISPLAY", None)

# ── Per-rank GPU isolation ──
# Must happen BEFORE any CUDA / Kit imports.
#
# usdrt (the Fabric/scenegraph bridge that returns rendered depth as CUDA
# tensors) hardcodes "cuda:0 only".  To work around this on multi-GPU we
# use CUDA_VISIBLE_DEVICES so each rank sees exactly one physical GPU as
# cuda:0.  The Vulkan renderer still enumerates all physical GPUs, so we
# set --/renderer/activeGpu to the *physical* index to keep Vulkan and
# CUDA pointing at the same device.
_real_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
_real_global_rank = int(os.environ.get("RANK", "0"))
_world_size = int(os.environ.get("WORLD_SIZE", "1"))
_distributed = _world_size > 1

# Allow gpu=N for single-GPU on non-default device (pre-parse before Kit init)
# Support both Hydra (gpu=N) and legacy (--gpu=N) syntax.
_gpu_override = None
for _a in sys.argv:
    if _a.startswith("gpu=") or _a.startswith("--gpu="):
        _gpu_override = int(_a.split("=")[1])

if _distributed:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_real_local_rank)
    os.environ["LOCAL_RANK"] = "0"
elif _gpu_override is not None:
    # Same CUDA_VISIBLE_DEVICES isolation as multi-GPU:
    # Make the target GPU appear as cuda:0 so Isaac Lab internals
    # (which hardcode cuda:0) work correctly.
    _real_local_rank = _gpu_override
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_override)
    os.environ["LOCAL_RANK"] = "0"

# Kit renderer args — injected into sys.argv inside main() AFTER Hydra
# parses its overrides, but BEFORE AppLauncher reads them.
# Vulkan activeGpu = physical GPU index (unchanged by CUDA_VISIBLE_DEVICES)
_kit_args = [
    f"--/renderer/activeGpu={_real_local_rank}",
    "--/renderer/multiGpu/enabled=false",
    "--/crashreporter/enabled=false",
]

import hydra
import torch
from omegaconf import DictConfig

# After CUDA_VISIBLE_DEVICES isolation, the target GPU is always cuda:0
torch.cuda.set_device(0)


@hydra.main(version_base="1.1", config_name="config", config_path="cfg")
def main(cfg: DictConfig):
    # Inject Kit renderer args now that Hydra is done parsing sys.argv
    sys.argv.extend(_kit_args)

    # ── Isaac Lab app launcher ──
    # With GPU isolation, each rank sees only cuda:0, so we do NOT pass
    # distributed=True (that would make AppLauncher read the overridden
    # LOCAL_RANK=0, which is correct, but we handle NCCL init ourselves
    # inside SAPGAgent).
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=cfg.headless,
        enable_cameras=cfg.enable_cameras or cfg.use_depth or cfg.video,
    )
    simulation_app = app_launcher.app

    # ── Diagnostic: confirm Kit renderer GPU ──
    import carb
    _settings = carb.settings.get_settings()
    print(f"[Rank {_real_global_rank}] Kit renderer GPU: {_settings.get('/renderer/activeGpu')}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}, "
          f"torch.cuda.current_device()={torch.cuda.current_device()}", flush=True)

    # ── Now safe to import Isaac Lab and our modules ──
    import gymnasium

    # Enable TF32 for performance
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Import our env (registers gymnasium env)
    import envs.isaaclab  # noqa: F401
    from envs.isaaclab.sim_tool_real_cfg import SimToolRealEnvCfg
    from rl.agent import SAPGAgent
    from rl.vec_env import IsaacLabVecEnv

    # ── Multi-GPU setup ──
    # With CUDA_VISIBLE_DEVICES isolation, every rank uses cuda:0.
    # We use _real_* values (saved before the override) for NCCL / logging.
    global_rank = _real_global_rank
    world_size = _world_size
    multi_gpu = cfg.multi_gpu and world_size > 1

    # ── Startup verification ──
    # After CUDA_VISIBLE_DEVICES isolation, target GPU is always cuda:0
    device_str = "cuda:0"
    print(f"[Rank {global_rank}] physGPU={_real_local_rank}  "
          f"torch.cuda.current_device()={torch.cuda.current_device()}  "
          f"GPU={torch.cuda.get_device_name(0)}", flush=True)

    # ── Build env config ──
    env_cfg = SimToolRealEnvCfg()
    env_cfg.sim.device = device_str

    if multi_gpu:
        assert cfg.num_envs % world_size == 0, (
            f"num_envs ({cfg.num_envs}) must be divisible by world_size ({world_size})"
        )
        env_cfg.scene.num_envs = cfg.num_envs // world_size
        print(f"[Rank {global_rank}] Running {env_cfg.scene.num_envs} envs "
              f"(total {cfg.num_envs}) on {device_str}", flush=True)
    else:
        env_cfg.scene.num_envs = cfg.num_envs

    env_cfg.use_depth_camera = cfg.use_depth
    env_cfg.randomize = cfg.randomize
    if cfg.video:
        env_cfg.capture_video = True
        env_cfg.capture_video_freq = 6000   # match IsaacGym SimToolReal.yaml
        env_cfg.capture_video_len = 600

    # Override object depenetration velocity if specified
    if cfg.object_max_depenetration_velocity is not None:
        env_cfg.object_cfg.spawn.rigid_props.max_depenetration_velocity = cfg.object_max_depenetration_velocity

    # Override dof speed scale if specified
    if cfg.dof_speed_scale is not None:
        env_cfg.dof_speed_scale = cfg.dof_speed_scale
        print(f"[Config] dof_speed_scale overridden to {cfg.dof_speed_scale}")
        print(f"[Rank {global_rank}] Object max_depenetration_velocity = {cfg.object_max_depenetration_velocity}", flush=True)

    # ── Create Isaac Lab env ──
    env = gymnasium.make("SimToolReal-Direct-v0", cfg=env_cfg)

    # ── Wrap for our RL pipeline ──
    vec_env = IsaacLabVecEnv(env)

    # ── Build training config ──
    # SAPGAgent reads LOCAL_RANK from env to set ppo_device = cuda:{local_rank}.
    # In distributed mode it also calls dist.init_process_group("nccl").
    config = {
        'name': cfg.experiment,
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
        'gamma': cfg.gamma,
        'tau': cfg.tau,
        'learning_rate': cfg.learning_rate,
        'lr_schedule': 'adaptive',
        'schedule_type': 'standard',
        'kl_threshold': 0.016,
        'score_to_win': 1_000_000,
        'max_epochs': cfg.max_epochs,
        'max_frames': 100_000_000_000_000,
        'save_best_after': 100,
        'save_frequency': 3000,
        'print_stats': True,
        'grad_norm': cfg.grad_norm,
        'entropy_coef': cfg.entropy_coef,
        'truncate_grads': True,
        'e_clip': cfg.e_clip,
        'minibatch_size': cfg.minibatch_size // world_size if multi_gpu else cfg.minibatch_size,
        'mini_epochs': cfg.mini_epochs,
        'critic_coef': cfg.critic_coef,
        'clip_value': True,
        'horizon_length': cfg.horizon_length,
        'seq_length': 16,
        'bounds_loss_coef': cfg.bounds_loss_coef,

        # SAPG exploration
        'expl_type': cfg.expl_type,
        'expl_reward_coef_embd_size': 32,
        'expl_reward_coef_scale': cfg.expl_reward_coef_scale,
        'expl_reward_type': cfg.expl_reward_type,
        'expl_coef_block_size': min(cfg.expl_coef_block_size, env_cfg.scene.num_envs),

        # Depth camera
        'use_depth_camera': cfg.use_depth,
        'depth_encoder_type': env_cfg.depth_encoder_type,
        'depth_feature_dim': env_cfg.depth_feature_dim,
        'freeze_depth_encoder': env_cfg.depth_freeze_encoder,
        'unfreeze_depth_after_epochs': env_cfg.depth_unfreeze_after_epochs,
        'depth_encoder_lr': env_cfg.depth_encoder_lr,
        'depth_image_height': env_cfg.depth_height,
        'depth_image_width': env_cfg.depth_width,

        # Asymmetric critic (central value network)
        'central_value_config': {
            'minibatch_size': min(cfg.minibatch_size, env_cfg.scene.num_envs * cfg.horizon_length),
            'mini_epochs': 2,
            'learning_rate': 1e-4,
            'kl_threshold': 0.016,
            'clip_value': True,
            'normalize_input': True,
            'truncate_grads': True,
        },

        # Misc
        'population_based_training': False,
        'pbt_idx': None,
        'full_experiment_name': cfg.experiment,
        'use_others_experience': cfg.use_others_experience,
        'off_policy_ratio': 1.0,
        'good_reset_boundary': 0,
        'value_bootstrap': True,
    }

    # ── WandB ──
    if cfg.wandb_activate and global_rank == 0:
        import wandb
        wandb_name = cfg.wandb_name or cfg.experiment
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            name=wandb_name,
            config={
                'num_envs': cfg.num_envs,
                'use_depth': cfg.use_depth,
                'multi_gpu': multi_gpu,
                'world_size': world_size,
                **config,
            },
            sync_tensorboard=True,
        )
        # Use epochs as x-axis for all metrics
        wandb.define_metric('info/epochs', hidden=False)
        wandb.define_metric('*', step_metric='info/epochs')
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

    if cfg.checkpoint:
        agent.restore(cfg.checkpoint)

    # ── Viser trajectory visualizer + diagnostics (rank 0 only) ──
    if global_rank == 0:
        from misc.viser_visualizer import setup_isaaclab_viser, make_isaaclab_diag_callback

        vs_cb, ve_cb = setup_isaaclab_viser(env, interval=10, port=cfg.viser_port)
        agent.on_step_callbacks.append(vs_cb)
        agent.on_epoch_end_callbacks.append(ve_cb)

        agent.on_epoch_end_callbacks.append(make_isaaclab_diag_callback(env, interval=10))

    agent.train()

    # ── Cleanup ──
    simulation_app.close()


if __name__ == "__main__":
    main()

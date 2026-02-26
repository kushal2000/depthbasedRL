#!/usr/bin/env python3
"""Multi-GPU test for Isaac Lab training (with per-rank renderer pinning).

Usage:
    # No depth:
    PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 test_multigpu_isaaclab.py --distributed --multi_gpu --num_envs 64

    # With depth:
    PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 test_multigpu_isaaclab.py --distributed --multi_gpu --use_depth --enable_cameras --num_envs 32
"""
import argparse
import os
import sys
import time

# ── Per-rank renderer pinning ──
# Must happen BEFORE AppLauncher / Kit initialises so the Vulkan renderer
# lands on the correct physical GPU.
_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
sys.argv.append(f"--/renderer/activeGpu={_local_rank}")

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true")
    args, _ = parser.parse_known_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=args.enable_cameras or args.use_depth,
        distributed=args.distributed,
    )
    simulation_app = app_launcher.app

    import gymnasium
    torch.set_float32_matmul_precision('high')

    import isaaclab_envs
    from isaaclab_envs.sim_tool_real_cfg import SimToolRealEnvCfg
    from rl.agent import SAPGAgent
    from rl.vec_env import IsaacLabVecEnv

    global_rank = app_launcher.global_rank
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    multi_gpu = args.multi_gpu and world_size > 1

    local_rank = app_launcher.local_rank
    device_str = f"cuda:{local_rank}"

    def log(msg):
        print(f"[Rank {global_rank}/{world_size} GPU{local_rank}] {msg}", flush=True)

    log(f"Starting — LOCAL_RANK={local_rank}  "
        f"torch.cuda.current_device()={torch.cuda.current_device()}  "
        f"GPU={torch.cuda.get_device_name(local_rank)}  "
        f"renderer/activeGpu={_local_rank}")

    env_cfg = SimToolRealEnvCfg()
    env_cfg.sim.device = device_str
    if multi_gpu:
        assert args.num_envs % world_size == 0
        env_cfg.scene.num_envs = args.num_envs // world_size
    else:
        env_cfg.scene.num_envs = args.num_envs
    env_cfg.use_depth_camera = args.use_depth

    log(f"Creating env with {env_cfg.scene.num_envs} envs...")
    t0 = time.time()
    env = gymnasium.make("SimToolReal-Direct-v0", cfg=env_cfg)
    vec_env = IsaacLabVecEnv(env)
    log(f"Env created in {time.time()-t0:.1f}s, device={vec_env.device}")

    log("Resetting...")
    obs = vec_env.reset()
    log(f"obs keys: {list(obs.keys())}, obs shape: {obs['obs'].shape}")
    if 'depth' in obs:
        d = obs['depth']
        log(f"depth shape: {d.shape}, range=[{d.min():.4f}, {d.max():.4f}], var={d.var():.6f}")
        assert d.var() > 0, "FAIL: depth has zero variance — renderer may not be working!"
        log("PASS: depth has non-zero variance after reset")

    log("Running 5 steps...")
    for i in range(5):
        actions = torch.randn(env_cfg.scene.num_envs, vec_env.num_actions, device=vec_env.device)
        obs, rew, done, info = vec_env.step(actions)
        if i == 0:
            log(f"step 0: reward mean={rew.mean():.4f}")
            if 'depth' in obs:
                d = obs['depth']
                log(f"step 0 depth: range=[{d.min():.4f}, {d.max():.4f}], var={d.var():.6f}")
                assert d.var() > 0, "FAIL: depth has zero variance after step!"

    config = {
        'name': 'test_multigpu',
        'device': device_str,
        'multi_gpu': multi_gpu,
        'ppo': True,
        'mixed_precision': False,
        'normalize_input': True,
        'normalize_value': True,
        'normalize_advantage': True,
        'reward_shaper': {'scale_value': 0.01},
        'num_actors': env_cfg.scene.num_envs,
        'gamma': 0.99,
        'tau': 0.95,
        'learning_rate': 1e-4,
        'lr_schedule': 'adaptive',
        'schedule_type': 'standard',
        'kl_threshold': 0.016,
        'score_to_win': 1_000_000,
        'max_epochs': 3,
        'max_frames': 100_000_000_000_000,
        'save_best_after': 100,
        'save_frequency': 3000,
        'print_stats': True,
        'grad_norm': 1.0,
        'entropy_coef': 0.0,
        'truncate_grads': True,
        'e_clip': 0.1,
        'minibatch_size': 64 // world_size if multi_gpu else 64,
        'mini_epochs': 1,
        'critic_coef': 4.0,
        'clip_value': True,
        'horizon_length': 4,
        'seq_length': 4,
        'bounds_loss_coef': 0.0001,
        'expl_type': 'none',
        'expl_reward_coef_embd_size': 32,
        'expl_reward_coef_scale': 1.0,
        'expl_reward_type': 'rnd',
        'expl_coef_block_size': 4096,
        'use_depth_camera': args.use_depth,
        'depth_encoder_type': env_cfg.depth_encoder_type,
        'depth_feature_dim': env_cfg.depth_feature_dim,
        'freeze_depth_encoder': env_cfg.depth_freeze_encoder,
        'unfreeze_depth_after_epochs': env_cfg.depth_unfreeze_after_epochs,
        'depth_encoder_lr': env_cfg.depth_encoder_lr,
        'depth_image_height': env_cfg.depth_height,
        'depth_image_width': env_cfg.depth_width,
        'population_based_training': False,
        'pbt_idx': None,
        'full_experiment_name': 'test_multigpu',
        'use_others_experience': 'none',
        'off_policy_ratio': 1.0,
        'good_reset_boundary': 0,
        'value_bootstrap': True,
    }

    log("Creating SAPGAgent...")
    agent = SAPGAgent(vec_env, config, writer=None)
    log(f"Agent created, ppo_device={agent.ppo_device}")

    log("Running init_tensors + env_reset...")
    agent.init_tensors()
    agent.obs = agent.env_reset()
    agent.curr_frames = agent.batch_size_envs

    log("Training 2 epochs...")
    for epoch in range(2):
        t0 = time.time()
        result = agent.train_epoch()
        elapsed = time.time() - t0
        a_losses, c_losses = result[4], result[5]
        a_loss = sum(a_losses) / len(a_losses) if isinstance(a_losses, list) else a_losses
        c_loss = sum(c_losses) / len(c_losses) if isinstance(c_losses, list) else c_losses
        log(f"epoch {epoch}: a_loss={a_loss:.4f} c_loss={c_loss:.4f} total={elapsed:.2f}s")

    log("ALL TESTS PASSED!")
    simulation_app.close()

if __name__ == "__main__":
    main()

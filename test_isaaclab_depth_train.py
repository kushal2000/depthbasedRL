#!/usr/bin/env python3
"""Quick test: Isaac Lab env with depth camera + SAPGAgent training.

Usage:
    conda run -n sapg_il python test_isaaclab_depth_train.py --enable_cameras --use_depth
"""
import argparse
import os
import sys
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--enable_cameras", action="store_true")
    args, _ = parser.parse_known_args()

    # Isaac Lab app launcher (must be before imports)
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=args.enable_cameras or args.use_depth,
    )
    simulation_app = app_launcher.app

    import gymnasium
    torch.set_float32_matmul_precision('high')

    import isaaclab_envs  # noqa: registers env
    from isaaclab_envs.sim_tool_real_cfg import SimToolRealEnvCfg
    from rl.agent import SAPGAgent
    from rl.vec_env import IsaacLabVecEnv

    # Build config
    env_cfg = SimToolRealEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.use_depth_camera = args.use_depth

    print(f"\n{'='*60}")
    print(f"Test: Isaac Lab + {'depth' if args.use_depth else 'no depth'}")
    print(f"num_envs={args.num_envs}")
    print(f"{'='*60}\n")

    # Create env
    print("[1/5] Creating env...")
    env = gymnasium.make("SimToolReal-Direct-v0", cfg=env_cfg)
    vec_env = IsaacLabVecEnv(env)
    print(f"  num_envs={vec_env.num_envs}, obs={vec_env.num_obs}, "
          f"states={vec_env.num_states}, actions={vec_env.num_actions}")

    # Reset
    print("[2/5] Resetting env...")
    obs = vec_env.reset()
    print(f"  obs keys: {list(obs.keys())}")
    print(f"  obs['obs'] shape: {obs['obs'].shape}")
    if 'states' in obs:
        print(f"  obs['states'] shape: {obs['states'].shape}")
    if 'depth' in obs:
        print(f"  obs['depth'] shape: {obs['depth'].shape}")
        print(f"  depth range: [{obs['depth'].min():.4f}, {obs['depth'].max():.4f}]")
    elif args.use_depth:
        print("  WARNING: use_depth=True but no 'depth' in obs!")

    # Step
    print("[3/5] Running 10 steps...")
    for i in range(10):
        actions = torch.randn(args.num_envs, vec_env.num_actions, device=vec_env.device)
        obs_dict, reward, done, info = vec_env.step(actions)
        if i == 0:
            print(f"  step 0: obs keys={list(obs_dict.keys())}, "
                  f"reward mean={reward.mean():.4f}")
            if 'depth' in obs_dict:
                d = obs_dict['depth']
                print(f"  depth shape={d.shape}, range=[{d.min():.4f}, {d.max():.4f}]")

    # Create agent
    print("[4/5] Creating SAPGAgent...")
    config = {
        'name': 'test_depth',
        'device': 'cuda:0',
        'multi_gpu': False,
        'ppo': True,
        'mixed_precision': False,
        'normalize_input': True,
        'normalize_value': True,
        'normalize_advantage': True,
        'reward_shaper': {'scale_value': 0.01},
        'num_actors': args.num_envs,
        'gamma': 0.99,
        'tau': 0.95,
        'learning_rate': 1e-4,
        'lr_schedule': 'adaptive',
        'schedule_type': 'standard',
        'kl_threshold': 0.016,
        'score_to_win': 1_000_000,
        'max_epochs': 5,
        'max_frames': 100_000_000_000_000,
        'save_best_after': 100,
        'save_frequency': 3000,
        'print_stats': True,
        'grad_norm': 1.0,
        'entropy_coef': 0.0,
        'truncate_grads': True,
        'e_clip': 0.1,
        'minibatch_size': 64,
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
        'full_experiment_name': 'test_depth',
        'use_others_experience': 'none',
        'off_policy_ratio': 1.0,
        'good_reset_boundary': 0,
        'value_bootstrap': True,
    }

    agent = SAPGAgent(vec_env, config, writer=None)
    print(f"  ActorNetwork use_depth={agent.model.use_depth}")
    if agent.model.depth_encoder is not None:
        print(f"  depth_encoder type: {type(agent.model.depth_encoder).__name__}")

    # Train 2 epochs
    print("[5/5] Training 2 epochs...")
    agent.init_tensors()
    agent.obs = agent.env_reset()
    agent.curr_frames = agent.batch_size_envs
    for epoch in range(2):
        result = agent.train_epoch()
        step_time, play_time, update_time, total_time = result[0], result[1], result[2], result[3]
        a_losses, c_losses = result[4], result[5]
        a_loss = sum(a_losses) / len(a_losses) if isinstance(a_losses, list) else a_losses
        c_loss = sum(c_losses) / len(c_losses) if isinstance(c_losses, list) else c_losses
        print(f"  epoch {epoch}: a_loss={a_loss:.4f} c_loss={c_loss:.4f} "
              f"step_time={step_time:.3f}s play_time={play_time:.3f}s "
              f"update_time={update_time:.3f}s")

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED!")
    print(f"{'='*60}\n")

    simulation_app.close()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Simple test: Isaac Lab depth + agent.play_steps() debug."""
import argparse
import os
import sys
import time
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--enable_cameras", action="store_true")
    args, _ = parser.parse_known_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=args.enable_cameras or args.use_depth,
    )
    simulation_app = app_launcher.app

    import gymnasium
    torch.set_float32_matmul_precision('high')

    import isaaclab_envs
    from isaaclab_envs.sim_tool_real_cfg import SimToolRealEnvCfg
    from rl.agent import SAPGAgent
    from rl.vec_env import IsaacLabVecEnv

    env_cfg = SimToolRealEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.use_depth_camera = args.use_depth

    print(f"Creating env (depth={args.use_depth})...")
    env = gymnasium.make("SimToolReal-Direct-v0", cfg=env_cfg)
    vec_env = IsaacLabVecEnv(env)

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

    print("Creating agent...")
    agent = SAPGAgent(vec_env, config, writer=None)
    agent.init_tensors()

    print("Running env_reset...")
    t0 = time.time()
    agent.obs = agent.env_reset()
    print(f"  env_reset took {time.time()-t0:.2f}s")
    print(f"  obs keys: {list(agent.obs.keys())}")
    if 'depth' in agent.obs:
        print(f"  depth shape: {agent.obs['depth'].shape}")

    agent.curr_frames = agent.batch_size_envs

    # Manual play_steps: step through the horizon
    print(f"\nManual play_steps (horizon={agent.horizon_length})...")
    for n in range(agent.horizon_length):
        print(f"  step {n}: getting action...", flush=True)
        t0 = time.time()
        res_dict = agent.get_action_values(agent.obs, agent.rnn_states)
        print(f"    get_action_values took {time.time()-t0:.3f}s", flush=True)

        agent.rnn_states = res_dict['rnn_states']

        print(f"  step {n}: env_step...", flush=True)
        t0 = time.time()
        agent.obs, rewards, intr_rewards, agent.dones, infos = agent.env_step(res_dict['actions'])
        print(f"    env_step took {time.time()-t0:.3f}s", flush=True)

        if 'depth' in agent.obs:
            d = agent.obs['depth']
            print(f"    depth: shape={d.shape}, range=[{d.min():.4f}, {d.max():.4f}]")

        print(f"    reward mean={rewards.mean():.4f}")

    print("\nNow running actual train_epoch...")
    t0 = time.time()
    result = agent.train_epoch()
    elapsed = time.time() - t0
    step_time, play_time, update_time = result[0], result[1], result[2]
    a_losses, c_losses = result[4], result[5]
    a_loss = sum(a_losses) / len(a_losses) if isinstance(a_losses, list) else a_losses
    c_loss = sum(c_losses) / len(c_losses) if isinstance(c_losses, list) else c_losses
    print(f"  train_epoch total: {elapsed:.2f}s")
    print(f"  a_loss={a_loss:.4f} c_loss={c_loss:.4f}")
    print(f"  step_time={step_time:.3f}s play_time={play_time:.3f}s update_time={update_time:.3f}s")

    print("\nALL TESTS PASSED!")
    simulation_app.close()

if __name__ == "__main__":
    main()

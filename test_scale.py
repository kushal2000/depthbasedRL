#!/usr/bin/env python3
"""Scale test: Isaac Lab env with depth camera at various env counts.

Usage:
    PYTHONUNBUFFERED=1 python test_scale.py --enable_cameras --use_depth --num_envs 256
"""
import argparse
import os
import sys
import time
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--steps", type=int, default=20, help="Number of env steps to time")
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
    from rl.vec_env import IsaacLabVecEnv

    env_cfg = SimToolRealEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.use_depth_camera = args.use_depth

    print(f"Creating env: num_envs={args.num_envs}, depth={args.use_depth}")
    t0 = time.time()
    env = gymnasium.make("SimToolReal-Direct-v0", cfg=env_cfg)
    vec_env = IsaacLabVecEnv(env)
    print(f"Env created in {time.time()-t0:.1f}s")

    print("Resetting...")
    t0 = time.time()
    obs = vec_env.reset()
    print(f"Reset in {time.time()-t0:.2f}s")
    if 'depth' in obs:
        print(f"Depth: shape={obs['depth'].shape}, range=[{obs['depth'].min():.4f}, {obs['depth'].max():.4f}]")

    # Warm up
    print("Warmup (5 steps)...")
    for _ in range(5):
        actions = torch.randn(args.num_envs, vec_env.num_actions, device=vec_env.device)
        obs, _, _, _ = vec_env.step(actions)

    # Benchmark
    print(f"Benchmarking {args.steps} steps...")
    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(args.steps):
        actions = torch.randn(args.num_envs, vec_env.num_actions, device=vec_env.device)
        obs, _, _, _ = vec_env.step(actions)
    torch.cuda.synchronize()
    elapsed = time.time() - t_start

    steps_per_sec = args.steps / elapsed
    fps = args.steps * args.num_envs / elapsed
    ms_per_step = elapsed / args.steps * 1000

    print(f"\nResults ({args.num_envs} envs, depth={'ON' if args.use_depth else 'OFF'}):")
    print(f"  {steps_per_sec:.1f} steps/sec ({ms_per_step:.1f} ms/step)")
    print(f"  {fps:.0f} env FPS")

    if 'depth' in obs:
        d = obs['depth']
        print(f"  Depth: shape={d.shape}, range=[{d.min():.4f}, {d.max():.4f}]")

    # GPU memory
    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Peak GPU memory: {mem:.1f} GB")

    print("\nDONE")
    simulation_app.close()

if __name__ == "__main__":
    main()

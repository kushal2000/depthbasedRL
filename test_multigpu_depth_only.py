#!/usr/bin/env python3
"""Minimal multi-GPU depth rendering test (no training / NCCL).

Verifies that each rank can render depth on its assigned GPU using
the --/renderer/activeGpu=<local_rank> Kit arg.

Usage:
    PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 --redirects 3 --log-dir /tmp/multigpu_logs \
        test_multigpu_depth_only.py --distributed --use_depth --enable_cameras --num_envs 16
"""
import argparse
import os
import sys
import time

# ── Per-rank renderer pinning (before Kit init) ──
_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
sys.argv.append(f"--/renderer/activeGpu={_local_rank}")

import torch


def log(msg):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    # Write directly to stderr to bypass any stdout redirection
    line = f"[Rank {rank}/{world} GPU{_local_rank}] {msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    sys.stderr.write(line)
    sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    args, _ = parser.parse_known_args()

    log(f"Starting Kit init (renderer/activeGpu={_local_rank})")

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=args.enable_cameras or args.use_depth,
        distributed=args.distributed,
    )
    simulation_app = app_launcher.app

    local_rank = app_launcher.local_rank
    device_str = f"cuda:{local_rank}"
    log(f"Kit initialized. local_rank={local_rank}, "
        f"torch.cuda.current_device()={torch.cuda.current_device()}, "
        f"GPU={torch.cuda.get_device_name(local_rank)}")

    import gymnasium
    torch.set_float32_matmul_precision('high')

    import isaaclab_envs  # noqa: F401
    from isaaclab_envs.sim_tool_real_cfg import SimToolRealEnvCfg
    from rl.vec_env import IsaacLabVecEnv

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    env_cfg = SimToolRealEnvCfg()
    env_cfg.sim.device = device_str
    env_cfg.scene.num_envs = args.num_envs // world_size
    env_cfg.use_depth_camera = args.use_depth

    log(f"Creating env with {env_cfg.scene.num_envs} envs on {device_str}...")
    t0 = time.time()
    env = gymnasium.make("SimToolReal-Direct-v0", cfg=env_cfg)
    vec_env = IsaacLabVecEnv(env)
    log(f"Env created in {time.time()-t0:.1f}s")

    log("Resetting...")
    obs = vec_env.reset()
    log(f"obs keys: {list(obs.keys())}, obs shape: {obs['obs'].shape}")

    if 'depth' in obs:
        d = obs['depth']
        log(f"RESET depth: shape={d.shape}, range=[{d.min():.4f}, {d.max():.4f}], var={d.var():.6f}")
        assert d.var() > 0, "FAIL: depth has zero variance after reset!"
        log("PASS: depth has non-zero variance after reset")
    else:
        log("WARNING: no depth in obs!")

    log("Running 3 steps...")
    for i in range(3):
        actions = torch.randn(env_cfg.scene.num_envs, vec_env.num_actions, device=vec_env.device)
        obs, rew, done, info = vec_env.step(actions)
        if 'depth' in obs:
            d = obs['depth']
            log(f"step {i}: depth range=[{d.min():.4f}, {d.max():.4f}], var={d.var():.6f}, reward={rew.mean():.4f}")
        else:
            log(f"step {i}: reward={rew.mean():.4f} (no depth)")

    log("ALL TESTS PASSED!")
    simulation_app.close()


if __name__ == "__main__":
    main()

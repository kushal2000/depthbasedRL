"""One-shot: build Cartpole scene, run current attach_record_camera, print
(eye, target) in world and env-local frames. These are the numbers that become
the default `record_camera_eye` / `record_camera_target` on CartpoleEnvCfg.

Run:
    /share/portal/kk837/depthbasedRL/.venv_isaacsim/bin/python \
        isaacsimenvs/tests/diag_camera_pose.py
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless = True
    args.enable_cameras = True
    app = AppLauncher(args).app

    import torch

    from isaacsimenvs.tasks.cartpole.cartpole_env import CartpoleEnv, CartpoleEnvCfg
    from isaacsimenvs.utils.video_capture import attach_record_camera

    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = 4  # keep small; irrelevant to this test
    env = CartpoleEnv(cfg=env_cfg)

    camera = attach_record_camera(env)
    # attach_record_camera triggers env.sim.reset(); step once so camera pose propagates.
    env.sim.step()
    camera.update(0.0)

    env_idx = int(torch.argmin(torch.norm(env.scene.env_origins, dim=1)).item())
    env_origin = env.scene.env_origins[env_idx]
    cart_root_w = env.scene.articulations["cartpole"].data.root_pos_w[env_idx]
    cart_root_local = cart_root_w - env_origin

    # Compute expected camera pose from cfg (env-local → world).
    eye_local = torch.tensor(env_cfg.record_camera_eye, device=env.device)
    target_local = torch.tensor(env_cfg.record_camera_target, device=env.device)
    expected_eye_w = env_origin + eye_local
    expected_target_w = env_origin + target_local

    print("=" * 60)
    print(f"env_idx picked           = {env_idx}")
    print(f"env_origin (w)           = {env_origin.cpu().tolist()}")
    print(f"cart root_pos_w          = {cart_root_w.cpu().tolist()}")
    print(f"cart root in env-local   = {cart_root_local.cpu().tolist()}")
    print(f"cfg.record_camera_eye    = {env_cfg.record_camera_eye}")
    print(f"cfg.record_camera_target = {env_cfg.record_camera_target}")
    print(f"expected eye (w)         = {expected_eye_w.cpu().tolist()}")
    print(f"expected target (w)      = {expected_target_w.cpu().tolist()}")
    print("Before refactor, behavior was: target=cart_root_w, eye=cart_root_w+(-3,0,0.5).")
    print(f"  pre-refactor eye (w)   = {(cart_root_w + torch.tensor([-3.0, 0.0, 0.5], device=env.device)).cpu().tolist()}")
    print(f"  pre-refactor target(w) = {cart_root_w.cpu().tolist()}")
    print("=" * 60)

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

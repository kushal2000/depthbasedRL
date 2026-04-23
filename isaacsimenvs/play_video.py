"""Run a trained policy and record an mp4 of the rollout.

Unlike `train.py test=true` (which calls rl_games' `player.run()` and gives us
no hook to capture frames), this script builds a manual rollout loop so we can
`camera.update(dt)` and append rgb frames each step.

    python isaacsimenvs/play_video.py \
        --checkpoint runs/0_cartpole_direct/nn/last_0_cartpole_direct_ep_<...>_.pth \
        --task Cartpole --train_cfg CartpolePPO \
        --num_envs 4 --steps 300

Output: `isaacsimenvs/videos/<task>_rollout.mp4`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = Path(__file__).resolve().parent / "videos"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to rl_games .pth checkpoint")
    parser.add_argument("--task", default="Cartpole", help="Task name from isaacsim_task_map")
    parser.add_argument("--train_cfg", default="CartpolePPO", help="Train yaml stem under cfg/train/")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--env_idx", type=int, default=0, help="Which env the camera follows (0..num_envs-1)")
    parser.add_argument("--steps", type=int, default=300, help="Physics steps to record")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--out", default=None, help="Output mp4 path (default: videos/<task>_rollout.mp4)")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy (mean)")
    my_args = parser.parse_args()

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = True
    app = AppLauncher(launcher_args).app

    import torch
    import yaml
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    from isaacsimenvs.tasks import isaacsim_task_map
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env

    # --- Env ---
    env_cls, cfg_cls = isaacsim_task_map[my_args.task]
    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = my_args.num_envs
    env = env_cls(cfg=env_cfg)

    # --- Camera sensor ---
    # Spawn with a placeholder pose; re-aim at the cart with set_world_poses_from_view
    # after the env's initial sim.reset. The cartpole rail is along +Y, so we view
    # from -X (perpendicular to rail) to see the cart slide + pole swing.
    camera_cfg = CameraCfg(
        prim_path="/World/RecordCamera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 10.0),  # placeholder
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )
    camera = Camera(cfg=camera_cfg)
    env.sim.reset()

    cart_pos = env.scene.articulations["cartpole"].data.root_pos_w[my_args.env_idx]
    # Look at the cart from along -X (perpendicular to the Y-axis rail), slightly up.
    env_origin = env.scene.env_origins[my_args.env_idx]
    eye = cart_pos + torch.tensor([-3.0, 0.0, 0.5], device=env.device)
    target = cart_pos.clone()
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))

    # Step the sim once to flush the pose change through PhysX/Hydra so
    # camera.data.pos_w updates.
    env.sim.step()
    camera.update(0.0)

    print(f"[diag] cart root_pos_w = {cart_pos.cpu().tolist()}")
    print(f"[diag] camera eye desired = {eye.cpu().tolist()}")
    print(f"[diag] camera pos_w actual = {camera.data.pos_w[0].cpu().tolist()}")
    print(f"[diag] camera quat_w actual = {camera.data.quat_w_world[0].cpu().tolist()}")

    # --- Wrap for rl_games + register ---
    wrapped = register_rlgames_env(
        env,
        rl_device=my_args.rl_device,
        clip_obs=5.0,
        clip_actions=1.0,
    )

    # --- Load train yaml to construct the Runner with matching arch ---
    train_yaml_path = REPO_ROOT / "isaacsimenvs" / "cfg" / "train" / f"{my_args.train_cfg}.yaml"
    with open(train_yaml_path) as f:
        rlg_cfg = yaml.safe_load(f)
    rlg_cfg["params"]["config"]["device"] = my_args.rl_device
    rlg_cfg["params"]["config"]["device_name"] = my_args.rl_device

    # --- Player ---
    from rl_games.torch_runner import Runner

    runner = Runner()
    runner.load(rlg_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(my_args.checkpoint)
    # `has_batch_dimension` is only set inside `player.run()`, which we bypass.
    # Our obs are always batched (num_envs, obs_dim), so set it explicitly.
    player.has_batch_dimension = True

    # --- Rollout + capture ---
    obs = player.env_reset(wrapped)
    dt = env.sim.get_physics_dt()
    capture_every = max(1, round((1.0 / my_args.video_fps) / dt))

    frames = []
    print(f"[play_video] Rolling out {my_args.steps} steps on {my_args.num_envs} envs...", flush=True)
    for step_i in range(my_args.steps):
        action = player.get_action(obs, is_deterministic=my_args.deterministic)
        obs, rew, dones, infos = player.env_step(wrapped, action)

        if step_i % capture_every == 0:
            camera.update(capture_every * dt)
            rgb = camera.data.output["rgb"]
            if rgb is not None and rgb.shape[0] > 0:
                frame = rgb[0].cpu().numpy()[:, :, :3]
                if not frames:
                    print(
                        f"[diag] first frame: shape={frame.shape} dtype={frame.dtype}"
                        f" min={frame.min()} max={frame.max()} mean={frame.mean():.2f}"
                    )
                    import imageio as _io
                    _debug_png = VIDEO_DIR / "cartpole_first_frame.png"
                    _debug_png.parent.mkdir(parents=True, exist_ok=True)
                    _io.imwrite(str(_debug_png), frame)
                    print(f"[diag] wrote first-frame png to {_debug_png}")
                frames.append(frame)

    # --- Save ---
    import imageio

    out_path = Path(my_args.out) if my_args.out else VIDEO_DIR / f"{my_args.task.lower()}_rollout.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), frames, fps=my_args.video_fps)
    print(f"[play_video] Wrote {len(frames)} frames to {out_path}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

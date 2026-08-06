"""Render a 5x5 grid of beam_0 parallel envs at diverse initial states.

For a presentation slide on "how we create diverse initial states for RL".
Boots IsaacSim with N parallel envs of the beam_3x part-0 assembly task, adds
ONE camera per env (regex prim path -> batched capture), resets once so every
env draws its own randomized init state, settles physics, then montages the N
per-env captures into a clean grid PNG.

Each tile shows the full training scene (Kuka arm + table + beam) with the beam
part at a different random initial pose -- i.e. the parallel-env "wall".

Usage (GPU node, isaacsim venv):
    .venv_isaacsim/bin/python plot_figures/fig2/render_beam0_parallel_grid.py

Writes:
    plot_figures/fig2/outputs/thesis_presentation/beam0_parallel_envs.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


def lookat_quat_wxyz(eye, target, up=(0.0, 1.0, 0.0)):
    import numpy as np
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    z_world = -fwd  # opengl: camera looks down local -Z
    x_world = np.cross(np.asarray(up, float), z_world)
    x_world /= np.linalg.norm(x_world)
    y_world = np.cross(z_world, x_world)
    R = np.stack([x_world, y_world, z_world], axis=1)
    tr = R.trace()
    qw = math.sqrt(max(0.0, 1 + tr)) / 2
    f = (4 * qw) if qw > 1e-6 else 1.0
    qx = (R[2, 1] - R[1, 2]) / f
    qy = (R[0, 2] - R[2, 0]) / f
    qz = (R[1, 0] - R[0, 1]) / f
    return (float(qw), float(qx), float(qy), float(qz))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Isaacsimenvs-PegInHole-Direct-v0")
    parser.add_argument("--problem", default="fabrica.beam_3x.part_0_matchedmass_sdf_hybrid")
    parser.add_argument("--num-envs", type=int, default=25)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--tile", type=int, default=320, help="per-env render size (px)")
    parser.add_argument("--gap", type=int, default=4, help="white gap between tiles (px)")
    # Camera framing (env-local). Target between table fixture and robot base so
    # both the Kuka arm and the beam are in frame.
    parser.add_argument("--cam-height", type=float, default=1.25)
    parser.add_argument("--cam-offset-y", type=float, default=-0.55)
    parser.add_argument("--target-y", type=float, default=0.18)
    parser.add_argument("--target-z", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch
    import yaml
    from PIL import Image

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg
    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spec = gym.spec(args.task)
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    with Path(spec.kwargs["env_cfg_yaml_entry_point"]).open() as f:
        cfg.from_dict(yaml.safe_load(f) or {})

    cfg.scene.num_envs = args.num_envs
    cfg.peg_in_hole.problem = args.problem
    cfg.peg_in_hole.goal_mode = "preInsertAndFinal"
    # Disable observation/action perturbations + wrench (visual cleanliness);
    # init-state randomization (object pose) is unaffected — that's the point.
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.domain_randomization.object_state_xyz_noise_std = 0.0
    cfg.domain_randomization.object_state_rotation_noise_degrees = 0.0
    cfg.domain_randomization.force_scale = 0.0
    cfg.domain_randomization.torque_scale = 0.0

    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    device = base.device
    zero_action = torch.zeros((args.num_envs, cfg.action_space), device=device,
                              dtype=torch.float32)

    print(f"[grid] {args.num_envs} envs constructed; adding per-env cameras", flush=True)

    cam_pos_local = (0.0, args.cam_offset_y, args.cam_height)
    cam_rot = lookat_quat_wxyz(cam_pos_local, (0.0, args.target_y, args.target_z))
    cam_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/GridCam",   # one camera per env -> batched
        update_period=0.0,
        height=args.tile,
        width=args.tile,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0, focus_distance=1.2,
            horizontal_aperture=20.955, clipping_range=(0.01, 6.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=cam_pos_local, rot=cam_rot, convention="opengl"),
    )
    cam = Camera(cam_cfg)

    base.sim.reset()
    for _ in range(2):
        base.sim.step()
    cam.update(dt=0.0)

    # One reset -> each env gets its own randomized initial object pose; settle.
    env.reset(seed=args.seed)
    for _ in range(args.settle_steps):
        env.step(zero_action)
    cam.update(dt=base.step_dt)
    rgb = cam.data.output["rgb"].detach().cpu().numpy()  # (N, H, W, 3 or 4)
    rgb = rgb[..., :3].astype(np.uint8)
    print(f"[grid] captured {rgb.shape}", flush=True)

    # Montage into a cols x rows grid with white gaps.
    n, H, W = rgb.shape[0], rgb.shape[1], rgb.shape[2]
    cols = args.cols
    rows = math.ceil(n / cols)
    gap = args.gap
    canvas = np.full((rows * H + (rows - 1) * gap, cols * W + (cols - 1) * gap, 3),
                     255, dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        y0 = r * (H + gap)
        x0 = c * (W + gap)
        canvas[y0:y0 + H, x0:x0 + W] = rgb[i]

    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "outputs" / "thesis_presentation"
        / "beam0_parallel_envs.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    print(f"wrote {out_path}  ({canvas.shape[1]}x{canvas.shape[0]})", flush=True)

    env.close()
    app.close()


if __name__ == "__main__":
    main()

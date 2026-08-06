"""Render per-env tiles of a beam assembly task at diverse init states.

Presentation asset (diverse initial states for RL). Boots IsaacSim with N
parallel envs, gives each its own camera, RE-COLORS the beam objects to their
original red (.mtl Kd 0.82,0.12,0.12) and the table to wood, sets up a proper
light rig, optionally path-traces for a clean look, resets once so each env
draws a different init pose, settles, and saves one PNG per env tile.

Run twice (beam_0 then beam_2), then montage with montage_beam_grid.py.

Usage (GPU node, isaacsim venv — run LOCALLY, no sbatch needed):
    .venv_isaacsim/bin/python plot_figures/fig2/render_beam_tiles.py \\
        --problem fabrica.beam_3x.part_0_matchedmass_sdf_hybrid --slug beam0 \\
        --num-envs 9 --path-tracing

Writes:
    plot_figures/fig2/outputs/thesis_presentation/tiles_<slug>/env_XX.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


def lookat_quat_wxyz(eye, target, up=(0.0, 1.0, 0.0)):
    import numpy as np
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(np.asarray(up, float), z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1); tr = R.trace()
    qw = math.sqrt(max(0.0, 1 + tr)) / 2
    f = (4 * qw) if qw > 1e-6 else 1.0
    return (float(qw), float((R[2, 1] - R[1, 2]) / f),
            float((R[0, 2] - R[2, 0]) / f), float((R[1, 0] - R[0, 1]) / f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaacsimenvs-PegInHole-Direct-v0")
    p.add_argument("--problem", required=True)
    p.add_argument("--slug", required=True, help="output subdir tiles_<slug>/")
    p.add_argument("--num-envs", type=int, default=9)
    p.add_argument("--settle-steps", type=int, default=100)
    p.add_argument("--tile", type=int, default=400)
    p.add_argument("--supersample", type=int, default=2, help="render at NxN then downscale (anti-grain)")
    p.add_argument("--cam-height", type=float, default=1.30)
    p.add_argument("--cam-offset-y", type=float, default=-0.58)
    p.add_argument("--target-y", type=float, default=0.16)
    p.add_argument("--target-z", type=float, default=0.50)
    p.add_argument("--focal", type=float, default=24.0, help="lens focal length (mm); higher=zoom")
    p.add_argument("--env-spacing", type=float, default=6.0, help="space envs apart so cams see only their own")
    p.add_argument("--beam-color", type=float, nargs=3, default=(0.74, 0.09, 0.09))
    p.add_argument("--table-color", type=float, nargs=3, default=(0.52, 0.34, 0.16))
    p.add_argument("--hdri", default=(
        ".venv_isaacsim/lib/python3.11/site-packages/isaacsim/extscache/"
        "omni.kit.widget.material_preview-1.0.16/data/domeLight/photo_studio_01_4k.hdr"),
        help="HDRI env map for image-based dome lighting ('' to disable)")
    p.add_argument("--dome-intensity", type=float, default=1100.0, help="dome/IBL intensity")
    p.add_argument("--ground-color", type=float, nargs=3, default=(0.62, 0.62, 0.64))
    p.add_argument("--sun-intensity", type=float, default=1800.0, help="directional key (fill on top of HDRI)")
    p.add_argument("--sun-elev", type=float, default=50.0, help="sun elevation above horizon (deg)")
    p.add_argument("--sun-azim", type=float, default=130.0, help="sun azimuth (deg)")
    p.add_argument("--path-tracing", action="store_true")
    p.add_argument("--spp", type=int, default=64, help="path-tracing samples/pixel")
    p.add_argument("--render-frames", type=int, default=24,
                   help="render iterations to accumulate before capture")
    p.add_argument("--seed", type=int, default=7)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.headless = True
    args.enable_cameras = True

    app = AppLauncher(args).app

    import carb
    import gymnasium as gym
    import numpy as np
    import torch
    import yaml
    from PIL import Image

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg
    from isaaclab.sim.utils import find_matching_prim_paths
    import isaacsimenvs  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.path_tracing:
        s = carb.settings.get_settings()
        s.set("/rtx/rendermode", "PathTracing")
        s.set("/rtx/pathtracing/spp", 8)              # samples per frame
        s.set("/rtx/pathtracing/totalSpp", int(args.spp))
        s.set("/rtx/pathtracing/clampSpp", int(args.spp))
        s.set("/rtx/pathtracing/maxBounces", 6)
        s.set("/rtx/pathtracing/optixDenoiser/enabled", True)  # key: kills grain
        s.set("/rtx/pathtracing/optixDenoiser/blendFactor", 0.0)

    spec = gym.spec(args.task)
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    with Path(spec.kwargs["env_cfg_yaml_entry_point"]).open() as f:
        cfg.from_dict(yaml.safe_load(f) or {})
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = args.env_spacing  # isolate each env from camera bleed
    cfg.peg_in_hole.problem = args.problem
    cfg.peg_in_hole.goal_mode = "preInsertAndFinal"
    for k, v in dict(use_obs_delay=False, use_action_delay=False,
                     use_object_state_delay_noise=False, object_state_xyz_noise_std=0.0,
                     object_state_rotation_noise_degrees=0.0, force_scale=0.0,
                     torque_scale=0.0).items():
        setattr(cfg.domain_randomization, k, v)

    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    device = base.device
    zero_action = torch.zeros((args.num_envs, cfg.action_space), device=device,
                              dtype=torch.float32)

    # --- Materials: recolor beam objects (original red) + wood table ----------
    def make_mat(path, rgb, roughness, metallic=0.0):
        m = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(rgb), roughness=roughness,
                                        metallic=metallic)
        m.func(path, m)
        return path

    beam_mat = make_mat("/World/Looks/BeamMat", args.beam_color, roughness=0.55)
    table_mat = make_mat("/World/Looks/TableMat", args.table_color, roughness=0.85)
    for obj_path in find_matching_prim_paths("/World/envs/env_.*/Object"):
        sim_utils.bind_visual_material(obj_path, beam_mat)
    for tbl_path in find_matching_prim_paths("/World/envs/env_.*/Table"):
        sim_utils.bind_visual_material(tbl_path, table_mat)
    print("[tiles] recolored beam + table", flush=True)

    # --- Light rig: soft dome fill + a strong angled "sun" key (directional
    # shadows + form). The distant light emits along its local -Z; orient it so
    # -Z points from the sun position (elev/azim) toward the scene.
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath("/World/Light").IsValid():
        stage.RemovePrim("/World/Light")  # drop the env's default dome
    hdri = str(Path(args.hdri).resolve()) if args.hdri else None
    dome = sim_utils.DomeLightCfg(intensity=args.dome_intensity, color=(1.0, 1.0, 1.0),
                                  texture_file=hdri)
    dome.func("/World/Light", dome)

    # Clean studio sweep instead of the default checkered debug grid.
    for g_path in find_matching_prim_paths("/World/ground"):
        sim_utils.bind_visual_material(
            g_path, make_mat("/World/Looks/GroundMat", args.ground_color, roughness=0.9))

    e = math.radians(args.sun_elev); a = math.radians(args.sun_azim)
    sun_pos = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
    sun = sim_utils.DistantLightCfg(intensity=args.sun_intensity,
                                    color=(1.0, 0.96, 0.88), angle=2.0)
    sun.func("/World/KeyLight", sun,
             orientation=lookat_quat_wxyz(sun_pos, (0.0, 0.0, 0.0)))

    # --- Per-env cameras ------------------------------------------------------
    render_px = args.tile * args.supersample
    cam_pos = (0.0, args.cam_offset_y, args.cam_height)
    cam_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/GridCam", update_period=0.0,
        height=render_px, width=render_px, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=args.focal, focus_distance=1.2,
                                         horizontal_aperture=20.955, clipping_range=(0.01, 6.0)),
        offset=CameraCfg.OffsetCfg(pos=cam_pos,
                                   rot=lookat_quat_wxyz(cam_pos, (0.0, args.target_y, args.target_z)),
                                   convention="opengl"),
    )
    cam = Camera(cam_cfg)

    base.sim.reset()
    for _ in range(2):
        base.sim.step()

    env.reset(seed=args.seed)
    for _ in range(args.settle_steps):
        env.step(zero_action)

    # Hide the goal-pose visualization — we only want the INITIAL states.
    from pxr import UsdGeom
    n_hidden = 0
    for gv in find_matching_prim_paths("/World/envs/env_.*/GoalViz"):
        prim = stage.GetPrimAtPath(gv)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
            n_hidden += 1
    print(f"[tiles] hid {n_hidden} GoalViz prims", flush=True)

    # Accumulate render frames: actually advance the renderer each iter so the
    # path tracer accumulates samples (cam.update alone re-pulls the same frame).
    for _ in range(args.render_frames):
        base.sim.render()
        cam.update(dt=base.step_dt)
    rgb = cam.data.output["rgb"].detach().cpu().numpy()[..., :3].astype(np.uint8)
    print(f"[tiles] captured {rgb.shape}", flush=True)

    out_dir = (Path(__file__).resolve().parent / "outputs" / "thesis_presentation"
               / f"tiles_{args.slug}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(rgb.shape[0]):
        im = Image.fromarray(rgb[i])
        if args.supersample > 1:  # downscale supersampled render -> smooth, grain-free
            im = im.resize((args.tile, args.tile), Image.LANCZOS)
        im.save(out_dir / f"env_{i:02d}.png")
    print(f"wrote {rgb.shape[0]} tiles -> {out_dir}", flush=True)

    env.close()
    app.close()


if __name__ == "__main__":
    main()

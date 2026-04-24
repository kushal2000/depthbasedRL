#!/usr/bin/env python3
"""Debug insertion test for peg-in-hole benchmark.

Teleports the peg along 3 waypoints (above hole → pre-insert → fully inserted),
then runs a free-physics settle phase. Tracks per-step position and keypoint
error vs desired, saves an MP4 + summary.

Usage:
    python peg_in_hole/validate_insertion.py --tolerance 1
    python peg_in_hole/validate_insertion.py --tolerance 0.5
"""

import argparse
import datetime
from pathlib import Path

from isaacgym import gymapi, gymtorch

import imageio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets/urdf/peg_in_hole"

# Peg rotated -90° about world Y so body +X (handle long axis) maps to world +Z.
# URDF origin is at HANDLE CENTER — tip ends up at (origin_z - 0.125), head-end
# at (origin_z + 0.125) after this rotation.
PEG_ORIENT_QUAT_XYZW = (0.0, -0.70710678, 0.0, 0.70710678)

# Handle extents (body frame) for keypoint metric — the inserted portion.
HANDLE_EXTENTS = np.array([0.25, 0.03, 0.02])
HALF_HANDLE = HANDLE_EXTENTS[0] / 2  # 0.125 — handle center -> tip offset

# Hole geometry (matches create_peg_and_holes.py).
HOLE_FLOOR_Z = 0.01
HOLE_OPENING_Z = 0.06

KEYPOINT_SCALE = 1.5
KEYPOINT_DIRS = np.array([
    [1, 1, 1],
    [1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1],
], dtype=np.float64)


def compute_keypoint_offsets(extents):
    return KEYPOINT_DIRS * extents[None, :] * KEYPOINT_SCALE / 2


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def compute_keypoints(pose, offsets):
    pos = pose[:3]
    R = quat_xyzw_to_matrix(pose[3:7])
    return pos[None, :] + (R @ offsets.T).T


def kp_max_dist(a, b, offsets):
    return np.linalg.norm(
        compute_keypoints(a, offsets) - compute_keypoints(b, offsets), axis=1
    ).max()


def slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    th0 = np.arccos(dot)
    th = th0 * t
    s_th = np.sin(th)
    s_th0 = np.sin(th0)
    s0 = np.cos(th) - dot * s_th / s_th0
    s1 = s_th / s_th0
    r = s0 * q0 + s1 * q1
    return r / np.linalg.norm(r)


def interpolate(waypoints, steps_per):
    poses = []
    for i in range(len(waypoints) - 1):
        a = np.asarray(waypoints[i], dtype=np.float64)
        b = np.asarray(waypoints[i + 1], dtype=np.float64)
        for s in range(steps_per):
            t = s / steps_per
            pos = a[:3] + t * (b[:3] - a[:3])
            quat = slerp(a[3:7], b[3:7], t)
            poses.append(np.concatenate([pos, quat]))
    poses.append(np.asarray(waypoints[-1], dtype=np.float64))
    return poses


def camera_for(waypoints):
    pts = np.array([w[:3] for w in waypoints])
    center = pts.mean(axis=0)
    span = pts.max(axis=0) - pts.min(axis=0)
    max_span = max(span.max(), 0.25)
    cam_pos = gymapi.Vec3(
        center[0] + max_span * 0.8,
        center[1] - max_span * 1.4,
        center[2] + max_span * 0.4,
    )
    cam_target = gymapi.Vec3(center[0], center[1], center[2])
    return cam_pos, cam_target


def fmt_tol(tol_mm):
    if tol_mm == int(tol_mm):
        return str(int(tol_mm))
    return str(tol_mm).replace(".", "p")


def make_waypoints(hole_xy=(0.0, 0.0)):
    """Waypoint positions refer to URDF origin (handle center).

    To place tip at world Z = T with the insertion orientation, we need
    center_z = T + HALF_HANDLE.
    """
    qx, qy, qz, qw = PEG_ORIENT_QUAT_XYZW
    hx, hy = hole_xy
    tip_above_z = 0.30
    tip_preins_z = HOLE_OPENING_Z + 0.01
    tip_ins_z = HOLE_FLOOR_Z + 0.001
    return [
        [hx, hy, tip_above_z + HALF_HANDLE, qx, qy, qz, qw],
        [hx, hy, tip_preins_z + HALF_HANDLE, qx, qy, qz, qw],
        [hx, hy, tip_ins_z + HALF_HANDLE, qx, qy, qz, qw],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="Hole tolerance in mm (must match a generated variant).")
    parser.add_argument("--steps-per-waypoint", type=int, default=100)
    parser.add_argument("--settle-frames", type=int, default=120)
    parser.add_argument("--output-dir", type=str,
                        default=str(REPO_ROOT / "peg_in_hole/debug_output"))
    parser.add_argument("--timestamp", type=str, default=None)
    args = parser.parse_args()

    tol_mm = args.tolerance
    tol_tag = fmt_tol(tol_mm)
    hname = f"hole_tol{tol_tag}mm"
    hole_dir = ASSETS_DIR / "holes" / hname
    hole_urdf = f"{hname}.urdf"
    assert (hole_dir / hole_urdf).exists(), f"Missing hole asset: {hole_dir / hole_urdf}"

    peg_dir = ASSETS_DIR / "peg"
    peg_urdf = "peg.urdf"
    assert (peg_dir / peg_urdf).exists(), f"Missing peg asset: {peg_dir / peg_urdf}"

    ts = args.timestamp or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_dir) / f"tol{tol_tag}mm" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    waypoints = make_waypoints()
    print(f"Tolerance {tol_mm}mm, waypoints:")
    for i, w in enumerate(waypoints):
        print(f"  [{i}] pos=({w[0]:.3f},{w[1]:.3f},{w[2]:.3f}) "
              f"quat=({w[3]:.3f},{w[4]:.3f},{w[5]:.3f},{w[6]:.3f})")

    gym = gymapi.acquire_gym()

    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sp.dt = 1.0 / 60.0
    sp.substeps = 2
    sp.use_gpu_pipeline = True
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 16
    sp.physx.num_velocity_iterations = 1
    sp.physx.rest_offset = 0.0
    sp.physx.contact_offset = 0.005
    sp.physx.friction_offset_threshold = 0.01
    sp.physx.friction_correlation_distance = 0.00625
    sp.physx.max_depenetration_velocity = 5.0
    sp.physx.max_gpu_contact_pairs = 16 * 1024 * 1024
    sp.physx.num_threads = 0
    sp.physx.use_gpu = True

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    assert sim is not None

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)

    hole_opts = gymapi.AssetOptions()
    hole_opts.fix_base_link = True
    hole_asset = gym.load_asset(sim, str(hole_dir), hole_urdf, hole_opts)
    print(f"Hole ({hname}): {gym.get_asset_rigid_body_count(hole_asset)} bodies, "
          f"{gym.get_asset_rigid_shape_count(hole_asset)} shapes")

    peg_opts = gymapi.AssetOptions()
    peg_opts.collapse_fixed_joints = True
    peg_asset = gym.load_asset(sim, str(peg_dir), peg_urdf, peg_opts)
    print(f"Peg: {gym.get_asset_rigid_body_count(peg_asset)} bodies, "
          f"{gym.get_asset_rigid_shape_count(peg_asset)} shapes")

    env = gym.create_env(sim,
                         gymapi.Vec3(-0.5, -0.5, 0.0),
                         gymapi.Vec3(0.5, 0.5, 0.5), 1)

    hole_pose = gymapi.Transform()
    hole_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    hole_pose.r = gymapi.Quat(0, 0, 0, 1)
    gym.create_actor(env, hole_asset, hole_pose, "hole", 0, 0)

    w0 = waypoints[0]
    peg_pose = gymapi.Transform()
    peg_pose.p = gymapi.Vec3(w0[0], w0[1], w0[2])
    peg_pose.r = gymapi.Quat(w0[3], w0[4], w0[5], w0[6])
    peg_actor = gym.create_actor(env, peg_asset, peg_pose, "peg", 0, 0, 0)

    cam = gymapi.CameraProperties()
    cam.width = 1280
    cam.height = 960
    cam.use_collision_geometry = False
    cam_h = gym.create_camera_sensor(env, cam)
    cp, ct = camera_for(waypoints)
    gym.set_camera_location(cam_h, env, cp, ct)

    gym.prepare_sim(sim)

    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    states = gymtorch.wrap_tensor(root_tensor)
    PART_IDX = 1

    offsets = compute_keypoint_offsets(HANDLE_EXTENTS)
    traj = interpolate(waypoints, args.steps_per_waypoint)
    teleport_steps = len(traj)
    total_steps = teleport_steps + args.settle_frames
    final_desired = np.array(traj[-1])
    print(f"Trajectory: {teleport_steps} teleport + {args.settle_frames} settle = {total_steps} total")

    frames_dir = out_dir / "video_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    deltas, kps = [], []
    settle_start = None
    settle_drift = []
    settle_kp = []

    for step in range(total_steps):
        if step < teleport_steps:
            des = traj[step]
            states[PART_IDX, 0:7] = torch.tensor(des, dtype=torch.float32, device="cuda")
            states[PART_IDX, 7:13] = 0.0
            idx = torch.tensor([PART_IDX], dtype=torch.int32, device="cuda")
            gym.set_actor_root_state_tensor_indexed(
                sim, gymtorch.unwrap_tensor(states),
                gymtorch.unwrap_tensor(idx), 1,
            )

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        gym.refresh_actor_root_state_tensor(sim)
        actual = states[PART_IDX, 0:7].cpu().numpy()

        if step < teleport_steps:
            des_np = traj[step]
            d = np.linalg.norm(des_np[:3] - actual[:3])
            k = kp_max_dist(actual, des_np, offsets)
            deltas.append(d)
            kps.append(k)
            if step % 50 == 0:
                print(f"  [{step:4d}/{total_steps}] des_z={des_np[2]:.4f} "
                      f"act_z={actual[2]:.4f} pos={d*1000:.2f}mm kp={k*1000:.2f}mm")
        else:
            if settle_start is None:
                settle_start = actual.copy()
            drift = np.linalg.norm(actual[:3] - settle_start[:3])
            kdr = kp_max_dist(actual, final_desired, offsets)
            settle_drift.append(drift)
            settle_kp.append(kdr)
            ss = step - teleport_steps
            if ss % 40 == 0:
                print(f"  settle [{ss:3d}/{args.settle_frames}] drift={drift*1000:.2f}mm "
                      f"kp_vs_des={kdr*1000:.2f}mm")

        if step % 5 == 0:
            img = gym.get_camera_image(sim, env, cam_h, gymapi.IMAGE_COLOR)
            if img.size > 0:
                img = img.reshape(cam.height, cam.width, 4)
                frames.append(img[:, :, :3].copy())

    if deltas:
        d_arr = np.array(deltas)
        k_arr = np.array(kps)
        print(f"\n--- {hname} teleport summary ---")
        print(f"  pos_delta  max: {d_arr.max()*1000:6.2f} mm   final: {d_arr[-1]*1000:.2f} mm")
        print(f"  kp_max     max: {k_arr.max()*1000:6.2f} mm   final: {k_arr[-1]*1000:.2f} mm")
        print(f"  steps pos>1mm: {(d_arr > 0.001).sum()}/{len(d_arr)}")
        print(f"  steps kp >1mm: {(k_arr > 0.001).sum()}/{len(k_arr)}")
        verdict = "No collisions" if d_arr.max() < 0.001 else f"COLLISIONS (max {d_arr.max()*1000:.2f}mm)"
        print(f"  {verdict}")

    if settle_drift:
        sd = np.array(settle_drift)
        sk = np.array(settle_kp)
        print(f"\n--- settle summary ({args.settle_frames} frames) ---")
        print(f"  drift_from_start max: {sd.max()*1000:.2f} mm   final: {sd[-1]*1000:.2f} mm")
        print(f"  kp_vs_desired final: {sk[-1]*1000:.2f} mm")

    gym.refresh_net_contact_force_tensor(sim)

    if frames:
        final_d = deltas[-1] * 1000 if deltas else 0.0
        final_k = kps[-1] * 1000 if kps else 0.0
        mp4 = out_dir / f"peg_in_{hname}_pos{final_d:.1f}mm_kp{final_k:.1f}mm.mp4"
        imageio.mimsave(str(mp4), frames, fps=12)
        print(f"\nSaved {len(frames)} frames")
        print(f"Video: {mp4}")
    else:
        print("WARNING: no frames captured")

    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()

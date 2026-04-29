#!/usr/bin/env python3
"""Validate FMB insertions by teleporting along scene trajectories in Isaac Gym.

For each (part, scene, start) combo, teleports the insertion piece along the
cached trajectory waypoints and checks for collisions / stability.

Usage:
    python -m fmb.validate_insertions --assembly fmb_board_1
    python -m fmb.validate_insertions --assembly fmb_board_1 --part-idx 0 --scene-idx 0 --start-idx 0
"""

import argparse
import json
from pathlib import Path

from isaacgym import gymapi, gymtorch

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fmb"

TABLE_Z = 0.38

KEYPOINT_SCALE = 1.5
KEYPOINT_DIRS = np.array([
    [1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1],
], dtype=np.float64)


def load_canonical_extents(assembly, part):
    path = ASSETS_DIR / assembly / "canonical_transforms.json"
    with open(path) as f:
        data = json.load(f)
    return np.array(data[part]["canonical_extents"])


def compute_keypoint_offsets(extents):
    return KEYPOINT_DIRS * extents[None, :] * KEYPOINT_SCALE / 2


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def keypoints_max_dist(pose_a, pose_b, kp_offsets):
    def kps(p):
        R = quat_xyzw_to_matrix(p[3:7])
        return p[:3][None, :] + (R @ kp_offsets.T).T
    return np.linalg.norm(kps(pose_a) - kps(pose_b), axis=1).max()


def slerp(q0, q1, t):
    q0, q1 = np.array(q0, np.float64), np.array(q1, np.float64)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1, dot = -q1, -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    r = s0 * q0 + s1 * q1
    return r / np.linalg.norm(r)


def interpolate_waypoints(waypoints, steps_per):
    poses = []
    for i in range(len(waypoints) - 1):
        wp0, wp1 = np.array(waypoints[i]), np.array(waypoints[i + 1])
        for s in range(steps_per):
            t = s / steps_per
            pos = wp0[:3] + t * (wp1[:3] - wp0[:3])
            quat = slerp(wp0[3:7], wp1[3:7], t)
            poses.append(np.concatenate([pos, quat]))
    poses.append(np.array(waypoints[-1]))
    return poses


def validate_one(gym, assembly, part_id, scene_urdf_path, waypoints, kp_offsets,
                 steps_per_waypoint=50, settle_frames=120):
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 16
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.005
    sim_params.physx.max_depenetration_velocity = 5.0
    sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024
    sim_params.physx.num_threads = 0
    sim_params.physx.use_gpu = True

    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Load scene (fixture) URDF
    scene_full_path = REPO_ROOT / "assets" / scene_urdf_path
    scene_root = str(scene_full_path.parent)
    scene_file = scene_full_path.name
    scene_options = gymapi.AssetOptions()
    scene_options.fix_base_link = True
    scene_options.collapse_fixed_joints = True
    scene_asset = gym.load_asset(sim, scene_root, scene_file, scene_options)

    # Load insertion part
    part_root = str(ASSETS_DIR / assembly / part_id / "coacd")
    part_file = f"{assembly}_{part_id}_coacd.urdf"
    part_options = gymapi.AssetOptions()
    part_options.collapse_fixed_joints = True
    part_options.replace_cylinder_with_capsule = True
    part_asset = gym.load_asset(sim, part_root, part_file, part_options)

    spacing = 0.5
    env = gym.create_env(sim, gymapi.Vec3(-spacing, -spacing, 0),
                         gymapi.Vec3(spacing, spacing, spacing), 1)

    scene_pose = gymapi.Transform()
    scene_pose.p = gymapi.Vec3(0, 0, TABLE_Z)
    scene_pose.r = gymapi.Quat(0, 0, 0, 1)
    gym.create_actor(env, scene_asset, scene_pose, "scene", 0, 0)

    wp0 = waypoints[0]
    part_pose = gymapi.Transform()
    part_pose.p = gymapi.Vec3(float(wp0[0]), float(wp0[1]), float(wp0[2]))
    part_pose.r = gymapi.Quat(float(wp0[3]), float(wp0[4]), float(wp0[5]), float(wp0[6]))
    gym.create_actor(env, part_asset, part_pose, "part", 0, 0, 0)

    gym.prepare_sim(sim)
    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_tensor)
    PART_IDX = 1

    traj_poses = interpolate_waypoints(waypoints, steps_per_waypoint)
    total_teleport = len(traj_poses)
    total_steps = total_teleport + settle_frames

    max_delta = 0.0
    max_kp_dist = 0.0

    for step in range(total_steps):
        if step < total_teleport:
            desired = traj_poses[step]
            root_states[PART_IDX, 0:7] = torch.tensor(desired, dtype=torch.float32, device="cuda")
            root_states[PART_IDX, 7:13] = 0.0
            actor_indices = torch.tensor([PART_IDX], dtype=torch.int32, device="cuda")
            gym.set_actor_root_state_tensor_indexed(
                sim, gymtorch.unwrap_tensor(root_states),
                gymtorch.unwrap_tensor(actor_indices), 1)

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)

        if step < total_teleport:
            actual = root_states[PART_IDX, 0:7].cpu().numpy()
            delta = np.linalg.norm(traj_poses[step][:3] - actual[:3])
            kp_d = keypoints_max_dist(actual, traj_poses[step], kp_offsets)
            max_delta = max(max_delta, delta)
            max_kp_dist = max(max_kp_dist, kp_d)

    # Settle check
    gym.refresh_actor_root_state_tensor(sim)
    final_pose = root_states[PART_IDX, 0:7].cpu().numpy()
    desired_final = np.array(traj_poses[-1])
    settle_kp = keypoints_max_dist(final_pose, desired_final, kp_offsets)

    gym.destroy_sim(sim)

    return {
        "max_pos_delta_mm": max_delta * 1000,
        "max_kp_dist_mm": max_kp_dist * 1000,
        "settle_kp_mm": settle_kp * 1000,
        "collision": max_kp_dist > 0.001,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate FMB insertions")
    parser.add_argument("--assembly", required=True)
    parser.add_argument("--part-idx", type=int, default=None,
                        help="Single part index. Default: all parts.")
    parser.add_argument("--scene-idx", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--num-scenes", type=int, default=3,
                        help="Number of scenes to validate per part.")
    parser.add_argument("--steps-per-waypoint", type=int, default=50)
    parser.add_argument("--settle-frames", type=int, default=120)
    args = parser.parse_args()

    data = np.load(ASSETS_DIR / args.assembly / "scenes.npz", allow_pickle=True)
    insertion_parts = [str(p) for p in data["insertion_parts"].tolist()]
    goals = data["goals"]
    traj_lengths = data["traj_lengths"]
    scene_urdf_paths = data["scene_urdf_paths"]

    P, N, M, T, _ = goals.shape
    print(f"Assembly: {args.assembly}")
    print(f"Parts: {insertion_parts}, Scenes: {N}, Starts: {M}\n")

    gym = gymapi.acquire_gym()

    part_indices = [args.part_idx] if args.part_idx is not None else range(P)

    results = []
    for p_idx in part_indices:
        pid = insertion_parts[p_idx]
        extents = load_canonical_extents(args.assembly, pid)
        kp_offsets = compute_keypoint_offsets(extents)

        scene_indices = range(min(args.num_scenes, N))
        for n_idx in scene_indices:
            m_idx = args.start_idx
            tl = int(traj_lengths[p_idx, n_idx, m_idx])
            waypoints = goals[p_idx, n_idx, m_idx, :tl].tolist()

            if len(waypoints) < 2:
                print(f"  [{pid}] scene {n_idx}: skip (traj_len={tl})")
                continue

            urdf_path = str(scene_urdf_paths[p_idx, n_idx])
            print(f"  [{pid}] scene {n_idx}, start {m_idx} (traj_len={tl})...", end=" ", flush=True)

            r = validate_one(
                gym, args.assembly, pid, urdf_path, waypoints, kp_offsets,
                args.steps_per_waypoint, args.settle_frames,
            )
            results.append((pid, n_idx, m_idx, r))

            status = "COLLISION" if r["collision"] else "OK"
            print(f"{status}  pos={r['max_pos_delta_mm']:.2f}mm  kp={r['max_kp_dist_mm']:.2f}mm  settle={r['settle_kp_mm']:.2f}mm")

    # Summary
    n_total = len(results)
    n_collisions = sum(1 for _, _, _, r in results if r["collision"])
    print(f"\n{'='*60}")
    print(f"Summary: {n_collisions}/{n_total} trajectories had collisions (kp > 1mm)")
    if results:
        max_kp = max(r["max_kp_dist_mm"] for _, _, _, r in results)
        avg_kp = np.mean([r["max_kp_dist_mm"] for _, _, _, r in results])
        print(f"  Max kp dist: {max_kp:.2f}mm")
        print(f"  Avg kp dist: {avg_kp:.2f}mm")


if __name__ == "__main__":
    main()

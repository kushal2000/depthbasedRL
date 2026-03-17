#!/usr/bin/env python3
"""Insertion test: teleport part 2 into part 6 with different collision methods.

Headless IsaacGym script that places part 6 fixed on the table, then teleports
part 2 along its descent trajectory (waypoints 8-11) to test insertion with
4 collision configurations: VHACD, CoACD (default), CoACD (tuned), SDF.

Usage:
    python fabrica/debug_insertion_test.py --method vhacd
    python fabrica/debug_insertion_test.py --method coacd_old
    python fabrica/debug_insertion_test.py --method coacd
    python fabrica/debug_insertion_test.py --method sdf
"""

import argparse
import json
from pathlib import Path

from isaacgym import gymapi, gymtorch

import imageio
import numpy as np
import torch


# Part 2 insertion waypoints 8-11 (from beam_2/pick_place.json goals[8:12])
INSERTION_WAYPOINTS = [
    [-0.124463, 0.04, 0.68,      0.0, -0.707107, 0.0, 0.707107],
    [-0.124463, 0.04, 0.617833,  0.0, -0.707107, 0.0, 0.707107],
    [-0.124463, 0.04, 0.580533,  0.0, -0.707107, 0.0, 0.707107],
    [-0.124463, 0.04, 0.5681,    0.0, -0.707107, 0.0, 0.707107],
]

# Part 6 goal pose (final waypoint from beam_6/pick_place.json)
PART6_POSE = [-0.08, 0.04, 0.5367, 0.0, 0.0, 0.0, 1.0]

TABLE_Z = 0.38


def slerp(q0, q1, t):
    """Spherical linear interpolation between quaternions (xyzw format)."""
    q0 = np.array(q0, dtype=np.float64)
    q1 = np.array(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


def interpolate_waypoints(waypoints, steps_per):
    """Interpolate between waypoints. Each waypoint is [x,y,z, qx,qy,qz,qw]."""
    poses = []
    for i in range(len(waypoints) - 1):
        wp0 = np.array(waypoints[i])
        wp1 = np.array(waypoints[i + 1])
        for s in range(steps_per):
            t = s / steps_per
            pos = wp0[:3] + t * (wp1[:3] - wp0[:3])
            quat = slerp(wp0[3:7], wp1[3:7], t)
            poses.append(np.concatenate([pos, quat]))
    poses.append(np.array(waypoints[-1]))
    return poses


def load_assets(gym, sim, method):
    """Load part 6 and part 2 assets with the specified collision method.

    Returns (part6_asset, part2_asset).
    """
    # Part 6
    p6_options = gymapi.AssetOptions()
    p6_options.fix_base_link = True
    p6_options.collapse_fixed_joints = True
    p6_options.replace_cylinder_with_capsule = True

    # Part 2
    p2_options = gymapi.AssetOptions()
    p2_options.collapse_fixed_joints = True
    p2_options.replace_cylinder_with_capsule = True

    if method == "vhacd":
        p6_root = "assets/urdf/fabrica/beam/6"
        p6_file = "beam_6.urdf"
        p6_options.vhacd_enabled = True
        p2_root = "assets/urdf/fabrica/beam/2"
        p2_file = "beam_2.urdf"
        p2_options.vhacd_enabled = True
    elif method == "coacd_old":
        p6_root = "assets/urdf/fabrica/beam/6/coacd_old"
        p6_file = "beam_6_coacd.urdf"
        p2_root = "assets/urdf/fabrica/beam/2/coacd_old"
        p2_file = "beam_2_coacd.urdf"
    elif method == "coacd":
        p6_root = "assets/urdf/fabrica/beam/6/coacd"
        p6_file = "beam_6_coacd.urdf"
        p2_root = "assets/urdf/fabrica/beam/2/coacd"
        p2_file = "beam_2_coacd.urdf"
    elif method == "sdf":
        p6_root = "assets/urdf/fabrica/beam/6"
        p6_file = "beam_6_sdf.urdf"
        p6_options.thickness = 0.0
        p2_root = "assets/urdf/fabrica/beam/2"
        p2_file = "beam_2_sdf.urdf"
        p2_options.thickness = 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

    p6_asset = gym.load_asset(sim, p6_root, p6_file, p6_options)
    print(f"Part 6 ({method}): {gym.get_asset_rigid_body_count(p6_asset)} bodies, "
          f"{gym.get_asset_rigid_shape_count(p6_asset)} shapes")

    p2_asset = gym.load_asset(sim, p2_root, p2_file, p2_options)
    print(f"Part 2 ({method}): {gym.get_asset_rigid_body_count(p2_asset)} bodies, "
          f"{gym.get_asset_rigid_shape_count(p2_asset)} shapes")

    return p6_asset, p2_asset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["vhacd", "coacd_old", "coacd", "sdf"])
    parser.add_argument("--num-frames", type=int, default=600)
    parser.add_argument("--steps-per-waypoint", type=int, default=50)
    parser.add_argument("--settle-frames", type=int, default=100,
                        help="Extra frames after teleport to let physics settle")
    parser.add_argument("--output-dir", type=str, default="debug_output/insertion")
    parser.add_argument("--compute-device", type=int, default=0)
    parser.add_argument("--graphics-device", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    gym = gymapi.acquire_gym()

    # Sim params
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
    sim_params.physx.friction_offset_threshold = 0.01
    sim_params.physx.friction_correlation_distance = 0.00625
    sim_params.physx.max_depenetration_velocity = 5.0
    sim_params.physx.num_threads = 0
    sim_params.physx.use_gpu = True

    sim = gym.create_sim(args.compute_device, args.graphics_device, gymapi.SIM_PHYSX, sim_params)
    assert sim is not None, "Failed to create sim"

    # Ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Table asset
    table_options = gymapi.AssetOptions()
    table_options.fix_base_link = True
    table_options.collapse_fixed_joints = True
    table_options.thickness = 0.001
    table_asset = gym.load_asset(sim, "assets/urdf/fabrica/environments", "empty_table.urdf", table_options)

    # Part assets
    p6_asset, p2_asset = load_assets(gym, sim, args.method)

    # Create env
    env_spacing = 0.5
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # Actor 0: Table
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.0, 0.0, TABLE_Z)
    table_pose.r = gymapi.Quat(0, 0, 0, 1)
    gym.create_actor(env, table_asset, table_pose, "table", 0, 0)

    # Actor 1: Part 6 (fixed on table)
    p6_pose = gymapi.Transform()
    p6_pose.p = gymapi.Vec3(PART6_POSE[0], PART6_POSE[1], PART6_POSE[2])
    p6_pose.r = gymapi.Quat(PART6_POSE[3], PART6_POSE[4], PART6_POSE[5], PART6_POSE[6])
    p6_handle = gym.create_actor(env, p6_asset, p6_pose, "part6", 0, 0, 0)

    # Set SDF shape properties for part 6 if needed
    if args.method == "sdf":
        shape_props = gym.get_actor_rigid_shape_properties(env, p6_handle)
        for sp in shape_props:
            sp.thickness = 0.0
        gym.set_actor_rigid_shape_properties(env, p6_handle, shape_props)

    # Actor 2: Part 2 (will be teleported)
    wp0 = INSERTION_WAYPOINTS[0]
    p2_pose = gymapi.Transform()
    p2_pose.p = gymapi.Vec3(wp0[0], wp0[1], wp0[2])
    p2_pose.r = gymapi.Quat(wp0[3], wp0[4], wp0[5], wp0[6])
    p2_handle = gym.create_actor(env, p2_asset, p2_pose, "part2", 0, 0, 0)

    # Set SDF shape properties for part 2 if needed
    if args.method == "sdf":
        shape_props = gym.get_actor_rigid_shape_properties(env, p2_handle)
        for sp in shape_props:
            sp.thickness = 0.0
        gym.set_actor_rigid_shape_properties(env, p2_handle, shape_props)

    # Camera
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 960
    cam_props.use_collision_geometry = True
    cam_handle = gym.create_camera_sensor(env, cam_props)
    # Camera positioned to see the insertion from side/above
    cam_pos = gymapi.Vec3(-0.02, -0.15, 0.68)
    cam_target = gymapi.Vec3(-0.10, 0.04, 0.56)
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)

    gym.prepare_sim(sim)

    # Get root state tensor for teleportation
    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_tensor)
    # Actors: 0=table, 1=part6, 2=part2
    PART2_IDX = 2

    # Generate interpolated trajectory
    traj_poses = interpolate_waypoints(INSERTION_WAYPOINTS, args.steps_per_waypoint)
    total_teleport_steps = len(traj_poses)
    total_steps = total_teleport_steps + args.settle_frames
    print(f"Trajectory: {total_teleport_steps} teleport steps + {args.settle_frames} settle = {total_steps} total")

    # Simulation loop
    frames_dir = output_dir / "video_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    deltas = []

    print(f"Running insertion test ({args.method})...")
    for step in range(total_steps):
        if step < total_teleport_steps:
            # Teleport part 2 to next pose
            desired = traj_poses[step]
            root_states[PART2_IDX, 0:7] = torch.tensor(desired, dtype=torch.float32, device="cuda")
            root_states[PART2_IDX, 7:13] = 0.0  # zero velocities

            actor_indices = torch.tensor([PART2_IDX], dtype=torch.int32, device="cuda")
            gym.set_actor_root_state_tensor_indexed(
                sim, gymtorch.unwrap_tensor(root_states),
                gymtorch.unwrap_tensor(actor_indices), 1
            )

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)

        # Read back actual pose and log delta
        gym.refresh_actor_root_state_tensor(sim)
        actual = root_states[PART2_IDX, 0:7].cpu().numpy()

        if step < total_teleport_steps:
            desired_np = traj_poses[step]
            delta = np.linalg.norm(desired_np[:3] - actual[:3])
            deltas.append(delta)
            if step % 25 == 0:
                print(f"  Step {step:4d}/{total_steps}: desired_z={desired_np[2]:.4f} "
                      f"actual_z={actual[2]:.4f} delta={delta:.6f}")
        else:
            settle_step = step - total_teleport_steps
            if settle_step % 25 == 0:
                print(f"  Settle {settle_step:4d}/{args.settle_frames}: "
                      f"pos=({actual[0]:.4f}, {actual[1]:.4f}, {actual[2]:.4f})")

        # Capture frame every 5 steps
        if step % 5 == 0:
            color_image = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
            if color_image.size > 0:
                color_image = color_image.reshape(cam_props.height, cam_props.width, 4)
                frames.append(color_image[:, :, :3].copy())
                frame_path = frames_dir / f"frame_{step:04d}.png"
                imageio.imwrite(str(frame_path), color_image[:, :, :3])

    # Summary
    if deltas:
        deltas = np.array(deltas)
        print(f"\n--- Collision Delta Summary ({args.method}) ---")
        print(f"  Max delta:  {deltas.max():.6f} m")
        print(f"  Mean delta: {deltas.mean():.6f} m")
        print(f"  Steps with delta > 1mm: {(deltas > 0.001).sum()}/{len(deltas)}")
        print(f"  Steps with delta > 5mm: {(deltas > 0.005).sum()}/{len(deltas)}")

    # Final poses
    gym.refresh_actor_root_state_tensor(sim)
    p6_final = root_states[1, 0:7].cpu().numpy()
    p2_final = root_states[2, 0:7].cpu().numpy()
    print(f"\nFinal part 6 pose: pos={p6_final[:3]}, quat={p6_final[3:7]}")
    print(f"Final part 2 pose: pos={p2_final[:3]}, quat={p2_final[3:7]}")

    # Contact forces
    contact_tensor = gym.acquire_net_contact_force_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    contacts = gymtorch.wrap_tensor(contact_tensor)
    print(f"Final contact force on part 6: {contacts[1].cpu().numpy()}")
    print(f"Final contact force on part 2: {contacts[2].cpu().numpy()}")

    # Save video
    if frames:
        video_path = output_dir / "insertion_test.mp4"
        imageio.mimsave(str(video_path), frames, fps=12)
        print(f"\nSaved {len(frames)} frames to {frames_dir}")
        print(f"Video: {video_path}")
    else:
        print("WARNING: No frames captured. Check graphics_device_id >= 0.")

    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()

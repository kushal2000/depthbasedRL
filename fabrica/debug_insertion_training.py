#!/usr/bin/env python3
"""Insertion debug using training-style scene URDFs.

Unlike debug_insertion_test.py (which loads empty_table + separate part 6),
this script loads the scene URDF (table + already-placed parts as a single
fixed asset) — matching what the RL training environment actually sees.

Usage:
    python fabrica/debug_insertion_training.py --method coacd
    python fabrica/debug_insertion_training.py --method sdf
    python fabrica/debug_insertion_training.py --method vhacd
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Debug insertion using training-style scene URDFs")
    parser.add_argument("--method", type=str, required=True,
                        choices=["vhacd", "coacd", "sdf"])
    parser.add_argument("--steps-per-waypoint", type=int, default=50)
    parser.add_argument("--settle-frames", type=int, default=100,
                        help="Extra frames after teleport to let physics settle")
    parser.add_argument("--output-dir", type=str, default="debug_output/insertion")
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

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    assert sim is not None, "Failed to create sim"

    # Ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # --- Scene asset (table + already-placed parts) ---
    scene_root = "assets/urdf/fabrica/environments/beam_2"
    scene_options = gymapi.AssetOptions()
    scene_options.fix_base_link = True
    scene_options.collapse_fixed_joints = True

    if args.method == "vhacd":
        scene_file = "scene.urdf"
        scene_options.vhacd_enabled = True
    elif args.method == "coacd":
        scene_file = "scene_coacd.urdf"
    elif args.method == "sdf":
        scene_file = "scene_sdf.urdf"
        scene_options.thickness = 0.0

    scene_asset = gym.load_asset(sim, scene_root, scene_file, scene_options)
    print(f"Scene ({args.method}): {gym.get_asset_rigid_body_count(scene_asset)} bodies, "
          f"{gym.get_asset_rigid_shape_count(scene_asset)} shapes")

    # --- Part 2 asset (active part) ---
    p2_options = gymapi.AssetOptions()
    p2_options.collapse_fixed_joints = True
    p2_options.replace_cylinder_with_capsule = True

    if args.method == "vhacd":
        p2_root = "assets/urdf/fabrica/beam/2"
        p2_file = "beam_2.urdf"
        p2_options.vhacd_enabled = True
    elif args.method == "coacd":
        p2_root = "assets/urdf/fabrica/beam/2/coacd"
        p2_file = "beam_2_coacd.urdf"
    elif args.method == "sdf":
        p2_root = "assets/urdf/fabrica/beam/2"
        p2_file = "beam_2_sdf.urdf"
        p2_options.thickness = 0.0

    p2_asset = gym.load_asset(sim, p2_root, p2_file, p2_options)
    print(f"Part 2 ({args.method}): {gym.get_asset_rigid_body_count(p2_asset)} bodies, "
          f"{gym.get_asset_rigid_shape_count(p2_asset)} shapes")

    # Create env
    env_spacing = 0.5
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # Actor 0: Scene (table + placed parts)
    scene_pose = gymapi.Transform()
    scene_pose.p = gymapi.Vec3(0.0, 0.0, TABLE_Z)
    scene_pose.r = gymapi.Quat(0, 0, 0, 1)
    gym.create_actor(env, scene_asset, scene_pose, "scene", 0, 0)

    # Actor 1: Part 2 (will be teleported)
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
    cam_pos = gymapi.Vec3(-0.02, -0.15, 0.68)
    cam_target = gymapi.Vec3(-0.10, 0.04, 0.56)
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)

    gym.prepare_sim(sim)

    # Get root state tensor for teleportation
    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_tensor)
    # Actors: 0=scene, 1=part2
    PART2_IDX = 1

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

    print(f"Running insertion test ({args.method}, training scene)...")
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
        print(f"\n--- Collision Delta Summary ({args.method}, training scene) ---")
        print(f"  Max delta:  {deltas.max():.6f} m")
        print(f"  Mean delta: {deltas.mean():.6f} m")
        print(f"  Steps with delta > 1mm: {(deltas > 0.001).sum()}/{len(deltas)}")
        print(f"  Steps with delta > 5mm: {(deltas > 0.005).sum()}/{len(deltas)}")

    # Final poses
    gym.refresh_actor_root_state_tensor(sim)
    scene_final = root_states[0, 0:7].cpu().numpy()
    p2_final = root_states[PART2_IDX, 0:7].cpu().numpy()
    print(f"\nFinal scene pose: pos={scene_final[:3]}, quat={scene_final[3:7]}")
    print(f"Final part 2 pose: pos={p2_final[:3]}, quat={p2_final[3:7]}")

    # Contact forces
    contact_tensor = gym.acquire_net_contact_force_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    contacts = gymtorch.wrap_tensor(contact_tensor)
    print(f"Final contact force on scene: {contacts[0].cpu().numpy()}")
    print(f"Final contact force on part 2: {contacts[PART2_IDX].cpu().numpy()}")

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

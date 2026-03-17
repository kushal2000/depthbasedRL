#!/usr/bin/env python3
"""Drop-test beam part using CoACD convex decomposition collision.

Headless IsaacGym script that drops a part onto the table and captures frames.

Usage:
    python fabrica/debug_coacd.py --part 2
    python fabrica/debug_coacd.py --part 6 --num-frames 500
"""

import argparse
import os
from pathlib import Path

from isaacgym import gymapi, gymtorch

import imageio
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default="2", choices=["2", "6"])
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="debug_output/coacd")
    parser.add_argument("--compute-device", type=int, default=0)
    parser.add_argument("--graphics-device", type=int, default=0)
    parser.add_argument("--empty-table", action="store_true", help="Use empty table (no fixture/parts)")
    parser.add_argument("--coacd-dir", type=str, default=None,
                        help="Override CoACD asset directory (default: assets/urdf/fabrica/beam/{part}/coacd)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / f"part_{args.part}"
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

    # Ground plane (z-up)
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Table asset
    if args.empty_table:
        table_root = "assets/urdf/fabrica/environments"
        table_file = "empty_table.urdf"
    else:
        table_root = f"assets/urdf/fabrica/environments/beam_{args.part}"
        table_file = "scene.urdf"
    table_options = gymapi.AssetOptions()
    table_options.fix_base_link = True
    table_options.collapse_fixed_joints = True
    table_options.thickness = 0.001
    table_asset = gym.load_asset(sim, table_root, table_file, table_options)

    # Object asset - CoACD
    beam_root = args.coacd_dir if args.coacd_dir else f"assets/urdf/fabrica/beam/{args.part}/coacd"
    beam_file = f"beam_{args.part}_coacd.urdf"
    beam_options = gymapi.AssetOptions()
    beam_options.vhacd_enabled = False
    beam_options.collapse_fixed_joints = True
    beam_options.replace_cylinder_with_capsule = True
    beam_asset = gym.load_asset(sim, beam_root, beam_file, beam_options)
    print(f"CoACD beam: {gym.get_asset_rigid_body_count(beam_asset)} bodies, "
          f"{gym.get_asset_rigid_shape_count(beam_asset)} shapes")

    # Create env
    env_spacing = 0.5
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # Table actor
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.0, 0.0, 0.38)
    table_pose.r = gymapi.Quat(0, 0, 0, 1)
    gym.create_actor(env, table_asset, table_pose, "table", 0, 0)

    # Object actor (drop from z=0.7)
    beam_pose = gymapi.Transform()
    beam_pose.p = gymapi.Vec3(0.0, 0.0, 0.7)
    beam_pose.r = gymapi.Quat(0, 0, 0, 1)
    obj_handle = gym.create_actor(env, beam_asset, beam_pose, "object", 0, 0, 0)

    # Camera sensor
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 960
    cam_props.use_collision_geometry = True
    cam_handle = gym.create_camera_sensor(env, cam_props)
    cam_pos = gymapi.Vec3(0.12, -0.12, 0.62)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.535)
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)

    gym.prepare_sim(sim)

    # Simulation loop
    frames_dir = output_dir / "video_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    print(f"Running {args.num_frames} frames...")
    for step in range(args.num_frames):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)

        if step % 10 == 0:
            color_image = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
            if color_image.size > 0:
                color_image = color_image.reshape(cam_props.height, cam_props.width, 4)
                frames.append(color_image[:, :, :3].copy())
                frame_path = frames_dir / f"frame_{step:04d}.png"
                imageio.imwrite(str(frame_path), color_image[:, :, :3])

    # Read final object pose
    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_tensor)
    # Object is actor index 1 (table=0, object=1)
    obj_state = root_states[1]
    pos = obj_state[:3].cpu().numpy()
    quat = obj_state[3:7].cpu().numpy()
    print(f"\nFinal object pose: pos={pos}, quat={quat}")

    # Read contact forces
    contact_tensor = gym.acquire_net_contact_force_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    contacts = gymtorch.wrap_tensor(contact_tensor)
    obj_contact = contacts[1].cpu().numpy()
    print(f"Final contact force on object: {obj_contact}")

    # Save video
    if frames:
        video_path = output_dir / "drop_test.mp4"
        imageio.mimsave(str(video_path), frames, fps=6)
        print(f"\nSaved {len(frames)} frames to {frames_dir}")
        print(f"Video: {video_path}")
    else:
        print("WARNING: No frames captured. Check graphics_device_id >= 0.")

    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()

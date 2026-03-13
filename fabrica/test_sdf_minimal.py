"""Minimal SDF test: progressively add env-like settings to find what breaks SDF.

Run with: python fabrica/test_sdf_minimal.py
"""
from isaacgym import gymapi, gymutil

gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="SDF debug")

args.use_gpu = True
args.use_gpu_pipeline = True

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = True

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 32
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.rest_offset = 0.0
sim_params.physx.contact_offset = 0.005
sim_params.physx.friction_offset_threshold = 0.01
sim_params.physx.friction_correlation_distance = 0.0005
sim_params.physx.num_threads = 0
sim_params.physx.use_gpu = True

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
assert sim is not None

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# ============================================================
# Assets
# ============================================================

# 1. Robot asset -- DISABLED
# robot_root = "assets/urdf/kuka_sharpa_description"
# robot_file = "iiwa14_left_sharpa_adjusted_restricted.urdf"
# robot_options = gymapi.AssetOptions()
# robot_options.flip_visual_attachments = False
# robot_options.fix_base_link = True
# robot_options.collapse_fixed_joints = True
# robot_options.disable_gravity = True
# robot_options.thickness = 0.001
# robot_options.angular_damping = 0.01
# robot_options.replace_cylinder_with_capsule = True
# robot_asset = gym.load_asset(sim, robot_root, robot_file, robot_options)

# 2. Table asset -- RE-ENABLED
table_root = "assets/urdf/fabrica/environments/beam_2"
table_file = "scene.urdf"
table_options = gymapi.AssetOptions()
table_options.fix_base_link = True
table_options.collapse_fixed_joints = True
table_options.thickness = 0.001
table_asset = gym.load_asset(sim, table_root, table_file, table_options)

# 3. Object asset (SDF)
beam_root = "assets/urdf/fabrica/beam/2"
beam_file = "beam_2_sdf.urdf"
beam_options = gymapi.AssetOptions()
beam_options.thickness = 0.0
beam_options.collapse_fixed_joints = True
beam_options.replace_cylinder_with_capsule = True
beam_asset = gym.load_asset(sim, beam_root, beam_file, beam_options)
print(f"Beam: {gym.get_asset_rigid_body_count(beam_asset)} bodies, "
      f"{gym.get_asset_rigid_shape_count(beam_asset)} shapes")

# 4. Goal asset (same URDF, default thickness)
goal_options = gymapi.AssetOptions()
goal_options.disable_gravity = True
goal_options.collapse_fixed_joints = True
goal_options.replace_cylinder_with_capsule = True
goal_asset = gym.load_asset(sim, beam_root, beam_file, goal_options)

# ============================================================
# Create env with all actors (same order as SimToolReal)
# ============================================================

env_spacing = 0.5
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
env = gym.create_env(sim, env_lower, env_upper, 1)

# Robot -- DISABLED
# robot_pose = gymapi.Transform()
# robot_pose.p = gymapi.Vec3(0.0, 0.8, 0.0)
# robot_pose.r = gymapi.Quat(0, 0, 0, 1)
# gym.create_actor(env, robot_asset, robot_pose, "robot", 0, 0)

# Object
beam_pose = gymapi.Transform()
beam_pose.p = gymapi.Vec3(0.0, 0.0, 0.6)
beam_pose.r = gymapi.Quat(0, 0, 0, 1)
gym.create_actor(env, beam_asset, beam_pose, "object", 0, 0, 0)

# Goal
goal_pose = gymapi.Transform()
goal_pose.p = gymapi.Vec3(0.0, 0.0, 0.55)
goal_pose.r = gymapi.Quat(0, 0, 0, 1)
gym.create_actor(env, goal_asset, goal_pose, "goal", 1, 0, 0)

# Table -- DISABLED to test
# table_pose = gymapi.Transform()
# table_pose.p = gymapi.Vec3(0.0, 0.0, 0.38)
# table_pose.r = gymapi.Quat(0, 0, 0, 1)
# gym.create_actor(env, table_asset, table_pose, "table", 0, 0)

gym.prepare_sim(sim)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
assert viewer is not None

cam_pos = gymapi.Vec3(0, -0.5, 0.8)
cam_target = gymapi.Vec3(0, 0, 0.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print("\nViewer ready. Robot + Table + Beam SDF + Goal. Press ESC to quit.")

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

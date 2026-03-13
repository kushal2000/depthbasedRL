"""Minimal SDF test: progressively add env-like settings to find what breaks SDF.

Run with: python fabrica/test_sdf_minimal.py
Toggle flags below one at a time to find the culprit.
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
# Toggle these flags one at a time to find what breaks SDF
# ============================================================
USE_COLLAPSE_FIXED_JOINTS = True       # env.py sets this
USE_REPLACE_CYLINDER = True            # env.py sets this
SET_SHAPE_PROPS_THICKNESS = True       # env.py does this after creation
USE_GPU_PIPELINE = True                # env uses True
# ============================================================

beam_root = "assets/urdf/fabrica/beam/2"
beam_file = "beam_2_sdf.urdf"

beam_options = gymapi.AssetOptions()
beam_options.thickness = 0.0
beam_options.collapse_fixed_joints = USE_COLLAPSE_FIXED_JOINTS
beam_options.replace_cylinder_with_capsule = USE_REPLACE_CYLINDER

print(f"\n{'='*60}")
print(f"collapse_fixed_joints = {USE_COLLAPSE_FIXED_JOINTS}")
print(f"replace_cylinder_with_capsule = {USE_REPLACE_CYLINDER}")
print(f"set_shape_props_thickness = {SET_SHAPE_PROPS_THICKNESS}")
print(f"{'='*60}\n")

beam_asset = gym.load_asset(sim, beam_root, beam_file, beam_options)

# Optionally modify shape properties after loading (like env.py does)
if SET_SHAPE_PROPS_THICKNESS:
    props = gym.get_asset_rigid_shape_properties(beam_asset)
    for i in range(len(props)):
        props[i].thickness = 0.0
    gym.set_asset_rigid_shape_properties(beam_asset, props)
    print(f"Set {len(props)} shape props thickness=0.0")

env_spacing = 0.5
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
env = gym.create_env(sim, env_lower, env_upper, 1)

beam_pose = gymapi.Transform()
beam_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
beam_pose.r = gymapi.Quat(0, 0, 0, 1)
gym.create_actor(env, beam_asset, beam_pose, "beam", 0, 0)

gym.prepare_sim(sim)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
assert viewer is not None

cam_pos = gymapi.Vec3(0, -0.3, 0.6)
cam_target = gymapi.Vec3(0, 0, 0.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print("Viewer ready. Check beam collision mesh. Press ESC to quit.")

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

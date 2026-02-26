"""Isaac Lab configuration for the SimToolReal environment.

Ports the IsaacGym YAML configs (SimToolReal.yaml + SimToolRealPPO.yaml)
to Isaac Lab's @configclass Python format.

Key convention changes from IsaacGym:
  - Quaternions: xyzw -> wxyz
  - Episode length: steps -> seconds
  - substeps -> decimation (with adjusted dt)
  - Joint ordering: depth-first -> breadth-first (remapped at runtime)
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

# Resolve asset paths relative to project root
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_ASSET_ROOT = os.path.join(_PROJECT_ROOT, "assets")
_USD_ROOT = os.path.join(_ASSET_ROOT, "usd")


# ══════════════════════════════════════════════════════════════════════
# Robot DOF properties (from isaacgymenvs/tasks/simtoolreal/utils.py)
# ══════════════════════════════════════════════════════════════════════

# Kuka arm (7 DOFs)
KUKA_STIFFNESSES = [600, 600, 500, 400, 200, 200, 200]
KUKA_DAMPINGS = [
    27.027026473513512, 27.027026473513512, 24.672186769721083,
    22.067474708266914, 9.752538131173853, 9.147747263670984, 9.147747263670984,
]
KUKA_EFFORTS = [300, 300, 300, 300, 300, 300, 300]

# Sharpa hand (22 DOFs)
HAND_STIFFNESSES = [
    6.95, 13.2, 4.76, 6.62, 0.9,     # thumb
    4.76, 6.62, 0.9, 0.9,             # index
    4.76, 6.62, 0.9, 0.9,             # middle
    4.76, 6.62, 0.9, 0.9,             # ring
    1.38, 4.76, 6.62, 0.9, 0.9,       # pinky
]
HAND_DAMPINGS = [
    0.28676845, 0.40845109, 0.20394083, 0.24044435, 0.04190723,
    0.20859232, 0.24595532, 0.04243185, 0.03504461,
    0.2085923, 0.24595532, 0.04243185, 0.03504461,
    0.20859226, 0.24595528, 0.04243183, 0.0350446,
    0.02782345, 0.20859229, 0.24595528, 0.04243183, 0.0350446,
]
HAND_ARMATURES = [
    0.0032, 0.0032, 0.00265, 0.00265, 0.0006,
    0.00265, 0.00265, 0.0006, 0.00042,
    0.00265, 0.00265, 0.0006, 0.00042,
    0.00265, 0.00265, 0.0006, 0.00042,
    0.00012, 0.00265, 0.00265, 0.0006, 0.00042,
]
HAND_FRICTIONS = [
    0.132, 0.132, 0.07456, 0.07456, 0.01276,
    0.07456, 0.07456, 0.01276, 0.00378738,
    0.07456, 0.07456, 0.01276, 0.00378738,
    0.07456, 0.07456, 0.01276, 0.00378738,
    0.012, 0.07456, 0.07456, 0.01276, 0.00378738,
]

# Combined
ALL_STIFFNESSES = KUKA_STIFFNESSES + HAND_STIFFNESSES
ALL_DAMPINGS = KUKA_DAMPINGS + HAND_DAMPINGS

NUM_ARM_DOFS = 7
NUM_HAND_DOFS = 22
NUM_HAND_ARM_DOFS = NUM_ARM_DOFS + NUM_HAND_DOFS  # 29

# Fingertip link names (left Sharpa hand)
FINGERTIP_NAMES = [
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_thumb_DP",
    "left_pinky_DP",
]
NUM_FINGERTIPS = 5

# Palm link
PALM_LINK = "iiwa14_link_7"

# Depth camera mount link
DEPTH_MOUNT_LINK = "iiwa14_link_6"


# ══════════════════════════════════════════════════════════════════════
# Domain randomization events
# ══════════════════════════════════════════════════════════════════════

@configclass
class EventCfg:
    """Domain randomization events — ported from IsaacGym's DR API.

    NOTE: Isaac Lab's EventTermCfg is imported and used at runtime
    inside the env since it requires mdp functions. This config class
    stores the parameters that the env's _setup_events() method uses.
    """
    # Robot DOF property randomization (applied at reset)
    robot_dof_damping_range: tuple = (0.7, 1.3)
    robot_dof_stiffness_range: tuple = (0.7, 1.3)
    robot_dof_effort_range: tuple = (0.7, 1.3)
    robot_dof_friction_range: tuple = (0.7, 1.3)
    robot_dof_armature_range: tuple = (0.7, 1.3)

    # Robot rigid body mass (applied once at startup)
    robot_mass_range: tuple = (0.7, 1.3)

    # Robot rigid shape friction (applied at reset)
    robot_friction_range: tuple = (0.7, 1.3)
    robot_restitution_range: tuple = (0.0, 0.3)

    # Object rigid body mass (applied once at startup)
    object_mass_range: tuple = (0.7, 1.3)

    # Object friction (applied at reset)
    object_friction_range: tuple = (0.7, 1.3)
    object_restitution_range: tuple = (0.0, 0.3)

    # Gravity perturbation
    gravity_noise_range: tuple = (0.0, 0.3)

    # Observation/action noise
    obs_noise_range: tuple = (0.0, 0.01)
    action_noise_range: tuple = (0.0, 0.01)


# ══════════════════════════════════════════════════════════════════════
# Main environment config
# ══════════════════════════════════════════════════════════════════════

@configclass
class SimToolRealEnvCfg(DirectRLEnvCfg):
    """Configuration for the SimToolReal Isaac Lab environment.

    Equivalent to SimToolReal.yaml + SimToolRealPPO.yaml from IsaacGym.
    """

    # ── Scene ──
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=1.2,
        replicate_physics=True,
    )

    # ── Simulation ──
    # IsaacGym: dt=1/60, substeps=2 -> effective physics dt = 1/120
    # Isaac Lab: dt=1/120, decimation=2 -> control rate = 1/60 (same)
    decimation: int = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,  # render every decimation steps
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,  # TGS
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8 * 1024 * 1024,
            gpu_found_lost_pairs_capacity=8 * 1024 * 1024,
            gpu_found_lost_aggregate_pairs_capacity=8 * 1024 * 1024,
        ),
    )

    # ── Episode ──
    # 600 steps * (1/60)s = 10s
    episode_length_s: float = 10.0

    # ── Robot articulation ──
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_USD_ROOT, "kuka_sharpa", "robot.usd"),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1000.0,
                angular_damping=0.01,
                linear_damping=0.01,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.8, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),  # wxyz
            joint_pos={
                # Kuka arm default pose
                "iiwa14_joint_1": -1.571,
                "iiwa14_joint_2": 1.571,
                "iiwa14_joint_3": 0.0,
                "iiwa14_joint_4": 1.376,
                "iiwa14_joint_5": 0.0,
                "iiwa14_joint_6": 1.485,
                "iiwa14_joint_7": 1.308,  # 60 deg offset for sharpa mount
                # Hand defaults to 0 (use regex that doesn't overlap arm joints)
                "left_.*": 0.0,
            },
            joint_vel={"left_.*": 0.0, "iiwa14_joint_.*": 0.0},
        ),
        actuators={
            "kuka_arm": ImplicitActuatorCfg(
                joint_names_expr=["iiwa14_joint_[1-7]"],
                effort_limit={
                    f"iiwa14_joint_{i+1}": float(KUKA_EFFORTS[i])
                    for i in range(NUM_ARM_DOFS)
                },
                stiffness={
                    f"iiwa14_joint_{i+1}": float(KUKA_STIFFNESSES[i])
                    for i in range(NUM_ARM_DOFS)
                },
                damping={
                    f"iiwa14_joint_{i+1}": float(KUKA_DAMPINGS[i])
                    for i in range(NUM_ARM_DOFS)
                },
            ),
            "sharpa_hand": ImplicitActuatorCfg(
                joint_names_expr=["left_.*"],
                stiffness={},  # Set per-joint at runtime via _build_joint_index_maps
                damping={},    # Set per-joint at runtime
            ),
        },
    )

    # ── Object (rigid body) ──
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_USD_ROOT, "dextoolbench", "hammer", "claw_hammer", "claw_hammer.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1000.0,
                angular_damping=0.0,
                linear_damping=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.63),
            rot=(1.0, 0.0, 0.0, 0.0),  # wxyz
        ),
    )

    # ── Table (static) ──
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_USD_ROOT, "table", "table_narrow.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.38),
            rot=(1.0, 0.0, 0.0, 0.0),  # wxyz
        ),
    )

    # ── TiledCamera (replaces IsaacGym per-env cameras) ──
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/iiwa14_link_6/DepthCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.04, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),  # wxyz (identity)
            convention="ros",
        ),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            clipping_range=(0.01, 1.5),
        ),
        width=64,
        height=64,
    )

    # ── Depth camera settings ──
    use_depth_camera: bool = False  # Toggle at runtime
    depth_width: int = 64
    depth_height: int = 64
    depth_near: float = 0.01
    depth_far: float = 1.5
    depth_hfov: float = 70.0
    depth_encoder_type: str = "scratch_cnn"
    depth_feature_dim: int = 512
    depth_freeze_encoder: bool = False
    depth_unfreeze_after_epochs: int = -1
    depth_encoder_lr: float = 1.0e-5

    # ── Observation / Action dimensions ──
    # obs_list: joint_pos(29) + joint_vel(29) + prev_action_targets(29) + palm_pos(3)
    #         + palm_rot(4) + object_rot(4) + fingertip_pos_rel_palm(15)
    #         + keypoints_rel_palm(12) + keypoints_rel_goal(12) + object_scales(3) = 140
    observation_space: int = 140
    # state_list: everything in obs + palm_vel(6) + object_vel(6) + closest_keypoint_max_dist(1)
    #           + closest_fingertip_dist(5) + lifted_object(1) + progress(1) + successes(1)
    #           + reward(1) = 162
    # Actually from YAML: the stateList includes all obs_list items plus extra privileged info
    state_space: int = 0  # Computed at runtime from state_list
    action_space: int = 29  # 7 arm + 22 hand DOFs

    # ── Obs / state list configuration ──
    state_list: tuple = (
        "joint_pos", "joint_vel", "prev_action_targets",
        "palm_pos", "palm_rot", "palm_vel",
        "object_rot", "object_vel",
        "fingertip_pos_rel_palm", "keypoints_rel_palm", "keypoints_rel_goal",
        "object_scales", "closest_keypoint_max_dist", "closest_fingertip_dist",
        "lifted_object", "progress", "successes", "reward",
    )
    obs_list: tuple = (
        "joint_pos", "joint_vel", "prev_action_targets",
        "palm_pos", "palm_rot",
        "object_rot",
        "fingertip_pos_rel_palm", "keypoints_rel_palm", "keypoints_rel_goal",
        "object_scales",
    )

    # ── Control parameters ──
    control_freq_inv: int = 1  # IsaacGym controlFrequencyInv (steps per action)
    dof_speed_scale: float = 10.0
    hand_moving_average: float = 0.1
    arm_moving_average: float = 0.1
    use_relative_control: bool = False
    clamp_abs_observations: float = 10.0

    # ── Reset noise ──
    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    reset_position_noise_z: float = 0.02
    randomize_object_rotation: bool = True
    reset_dof_pos_noise_fingers: float = 0.1
    reset_dof_pos_noise_arm: float = 0.1
    reset_dof_vel_noise: float = 0.5

    # ── Table reset ──
    table_reset_z: float = 0.38
    table_reset_z_range: float = 0.01
    table_object_z_offset: float = 0.25

    # ── Random forces on object ──
    force_scale: float = 20.0
    force_prob_range: tuple = (0.001, 0.1)
    force_decay: float = 0.0
    force_decay_interval: float = 0.08
    force_only_when_lifted: bool = True
    torque_scale: float = 2.0
    torque_prob_range: tuple = (0.001, 0.1)
    torque_decay: float = 0.0
    torque_decay_interval: float = 0.08
    torque_only_when_lifted: bool = True

    # ── Reward scales ──
    lifting_rew_scale: float = 20.0
    lifting_bonus: float = 300.0
    lifting_bonus_threshold: float = 0.15
    keypoint_rew_scale: float = 200.0
    distance_delta_rew_scale: float = 50.0
    reach_goal_bonus: float = 1000.0
    kuka_actions_penalty_scale: float = 0.03
    hand_actions_penalty_scale: float = 0.003
    fall_distance: float = 0.24
    fall_penalty: float = 0.0
    object_lin_vel_penalty_scale: float = 0.0
    object_ang_vel_penalty_scale: float = 0.0

    # ── Keypoint settings ──
    keypoint_scale: float = 1.5
    object_base_size: float = 0.04
    fixed_size_keypoint_reward: bool = True
    fixed_size: tuple = (0.141, 0.03025, 0.0271)
    num_keypoints: int = 4
    keypoint_offsets: tuple = (
        (1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    )

    # ── Success / curriculum ──
    success_tolerance: float = 0.075
    target_success_tolerance: float = 0.01
    tolerance_curriculum_increment: float = 0.9
    tolerance_curriculum_interval: int = 3000
    max_consecutive_successes: int = 50
    success_steps: int = 10
    force_consecutive_near_goal_steps: bool = False

    # ── Goal sampling ──
    goal_sampling_type: str = "delta"
    target_volume_region_scale: float = 1.0
    delta_goal_distance: float = 0.1
    delta_rotation_degrees: float = 90.0
    target_volume_mins: tuple = (-0.35, -0.2, 0.6)
    target_volume_maxs: tuple = (0.35, 0.2, 0.95)

    # ── Delay and noise ──
    use_obs_delay: bool = True
    obs_delay_max: int = 3
    use_action_delay: bool = True
    action_delay_max: int = 3
    use_object_state_delay_noise: bool = False
    object_state_delay_max: int = 10
    object_state_xyz_noise_std: float = 0.01
    object_state_rotation_noise_degrees: float = 5.0
    joint_velocity_obs_noise_std: float = 0.1

    # ── Friction settings ──
    modify_asset_frictions: bool = True
    robot_friction: float = 0.5
    fingertip_friction: float = 1.5
    object_friction: float = 0.5
    table_friction: float = 0.5

    # ── Object configuration ──
    object_name: str = "handle_head_primitives"
    handle_head_types: tuple = ("hammer", "screwdriver", "marker", "spatula", "eraser", "brush")

    # ── Misc ──
    privileged_actions: bool = False
    start_arm_higher: bool = False
    capture_video: bool = True
    capture_video_freq: int = 6000
    capture_video_len: int = 600

    # ── Domain randomization ──
    # NOTE: We use 'dr_params' instead of 'events' because the base DirectRLEnvCfg
    # passes 'events' to Isaac Lab's EventManager which expects EventTermCfg objects.
    # Our DR is applied manually in the env code.
    randomize: bool = False
    dr_params: EventCfg = EventCfg()

    # ── Curriculum flags ──
    turn_off_extra_obs: bool = False
    turn_off_extra_obs_slowly: bool = False
    turn_off_palm_vel_obs: bool = False
    turn_off_palm_vel_obs_slowly: bool = False
    turn_off_object_vel_obs: bool = False
    turn_off_object_vel_obs_slowly: bool = False
    use_obs_dropout: bool = False
    curriculum_success_ratio: float = 0.6

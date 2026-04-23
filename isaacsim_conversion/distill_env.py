from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from isaacgymenvs.utils.observation_action_utils_sharpa import (
    _compute_keypoint_positions,
    compute_joint_pos_targets,
    matrix_to_quaternion_xyzw_scipy,
    OBJECT_KEYPOINT_OFFSETS_np,
    PALM_OFFSET_np,
    FINGERTIP_OFFSETS_np,
    Q_LOWER_LIMITS_np,
    Q_UPPER_LIMITS_np,
)
from isaacgymenvs.utils.torch_jit_utils import quat_rotate, unscale
from isaacgymenvs.utils.torch_jit_utils import quaternion_to_matrix, matrix_to_quaternion
from isaacsim_conversion.isaacsim_env import (
    CONTROL_DT,
    DEFAULT_ASSET_FRICTION,
    DEFAULT_JOINT_POS,
    FINGERTIP_FRICTION,
    FINGERTIP_LINK_NAMES,
    JOINT_DAMPINGS,
    JOINT_DAMPINGS_COMPENSATED,
    JOINT_NAMES_ISAACGYM,
    JOINT_STIFFNESSES,
    JOINT_STIFFNESSES_COMPENSATED,
    PHYSICS_DT,
    PHYSICS_SUBSTEPS,
    _log,
)
from isaacsim_conversion.image_robustness import ImageRobustnessCfg, VisualDRCfg
from isaacsim_conversion.task_utils import CameraPose, TaskSpec, xyzw_to_wxyz
from isaacsim_conversion.task_utils import CameraIntrinsics


OBS_LIST = [
    "joint_pos",
    "joint_vel",
    "prev_action_targets",
    "palm_pos",
    "palm_rot",
    "object_rot",
    "fingertip_pos_rel_palm",
    "keypoints_rel_palm",
    "keypoints_rel_goal",
    "object_scales",
]
OBJECT_BASE_SIZE = 0.04
KEYPOINT_SCALE = 1.5


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / torch.clamp(angle, min=1e-8)
    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]
    zeros = torch.zeros_like(x)
    k = torch.stack(
        (
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).repeat(axis_angle.shape[0], 1, 1)
    sin_term = torch.sin(angle).unsqueeze(-1)
    cos_term = (1.0 - torch.cos(angle)).unsqueeze(-1)
    outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)
    rot = eye + sin_term * k + cos_term * (outer - eye)
    small = (angle.squeeze(-1) < 1e-8)
    if torch.any(small):
        rot[small] = eye[small]
    return rot


@dataclass
class SimState:
    q: np.ndarray
    qd: np.ndarray
    object_pose: np.ndarray
    goal_pose: np.ndarray
    goal_idx: np.ndarray
    kp_dist: np.ndarray
    near_goal_steps: np.ndarray
    object_pos_world_env_frame: np.ndarray


class IsaacSimDistillEnv:
    def __init__(
        self,
        task_spec: TaskSpec,
        app: Any,
        headless: bool,
        camera_modality: str = "depth",
        use_real_camera_transform: bool = True,
        camera_pose_override: CameraPose | None = None,
        camera_intrinsics: CameraIntrinsics | None = None,
        num_envs: int = 1,
        env_spacing: float = 2.0,
        object_start_mode: str = "fixed",
        object_pos_noise_xyz: tuple[float, float, float] = (0.03, 0.03, 0.01),
        object_yaw_noise_deg: float = 20.0,
        enable_camera: bool = True,
        camera_backend: str = "tiled",
        ground_plane_size: float = 500.0,
        enable_scene_query_support: bool = False,
        depth_preprocess_mode: str = "clip_divide",
        depth_min_m: float = 0.0,
        depth_max_m: float = 5.0,
        episode_length: int = 600,
        reset_when_dropped: bool = False,
        use_obs_delay: bool = False,
        obs_delay_max: int = 1,
        use_action_delay: bool = False,
        action_delay_max: int = 1,
        use_object_state_delay_noise: bool = False,
        object_state_delay_max: int = 1,
        object_state_xyz_noise_std: float = 0.0,
        object_state_rotation_noise_degrees: float = 0.0,
        joint_velocity_obs_noise_std: float = 0.0,
        goal_xy_obs_noise: float = 0.0,
        object_scale_noise_multiplier_range: tuple[float, float] = (1.0, 1.0),
        default_asset_friction: float = DEFAULT_ASSET_FRICTION,
        fingertip_friction: float = FINGERTIP_FRICTION,
        start_arm_higher: bool = False,
        hand_moving_average: float = 0.1,
        arm_moving_average: float = 0.1,
        dof_speed_scale: float = 1.5,
        image_robustness: ImageRobustnessCfg | None = None,
    ):
        self.task_spec = task_spec
        self.camera_modality = camera_modality
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.object_start_mode = object_start_mode
        self.object_pos_noise_xyz = np.array(object_pos_noise_xyz, dtype=np.float32)
        self.object_yaw_noise_deg = float(object_yaw_noise_deg)
        self.app = app
        self.headless = headless
        self.camera_pose = camera_pose_override or task_spec.camera_pose
        self.camera_intrinsics = camera_intrinsics or CameraIntrinsics()
        self.use_real_camera_transform = use_real_camera_transform
        self.enable_camera = enable_camera
        self.camera_backend = camera_backend
        self.ground_plane_size = float(ground_plane_size)
        self.enable_scene_query_support = bool(enable_scene_query_support)
        if self.camera_backend not in {"tiled", "standard"}:
            raise ValueError(f"Unsupported camera_backend={self.camera_backend!r}")
        if self.ground_plane_size <= 0:
            raise ValueError("ground_plane_size must be > 0")
        self.depth_preprocess_mode = depth_preprocess_mode
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.episode_length = int(episode_length)
        self.reset_when_dropped = bool(reset_when_dropped)
        self.use_obs_delay = bool(use_obs_delay)
        self.obs_delay_max = int(max(obs_delay_max, 1))
        self.use_action_delay = bool(use_action_delay)
        self.action_delay_max = int(max(action_delay_max, 1))
        self.use_object_state_delay_noise = bool(use_object_state_delay_noise)
        self.object_state_delay_max = int(max(object_state_delay_max, 1))
        self.object_state_xyz_noise_std = float(object_state_xyz_noise_std)
        self.object_state_rotation_noise_degrees = float(object_state_rotation_noise_degrees)
        self.joint_velocity_obs_noise_std = float(joint_velocity_obs_noise_std)
        self.goal_xy_obs_noise = float(goal_xy_obs_noise)
        self.object_scale_noise_multiplier_range = (
            float(object_scale_noise_multiplier_range[0]),
            float(object_scale_noise_multiplier_range[1]),
        )
        self.default_asset_friction = float(default_asset_friction)
        self.fingertip_friction = float(fingertip_friction)
        self.start_arm_higher = bool(start_arm_higher)
        self.hand_moving_average = float(hand_moving_average)
        self.arm_moving_average = float(arm_moving_average)
        self.dof_speed_scale = float(dof_speed_scale)
        self.image_robustness = image_robustness or ImageRobustnessCfg()
        self.visual_dr_cfg: VisualDRCfg = self.image_robustness.visual_dr
        if self.depth_preprocess_mode not in {"clip_divide", "window_normalize", "metric"}:
            raise ValueError(f"Unsupported depth_preprocess_mode={self.depth_preprocess_mode!r}")
        if self.depth_max_m <= self.depth_min_m:
            raise ValueError("depth_max_m must be greater than depth_min_m")
        self.action_dim = 29
        self.student_proprio_dim = 29 + 29 + 29 + 1
        self.object_scales_batch = np.repeat(self.task_spec.object_scales.astype(np.float32), self.num_envs, axis=0)
        success_scales = None
        if self.task_spec.metadata is not None and self.task_spec.metadata.get("fixed_size_keypoint_reward", False):
            fixed_size = self.task_spec.metadata.get("fixed_size_scales")
            if fixed_size is not None:
                success_scales = np.array([fixed_size], dtype=np.float32)
        if success_scales is None:
            success_scales = self.task_spec.object_scales.astype(np.float32)
        self.success_object_scales_batch = np.repeat(success_scales, self.num_envs, axis=0)
        self.camera_data_types = self._data_types_for_modality(camera_modality)
        self.goal_idx = np.zeros(self.num_envs, dtype=np.int32)
        self.max_goal_idx = np.zeros(self.num_envs, dtype=np.int32)
        self.near_goal_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.progress_buf = np.zeros(self.num_envs, dtype=np.int32)
        self.lifted_object = np.zeros(self.num_envs, dtype=bool)
        self.reset_reason_counts = {
            "object_z_low": 0,
            "time_limit": 0,
            "max_goals": 0,
            "hand_far": 0,
            "dropped_after_lift": 0,
        }
        self.goal_pose = np.repeat(self.task_spec.goals[0][None], self.num_envs, axis=0).astype(np.float32)
        self.current_start_pose = np.repeat(self.task_spec.start_pose[None], self.num_envs, axis=0).astype(np.float32)
        self.prev_targets: np.ndarray | None = None
        self.goal_pos_obs_noise = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.object_scale_noise_multiplier = np.ones((self.num_envs, 3), dtype=np.float32)
        self.camera_world_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.camera_world_quat_wxyz = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.camera_pos_jitter = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.camera_rot_jitter_deg = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.permutation: np.ndarray | None = None
        self.inverse_permutation: np.ndarray | None = None
        self.permutation_torch: torch.Tensor | None = None
        self.inverse_permutation_torch: torch.Tensor | None = None
        self.palm_body_idx: int | None = None
        self.fingertip_body_indices: torch.Tensor | None = None
        self.obs_queue: torch.Tensor | None = None
        self.action_queue: torch.Tensor | None = None
        self.object_state_queue: torch.Tensor | None = None
        self._obs_queue_needs_reset = np.ones(self.num_envs, dtype=bool)
        self._action_queue_needs_reset = np.ones(self.num_envs, dtype=bool)
        self._object_state_queue_needs_reset = np.ones(self.num_envs, dtype=bool)

        self._setup_scene()

    def _sample_visual_randomization(self, env_ids_np: np.ndarray):
        if not self.visual_dr_cfg.enabled:
            self.camera_pos_jitter[env_ids_np] = 0.0
            self.camera_rot_jitter_deg[env_ids_np] = 0.0
            return
        pos_range = np.asarray(self.visual_dr_cfg.camera_pos_jitter_m, dtype=np.float32)
        rot_range = np.asarray(self.visual_dr_cfg.camera_rot_jitter_deg, dtype=np.float32)
        self.camera_pos_jitter[env_ids_np] = np.random.uniform(-pos_range, pos_range, size=(len(env_ids_np), 3)).astype(np.float32)
        self.camera_rot_jitter_deg[env_ids_np] = np.random.uniform(-rot_range, rot_range, size=(len(env_ids_np), 3)).astype(np.float32)

    def _init_teacher_obs_queues(self):
        self.obs_queue = torch.zeros(
            self.num_envs, self.obs_delay_max, 140, dtype=torch.float32, device=self.device
        )
        self.action_queue = torch.zeros(
            self.num_envs, self.action_delay_max, self.action_dim, dtype=torch.float32, device=self.device
        )
        self.object_state_queue = torch.zeros(
            self.num_envs, self.object_state_delay_max, 13, dtype=torch.float32, device=self.device
        )

    def _update_queue(self, queue: torch.Tensor, current_values: torch.Tensor, needs_reset: np.ndarray) -> torch.Tensor:
        if queue is None:
            raise RuntimeError("Queue not initialized")
        if np.any(needs_reset):
            env_ids = torch.as_tensor(np.where(needs_reset)[0], device=self.device, dtype=torch.long)
            queue[env_ids] = current_values[env_ids].unsqueeze(1).repeat(1, queue.shape[1], 1)
        queue[:, 1:] = queue[:, :-1].clone()
        queue[:, 0] = current_values.clone()
        return queue

    def _sample_delta_quat_xyzw(self, input_quat_xyzw: torch.Tensor, delta_rotation_degrees: float) -> torch.Tensor:
        if delta_rotation_degrees <= 0.0:
            return input_quat_xyzw.clone()
        quat_wxyz = torch.cat((input_quat_xyzw[:, 3:], input_quat_xyzw[:, :3]), dim=1)
        quat_matrix = quaternion_to_matrix(quat_wxyz)
        delta_rotation_radians = delta_rotation_degrees * np.pi / 180.0
        random_direction = torch.randn((input_quat_xyzw.shape[0], 3), device=self.device, dtype=torch.float32)
        random_direction = random_direction / torch.clamp(torch.norm(random_direction, dim=1, keepdim=True), min=1e-8)
        sampled_rotation_magnitude = torch.empty((input_quat_xyzw.shape[0], 1), device=self.device, dtype=torch.float32).uniform_(
            -delta_rotation_radians, delta_rotation_radians
        )
        sampled_rotation_axis_angles = random_direction * sampled_rotation_magnitude
        sampled_rotation_matrix = _axis_angle_to_matrix(sampled_rotation_axis_angles)
        new_matrix = quat_matrix @ sampled_rotation_matrix
        new_quat_wxyz = matrix_to_quaternion(new_matrix)
        return torch.cat((new_quat_wxyz[:, 1:], new_quat_wxyz[:, 0:1]), dim=1).clone()

    def _camera_mount_mode(self) -> str:
        return getattr(self.camera_pose, "mount", "world")

    def _data_types_for_modality(self, modality: str) -> list[str]:
        need_depth_for_rgb_aug = (
            self.image_robustness.enabled
            and self.image_robustness.rgb_aug.enabled
            and self.image_robustness.rgb_aug.background_prob > 0.0
        )
        if modality == "depth":
            return ["distance_to_image_plane"]
        if modality == "rgb":
            return ["rgb", "distance_to_image_plane"] if need_depth_for_rgb_aug else ["rgb"]
        if modality == "rgbd":
            return ["rgb", "distance_to_image_plane"]
        raise ValueError(f"Unsupported student camera modality: {modality}")

    def _make_scene_cfg(self, sim_utils):
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass

        robot_joint_vel = {name: 0.0 for name in JOINT_NAMES_ISAACGYM}
        default_joint_pos = dict(DEFAULT_JOINT_POS)
        if self.start_arm_higher:
            # Match Isaac Gym eval mode (simtoolreal/env.py START_HIGHER path):
            # raise the arm slightly before rollout to improve the grasp approach.
            default_joint_pos["iiwa14_joint_2"] -= float(np.deg2rad(10.0))
            default_joint_pos["iiwa14_joint_4"] += float(np.deg2rad(10.0))
        if self.enable_camera:
            from isaaclab.sensors import CameraCfg, TiledCameraCfg

            camera_types = list(self.camera_data_types)
            camera_cfg_cls = TiledCameraCfg if self.camera_backend == "tiled" else CameraCfg
            camera_mount = self._camera_mount_mode()
            spawn_tiled_at_parent_pose = self.camera_backend == "tiled" and camera_mount in {"world", "wrist"}
            camera_offset_pos = self.camera_pose.pos if spawn_tiled_at_parent_pose else (0.0, 0.0, 0.0)
            camera_offset_rot = self.camera_pose.quat_wxyz if spawn_tiled_at_parent_pose else (1.0, 0.0, 0.0, 0.0)
            camera_offset_convention = self.camera_pose.convention if spawn_tiled_at_parent_pose else "ros"
            if self.camera_backend == "tiled" and camera_mount == "wrist":
                camera_link_name = self.camera_pose.link_name or "iiwa14_link_7"
                camera_prim_path = f"{{ENV_REGEX_NS}}/Robot/{camera_link_name}/DistillCamera"
            else:
                camera_prim_path = "{ENV_REGEX_NS}/DistillCamera"

            @configclass
            class DistillSceneCfg(InteractiveSceneCfg):
                robot = ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/Robot",
                    spawn=sim_utils.UrdfFileCfg(
                        asset_path=self.task_spec.robot_urdf,
                        usd_dir="/tmp/isaaclab_usd_cache_distill/robot",
                        force_usd_conversion=True,
                        make_instanceable=False,
                        fix_base=True,
                        merge_fixed_joints=True,
                        self_collision=False,
                        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                            drive_type="force",
                            target_type="position",
                            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                                stiffness=JOINT_STIFFNESSES_COMPENSATED,
                                damping=JOINT_DAMPINGS_COMPENSATED,
                            ),
                        ),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True,
                            max_depenetration_velocity=1000.0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.002,
                            rest_offset=0.0,
                        ),
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            enabled_self_collisions=False,
                            solver_position_iteration_count=8,
                            solver_velocity_iteration_count=0,
                        ),
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.8, 0.0),
                        joint_pos=default_joint_pos,
                        joint_vel=robot_joint_vel,
                    ),
                    actuators={
                        "arm": ImplicitActuatorCfg(
                            joint_names_expr=["iiwa14_joint_.*"],
                            stiffness={k: v for k, v in JOINT_STIFFNESSES.items() if k.startswith("iiwa14")},
                            damping={k: v for k, v in JOINT_DAMPINGS.items() if k.startswith("iiwa14")},
                        ),
                        "hand": ImplicitActuatorCfg(
                            joint_names_expr=["left_.*"],
                            stiffness={k: v for k, v in JOINT_STIFFNESSES.items() if k.startswith("left")},
                            damping={k: v for k, v in JOINT_DAMPINGS.items() if k.startswith("left")},
                        ),
                    },
                )

                table = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/Table",
                    spawn=sim_utils.UrdfFileCfg(
                        asset_path=self.task_spec.table_urdf,
                        usd_dir="/tmp/isaaclab_usd_cache_distill/table",
                        force_usd_conversion=True,
                        make_instanceable=False,
                        fix_base=True,
                        merge_fixed_joints=True,
                        joint_drive=None,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True,
                            linear_damping=0.0,
                            angular_damping=0.0,
                            max_depenetration_velocity=1000.0,
                            solver_position_iteration_count=8,
                            solver_velocity_iteration_count=0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.002,
                            rest_offset=0.0,
                        ),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38)),
                )

                object = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    spawn=sim_utils.UrdfFileCfg(
                        asset_path=self.task_spec.object_urdf,
                        usd_dir="/tmp/isaaclab_usd_cache_distill/object",
                        force_usd_conversion=True,
                        make_instanceable=False,
                        fix_base=False,
                        merge_fixed_joints=True,
                        joint_drive=None,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=False,
                            linear_damping=0.01,
                            angular_damping=0.01,
                            max_depenetration_velocity=1000.0,
                            solver_position_iteration_count=8,
                            solver_velocity_iteration_count=0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.002,
                            rest_offset=0.0,
                        ),
                    ),
                )

                camera = camera_cfg_cls(
                    prim_path=camera_prim_path,
                    update_period=0,
                    update_latest_camera_pose=True,
                    height=self.camera_intrinsics.height,
                    width=self.camera_intrinsics.width,
                    data_types=camera_types,
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=self.camera_intrinsics.focal_length,
                        focus_distance=self.camera_intrinsics.focus_distance,
                        horizontal_aperture=self.camera_intrinsics.horizontal_aperture,
                        clipping_range=self.camera_intrinsics.clipping_range,
                    ),
                    offset=camera_cfg_cls.OffsetCfg(
                        # Static world TiledCamera render products do not reliably
                        # pick up poses written after initialization, so spawn them
                        # at the per-env local camera pose. Tiled wrist cameras are
                        # spawned under the wrist link with the configured local
                        # offset so articulation motion drives the camera transform.
                        pos=camera_offset_pos,
                        rot=camera_offset_rot,
                        convention=camera_offset_convention,
                    ),
                )

                light = AssetBaseCfg(
                    prim_path="/World/DomeLight",
                    spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
                )
        else:
            @configclass
            class DistillSceneCfg(InteractiveSceneCfg):
                robot = ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/Robot",
                    spawn=sim_utils.UrdfFileCfg(
                        asset_path=self.task_spec.robot_urdf,
                        usd_dir="/tmp/isaaclab_usd_cache_distill/robot",
                        force_usd_conversion=True,
                        make_instanceable=False,
                        fix_base=True,
                        merge_fixed_joints=True,
                        self_collision=False,
                        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                            drive_type="force",
                            target_type="position",
                            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                                stiffness=JOINT_STIFFNESSES_COMPENSATED,
                                damping=JOINT_DAMPINGS_COMPENSATED,
                            ),
                        ),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True,
                            max_depenetration_velocity=1000.0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.002,
                            rest_offset=0.0,
                        ),
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            enabled_self_collisions=False,
                            solver_position_iteration_count=8,
                            solver_velocity_iteration_count=0,
                        ),
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.8, 0.0),
                        joint_pos=default_joint_pos,
                        joint_vel=robot_joint_vel,
                    ),
                    actuators={
                        "arm": ImplicitActuatorCfg(
                            joint_names_expr=["iiwa14_joint_.*"],
                            stiffness={k: v for k, v in JOINT_STIFFNESSES.items() if k.startswith("iiwa14")},
                            damping={k: v for k, v in JOINT_DAMPINGS.items() if k.startswith("iiwa14")},
                        ),
                        "hand": ImplicitActuatorCfg(
                            joint_names_expr=["left_.*"],
                            stiffness={k: v for k, v in JOINT_STIFFNESSES.items() if k.startswith("left")},
                            damping={k: v for k, v in JOINT_DAMPINGS.items() if k.startswith("left")},
                        ),
                    },
                )

                table = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/Table",
                    spawn=sim_utils.UrdfFileCfg(
                        asset_path=self.task_spec.table_urdf,
                        usd_dir="/tmp/isaaclab_usd_cache_distill/table",
                        force_usd_conversion=True,
                        make_instanceable=False,
                        fix_base=True,
                        merge_fixed_joints=True,
                        joint_drive=None,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True,
                            linear_damping=0.0,
                            angular_damping=0.0,
                            max_depenetration_velocity=1000.0,
                            solver_position_iteration_count=8,
                            solver_velocity_iteration_count=0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.002,
                            rest_offset=0.0,
                        ),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38)),
                )

                object = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    spawn=sim_utils.UrdfFileCfg(
                        asset_path=self.task_spec.object_urdf,
                        usd_dir="/tmp/isaaclab_usd_cache_distill/object",
                        force_usd_conversion=True,
                        make_instanceable=False,
                        fix_base=False,
                        merge_fixed_joints=True,
                        joint_drive=None,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=False,
                            linear_damping=0.01,
                            angular_damping=0.01,
                            max_depenetration_velocity=1000.0,
                            solver_position_iteration_count=8,
                            solver_velocity_iteration_count=0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.002,
                            rest_offset=0.0,
                        ),
                    ),
                )

                light = AssetBaseCfg(
                    prim_path="/World/DomeLight",
                    spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
                )

        return DistillSceneCfg

    def _setup_scene(self):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext

        # IsaacLab's default ground plane is 100m x 100m. With many envs at
        # env_spacing=4.0, far envs can render off-plane as a white background,
        # which corrupts camera policy inputs while physics still runs.
        ground_plane_cfg = sim_utils.GroundPlaneCfg(size=(self.ground_plane_size, self.ground_plane_size))
        ground_plane_cfg.func("/World/GroundPlane", ground_plane_cfg)
        sim_cfg = SimulationCfg(
            dt=PHYSICS_DT,
            render_interval=PHYSICS_SUBSTEPS,
            gravity=(0.0, 0.0, -9.81),
            enable_scene_query_support=self.enable_scene_query_support,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=DEFAULT_ASSET_FRICTION,
                dynamic_friction=DEFAULT_ASSET_FRICTION,
                restitution=0.0,
            ),
            physx=PhysxCfg(
                solver_type=1,
                bounce_threshold_velocity=0.2,
                friction_offset_threshold=0.04,
                friction_correlation_distance=0.025,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        _log(
            "SimulationContext created "
            f"(control_dt={CONTROL_DT:.6f}, physics_dt={PHYSICS_DT:.6f}, substeps={PHYSICS_SUBSTEPS}, "
            f"enable_scene_query_support={self.enable_scene_query_support})"
        )

        scene_cfg_type = self._make_scene_cfg(sim_utils)
        self.scene = InteractiveScene(
            scene_cfg_type(
                num_envs=self.num_envs,
                env_spacing=self.env_spacing,
                lazy_sensor_update=False,
                # Single-env debugging is easier and more faithful when we do
                # not replicate physics as instanced prims: it avoids a class
                # of "cannot modify collision properties on instanced prims"
                # issues during peg contact debugging. Multi-env runs still use
                # replication for performance.
                replicate_physics=self.num_envs > 1,
            )
        )
        self.robot = self.scene.articulations["robot"]
        self.object_rigid = self.scene.rigid_objects["object"]
        self.camera = self.scene.sensors["camera"] if self.enable_camera else None
        self.env_origins = self.scene.env_origins
        self.device = self.sim.device
        self.q_lower_limits_t = torch.tensor(Q_LOWER_LIMITS_np, dtype=torch.float32, device=self.device)
        self.q_upper_limits_t = torch.tensor(Q_UPPER_LIMITS_np, dtype=torch.float32, device=self.device)
        self.palm_offset_t = torch.tensor(PALM_OFFSET_np, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.fingertip_offsets_t = torch.tensor(FINGERTIP_OFFSETS_np, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.object_keypoint_offsets_t = (
            torch.tensor(OBJECT_KEYPOINT_OFFSETS_np, dtype=torch.float32, device=self.device)
            * (OBJECT_BASE_SIZE * KEYPOINT_SCALE / 2.0)
        ).unsqueeze(0)
        self._apply_physics_material_overrides(sim_utils)
        self.sim.reset()
        self.scene.reset()
        self._log_object_asset_diagnostics()
        self._validate_joint_ordering()
        if self.camera is not None:
            _log(
                "Distill camera created "
                f"(backend={self.camera_backend}, instances={self.camera.num_instances}, modality={self.camera_modality}, pos={self.camera_pose.pos}, "
                f"quat_wxyz={self.camera_pose.quat_wxyz}, convention={self.camera_pose.convention})"
            )
        else:
            _log("Distill camera disabled for this run")
        self._init_teacher_obs_queues()
        self.reset()

    def _validate_joint_ordering(self):
        sim_names = list(self.robot.joint_names)
        self.palm_body_idx = self.robot.body_names.index("iiwa14_link_7")
        self.fingertip_body_indices = torch.tensor(
            [self.robot.body_names.index(name) for name in FINGERTIP_LINK_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        if sim_names == JOINT_NAMES_ISAACGYM:
            self.permutation = None
            self.inverse_permutation = None
            self.permutation_torch = None
            self.inverse_permutation_torch = None
            _log("Distill joint ordering matches Isaac Gym")
            return
        self.permutation = np.array([sim_names.index(name) for name in JOINT_NAMES_ISAACGYM], dtype=np.int32)
        self.inverse_permutation = np.zeros_like(self.permutation)
        for gym_idx, sim_idx in enumerate(self.permutation):
            self.inverse_permutation[sim_idx] = gym_idx
        self.permutation_torch = torch.tensor(self.permutation, dtype=torch.long, device=self.device)
        self.inverse_permutation_torch = torch.tensor(self.inverse_permutation, dtype=torch.long, device=self.device)
        _log(f"Distill joint permutation set: {self.permutation}")

    def _sim_to_gym_order(self, values: np.ndarray) -> np.ndarray:
        if self.permutation is None:
            return values
        return values[:, self.permutation]

    def _gym_to_sim_order(self, values: np.ndarray) -> np.ndarray:
        if self.permutation is None:
            return values
        reordered = np.zeros_like(values)
        reordered[:, self.permutation] = values
        return reordered

    def _iter_collision_prim_paths(self, root_prim_path: str) -> list[str]:
        from isaaclab.sim.utils import get_current_stage

        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim.IsValid():
            return []

        collision_paths: list[str] = []
        prims_to_visit = [root_prim]
        while prims_to_visit:
            prim = prims_to_visit.pop()
            if prim.GetName() == "collisions":
                collision_paths.append(str(prim.GetPath()))
                descendant_stack = list(prim.GetChildren())
                while descendant_stack:
                    child = descendant_stack.pop()
                    collision_paths.append(str(child.GetPath()))
                    descendant_stack.extend(list(child.GetChildren()))
            prims_to_visit.extend(list(prim.GetChildren()))
        return list(dict.fromkeys(collision_paths))

    def _apply_material_to_rigid_prims(self, material_path: str, prim_paths: list[str]):
        from isaaclab.sim.utils import bind_physics_material

        raw_bind_physics_material = getattr(bind_physics_material, "__wrapped__", bind_physics_material)
        for prim_path in prim_paths:
            raw_bind_physics_material(prim_path, material_path)

    def _apply_physics_material_overrides(self, sim_utils):
        all_rigid_prim_paths: list[str] = []
        fingertip_rigid_prim_paths: list[str] = []
        for env_id in range(self.num_envs):
            env_ns = f"/World/envs/env_{env_id}"
            robot_root = f"{env_ns}/Robot"
            table_root = f"{env_ns}/Table"
            object_root = f"{env_ns}/Object"
            for root in (robot_root, table_root, object_root):
                for collision_path in self._iter_collision_prim_paths(root):
                    if "/collisions" in collision_path:
                        all_rigid_prim_paths.append(collision_path.split("/collisions", 1)[0])
            for link_name in FINGERTIP_LINK_NAMES:
                fingertip_rigid_prim_paths.append(f"{robot_root}/{link_name}")

        all_rigid_prim_paths = list(dict.fromkeys(all_rigid_prim_paths))
        fingertip_rigid_prim_paths = list(dict.fromkeys(fingertip_rigid_prim_paths))

        default_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.default_asset_friction,
            dynamic_friction=self.default_asset_friction,
            restitution=0.0,
        )
        default_material_path = "/World/PhysicsMaterials/distill_default_asset_material"
        default_material_cfg.func(default_material_path, default_material_cfg)
        self._apply_material_to_rigid_prims(default_material_path, all_rigid_prim_paths)

        fingertip_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.fingertip_friction,
            dynamic_friction=self.fingertip_friction,
            restitution=0.0,
        )
        fingertip_material_path = "/World/PhysicsMaterials/distill_fingertip_material"
        fingertip_material_cfg.func(fingertip_material_path, fingertip_material_cfg)
        self._apply_material_to_rigid_prims(fingertip_material_path, fingertip_rigid_prim_paths)
        _log(
            "Distill physics materials applied "
            f"(default_friction={self.default_asset_friction}, fingertip_friction={self.fingertip_friction}, "
            f"rigid_prims={len(all_rigid_prim_paths)}, offsets_via_spawn_cfg=True)"
        )

    def _log_object_asset_diagnostics(self):
        from isaaclab.sim.utils import get_current_stage
        from pxr import Usd, UsdPhysics

        stage = get_current_stage()
        object_root = stage.GetPrimAtPath("/World/envs/env_0/Object")
        if not object_root.IsValid():
            _log("Object diagnostics skipped: /World/envs/env_0/Object not found")
            return

        collision_paths = self._iter_collision_prim_paths("/World/envs/env_0/Object")
        _log(f"Object diagnostics: collision_prim_count={len(collision_paths)}")

        for prim in Usd.PrimRange(object_root):
            prim_path = str(prim.GetPath())
            mass_api = UsdPhysics.MassAPI(prim)
            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            if not mass_api and not rigid_api:
                continue
            mass_attr = mass_api.GetMassAttr() if mass_api else None
            com_attr = mass_api.GetCenterOfMassAttr() if mass_api else None
            inertia_attr = mass_api.GetDiagonalInertiaAttr() if mass_api else None
            principal_axes_attr = mass_api.GetPrincipalAxesAttr() if mass_api else None
            mass_val = mass_attr.Get() if mass_attr and mass_attr.HasAuthoredValueOpinion() else None
            com_val = com_attr.Get() if com_attr and com_attr.HasAuthoredValueOpinion() else None
            inertia_val = (
                inertia_attr.Get() if inertia_attr and inertia_attr.HasAuthoredValueOpinion() else None
            )
            axes_val = (
                principal_axes_attr.Get()
                if principal_axes_attr and principal_axes_attr.HasAuthoredValueOpinion()
                else None
            )
            _log(
                "Object mass diagnostics: "
                f"prim={prim_path}, mass={mass_val}, com={com_val}, inertia={inertia_val}, principal_axes={axes_val}"
            )

    def _iter_gprims(self, root_prim_path: str):
        from isaaclab.sim.utils import get_current_stage
        from pxr import Usd, UsdGeom

        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim.IsValid():
            return []
        gprims = []
        for prim in Usd.PrimRange(root_prim):
            gprim = UsdGeom.Gprim(prim)
            if gprim:
                gprims.append(gprim)
        return gprims

    def _apply_display_color(self, root_prim_path: str, rgb: tuple[float, float, float]):
        from pxr import Gf

        for gprim in self._iter_gprims(root_prim_path):
            try:
                gprim.GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
            except Exception:
                continue

    def _apply_visual_randomization(self, env_ids_np: np.ndarray):
        if not self.visual_dr_cfg.enabled:
            return
        from isaaclab.sim.utils import get_current_stage
        from pxr import Gf

        stage = get_current_stage()
        if self.visual_dr_cfg.dome_light_randomization:
            dome = stage.GetPrimAtPath("/World/DomeLight")
            if dome.IsValid():
                intensity = float(np.random.uniform(*self.visual_dr_cfg.dome_light_intensity_range))
                dome.GetAttribute("inputs:intensity").Set(intensity)
                base = np.array([0.9, 0.9, 0.9], dtype=np.float32)
                jitter = np.random.uniform(
                    -self.visual_dr_cfg.dome_light_color_jitter,
                    self.visual_dr_cfg.dome_light_color_jitter,
                    size=3,
                ).astype(np.float32)
                color = np.clip(base + jitter, 0.2, 1.0)
                dome.GetAttribute("inputs:color").Set(Gf.Vec3f(*[float(x) for x in color]))
        if not self.visual_dr_cfg.material_color_randomization:
            return
        for env_id in env_ids_np:
            env_ns = f"/World/envs/env_{env_id}"
            obj_val = tuple(float(x) for x in np.random.uniform(*self.visual_dr_cfg.object_color_range, size=3))
            table_val = tuple(float(x) for x in np.random.uniform(*self.visual_dr_cfg.table_color_range, size=3))
            robot_val = tuple(float(x) for x in np.random.uniform(*self.visual_dr_cfg.robot_color_range, size=3))
            self._apply_display_color(f"{env_ns}/Object", obj_val)
            self._apply_display_color(f"{env_ns}/Table", table_val)
            self._apply_display_color(f"{env_ns}/Robot", robot_val)

    def _apply_camera_world_poses(self, env_ids: torch.Tensor | None = None):
        if self.camera is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        env_ids_np = env_ids.detach().cpu().numpy()
        if self._camera_mount_mode() == "wrist":
            positions, orientations = self._compute_wrist_camera_world_poses(env_ids_np)
        else:
            env_origins = self.env_origins[env_ids].detach().cpu().numpy()
            base_pos = np.asarray(self.camera_pose.pos, dtype=np.float32)[None, :]
            positions = env_origins + base_pos + self.camera_pos_jitter[env_ids_np]
            base_rot = R.from_quat(np.asarray(self.camera_pose.quat_wxyz, dtype=np.float32)[[1, 2, 3, 0]])
            jitter = R.from_euler("xyz", self.camera_rot_jitter_deg[env_ids_np], degrees=True)
            orientations_xyzw = (jitter * base_rot).as_quat().astype(np.float32)
            orientations = orientations_xyzw[:, [3, 0, 1, 2]]
        if self.camera_backend == "tiled":
            self.camera_world_pos[env_ids_np] = positions
            self.camera_world_quat_wxyz[env_ids_np] = orientations
            return
        self.camera.set_world_poses(
            positions=positions,
            orientations=orientations,
            env_ids=env_ids,
            convention=self.camera_pose.convention,
        )
        self.camera_world_pos[env_ids_np] = positions
        self.camera_world_quat_wxyz[env_ids_np] = orientations

    def _compute_wrist_camera_world_poses(self, env_ids_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        link_name = self.camera_pose.link_name or "iiwa14_link_7"
        link_idx = self.robot.body_names.index(link_name)
        body_state = self.robot.data.body_state_w[env_ids_np, link_idx]
        link_pos = body_state[:, :3].detach().cpu().numpy()
        link_quat_wxyz = body_state[:, 3:7].detach().cpu().numpy()
        link_rot = R.from_quat(link_quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()

        cam_offset = np.asarray(self.camera_pose.pos, dtype=np.float32)[None, :] + self.camera_pos_jitter[env_ids_np]
        cam_pos = link_pos + np.einsum("nij,nj->ni", link_rot, cam_offset)

        rel_quat_wxyz = np.asarray(self.camera_pose.quat_wxyz, dtype=np.float32)
        rel_rot = R.from_quat(rel_quat_wxyz[[1, 2, 3, 0]]).as_matrix()
        jitter_rot = R.from_euler("xyz", self.camera_rot_jitter_deg[env_ids_np], degrees=True).as_matrix()
        cam_rot = link_rot @ rel_rot[None, :, :] @ jitter_rot
        cam_quat_xyzw = matrix_to_quaternion_xyzw_scipy(cam_rot)
        cam_quat_wxyz = cam_quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
        return cam_pos.astype(np.float32), cam_quat_wxyz

    def _sample_start_poses(self) -> np.ndarray:
        start = np.repeat(self.task_spec.start_pose[None], self.num_envs, axis=0).astype(np.float32)
        if self.object_start_mode == "fixed":
            return start
        if self.object_start_mode != "randomized":
            raise ValueError(f"Unsupported object_start_mode: {self.object_start_mode}")

        noise = (np.random.uniform(-1.0, 1.0, size=(self.num_envs, 3)).astype(np.float32) * self.object_pos_noise_xyz[None])
        start[:, :3] += noise
        yaw_noise = np.deg2rad(
            np.random.uniform(-self.object_yaw_noise_deg, self.object_yaw_noise_deg, size=self.num_envs).astype(np.float32)
        )
        for i in range(self.num_envs):
            base = R.from_quat(start[i, 3:7])
            yaw = R.from_euler("z", float(yaw_noise[i]))
            start[i, 3:7] = (yaw * base).as_quat().astype(np.float32)
        start[:, 2] = np.maximum(start[:, 2], self.task_spec.start_pose[2] - self.object_pos_noise_xyz[2])
        return start

    def _local_pose_to_world_pose(self, local_pose_xyzw: np.ndarray) -> torch.Tensor:
        world_pose = np.zeros((self.num_envs, 7), dtype=np.float32)
        world_pose[:, :3] = local_pose_xyzw[:, :3] + self.env_origins.cpu().numpy()
        for i in range(self.num_envs):
            world_pose[i, 3:] = xyzw_to_wxyz(local_pose_xyzw[i, 3:])
        return torch.tensor(world_pose, dtype=torch.float32, device=self.device)

    def reset(self):
        for key in self.reset_reason_counts:
            self.reset_reason_counts[key] = 0
        self.max_goal_idx[:] = 0
        self._reset_envs(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def _reset_envs(self, env_ids: torch.Tensor, reset_reasons: dict[str, np.ndarray] | None = None):
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        env_ids_np = env_ids.detach().cpu().numpy()
        if reset_reasons is not None:
            for key, mask in reset_reasons.items():
                if key in self.reset_reason_counts:
                    self.reset_reason_counts[key] += int(np.count_nonzero(mask))
        root_state = self.robot.data.default_root_state.clone()
        root_state[env_ids, :3] += self.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_state[env_ids, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_state[env_ids, 7:], env_ids=env_ids)
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos[env_ids], env_ids=env_ids)

        start_poses = self._sample_start_poses()[env_ids_np]
        self.current_start_pose[env_ids_np] = start_poses
        self._sample_visual_randomization(env_ids_np)
        object_pose_w = self._local_pose_to_world_pose(self.current_start_pose)
        zeros_vel = torch.zeros((len(env_ids_np), 6), dtype=torch.float32, device=self.device)
        self.object_rigid.write_root_pose_to_sim(object_pose_w[env_ids], env_ids=env_ids)
        self.object_rigid.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)
        self._apply_camera_world_poses(env_ids)
        self._apply_visual_randomization(env_ids_np)

        self.scene.write_data_to_sim()
        self.sim.step(render=not self.headless)
        self.scene.update(PHYSICS_DT)
        if self.camera is not None:
            self.sim.render()
        if self.camera is not None:
            self.camera.update(0.0, force_recompute=True)
        joint_pos_sim = self.robot.data.joint_pos.detach().cpu().numpy().copy()
        if self.prev_targets is None:
            self.prev_targets = self._sim_to_gym_order(joint_pos_sim)
        else:
            self.prev_targets[env_ids_np] = self._sim_to_gym_order(joint_pos_sim)[env_ids_np]
        self.goal_idx[env_ids_np] = 0
        self.near_goal_steps[env_ids_np] = 0
        self.progress_buf[env_ids_np] = 0
        self.lifted_object[env_ids_np] = False
        self.goal_pose[env_ids_np] = np.repeat(self.task_spec.goals[0][None], len(env_ids_np), axis=0).astype(np.float32)
        if self.goal_xy_obs_noise > 0.0:
            self.goal_pos_obs_noise[env_ids_np, 0:2] = np.random.uniform(
                -self.goal_xy_obs_noise,
                self.goal_xy_obs_noise,
                size=(len(env_ids_np), 2),
            ).astype(np.float32)
        else:
            self.goal_pos_obs_noise[env_ids_np] = 0.0
        noise_min, noise_max = self.object_scale_noise_multiplier_range
        if noise_min == 1.0 and noise_max == 1.0:
            self.object_scale_noise_multiplier[env_ids_np] = 1.0
        else:
            self.object_scale_noise_multiplier[env_ids_np] = np.random.uniform(
                noise_min,
                noise_max,
                size=(len(env_ids_np), 3),
            ).astype(np.float32)
        self._obs_queue_needs_reset[env_ids_np] = True
        self._action_queue_needs_reset[env_ids_np] = True
        self._object_state_queue_needs_reset[env_ids_np] = True
        self._post_reset_zero_action_step(env_ids_np)

    def _post_reset_zero_action_step(self, env_ids_np: np.ndarray):
        zero_actions = np.zeros((len(env_ids_np), self.action_dim), dtype=np.float32)
        prev_targets_subset = self.prev_targets[env_ids_np].astype(np.float32)
        zero_targets = compute_joint_pos_targets(
            actions=zero_actions,
            prev_targets=prev_targets_subset,
            hand_moving_average=self.hand_moving_average,
            arm_moving_average=self.arm_moving_average,
            hand_dof_speed_scale=self.dof_speed_scale,
            dt=CONTROL_DT,
        )
        full_targets = self.prev_targets.copy()
        full_targets[env_ids_np] = zero_targets
        self.apply_action(full_targets)
        self.step(render=not self.headless)

    def apply_action(self, targets: np.ndarray):
        self.prev_targets = targets.copy()
        targets_sim = self._gym_to_sim_order(targets)
        targets_t = torch.tensor(targets_sim, dtype=torch.float32, device=self.device)
        self.robot.set_joint_position_target(targets_t)

    def delay_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_queue is None:
            raise RuntimeError("Action queue not initialized")
        self._update_queue(self.action_queue, actions, self._action_queue_needs_reset)
        self._action_queue_needs_reset[:] = False
        if not self.use_action_delay:
            return actions
        delay_index = torch.randint(
            0, self.action_queue.shape[1], (self.num_envs,), device=self.device
        )
        return self.action_queue[torch.arange(self.num_envs, device=self.device), delay_index].clone()

    def step(self, render: bool = True):
        self.scene.write_data_to_sim()
        for _ in range(PHYSICS_SUBSTEPS):
            self.sim.step(render=render)
        self.scene.update(PHYSICS_DT)
        self.progress_buf += 1
        if self.camera is not None and self._camera_mount_mode() == "wrist":
            self._apply_camera_world_poses()
            if self.camera_backend != "tiled":
                # Standard cameras are moved explicitly; push those dynamic wrist
                # poses before forcing a render/update.
                self.scene.write_data_to_sim()
        if self.camera is not None:
            self.sim.render()
            self.camera.update(PHYSICS_DT, force_recompute=True)

    def compute_sim_state(self) -> SimState:
        q = self._sim_to_gym_order(self.robot.data.joint_pos.detach().cpu().numpy())
        qd = self._sim_to_gym_order(self.robot.data.joint_vel.detach().cpu().numpy())
        object_pos_w = self.object_rigid.data.root_pos_w.detach().cpu().numpy()
        object_quat_w = self.object_rigid.data.root_quat_w.detach().cpu().numpy()

        object_pose = np.zeros((self.num_envs, 7), dtype=np.float32)
        # Keep object pose env-local by subtracting scene env origins. This matches the
        # DEXTRAH env convention:
        #   object_pos = self.object.data.root_pos_w - self.scene.env_origins
        # and is the frame used for aux_info["object_pos"] in their distillation code.
        object_pose[:, :3] = object_pos_w - self.env_origins.cpu().numpy()
        object_pose[:, 3:] = object_quat_w[:, [1, 2, 3, 0]]

        object_kps = _compute_keypoint_positions(object_pose, self.success_object_scales_batch)
        goal_kps = _compute_keypoint_positions(self.goal_pose, self.success_object_scales_batch)
        kp_dist = np.max(np.linalg.norm(object_kps - goal_kps, axis=-1), axis=-1).astype(np.float32)
        return SimState(
            q=q.copy(),
            qd=qd.copy(),
            object_pose=object_pose.copy(),
            goal_pose=self.goal_pose.copy(),
            goal_idx=self.goal_idx.copy(),
            kp_dist=kp_dist.copy(),
            near_goal_steps=self.near_goal_steps.copy(),
            # This is the DEXTRAH-style auxiliary target frame: env-local world position,
            # not camera-relative and not robot-relative.
            object_pos_world_env_frame=object_pose[:, :3].copy(),
        )

    def capture_viewer_frame(self, env_id: int, sim_state: SimState | None = None) -> dict[str, np.ndarray | list[str] | int | float]:
        if env_id < 0 or env_id >= self.num_envs:
            raise ValueError(f"viewer env_id {env_id} out of range for num_envs={self.num_envs}")
        if sim_state is None:
            sim_state = self.compute_sim_state()
        env_origin = self.env_origins[env_id].detach().cpu().numpy()
        body_state = self.robot.data.body_state_w[env_id].detach().cpu().numpy()
        body_positions = body_state[:, :3] - env_origin[None, :]
        robot_root_state = self.robot.data.root_state_w[env_id].detach().cpu().numpy()
        robot_base_pose = np.zeros(7, dtype=np.float32)
        robot_base_pose[:3] = robot_root_state[:3] - env_origin
        robot_base_pose[3:] = robot_root_state[3:7][[1, 2, 3, 0]]
        table_pose = self.task_spec.table_pose.astype(np.float32)
        return {
            "env_id": int(env_id),
            "robot_joint_names": list(self.robot.joint_names),
            "robot_joint_pos": self.robot.data.joint_pos[env_id].detach().cpu().numpy().astype(np.float32),
            "robot_body_names": list(self.robot.body_names),
            "robot_body_positions": body_positions.astype(np.float32),
            "robot_base_pose": robot_base_pose.astype(np.float32),
            "object_pose": sim_state.object_pose[env_id].astype(np.float32),
            "goal_pose": sim_state.goal_pose[env_id].astype(np.float32),
            "table_pose": table_pose,
            "goal_idx": int(sim_state.goal_idx[env_id]),
            "kp_dist": float(sim_state.kp_dist[env_id]),
        }

    def maybe_advance_goal(self, sim_state: SimState) -> bool:
        lifted_now = sim_state.object_pose[:, 2] > (self.current_start_pose[:, 2] + 0.1)
        self.lifted_object |= lifted_now
        for env_id in range(self.num_envs):
            if self.goal_idx[env_id] >= len(self.task_spec.goals):
                continue
            if sim_state.kp_dist[env_id] < self.task_spec.keypoint_tolerance:
                self.near_goal_steps[env_id] += 1
            else:
                self.near_goal_steps[env_id] = 0
            if self.near_goal_steps[env_id] < self.task_spec.success_steps:
                continue
            self.goal_idx[env_id] += 1
            self.max_goal_idx[env_id] = max(self.max_goal_idx[env_id], self.goal_idx[env_id])
            self.near_goal_steps[env_id] = 0
            if self.goal_idx[env_id] < len(self.task_spec.goals):
                self.goal_pose[env_id] = self.task_spec.goals[self.goal_idx[env_id]]
        return bool(np.all(self.goal_idx >= len(self.task_spec.goals)))

    def compute_reset_masks(self, sim_state: SimState) -> dict[str, np.ndarray]:
        object_z_low = sim_state.object_pose[:, 2] < 0.1
        time_limit = self.progress_buf >= self.episode_length - 1
        max_goals = self.goal_idx >= len(self.task_spec.goals)
        hand_far = self._compute_hand_far_mask()
        dropped_after_lift = (
            (sim_state.object_pose[:, 2] < self.current_start_pose[:, 2])
            & self.lifted_object
            if self.reset_when_dropped
            else np.zeros(self.num_envs, dtype=bool)
        )
        return {
            "object_z_low": object_z_low,
            "time_limit": time_limit,
            "max_goals": max_goals,
            "hand_far": hand_far,
            "dropped_after_lift": dropped_after_lift,
        }

    def _compute_hand_far_mask(self) -> np.ndarray:
        body_state = self.robot.data.body_state_w
        fingertip_state = body_state[:, self.fingertip_body_indices]
        fingertip_pos = fingertip_state[:, :, :3] - self.env_origins.unsqueeze(1)
        object_pos = self.object_rigid.data.root_pos_w - self.env_origins
        distances = torch.linalg.norm(fingertip_pos - object_pos.unsqueeze(1), dim=-1)
        return (torch.max(distances, dim=-1).values > 1.5).detach().cpu().numpy()

    def reset_done_envs(self, sim_state: SimState) -> np.ndarray:
        reset_masks = self.compute_reset_masks(sim_state)
        done_mask = np.zeros(self.num_envs, dtype=bool)
        for mask in reset_masks.values():
            done_mask |= mask
        done_env_ids = np.where(done_mask)[0].astype(np.int64)
        if done_env_ids.size == 0:
            return done_env_ids
        reason_subset = {key: mask[done_env_ids] for key, mask in reset_masks.items()}
        self._reset_envs(torch.tensor(done_env_ids, device=self.device, dtype=torch.long), reset_reasons=reason_subset)
        return done_env_ids

    def task_finished(self, sim_state: SimState) -> bool:
        return bool(np.all(self.goal_idx >= len(self.task_spec.goals)))

    def compute_progress_metrics(self, sim_state: SimState) -> dict[str, float]:
        progress_goal_idx = np.maximum(self.goal_idx, self.max_goal_idx)
        goal_completion = progress_goal_idx.astype(np.float32) / len(self.task_spec.goals)
        return {
            "goal_idx": float(np.mean(progress_goal_idx)),
            "goal_completion_ratio": float(np.mean(goal_completion)),
            "kp_dist": float(np.mean(sim_state.kp_dist)),
            "near_goal_steps": float(np.mean(self.near_goal_steps)),
        }

    def reset_dropped_envs(self, sim_state: SimState) -> np.ndarray:
        reset_masks = {"object_z_low": sim_state.object_pose[:, 2] < 0.1}
        dropped_env_ids = np.where(reset_masks["object_z_low"])[0].astype(np.int64)
        if dropped_env_ids.size == 0:
            return dropped_env_ids
        _log(f"Resetting dropped envs: {dropped_env_ids.tolist()}")
        self._reset_envs(
            torch.tensor(dropped_env_ids, device=self.device, dtype=torch.long),
            reset_reasons={"object_z_low": reset_masks["object_z_low"][dropped_env_ids]},
        )
        return dropped_env_ids

    def build_teacher_obs(self, sim_state: SimState) -> np.ndarray:
        if self.prev_targets is None:
            raise RuntimeError("Environment must be reset before building teacher observations")
        if self.obs_queue is None or self.object_state_queue is None:
            raise RuntimeError("Teacher observation queues not initialized")
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        if self.permutation_torch is not None:
            joint_pos = joint_pos[:, self.permutation_torch]
            joint_vel = joint_vel[:, self.permutation_torch]
        joint_pos_unscaled = unscale(joint_pos, self.q_lower_limits_t, self.q_upper_limits_t)

        body_state = self.robot.data.body_state_w
        env_origins = self.env_origins

        palm_state = body_state[:, self.palm_body_idx]
        palm_pos = palm_state[:, :3] - env_origins
        palm_quat_wxyz = palm_state[:, 3:7]
        palm_quat_xyzw = palm_quat_wxyz[:, [1, 2, 3, 0]]
        palm_center_pos = palm_pos + quat_rotate(palm_quat_xyzw, self.palm_offset_t)

        fingertip_state = body_state[:, self.fingertip_body_indices]
        fingertip_pos = fingertip_state[:, :, :3] - env_origins.unsqueeze(1)
        fingertip_quat_xyzw = fingertip_state[:, :, 3:7][:, :, [1, 2, 3, 0]]
        fingertip_pos_offset = fingertip_pos + quat_rotate(
            fingertip_quat_xyzw.reshape(-1, 4),
            self.fingertip_offsets_t.reshape(-1, 3),
        ).reshape(self.num_envs, len(FINGERTIP_LINK_NAMES), 3)
        fingertip_pos_rel_palm = fingertip_pos_offset - palm_center_pos.unsqueeze(1)

        object_root_state = self.object_rigid.data.root_state_w
        object_pos = object_root_state[:, :3] - env_origins
        object_quat_xyzw = object_root_state[:, 3:7][:, [1, 2, 3, 0]]
        object_linvel = object_root_state[:, 7:10]
        object_angvel = object_root_state[:, 10:13]
        object_state = torch.cat((object_pos, object_quat_xyzw, object_linvel, object_angvel), dim=-1)
        self.object_state_queue = self._update_queue(
            self.object_state_queue,
            object_state,
            self._object_state_queue_needs_reset,
        )
        self._object_state_queue_needs_reset[:] = False
        observed_object_state = object_state.clone()
        if self.use_object_state_delay_noise:
            delay_index = torch.randint(
                0,
                self.object_state_queue.shape[1],
                (self.num_envs,),
                device=self.device,
            )
            observed_object_state = self.object_state_queue[
                torch.arange(self.num_envs, device=self.device),
                delay_index,
            ].clone()
            if self.object_state_xyz_noise_std > 0.0:
                observed_object_state[:, :3] += (
                    torch.randn_like(observed_object_state[:, :3]) * self.object_state_xyz_noise_std
                )
            observed_object_state[:, 3:7] = self._sample_delta_quat_xyzw(
                observed_object_state[:, 3:7],
                self.object_state_rotation_noise_degrees,
            )
        observed_object_pos = observed_object_state[:, :3]
        observed_object_quat_xyzw = observed_object_state[:, 3:7]

        goal_pose_t = torch.tensor(self.goal_pose, dtype=torch.float32, device=self.device)
        object_scales_t = torch.tensor(self.object_scales_batch, dtype=torch.float32, device=self.device)
        object_scale_noise_multiplier_t = torch.tensor(
            self.object_scale_noise_multiplier,
            dtype=torch.float32,
            device=self.device,
        )
        object_scales_obs_t = object_scales_t * object_scale_noise_multiplier_t
        object_keypoint_offsets_obs = (
            self.object_keypoint_offsets_t.repeat(self.num_envs, 1, 1) * object_scales_obs_t.unsqueeze(1)
        )
        goal_keypoint_pos = goal_pose_t[:, :3].unsqueeze(1) + quat_rotate(
            goal_pose_t[:, 3:7].unsqueeze(1).repeat(1, object_keypoint_offsets_obs.shape[1], 1).reshape(-1, 4),
            object_keypoint_offsets_obs.reshape(-1, 3),
        ).reshape(self.num_envs, object_keypoint_offsets_obs.shape[1], 3)
        observed_obj_keypoint_pos = observed_object_pos.unsqueeze(1) + quat_rotate(
            observed_object_quat_xyzw.unsqueeze(1).repeat(1, object_keypoint_offsets_obs.shape[1], 1).reshape(-1, 4),
            object_keypoint_offsets_obs.reshape(-1, 3),
        ).reshape(self.num_envs, object_keypoint_offsets_obs.shape[1], 3)
        keypoints_rel_palm = observed_obj_keypoint_pos - palm_center_pos.unsqueeze(1)
        keypoints_rel_goal = observed_obj_keypoint_pos - goal_keypoint_pos
        if self.goal_xy_obs_noise > 0.0:
            goal_noise = torch.tensor(self.goal_pos_obs_noise, dtype=torch.float32, device=self.device)
            keypoints_rel_goal = keypoints_rel_goal - goal_noise.unsqueeze(1)

        prev_targets_t = torch.tensor(self.prev_targets, dtype=torch.float32, device=self.device)
        joint_vel_obs = joint_vel.clone()
        if self.joint_velocity_obs_noise_std > 0.0:
            joint_vel_obs = joint_vel_obs + (
                torch.randn_like(joint_vel_obs) * self.joint_velocity_obs_noise_std
            )
        obs_dict = {
            "joint_pos": joint_pos_unscaled,
            "joint_vel": joint_vel_obs,
            "prev_action_targets": prev_targets_t,
            "palm_pos": palm_center_pos,
            "palm_rot": palm_quat_xyzw,
            "object_rot": observed_object_quat_xyzw,
            "fingertip_pos_rel_palm": fingertip_pos_rel_palm.reshape(self.num_envs, -1),
            "keypoints_rel_palm": keypoints_rel_palm.reshape(self.num_envs, -1),
            "keypoints_rel_goal": keypoints_rel_goal.reshape(self.num_envs, -1),
            "object_scales": object_scales_obs_t,
        }
        obs = torch.cat([obs_dict[key] for key in OBS_LIST], dim=-1)
        self.obs_queue = self._update_queue(self.obs_queue, obs, self._obs_queue_needs_reset)
        self._obs_queue_needs_reset[:] = False
        obs_out = obs
        if self.use_obs_delay:
            delay_index = torch.randint(
                0,
                self.obs_queue.shape[1],
                (self.num_envs,),
                device=self.device,
            )
            obs_out = self.obs_queue[
                torch.arange(self.num_envs, device=self.device),
                delay_index,
            ].clone()
        return obs_out.detach().cpu().numpy()

    def _read_camera_outputs(self) -> dict[str, torch.Tensor]:
        if self.camera is None:
            raise RuntimeError("Camera is disabled for this environment")
        outputs = self.camera.data.output
        available = {key: value for key, value in outputs.items() if value is not None}
        if not available:
            raise RuntimeError("Camera produced no outputs")
        return available

    def _preprocess_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.float()
        if self.depth_preprocess_mode == "clip_divide":
            return torch.clamp(depth, self.depth_min_m, self.depth_max_m) / self.depth_max_m
        valid = (depth >= self.depth_min_m) & (depth <= self.depth_max_m)
        if self.depth_preprocess_mode == "metric":
            return torch.where(valid, depth, torch.zeros_like(depth))
        normalized = (depth - self.depth_min_m) / (self.depth_max_m - self.depth_min_m)
        return torch.where(valid, normalized, torch.zeros_like(depth))

    def build_student_obs(self, sim_state: SimState, camera_modality: str | None = None) -> dict[str, torch.Tensor]:
        modality = camera_modality or self.camera_modality
        outputs = self._read_camera_outputs()
        image_dict: dict[str, torch.Tensor] = {}
        depth = outputs.get("distance_to_image_plane")
        if depth is not None:
            if depth.dim() == 4 and depth.shape[-1] == 1:
                depth = depth.permute(0, 3, 1, 2)
            elif depth.dim() == 3:
                depth = depth.unsqueeze(1)
            image_dict["depth"] = self._preprocess_depth(depth)
        if modality in ("depth", "rgbd"):
            if "depth" not in image_dict:
                raise RuntimeError(f"Depth output missing. Available camera outputs: {list(outputs.keys())}")
        if modality in ("rgb", "rgbd"):
            rgb = outputs.get("rgb")
            if rgb is None:
                raise RuntimeError(f"RGB output missing. Available camera outputs: {list(outputs.keys())}")
            rgb = rgb[..., :3].permute(0, 3, 1, 2).float() / 255.0
            image_dict["rgb"] = rgb

        progress = (self.goal_idx.astype(np.float32) / len(self.task_spec.goals))[:, None]
        proprio = torch.from_numpy(
            np.concatenate(
                [
                    sim_state.q.astype(np.float32),
                    sim_state.qd.astype(np.float32),
                    self.prev_targets.astype(np.float32),
                    progress.astype(np.float32),
                ],
                axis=1,
            )
        ).to(self.device)
        return {
            "images": image_dict,
            "proprio": proprio,
        }

    def close(self):
        self.scene = None
        self.camera = None
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

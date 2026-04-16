from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from isaacgymenvs.utils.observation_action_utils_sharpa import (
    _compute_keypoint_positions,
    compute_observation,
    create_urdf_object,
)
from isaacsim_conversion.isaacsim_env import (
    CONTROL_DT,
    DEFAULT_ASSET_FRICTION,
    DEFAULT_JOINT_POS,
    JOINT_DAMPINGS,
    JOINT_DAMPINGS_COMPENSATED,
    JOINT_NAMES_ISAACGYM,
    JOINT_STIFFNESSES,
    JOINT_STIFFNESSES_COMPENSATED,
    PHYSICS_DT,
    PHYSICS_SUBSTEPS,
    _log,
)
from isaacsim_conversion.task_utils import CameraPose, TaskSpec, xyzw_to_wxyz


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
        num_envs: int = 1,
        env_spacing: float = 2.0,
        object_start_mode: str = "fixed",
        object_pos_noise_xyz: tuple[float, float, float] = (0.03, 0.03, 0.01),
        object_yaw_noise_deg: float = 20.0,
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
        self._fk_urdf = create_urdf_object("iiwa14_left_sharpa_adjusted_restricted")
        self.camera_pose = camera_pose_override or task_spec.camera_pose
        self.use_real_camera_transform = use_real_camera_transform
        self.action_dim = 29
        self.student_proprio_dim = 29 + 29 + 29 + 1
        self.object_scales_batch = np.repeat(self.task_spec.object_scales.astype(np.float32), self.num_envs, axis=0)
        self.camera_data_types = self._data_types_for_modality(camera_modality)
        self.goal_idx = np.zeros(self.num_envs, dtype=np.int32)
        self.near_goal_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.goal_pose = np.repeat(self.task_spec.goals[0][None], self.num_envs, axis=0).astype(np.float32)
        self.current_start_pose = np.repeat(self.task_spec.start_pose[None], self.num_envs, axis=0).astype(np.float32)
        self.prev_targets: np.ndarray | None = None

        self._setup_scene()

    def _data_types_for_modality(self, modality: str) -> list[str]:
        if modality == "depth":
            return ["distance_to_image_plane"]
        if modality == "rgb":
            return ["rgb"]
        if modality == "rgbd":
            return ["rgb", "distance_to_image_plane"]
        raise ValueError(f"Unsupported student camera modality: {modality}")

    def _convert_robot_urdf(self, sim_utils, UrdfConverterCfg, UrdfConverter, urdf_path: str) -> str:
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir="/tmp/isaaclab_usd_cache_distill/robot",
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=True,
            merge_fixed_joints=True,
            self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=JOINT_STIFFNESSES_COMPENSATED,
                    damping=JOINT_DAMPINGS_COMPENSATED,
                ),
            ),
        )
        return UrdfConverter(cfg).usd_path

    def _convert_static_urdf(self, UrdfConverterCfg, UrdfConverter, urdf_path: str, usd_dir: str, fix_base: bool) -> str:
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=usd_dir,
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=fix_base,
            merge_fixed_joints=True,
            joint_drive=None,
        )
        return UrdfConverter(cfg).usd_path

    def _make_scene_cfg(self, sim_utils, robot_usd: str, table_usd: str, object_usd: str):
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils import configclass

        camera_pose = self.camera_pose
        camera_types = list(self.camera_data_types)
        robot_joint_vel = {name: 0.0 for name in JOINT_NAMES_ISAACGYM}
        default_joint_pos = dict(DEFAULT_JOINT_POS)

        @configclass
        class DistillSceneCfg(InteractiveSceneCfg):
            robot = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=robot_usd,
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
                spawn=sim_utils.UsdFileCfg(
                    usd_path=table_usd,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002,
                        rest_offset=0.0,
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38)),
            )

            object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=object_usd,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002,
                        rest_offset=0.0,
                    ),
                ),
            )

            camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/DistillCamera",
                update_period=0,
                update_latest_camera_pose=True,
                height=480,
                width=640,
                data_types=camera_types,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 100.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=camera_pose.pos,
                    rot=camera_pose.quat_wxyz,
                    convention=camera_pose.convention,
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
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
        sim_cfg = SimulationCfg(
            dt=PHYSICS_DT,
            render_interval=PHYSICS_SUBSTEPS,
            gravity=(0.0, 0.0, -9.81),
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
            f"(control_dt={CONTROL_DT:.6f}, physics_dt={PHYSICS_DT:.6f}, substeps={PHYSICS_SUBSTEPS})"
        )

        robot_usd = self._convert_robot_urdf(sim_utils, UrdfConverterCfg, UrdfConverter, self.task_spec.robot_urdf)
        table_usd = self._convert_static_urdf(
            UrdfConverterCfg, UrdfConverter, self.task_spec.table_urdf, "/tmp/isaaclab_usd_cache_distill/table", True
        )
        object_usd = self._convert_static_urdf(
            UrdfConverterCfg, UrdfConverter, self.task_spec.object_urdf, "/tmp/isaaclab_usd_cache_distill/object", False
        )

        scene_cfg_type = self._make_scene_cfg(sim_utils, robot_usd, table_usd, object_usd)
        self.scene = InteractiveScene(
            scene_cfg_type(
                num_envs=self.num_envs,
                env_spacing=self.env_spacing,
                lazy_sensor_update=False,
                replicate_physics=True,
            )
        )
        self.robot = self.scene.articulations["robot"]
        self.object_rigid = self.scene.rigid_objects["object"]
        self.camera = self.scene.sensors["camera"]
        self.env_origins = self.scene.env_origins
        self.device = self.sim.device
        self.sim.reset()
        self.scene.reset()
        _log(
            "Distill camera created "
            f"(instances={self.camera.num_instances}, modality={self.camera_modality}, pos={self.camera_pose.pos}, "
            f"quat_wxyz={self.camera_pose.quat_wxyz}, convention={self.camera_pose.convention})"
        )
        self.reset()

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
        root_state = self.robot.data.default_root_state.clone()
        root_state[:, :3] += self.env_origins
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.set_joint_position_target(joint_pos)

        self.current_start_pose = self._sample_start_poses()
        object_pose_w = self._local_pose_to_world_pose(self.current_start_pose)
        zeros_vel = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self.object_rigid.write_root_pose_to_sim(object_pose_w)
        self.object_rigid.write_root_velocity_to_sim(zeros_vel)

        self.scene.write_data_to_sim()
        self.sim.step(render=not self.headless)
        self.scene.update(PHYSICS_DT)
        self.prev_targets = self.robot.data.joint_pos.detach().cpu().numpy().copy()
        self.goal_idx = np.zeros(self.num_envs, dtype=np.int32)
        self.near_goal_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.goal_pose = np.repeat(self.task_spec.goals[0][None], self.num_envs, axis=0).astype(np.float32)

    def apply_action(self, targets: np.ndarray):
        self.prev_targets = targets.copy()
        targets_t = torch.tensor(targets, dtype=torch.float32, device=self.device)
        self.robot.set_joint_position_target(targets_t)

    def step(self, render: bool = True):
        self.scene.write_data_to_sim()
        for _ in range(PHYSICS_SUBSTEPS):
            self.sim.step(render=render)
            self.scene.update(PHYSICS_DT)
        self.camera.update(PHYSICS_DT)

    def compute_sim_state(self) -> SimState:
        q = self.robot.data.joint_pos.detach().cpu().numpy()
        qd = self.robot.data.joint_vel.detach().cpu().numpy()
        object_pos_w = self.object_rigid.data.root_pos_w.detach().cpu().numpy()
        object_quat_w = self.object_rigid.data.root_quat_w.detach().cpu().numpy()

        object_pose = np.zeros((self.num_envs, 7), dtype=np.float32)
        object_pose[:, :3] = object_pos_w - self.env_origins.cpu().numpy()
        object_pose[:, 3:] = object_quat_w[:, [1, 2, 3, 0]]

        object_kps = _compute_keypoint_positions(object_pose, self.object_scales_batch)
        goal_kps = _compute_keypoint_positions(self.goal_pose, self.object_scales_batch)
        kp_dist = np.max(np.linalg.norm(object_kps - goal_kps, axis=-1), axis=-1).astype(np.float32)
        return SimState(
            q=q.copy(),
            qd=qd.copy(),
            object_pose=object_pose.copy(),
            goal_pose=self.goal_pose.copy(),
            goal_idx=self.goal_idx.copy(),
            kp_dist=kp_dist.copy(),
            near_goal_steps=self.near_goal_steps.copy(),
            object_pos_world_env_frame=object_pose[:, :3].copy(),
        )

    def maybe_advance_goal(self, sim_state: SimState) -> bool:
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
            self.near_goal_steps[env_id] = 0
            if self.goal_idx[env_id] < len(self.task_spec.goals):
                self.goal_pose[env_id] = self.task_spec.goals[self.goal_idx[env_id]]
        return bool(np.all(self.goal_idx >= len(self.task_spec.goals)))

    def task_finished(self, sim_state: SimState) -> bool:
        return bool(np.all(self.goal_idx >= len(self.task_spec.goals)))

    def compute_progress_metrics(self, sim_state: SimState) -> dict[str, float]:
        goal_completion = self.goal_idx.astype(np.float32) / len(self.task_spec.goals)
        return {
            "goal_idx": float(np.mean(self.goal_idx)),
            "goal_completion_ratio": float(np.mean(goal_completion)),
            "kp_dist": float(np.mean(sim_state.kp_dist)),
            "near_goal_steps": float(np.mean(self.near_goal_steps)),
        }

    def build_teacher_obs(self, sim_state: SimState) -> np.ndarray:
        if self.prev_targets is None:
            raise RuntimeError("Environment must be reset before building teacher observations")
        return compute_observation(
            q=sim_state.q,
            qd=sim_state.qd,
            prev_action_targets=self.prev_targets,
            object_pose=sim_state.object_pose,
            goal_object_pose=self.goal_pose,
            object_scales=self.object_scales_batch,
            urdf=self._fk_urdf,
            obs_list=OBS_LIST,
        )

    def _read_camera_outputs(self) -> dict[str, torch.Tensor]:
        outputs = self.camera.data.output
        available = {key: value for key, value in outputs.items() if value is not None}
        if not available:
            raise RuntimeError("Camera produced no outputs")
        return available

    def build_student_obs(self, sim_state: SimState, camera_modality: str | None = None) -> dict[str, torch.Tensor]:
        modality = camera_modality or self.camera_modality
        outputs = self._read_camera_outputs()
        image_dict: dict[str, torch.Tensor] = {}
        if modality in ("depth", "rgbd"):
            depth = outputs.get("distance_to_image_plane")
            if depth is None:
                raise RuntimeError(f"Depth output missing. Available camera outputs: {list(outputs.keys())}")
            if depth.dim() == 4 and depth.shape[-1] == 1:
                depth = depth.permute(0, 3, 1, 2)
            elif depth.dim() == 3:
                depth = depth.unsqueeze(1)
            image_dict["depth"] = torch.clamp(depth.float(), 0.0, 5.0) / 5.0
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

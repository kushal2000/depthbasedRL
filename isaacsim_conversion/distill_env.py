from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from isaacgymenvs.utils.observation_action_utils_sharpa import (
    _compute_keypoint_positions,
    compute_observation,
    create_urdf_object,
)
from isaacsim_conversion.isaacsim_env import IsaacSimEnv, _log
from isaacsim_conversion.task_utils import CameraPose, TaskSpec


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
    goal_idx: int
    kp_dist: float
    near_goal_steps: int
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
    ):
        self.task_spec = task_spec
        self.camera_modality = camera_modality
        self.env = IsaacSimEnv(
            robot_urdf=task_spec.robot_urdf,
            table_urdf=task_spec.table_urdf,
            object_urdf=task_spec.object_urdf,
            headless=headless,
            app=app,
        )
        self.app = app
        self._fk_urdf = create_urdf_object("iiwa14_left_sharpa_adjusted_restricted")
        self.prev_targets: np.ndarray | None = None
        self.goal_idx = 0
        self.near_goal_steps = 0
        self.goal_pose = self.task_spec.goals[0][None].copy()
        self.action_dim = 29
        self.student_proprio_dim = 29 + 29 + 29 + 1
        self.camera_pose = camera_pose_override or task_spec.camera_pose
        self.use_real_camera_transform = use_real_camera_transform
        self.camera = None
        self.camera_data_types = self._data_types_for_modality(camera_modality)
        self._sim_utils = None
        self._create_camera()

    def _data_types_for_modality(self, modality: str) -> list[str]:
        if modality == "depth":
            return ["distance_to_image_plane"]
        if modality == "rgb":
            return ["rgb"]
        if modality == "rgbd":
            return ["rgb", "distance_to_image_plane"]
        raise ValueError(f"Unsupported student camera modality: {modality}")

    def _create_camera(self):
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg

        self._sim_utils = sim_utils
        camera_cfg = CameraCfg(
            prim_path="/World/DistillCamera",
            update_period=0,
            height=480,
            width=640,
            data_types=self.camera_data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=self.camera_pose.pos,
                rot=self.camera_pose.quat_wxyz,
                convention=self.camera_pose.convention,
            ),
        )
        self.camera = Camera(cfg=camera_cfg)
        self.env.sim.reset()
        _log(
            "Distill camera created "
            f"(modality={self.camera_modality}, pos={self.camera_pose.pos}, "
            f"quat_wxyz={self.camera_pose.quat_wxyz}, convention={self.camera_pose.convention})"
        )

    def reset(self):
        self.env.set_object_pose(self.task_spec.start_pose[:3], self.task_spec.start_pose[3:7])
        self.env.reset_robot_to_default_pose(render=False)
        self.env.step(render=True)
        q, _ = self.env.get_robot_state()
        self.prev_targets = q[None].copy()
        self.goal_idx = 0
        self.near_goal_steps = 0
        self.goal_pose = self.task_spec.goals[0][None].copy()

    def apply_action(self, targets: np.ndarray):
        self.prev_targets = targets[None].copy()
        self.env.set_joint_position_targets(targets)

    def step(self, render: bool = True):
        self.env.step(render=render)
        self.camera.update(self.env.sim.get_physics_dt())

    def compute_sim_state(self) -> SimState:
        q, qd = self.env.get_robot_state()
        object_pose = self.env.get_object_pose_xyzw()[None]
        object_kps = _compute_keypoint_positions(object_pose, self.task_spec.object_scales)
        goal_kps = _compute_keypoint_positions(self.goal_pose, self.task_spec.object_scales)
        kp_dist = float(np.max(np.linalg.norm(object_kps[0] - goal_kps[0], axis=-1)))
        return SimState(
            q=q.copy(),
            qd=qd.copy(),
            object_pose=object_pose[0].copy(),
            goal_pose=self.goal_pose[0].copy(),
            goal_idx=self.goal_idx,
            kp_dist=kp_dist,
            near_goal_steps=self.near_goal_steps,
            object_pos_world_env_frame=object_pose[0, :3].copy(),
        )

    def maybe_advance_goal(self, sim_state: SimState) -> bool:
        if sim_state.kp_dist < self.task_spec.keypoint_tolerance:
            self.near_goal_steps += 1
        else:
            self.near_goal_steps = 0
        if self.near_goal_steps < self.task_spec.success_steps:
            return False
        self.goal_idx += 1
        self.near_goal_steps = 0
        if self.goal_idx >= len(self.task_spec.goals):
            return True
        self.goal_pose = self.task_spec.goals[self.goal_idx][None].copy()
        return False

    def task_finished(self, sim_state: SimState) -> bool:
        return self.goal_idx >= len(self.task_spec.goals)

    def compute_progress_metrics(self, sim_state: SimState) -> dict[str, float]:
        return {
            "goal_idx": float(self.goal_idx),
            "goal_completion_ratio": float(self.goal_idx / len(self.task_spec.goals)),
            "kp_dist": float(sim_state.kp_dist),
            "near_goal_steps": float(self.near_goal_steps),
        }

    def build_teacher_obs(self, sim_state: SimState) -> np.ndarray:
        if self.prev_targets is None:
            raise RuntimeError("Environment must be reset before building teacher observations")
        return compute_observation(
            q=sim_state.q[None],
            qd=sim_state.qd[None],
            prev_action_targets=self.prev_targets,
            object_pose=sim_state.object_pose[None],
            goal_object_pose=self.goal_pose,
            object_scales=self.task_spec.object_scales,
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

        progress = torch.tensor(
            [[self.goal_idx / len(self.task_spec.goals)]],
            dtype=torch.float32,
            device=self.env.sim.device,
        )
        proprio = torch.from_numpy(
            np.concatenate(
                [
                    sim_state.q.astype(np.float32),
                    sim_state.qd.astype(np.float32),
                    self.prev_targets[0].astype(np.float32),
                    np.array([progress.item()], dtype=np.float32),
                ]
            )[None]
        ).to(self.env.sim.device)
        return {
            "images": image_dict,
            "proprio": proprio,
        }

    def close(self):
        self.camera = None
        self.env.close()

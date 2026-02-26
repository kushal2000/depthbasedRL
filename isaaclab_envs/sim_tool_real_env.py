"""Isaac Lab DirectRLEnv implementation of the SimToolReal environment.

Ports isaacgymenvs/tasks/simtoolreal/env.py (IsaacGym) to Isaac Lab's
DirectRLEnv API.  The RL training code (rl/) stays unchanged — only the
environment backend swaps.

Key differences from IsaacGym version:
  - No per-env loop for scene creation (Cloner API handles replication)
  - No gym.refresh_*_tensor() calls (Isaac Lab auto-updates data.*)
  - Quaternions in wxyz convention (Isaac Lab) instead of xyzw (IsaacGym)
  - Joint ordering is breadth-first (Isaac Lab) instead of depth-first (IsaacGym)
  - TiledCamera for batched depth rendering instead of per-env cameras
  - Episode length in seconds instead of steps
  - Resets happen AFTER rewards: _get_dones() -> _get_rewards() -> _reset_idx() -> _get_observations()
"""

from __future__ import annotations

import math
import os
import random
import tempfile
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab_envs.sim_tool_real_cfg import (
    NUM_ARM_DOFS,
    NUM_FINGERTIPS,
    NUM_HAND_ARM_DOFS,
    NUM_HAND_DOFS,
    FINGERTIP_NAMES,
    PALM_LINK,
    DEPTH_MOUNT_LINK,
    SimToolRealEnvCfg,
)
from isaaclab_envs.utils import (
    get_axis_params,
    quat_rotate_wxyz,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    scale,
    tensor_clamp,
    torch_rand_float,
    unscale,
)


class SimToolRealEnv(DirectRLEnv):
    """Isaac Lab environment for dexterous tool manipulation with a Kuka arm + Sharpa hand.

    This is a direct port of the IsaacGym SimToolReal environment. The reward structure,
    observation space, curriculum, and domain randomization logic are preserved exactly.
    """

    cfg: SimToolRealEnvCfg

    def __init__(self, cfg: SimToolRealEnvCfg, render_mode: str | None = None, **kwargs):
        # Pre-compute obs/state sizes from config lists before super().__init__
        obs_type_size_dict = self._obs_type_size_dict()
        cfg.observation_space = sum(obs_type_size_dict[k] for k in cfg.obs_list)
        cfg.state_space = sum(obs_type_size_dict[k] for k in cfg.state_list)

        # DirectRLEnv reads these from cfg directly (no num_* needed)

        super().__init__(cfg, render_mode, **kwargs)

        # ── Constants ──
        self.num_arm_dofs = NUM_ARM_DOFS
        self.num_hand_dofs = NUM_HAND_DOFS
        self.num_hand_arm_dofs = NUM_HAND_ARM_DOFS
        self.num_fingertips = NUM_FINGERTIPS
        self.num_keypoints = cfg.num_keypoints
        self.keypoint_offsets_np = np.array(cfg.keypoint_offsets, dtype=np.float32)

        # self.step_dt and self.max_episode_length are computed by DirectRLEnv
        # from cfg.sim.dt, cfg.decimation, and cfg.episode_length_s.

        # ── Build joint index maps ──
        self._build_joint_index_maps()

        # ── Allocate persistent buffers ──
        self._allocate_buffers()

        # ── TiledCamera depth buffer ──
        if cfg.use_depth_camera:
            self.depth_buf = torch.zeros(
                self.num_envs, 1, cfg.depth_height, cfg.depth_width,
                device=self.device,
            )
        else:
            self.depth_buf = None

    # ══════════════════════════════════════════════════════════════════
    # Scene setup
    # ══════════════════════════════════════════════════════════════════

    def _setup_scene(self):
        """Create the scene: robot, object, table, camera, ground plane."""
        # Spawn assets
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        # TiledCamera for depth
        if self.cfg.use_depth_camera:
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Ground plane
        cfg_ground = sim_utils.GroundPlaneCfg()
        cfg_ground.func("/World/ground", cfg_ground)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # Register with scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ══════════════════════════════════════════════════════════════════
    # Joint index mapping
    # ══════════════════════════════════════════════════════════════════

    def _build_joint_index_maps(self):
        """Build mapping from our canonical joint ordering to Isaac Lab's breadth-first order.

        IsaacGym uses depth-first ordering, Isaac Lab uses breadth-first.
        We need index maps to reorder joint data consistently.
        """
        joint_names = self.robot.data.joint_names

        # Find arm joint indices
        arm_joint_names = [f"iiwa14_joint_{i}" for i in range(1, 8)]
        self._arm_joint_ids = []
        for name in arm_joint_names:
            try:
                idx = joint_names.index(name)
                self._arm_joint_ids.append(idx)
            except ValueError:
                raise ValueError(f"Arm joint '{name}' not found in robot. Available: {joint_names}")

        # Find hand joint indices — we need to discover them from the URDF
        # The hand joints are all joints that are NOT arm joints
        arm_set = set(self._arm_joint_ids)
        self._hand_joint_ids = [i for i in range(len(joint_names)) if i not in arm_set]

        assert len(self._arm_joint_ids) == self.num_arm_dofs, (
            f"Expected {self.num_arm_dofs} arm joints, found {len(self._arm_joint_ids)}"
        )

        # Combined ordering: arm first, then hand (matching IsaacGym convention)
        self._all_joint_ids = self._arm_joint_ids + self._hand_joint_ids
        self._all_joint_ids_t = torch.tensor(self._all_joint_ids, device=self.device, dtype=torch.long)

        # Inverse mapping: from our order back to Isaac Lab order
        self._inv_joint_ids = [0] * len(self._all_joint_ids)
        for our_idx, lab_idx in enumerate(self._all_joint_ids):
            self._inv_joint_ids[lab_idx] = our_idx
        self._inv_joint_ids_t = torch.tensor(self._inv_joint_ids, device=self.device, dtype=torch.long)

        # Find body indices for fingertips and palm
        body_names = self.robot.data.body_names

        self._fingertip_body_ids = []
        for name in FINGERTIP_NAMES:
            try:
                idx = body_names.index(name)
                self._fingertip_body_ids.append(idx)
            except ValueError:
                raise ValueError(f"Fingertip body '{name}' not found. Available: {body_names}")

        try:
            self._palm_body_id = body_names.index(PALM_LINK)
        except ValueError:
            raise ValueError(f"Palm body '{PALM_LINK}' not found. Available: {body_names}")

        # Get joint limits in our canonical ordering
        joint_pos_limits = self.robot.data.soft_joint_pos_limits  # (num_envs, num_joints, 2)
        self.arm_hand_dof_lower_limits = joint_pos_limits[0, self._all_joint_ids_t, 0]
        self.arm_hand_dof_upper_limits = joint_pos_limits[0, self._all_joint_ids_t, 1]

        # Default DOF position
        self.hand_arm_default_dof_pos = torch.zeros(
            self.num_hand_arm_dofs, dtype=torch.float, device=self.device
        )
        desired_kuka_pos = torch.tensor(
            [-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308], device=self.device
        )
        if self.cfg.start_arm_higher:
            desired_kuka_pos[1] -= math.radians(10)
            desired_kuka_pos[3] += math.radians(10)
        self.hand_arm_default_dof_pos[:7] = desired_kuka_pos

        print(f"[SimToolRealEnv] Joint names: {joint_names}")
        print(f"[SimToolRealEnv] Arm joint IDs: {self._arm_joint_ids}")
        print(f"[SimToolRealEnv] Hand joint IDs (first 5): {self._hand_joint_ids[:5]}...")
        print(f"[SimToolRealEnv] Body names: {body_names}")
        print(f"[SimToolRealEnv] Fingertip body IDs: {self._fingertip_body_ids}")
        print(f"[SimToolRealEnv] Palm body ID: {self._palm_body_id}")

    def _get_joint_pos_in_our_order(self) -> Tensor:
        """Get joint positions reordered to our canonical (arm+hand) ordering."""
        return self.robot.data.joint_pos[:, self._all_joint_ids_t]

    def _get_joint_vel_in_our_order(self) -> Tensor:
        """Get joint velocities reordered to our canonical (arm+hand) ordering."""
        return self.robot.data.joint_vel[:, self._all_joint_ids_t]

    def _set_joint_targets_from_our_order(self, targets: Tensor):
        """Write joint position targets from our canonical ordering to Isaac Lab ordering."""
        # Expand to full joint count
        full_targets = torch.zeros(
            self.num_envs, len(self.robot.data.joint_names),
            device=self.device,
        )
        full_targets[:, self._all_joint_ids_t] = targets
        self.robot.set_joint_position_target(full_targets)

    # ══════════════════════════════════════════════════════════════════
    # Buffer allocation
    # ══════════════════════════════════════════════════════════════════

    def _allocate_buffers(self):
        """Allocate all persistent tensors used across steps."""
        N = self.num_envs
        D = self.num_hand_arm_dofs
        device = self.device

        # Action targets
        self.prev_targets = torch.zeros(N, D, device=device)
        self.cur_targets = torch.zeros(N, D, device=device)
        self.actions = torch.zeros(N, self.cfg.action_space, device=device)

        # Action / obs delay queues
        self.action_queue = torch.zeros(
            N, self.cfg.action_delay_max, self.cfg.action_space, device=device
        )
        self.obs_queue = torch.zeros(
            N, self.cfg.obs_delay_max, self.cfg.observation_space, device=device
        )
        self.object_state_queue = torch.zeros(
            N, self.cfg.object_state_delay_max, 13, device=device
        )

        # Episode tracking
        self.successes = torch.zeros(N, device=device)
        self.prev_episode_successes = torch.zeros(N, device=device)
        self.near_goal_steps = torch.zeros(N, dtype=torch.long, device=device)
        self.lifted_object = torch.zeros(N, dtype=torch.bool, device=device)
        self.closest_fingertip_dist = torch.full((N, self.num_fingertips), -1.0, device=device)
        self.furthest_hand_dist = torch.full((N,), -1.0, device=device)
        self.closest_keypoint_max_dist = torch.full((N,), -1.0, device=device)
        self.closest_keypoint_max_dist_fixed_size = torch.full((N,), -1.0, device=device)
        self.total_episode_closest_keypoint_max_dist = torch.zeros(N, device=device)
        self.prev_total_episode_closest_keypoint_max_dist = torch.zeros(N, device=device)
        self.prev_episode_closest_keypoint_max_dist = torch.zeros(N, device=device)
        self.prev_episode_true_objective = torch.zeros(N, device=device)
        self.true_objective = torch.zeros(N, device=device)

        # Goal states: [pos(3) + quat_wxyz(4) + linvel(3) + angvel(3)] = 13
        self.goal_states = torch.zeros(N, 13, device=device)

        # Object init state (recorded at reset for lifting reward reference)
        self.object_init_state = torch.zeros(N, 13, device=device)

        # Reward tracking
        self.rewards_episode = {}
        reward_names = [
            "raw_fingertip_delta_rew", "raw_hand_delta_penalty", "raw_lifting_rew",
            "raw_keypoint_rew", "raw_object_lin_vel_penalty", "raw_object_ang_vel_penalty",
            "fingertip_delta_rew", "hand_delta_penalty", "lifting_rew", "lift_bonus_rew",
            "keypoint_rew", "kuka_actions_penalty", "hand_actions_penalty",
            "bonus_rew", "object_lin_vel_penalty", "object_ang_vel_penalty", "total_reward",
        ]
        for name in reward_names:
            self.rewards_episode[name] = torch.zeros(N, device=device)

        # Reset goal buffer
        self.reset_goal_buf = torch.zeros(N, dtype=torch.bool, device=device)

        # Random force probabilities
        self.random_force_prob = torch.zeros(N, device=device)
        self.random_torque_prob = torch.zeros(N, device=device)
        self.random_lin_vel_impulse_prob = torch.zeros(N, device=device)
        self.random_ang_vel_impulse_prob = torch.zeros(N, device=device)

        # Random forces/torques applied to rigid bodies
        num_bodies = self.robot.data.body_names.__len__() + 1 + 1  # robot + object + table
        self.rb_forces = torch.zeros(N, num_bodies, 3, device=device)
        self.rb_torques = torch.zeros(N, num_bodies, 3, device=device)

        # Keypoint buffers
        self.obj_keypoint_pos = torch.zeros(N, self.num_keypoints, 3, device=device)
        self.goal_keypoint_pos = torch.zeros(N, self.num_keypoints, 3, device=device)
        self.observed_obj_keypoint_pos = torch.zeros(N, self.num_keypoints, 3, device=device)
        self.obj_keypoint_pos_fixed_size = torch.zeros(N, self.num_keypoints, 3, device=device)
        self.goal_keypoint_pos_fixed_size = torch.zeros(N, self.num_keypoints, 3, device=device)

        # Keypoint offsets — will be set per-object
        base_size = self.cfg.object_base_size
        kp_scale = self.cfg.keypoint_scale
        offsets = torch.tensor(self.keypoint_offsets_np, device=device, dtype=torch.float)
        self.object_keypoint_offsets = (offsets * base_size * kp_scale).unsqueeze(0).expand(N, -1, -1).clone()

        fixed_size = torch.tensor(self.cfg.fixed_size, device=device, dtype=torch.float)
        self.object_keypoint_offsets_fixed_size = (
            (offsets * fixed_size.unsqueeze(0) * kp_scale).unsqueeze(0).expand(N, -1, -1).clone()
        )

        # Object scales (for procedural objects; set to 1.0 initially)
        self.object_scales = torch.ones(N, 3, device=device)
        self.object_scale_noise_multiplier = torch.ones(N, 3, device=device)

        # Fingertip offsets
        self.fingertip_offsets_np = np.array([
            [0.02, 0.002, 0], [0.02, 0.002, 0], [0.02, 0.002, 0],
            [0.02, 0.002, 0], [0.02, 0.002, 0],
        ], dtype=np.float32)
        self.fingertip_offsets_t = torch.from_numpy(self.fingertip_offsets_np).to(device)
        self.fingertip_offsets_t = self.fingertip_offsets_t.unsqueeze(0).expand(N, -1, -1).clone()

        # Palm offset
        self.palm_offset = torch.tensor([-0.0, -0.02, 0.16], device=device).unsqueeze(0).expand(N, -1).clone()

        # Finger reward coefficients (equal weight per fingertip)
        self.finger_rew_coeffs = torch.ones(N, self.num_fingertips, device=device)

        # Previous fingertip distances (for delta reward)
        self.fingertip_pos_rel_object_prev = None

        # Tolerance curriculum
        self.success_tolerance = self.cfg.success_tolerance
        self.initial_tolerance = self.cfg.success_tolerance
        self.target_tolerance = self.cfg.target_success_tolerance
        self.last_curriculum_update = 0

        # Tyler curriculum (obs dropout)
        self._tyler_curriculum_scale = self.cfg.init_tyler_curriculum_scale if hasattr(self.cfg, 'init_tyler_curriculum_scale') else 0.0

        # Target volume
        mins = torch.tensor(self.cfg.target_volume_mins, device=device)
        maxs = torch.tensor(self.cfg.target_volume_maxs, device=device)
        self.target_volume_origin = (mins + maxs) / 2
        self.target_volume_extent = torch.stack([
            -(maxs - mins) / 2,
            (maxs - mins) / 2,
        ], dim=1) * self.cfg.target_volume_region_scale

        # Control step counter
        self.control_steps = 0
        self.frame_since_restart = 0

        # Reset reason tracking
        self.recent_reset_reason_history = deque(maxlen=4096)

    # ══════════════════════════════════════════════════════════════════
    # Obs type size dictionary
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _obs_type_size_dict() -> dict:
        """Size of each observation component."""
        return {
            "joint_pos": NUM_HAND_ARM_DOFS,
            "joint_vel": NUM_HAND_ARM_DOFS,
            "prev_action_targets": NUM_HAND_ARM_DOFS,
            "palm_pos": 3,
            "palm_rot": 4,
            "palm_vel": 6,
            "object_rot": 4,
            "object_vel": 6,
            "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS,
            "keypoints_rel_palm": 3 * 4,  # num_keypoints=4
            "keypoints_rel_goal": 3 * 4,
            "object_scales": 3,
            "closest_keypoint_max_dist": 1,
            "closest_fingertip_dist": NUM_FINGERTIPS,
            "lifted_object": 1,
            "progress": 1,
            "successes": 1,
            "reward": 1,
        }

    # ══════════════════════════════════════════════════════════════════
    # DirectRLEnv API: _pre_physics_step
    # ══════════════════════════════════════════════════════════════════

    def _pre_physics_step(self, actions: Tensor):
        """Process actions before physics simulation.

        Called once per control step (before the decimation loop).
        Ports: pre_physics_step() from IsaacGym env.
        """
        actions = actions.to(self.device)

        # Update action queue for delay
        self.action_queue = self._update_queue(self.action_queue, actions)

        if self.cfg.use_action_delay:
            delay_index = torch.randint(
                0, self.action_queue.shape[1], (self.num_envs,), device=self.device
            )
            actions = self.action_queue[torch.arange(self.num_envs), delay_index].clone()

        self.actions = actions.clone()

        # Compute joint position targets
        arm_actions = actions[:, :7]
        hand_actions = actions[:, 7:self.num_hand_arm_dofs]

        # Arm: delta from previous target
        if self.cfg.use_relative_control:
            arm_targets = (
                self._get_joint_pos_in_our_order()[:, :7]
                + self.cfg.dof_speed_scale * self.step_dt * arm_actions
            )
        else:
            arm_targets = (
                self.prev_targets[:, :7]
                + self.cfg.dof_speed_scale * self.step_dt * arm_actions
            )

        arm_targets = tensor_clamp(
            arm_targets,
            self.arm_hand_dof_lower_limits[:7],
            self.arm_hand_dof_upper_limits[:7],
        )

        # Smooth arm
        arm_targets = (
            self.cfg.arm_moving_average * arm_targets
            + (1.0 - self.cfg.arm_moving_average) * self.prev_targets[:, :7]
        )

        # Hand: absolute position from [-1, 1] scaled to joint limits
        hand_targets = scale(
            hand_actions,
            self.arm_hand_dof_lower_limits[7:self.num_hand_arm_dofs],
            self.arm_hand_dof_upper_limits[7:self.num_hand_arm_dofs],
        )
        hand_targets = (
            self.cfg.hand_moving_average * hand_targets
            + (1.0 - self.cfg.hand_moving_average) * self.prev_targets[:, 7:self.num_hand_arm_dofs]
        )
        hand_targets = tensor_clamp(
            hand_targets,
            self.arm_hand_dof_lower_limits[7:self.num_hand_arm_dofs],
            self.arm_hand_dof_upper_limits[7:self.num_hand_arm_dofs],
        )

        self.cur_targets[:, :7] = arm_targets
        self.cur_targets[:, 7:self.num_hand_arm_dofs] = hand_targets

        self.prev_targets[:, :] = self.cur_targets[:, :]

        # Apply joint position targets
        self._set_joint_targets_from_our_order(self.cur_targets)

        # Apply random forces to object
        self._apply_random_forces()

    def _apply_action(self):
        """Called once per decimation sub-step (physics step).

        Forces are already set in _pre_physics_step, so nothing extra needed here.
        Isaac Lab calls this before each sub-step within the decimation loop.
        """
        pass

    # ══════════════════════════════════════════════════════════════════
    # DirectRLEnv API: _get_observations
    # ══════════════════════════════════════════════════════════════════

    def _get_observations(self) -> dict:
        """Compute observations for policy and critic.

        Ports: populate_sim_buffers() + populate_obs_and_states_buffers()
        """
        self._populate_sim_buffers()

        if self.cfg.use_depth_camera:
            self._render_depth()

        obs_buf, states_buf = self._compute_obs_and_states()

        # Clamp observations
        obs_buf = torch.clamp(obs_buf, -self.cfg.clamp_abs_observations, self.cfg.clamp_abs_observations)

        result = {"policy": obs_buf}
        if states_buf is not None:
            result["critic"] = states_buf
        if self.cfg.use_depth_camera and self.depth_buf is not None:
            result["depth"] = self.depth_buf

        return result

    # ══════════════════════════════════════════════════════════════════
    # DirectRLEnv API: _get_rewards
    # ══════════════════════════════════════════════════════════════════

    def _get_rewards(self) -> Tensor:
        """Compute rewards.

        Ports: compute_kuka_reward()
        """
        return self._compute_kuka_reward()

    # ══════════════════════════════════════════════════════════════════
    # DirectRLEnv API: _get_dones
    # ══════════════════════════════════════════════════════════════════

    def _get_dones(self) -> Tuple[Tensor, Tensor]:
        """Compute termination and truncation signals.

        Returns:
            terminated: environments that hit a terminal condition (fall, max successes, etc.)
            truncated: environments that hit the episode time limit
        """
        ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Object fell below threshold
        object_pos = self.object.data.root_pos_w  # (N, 3)
        object_z_low = object_pos[:, 2] < 0.1

        # Max consecutive successes
        if self.cfg.max_consecutive_successes > 0:
            max_succ_reached = self.successes >= self.cfg.max_consecutive_successes
        else:
            max_succ_reached = zeros

        # Hand far from object
        if hasattr(self, 'curr_fingertip_distances'):
            hand_far = self.curr_fingertip_distances.max(dim=-1).values > 1.5
        else:
            hand_far = zeros

        terminated = object_z_low | max_succ_reached | hand_far

        # Time limit (truncation)
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    # ══════════════════════════════════════════════════════════════════
    # DirectRLEnv API: _reset_idx
    # ══════════════════════════════════════════════════════════════════

    def _reset_idx(self, env_ids: Tensor):
        """Reset specific environments.

        Ports: reset_idx()
        """
        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        # Reset episode tracking
        self.prev_episode_successes[env_ids] = self.successes[env_ids]
        self.successes[env_ids] = 0
        self.near_goal_steps[env_ids] = 0
        self.lifted_object[env_ids] = False
        self.closest_fingertip_dist[env_ids] = -1.0
        self.furthest_hand_dist[env_ids] = -1.0
        self.closest_keypoint_max_dist[env_ids] = -1.0
        self.closest_keypoint_max_dist_fixed_size[env_ids] = -1.0

        self.prev_episode_true_objective[env_ids] = self.true_objective[env_ids]
        self.true_objective[env_ids] = 0

        for key in self.rewards_episode:
            self.rewards_episode[key][env_ids] = 0

        # Reset forces
        self.rb_forces[env_ids] = 0.0
        self.rb_torques[env_ids] = 0.0

        # Reset random force probabilities
        self.random_force_prob[env_ids] = self._sample_log_uniform(
            self.cfg.force_prob_range[0], self.cfg.force_prob_range[1], len(env_ids)
        )
        self.random_torque_prob[env_ids] = self._sample_log_uniform(
            self.cfg.torque_prob_range[0], self.cfg.torque_prob_range[1], len(env_ids)
        )

        # Reset robot joint positions with noise
        delta_max = self.arm_hand_dof_upper_limits - self.hand_arm_default_dof_pos
        delta_min = self.arm_hand_dof_lower_limits - self.hand_arm_default_dof_pos

        rand_floats = torch_rand_float(
            0.0, 1.0, (len(env_ids), self.num_hand_arm_dofs), self.device
        )
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats

        noise_coeff = torch.zeros_like(self.hand_arm_default_dof_pos)
        noise_coeff[:7] = self.cfg.reset_dof_pos_noise_arm
        noise_coeff[7:] = self.cfg.reset_dof_pos_noise_fingers

        robot_pos = self.hand_arm_default_dof_pos + noise_coeff * rand_delta
        robot_pos = tensor_clamp(
            robot_pos, self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits
        )

        # Write joint state to sim (need to convert to Isaac Lab ordering)
        full_joint_pos = torch.zeros(
            len(env_ids), len(self.robot.data.joint_names), device=self.device
        )
        full_joint_vel = torch.zeros_like(full_joint_pos)
        full_joint_pos[:, self._all_joint_ids_t] = robot_pos

        rand_vel = torch_rand_float(
            -1.0, 1.0, (len(env_ids), self.num_hand_arm_dofs), self.device
        ) * self.cfg.reset_dof_vel_noise
        full_joint_vel[:, self._all_joint_ids_t] = rand_vel

        self.robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, env_ids=env_ids)

        # Update prev/cur targets
        self.prev_targets[env_ids, :self.num_hand_arm_dofs] = robot_pos
        self.cur_targets[env_ids, :self.num_hand_arm_dofs] = robot_pos

        # Set joint position targets
        full_targets = torch.zeros(
            len(env_ids), len(self.robot.data.joint_names), device=self.device
        )
        full_targets[:, self._all_joint_ids_t] = robot_pos
        self.robot.set_joint_position_target(full_targets, env_ids=env_ids)

        # Reset object pose with noise
        self._reset_object_pose(env_ids)

        # Reset goal
        self._reset_target_pose(env_ids, is_first_goal=True)

        # Record object init state for lifting reward reference
        self.object_init_state[env_ids, :3] = self.object.data.root_pos_w[env_ids]
        self.object_init_state[env_ids, 3:7] = self.object.data.root_quat_w[env_ids]

        # Reset delay queues
        self.action_queue[env_ids] = 0.0
        self.obs_queue[env_ids] = 0.0
        self.object_state_queue[env_ids] = 0.0

    # ══════════════════════════════════════════════════════════════════
    # Sim buffer population (replaces gym.refresh_*_tensor)
    # ══════════════════════════════════════════════════════════════════

    def _populate_sim_buffers(self):
        """Read simulation state into local convenience variables.

        In Isaac Lab, data is auto-updated. We just create convenient aliases.
        Ports: populate_sim_buffers()
        """
        # Robot state
        self.arm_hand_dof_pos = self._get_joint_pos_in_our_order()
        self.arm_hand_dof_vel = self._get_joint_vel_in_our_order()

        # Object state (wxyz quaternion in Isaac Lab)
        self.object_pos = self.object.data.root_pos_w  # (N, 3)
        self.object_quat_wxyz = self.object.data.root_quat_w  # (N, 4) wxyz
        self.object_rot = quat_wxyz_to_xyzw(self.object_quat_wxyz)  # convert to xyzw for reward math compatibility
        self.object_linvel = self.object.data.root_lin_vel_w  # (N, 3)
        self.object_angvel = self.object.data.root_ang_vel_w  # (N, 3)

        # Object state vector (13-dim, with quat in xyzw for compatibility)
        self.object_state = torch.cat([
            self.object_pos, self.object_rot, self.object_linvel, self.object_angvel
        ], dim=-1)

        # Observed object state (may have delay/noise)
        self.observed_object_state = self.object_state.clone()
        if self.cfg.use_object_state_delay_noise:
            self.object_state_queue = self._update_queue(self.object_state_queue, self.object_state)
            delay_index = torch.randint(
                0, self.object_state_queue.shape[1], (self.num_envs,), device=self.device
            )
            self.observed_object_state[:] = self.object_state_queue[
                torch.arange(self.num_envs), delay_index
            ].clone()
            self.observed_object_state[:, :3] += (
                torch.randn_like(self.observed_object_state[:, :3]) * self.cfg.object_state_xyz_noise_std
            )

        self.observed_object_pos = self.observed_object_state[:, :3]
        self.observed_object_rot = self.observed_object_state[:, 3:7]

        # Goal state
        self.goal_pos = self.goal_states[:, :3]
        self.goal_rot = self.goal_states[:, 3:7]  # xyzw

        # Palm state (wxyz from Isaac Lab, keep xyzw internally for compat)
        palm_pos_w = self.robot.data.body_pos_w[:, self._palm_body_id]  # (N, 3)
        palm_quat_wxyz = self.robot.data.body_quat_w[:, self._palm_body_id]  # (N, 4) wxyz
        palm_linvel = self.robot.data.body_lin_vel_w[:, self._palm_body_id]  # (N, 3)
        palm_angvel = self.robot.data.body_ang_vel_w[:, self._palm_body_id]  # (N, 3)
        palm_rot_xyzw = quat_wxyz_to_xyzw(palm_quat_wxyz)

        self._palm_pos = palm_pos_w
        self._palm_rot = palm_rot_xyzw
        self._palm_state = torch.cat([
            palm_pos_w, palm_rot_xyzw, palm_linvel, palm_angvel
        ], dim=-1)

        # Palm center (with offset)
        self.palm_center_pos = self._palm_pos + self._quat_rotate_xyzw(
            self._palm_rot, self.palm_offset
        )

        # Fingertip positions & rotations
        fingertip_ids = self._fingertip_body_ids
        self.fingertip_pos = self.robot.data.body_pos_w[:, fingertip_ids]  # (N, F, 3)
        fingertip_quat_wxyz = self.robot.data.body_quat_w[:, fingertip_ids]  # (N, F, 4) wxyz
        self.fingertip_rot = torch.stack([
            quat_wxyz_to_xyzw(fingertip_quat_wxyz[:, i]) for i in range(self.num_fingertips)
        ], dim=1)  # (N, F, 4) xyzw

        # Fingertip with offset
        self.fingertip_pos_offset = torch.zeros_like(self.fingertip_pos)
        for i in range(self.num_fingertips):
            self.fingertip_pos_offset[:, i] = self.fingertip_pos[:, i] + self._quat_rotate_xyzw(
                self.fingertip_rot[:, i], self.fingertip_offsets_t[:, i]
            )

        # Prev fingertip distances
        if self.fingertip_pos_rel_object_prev is not None:
            self.fingertip_pos_rel_object_prev_saved = self.fingertip_pos_rel_object_prev.clone()

        obj_pos_repeat = self.object_pos.unsqueeze(1).expand(-1, self.num_fingertips, -1)
        self.fingertip_pos_rel_object = self.fingertip_pos_offset - obj_pos_repeat
        self.curr_fingertip_distances = torch.norm(self.fingertip_pos_rel_object, dim=-1)

        if self.fingertip_pos_rel_object_prev is None:
            self.fingertip_pos_rel_object_prev = self.fingertip_pos_rel_object.clone()

        # Update closest fingertip distance
        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0,
            self.curr_fingertip_distances,
            self.closest_fingertip_dist,
        )
        self.furthest_hand_dist = torch.where(
            self.furthest_hand_dist < 0.0,
            self.curr_fingertip_distances[:, 0],
            self.furthest_hand_dist,
        )

        # Fingertip relative to palm
        palm_repeat = self.palm_center_pos.unsqueeze(1).expand(-1, self.num_fingertips, -1)
        self.fingertip_pos_rel_palm = self.fingertip_pos_offset - palm_repeat

        # Keypoints
        for i in range(self.num_keypoints):
            self.obj_keypoint_pos[:, i] = self.object_pos + self._quat_rotate_xyzw(
                self.object_rot, self.object_keypoint_offsets[:, i] * self.object_scale_noise_multiplier
            )
            self.goal_keypoint_pos[:, i] = self.goal_pos + self._quat_rotate_xyzw(
                self.goal_rot, self.object_keypoint_offsets[:, i] * self.object_scale_noise_multiplier
            )
            self.observed_obj_keypoint_pos[:, i] = self.observed_object_pos + self._quat_rotate_xyzw(
                self.observed_object_rot, self.object_keypoint_offsets[:, i] * self.object_scale_noise_multiplier
            )
            self.obj_keypoint_pos_fixed_size[:, i] = self.object_pos + self._quat_rotate_xyzw(
                self.object_rot, self.object_keypoint_offsets_fixed_size[:, i]
            )
            self.goal_keypoint_pos_fixed_size[:, i] = self.goal_pos + self._quat_rotate_xyzw(
                self.goal_rot, self.object_keypoint_offsets_fixed_size[:, i]
            )

        self.keypoints_rel_goal = self.obj_keypoint_pos - self.goal_keypoint_pos
        self.observed_keypoints_rel_goal = self.observed_obj_keypoint_pos - self.goal_keypoint_pos
        self.keypoints_rel_goal_fixed_size = self.obj_keypoint_pos_fixed_size - self.goal_keypoint_pos_fixed_size

        palm_kp_repeat = self.palm_center_pos.unsqueeze(1).expand(-1, self.num_keypoints, -1)
        self.keypoints_rel_palm = self.obj_keypoint_pos - palm_kp_repeat
        self.observed_keypoints_rel_palm = self.observed_obj_keypoint_pos - palm_kp_repeat

        self.keypoint_distances_l2 = torch.norm(self.keypoints_rel_goal, dim=-1)
        self.keypoint_distances_l2_fixed_size = torch.norm(self.keypoints_rel_goal_fixed_size, dim=-1)

        self.keypoints_max_dist = self.keypoint_distances_l2.max(dim=-1).values
        self.keypoints_max_dist_fixed_size = self.keypoint_distances_l2_fixed_size.max(dim=-1).values

        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0,
            self.keypoints_max_dist,
            self.closest_keypoint_max_dist,
        )
        self.closest_keypoint_max_dist_fixed_size = torch.where(
            self.closest_keypoint_max_dist_fixed_size < 0.0,
            self.keypoints_max_dist_fixed_size,
            self.closest_keypoint_max_dist_fixed_size,
        )

    # ══════════════════════════════════════════════════════════════════
    # Observation computation
    # ══════════════════════════════════════════════════════════════════

    def _compute_obs_and_states(self) -> Tuple[Tensor, Tensor]:
        """Compute policy obs and critic states.

        Ports: populate_obs_and_states_buffers()
        """
        N = self.num_envs
        obs_dict = {}

        # Joint positions (unscaled to [-1, 1])
        obs_dict["joint_pos"] = unscale(
            self.arm_hand_dof_pos, self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits
        )
        obs_dict["joint_vel"] = self.arm_hand_dof_vel.clone()
        obs_dict["prev_action_targets"] = self.prev_targets.clone()
        obs_dict["palm_pos"] = self.palm_center_pos
        obs_dict["palm_rot"] = self._palm_state[:, 3:7]
        obs_dict["object_rot"] = self.object_state[:, 3:7]
        obs_dict["keypoints_rel_palm"] = self.keypoints_rel_palm.reshape(N, -1)
        obs_dict["keypoints_rel_goal"] = self.keypoints_rel_goal.reshape(N, -1)
        obs_dict["fingertip_pos_rel_palm"] = self.fingertip_pos_rel_palm.reshape(N, -1)
        obs_dict["object_scales"] = self.object_scales * self.object_scale_noise_multiplier

        # Critic-only observations
        obs_dict["palm_vel"] = self._palm_state[:, 7:13]
        obs_dict["object_vel"] = self.object_state[:, 7:13]
        obs_dict["closest_keypoint_max_dist"] = self.closest_keypoint_max_dist.unsqueeze(-1)
        if self.cfg.fixed_size_keypoint_reward:
            obs_dict["closest_keypoint_max_dist"] = self.closest_keypoint_max_dist_fixed_size.unsqueeze(-1)
        obs_dict["closest_fingertip_dist"] = self.closest_fingertip_dist
        obs_dict["lifted_object"] = self.lifted_object.float().unsqueeze(-1)
        obs_dict["progress"] = torch.log(self.episode_length_buf.float() / 10 + 1).unsqueeze(-1)
        obs_dict["successes"] = torch.log(self.successes + 1).unsqueeze(-1)
        obs_dict["reward"] = 0.01 * self.reward_buf if hasattr(self, 'reward_buf') and self.reward_buf is not None else torch.zeros(N, device=self.device)
        if obs_dict["reward"].dim() == 1:
            obs_dict["reward"] = obs_dict["reward"].unsqueeze(-1) if obs_dict["reward"].shape[0] == N else torch.zeros(N, 1, device=self.device)

        # Build state buffer (critic)
        states_buf = torch.cat(
            [obs_dict[k].reshape(N, -1) for k in self.cfg.state_list], dim=-1
        )

        # Policy observations: add noise to joint velocities
        obs_dict["joint_vel"] = obs_dict["joint_vel"] + (
            torch.randn_like(obs_dict["joint_vel"]) * self.cfg.joint_velocity_obs_noise_std
        )

        # Build obs buffer (policy)
        obs_buf = torch.cat(
            [obs_dict[k].reshape(N, -1) for k in self.cfg.obs_list], dim=-1
        )

        # Obs delay
        self.obs_queue = self._update_queue(self.obs_queue, obs_buf)
        if self.cfg.use_obs_delay:
            delay_index = torch.randint(
                0, self.obs_queue.shape[1], (N,), device=self.device
            )
            obs_buf = self.obs_queue[torch.arange(N), delay_index].clone()

        return obs_buf, states_buf

    # ══════════════════════════════════════════════════════════════════
    # Reward computation
    # ══════════════════════════════════════════════════════════════════

    def _compute_kuka_reward(self) -> Tensor:
        """Compute all reward components.

        Ports: compute_kuka_reward()
        """
        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew, hand_delta_penalty = self._distance_delta_rewards(lifted_object)
        keypoint_rew, keypoint_rew_fixed_size = self._keypoint_reward(lifted_object)
        if self.cfg.fixed_size_keypoint_reward:
            keypoint_rew = keypoint_rew_fixed_size

        keypoint_success_tolerance = self.success_tolerance * self.cfg.keypoint_scale

        near_goal = self.keypoints_max_dist <= keypoint_success_tolerance
        near_goal_fixed_size = self.keypoints_max_dist_fixed_size <= keypoint_success_tolerance
        if self.cfg.fixed_size_keypoint_reward:
            near_goal = near_goal_fixed_size

        if self.cfg.force_consecutive_near_goal_steps:
            self.near_goal_steps = (self.near_goal_steps + near_goal.long()) * near_goal.long()
        else:
            self.near_goal_steps += near_goal.long()

        is_success = self.near_goal_steps >= self.cfg.success_steps
        self.successes += is_success.float()
        self.reset_goal_buf[:] = is_success

        # Scale rewards
        object_lin_vel_penalty = -torch.sum(torch.square(self.object_linvel), dim=-1)
        object_ang_vel_penalty = -torch.sum(torch.square(self.object_angvel), dim=-1)

        fingertip_delta_rew_scaled = fingertip_delta_rew * self.cfg.distance_delta_rew_scale
        lifting_rew_scaled = lifting_rew * self.cfg.lifting_rew_scale
        keypoint_rew_scaled = keypoint_rew * self.cfg.keypoint_rew_scale
        object_lin_vel_penalty_scaled = object_lin_vel_penalty * self.cfg.object_lin_vel_penalty_scale
        object_ang_vel_penalty_scaled = object_ang_vel_penalty * self.cfg.object_ang_vel_penalty_scale

        kuka_actions_penalty, hand_actions_penalty = self._action_penalties()

        bonus_rew = near_goal.float() * (self.cfg.reach_goal_bonus / self.cfg.success_steps)
        if self.cfg.force_consecutive_near_goal_steps:
            bonus_rew = is_success.float() * self.cfg.reach_goal_bonus

        reward = (
            fingertip_delta_rew_scaled
            + lifting_rew_scaled
            + lift_bonus_rew
            + keypoint_rew_scaled
            + kuka_actions_penalty
            + hand_actions_penalty
            + bonus_rew
            + object_lin_vel_penalty_scaled
            + object_ang_vel_penalty_scaled
        )

        # Track rewards
        self.rewards_episode["fingertip_delta_rew"] += fingertip_delta_rew_scaled
        self.rewards_episode["lifting_rew"] += lifting_rew_scaled
        self.rewards_episode["lift_bonus_rew"] += lift_bonus_rew
        self.rewards_episode["keypoint_rew"] += keypoint_rew_scaled
        self.rewards_episode["kuka_actions_penalty"] += kuka_actions_penalty
        self.rewards_episode["hand_actions_penalty"] += hand_actions_penalty
        self.rewards_episode["bonus_rew"] += bonus_rew
        self.rewards_episode["total_reward"] += reward

        # Extras for logging
        self.extras["successes"] = self.prev_episode_successes
        self.extras["success_ratio"] = (
            self.prev_episode_successes.mean().item() / self.cfg.max_consecutive_successes
        )

        # Tolerance curriculum
        self._update_tolerance_curriculum()

        # Control step counter
        self.control_steps += 1
        self.frame_since_restart += 1

        return reward

    # ── Reward sub-functions ──

    def _lifting_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting object off table."""
        z_lift = 0.05 + self.object_pos[:, 2] - self.object_init_state[:, 2]
        lifting_rew = torch.clip(z_lift, 0, 0.5)

        lifted_object = (z_lift > self.cfg.lifting_bonus_threshold) | self.lifted_object
        just_lifted = lifted_object & ~self.lifted_object
        lift_bonus_rew = self.cfg.lifting_bonus * just_lifted.float()

        lifting_rew *= (~lifted_object).float()
        self.lifted_object = lifted_object

        return lifting_rew, lift_bonus_rew, lifted_object

    def _distance_delta_rewards(self, lifted_object: Tensor) -> Tuple[Tensor, Tensor]:
        """Rewards for fingertips approaching the object."""
        fingertip_deltas_closest = self.closest_fingertip_dist - self.curr_fingertip_distances
        self.closest_fingertip_dist = torch.minimum(
            self.closest_fingertip_dist, self.curr_fingertip_distances
        )

        hand_deltas_furthest = self.furthest_hand_dist - self.curr_fingertip_distances[:, 0]
        self.furthest_hand_dist = torch.maximum(
            self.furthest_hand_dist, self.curr_fingertip_distances[:, 0]
        )

        fingertip_deltas = torch.clip(fingertip_deltas_closest, 0, 10)
        fingertip_deltas *= self.finger_rew_coeffs
        fingertip_delta_rew = torch.sum(fingertip_deltas, dim=-1)
        fingertip_delta_rew *= (~lifted_object).float()

        hand_delta_penalty = torch.clip(hand_deltas_furthest, -10, 0)
        hand_delta_penalty *= (~lifted_object).float()
        hand_delta_penalty *= self.num_fingertips

        return fingertip_delta_rew, hand_delta_penalty

    def _keypoint_reward(self, lifted_object: Tensor) -> Tuple[Tensor, Tensor]:
        """Reward for getting keypoints closer to goal."""
        max_keypoint_deltas = self.closest_keypoint_max_dist - self.keypoints_max_dist
        max_keypoint_deltas_fixed_size = (
            self.closest_keypoint_max_dist_fixed_size - self.keypoints_max_dist_fixed_size
        )

        self.closest_keypoint_max_dist = torch.minimum(
            self.closest_keypoint_max_dist, self.keypoints_max_dist
        )
        self.closest_keypoint_max_dist_fixed_size = torch.minimum(
            self.closest_keypoint_max_dist_fixed_size, self.keypoints_max_dist_fixed_size
        )

        max_keypoint_deltas = torch.clip(max_keypoint_deltas, 0, 100)
        max_keypoint_deltas_fixed_size = torch.clip(max_keypoint_deltas_fixed_size, 0, 100)

        keypoint_rew = max_keypoint_deltas * lifted_object.float()
        keypoint_rew_fixed_size = max_keypoint_deltas_fixed_size * lifted_object.float()

        return keypoint_rew, keypoint_rew_fixed_size

    def _action_penalties(self) -> Tuple[Tensor, Tensor]:
        """Penalties for large actions."""
        kuka_penalty = (
            -torch.sum(torch.abs(self.arm_hand_dof_vel[:, :7]), dim=-1)
            * self.cfg.kuka_actions_penalty_scale
        )
        hand_penalty = (
            -torch.sum(torch.abs(self.arm_hand_dof_vel[:, 7:self.num_hand_arm_dofs]), dim=-1)
            * self.cfg.hand_actions_penalty_scale
        )
        return kuka_penalty, hand_penalty

    def _update_tolerance_curriculum(self):
        """Update success tolerance curriculum."""
        if self.frame_since_restart - self.last_curriculum_update < self.cfg.tolerance_curriculum_interval:
            return
        mean_succ = self.prev_episode_successes.mean().item()
        if mean_succ < 3.0:
            return
        self.success_tolerance *= self.cfg.tolerance_curriculum_increment
        self.success_tolerance = min(self.success_tolerance, self.initial_tolerance)
        self.success_tolerance = max(self.success_tolerance, self.target_tolerance)
        self.last_curriculum_update = self.frame_since_restart
        print(f"[Curriculum] tolerance -> {self.success_tolerance:.4f} (mean_succ={mean_succ:.1f})")

    # ══════════════════════════════════════════════════════════════════
    # Object reset helpers
    # ══════════════════════════════════════════════════════════════════

    def _reset_object_pose(self, env_ids: Tensor):
        """Reset object to start pose with noise."""
        N = len(env_ids)
        # Base position: table top + offset
        pos = torch.zeros(N, 3, device=self.device)
        pos[:, 0] = 0.0  # x = robot x
        pos[:, 1] = 0.0  # y = table y
        pos[:, 2] = self.cfg.table_reset_z + self.cfg.table_object_z_offset

        # Add position noise
        pos[:, 0] += torch_rand_float(
            -self.cfg.reset_position_noise_x, self.cfg.reset_position_noise_x, (N, 1), self.device
        ).squeeze(-1)
        pos[:, 1] += torch_rand_float(
            -self.cfg.reset_position_noise_y, self.cfg.reset_position_noise_y, (N, 1), self.device
        ).squeeze(-1)
        pos[:, 2] += torch_rand_float(
            -self.cfg.reset_position_noise_z, self.cfg.reset_position_noise_z, (N, 1), self.device
        ).squeeze(-1)

        # Random rotation
        if self.cfg.randomize_object_rotation:
            quat_wxyz = self._random_quaternion(N)
        else:
            quat_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(N, -1)

        vel = torch.zeros(N, 6, device=self.device)

        self.object.write_root_pose_to_sim(
            torch.cat([pos, quat_wxyz], dim=-1), env_ids=env_ids
        )
        self.object.write_root_velocity_to_sim(vel, env_ids=env_ids)

    def _reset_target_pose(self, env_ids: Tensor, is_first_goal: bool = True):
        """Sample a new goal pose for the given environments."""
        N = len(env_ids)
        if N == 0:
            return

        if self.cfg.goal_sampling_type == "delta":
            # Sample goal as delta from current object position
            obj_pos = self.object.data.root_pos_w[env_ids]  # (N, 3)
            obj_rot_wxyz = self.object.data.root_quat_w[env_ids]  # (N, 4) wxyz
            obj_rot_xyzw = quat_wxyz_to_xyzw(obj_rot_wxyz)

            # Random direction
            direction = torch.randn(N, 3, device=self.device)
            direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            goal_pos = obj_pos + direction * self.cfg.delta_goal_distance

            # Clamp to target volume
            mins = torch.tensor(self.cfg.target_volume_mins, device=self.device)
            maxs = torch.tensor(self.cfg.target_volume_maxs, device=self.device)
            goal_pos = torch.max(torch.min(goal_pos, maxs), mins)

            # Random rotation delta
            goal_rot_xyzw = self._sample_delta_quat_xyzw(
                obj_rot_xyzw, self.cfg.delta_rotation_degrees
            )
        else:
            # Absolute sampling in target volume
            goal_pos = torch.zeros(N, 3, device=self.device)
            for d in range(3):
                goal_pos[:, d] = torch_rand_float(
                    self.cfg.target_volume_mins[d],
                    self.cfg.target_volume_maxs[d],
                    (N, 1), self.device,
                ).squeeze(-1)
            goal_rot_xyzw = self._random_quaternion_xyzw(N)

        self.goal_states[env_ids, :3] = goal_pos
        self.goal_states[env_ids, 3:7] = goal_rot_xyzw

        # Reset keypoint tracking for new goal
        self.closest_keypoint_max_dist[env_ids] = -1.0
        self.closest_keypoint_max_dist_fixed_size[env_ids] = -1.0
        self.closest_fingertip_dist[env_ids] = -1.0
        self.furthest_hand_dist[env_ids] = -1.0
        self.near_goal_steps[env_ids] = 0

    # ══════════════════════════════════════════════════════════════════
    # Random forces
    # ══════════════════════════════════════════════════════════════════

    def _apply_random_forces(self):
        """Apply random forces/torques to the object.

        Ports the random force logic from pre_physics_step().
        """
        if self.cfg.force_scale <= 0.0 and self.cfg.torque_scale <= 0.0:
            return

        # Decay existing forces
        if self.cfg.force_scale > 0.0 and self.cfg.force_decay > 0.0:
            self.rb_forces *= torch.pow(
                torch.tensor(self.cfg.force_decay, device=self.device),
                self.step_dt / self.cfg.force_decay_interval,
            )

        if self.cfg.torque_scale > 0.0 and self.cfg.torque_decay > 0.0:
            self.rb_torques *= torch.pow(
                torch.tensor(self.cfg.torque_decay, device=self.device),
                self.step_dt / self.cfg.torque_decay_interval,
            )

        # Apply new random forces
        if self.cfg.force_scale > 0.0:
            force_mask = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob)
            if force_mask.any():
                force_envs = force_mask.nonzero(as_tuple=False).squeeze(-1)
                new_forces = torch.randn(len(force_envs), 3, device=self.device) * self.cfg.force_scale * 0.1
                if self.cfg.force_only_when_lifted:
                    new_forces *= self.lifted_object[force_envs].float().unsqueeze(-1)
                # Apply force to object (using external force API)
                self.object.set_external_force_and_torque(
                    forces=new_forces.unsqueeze(1),
                    torques=torch.zeros_like(new_forces).unsqueeze(1),
                    env_ids=force_envs,
                )

        if self.cfg.torque_scale > 0.0:
            torque_mask = (torch.rand(self.num_envs, device=self.device) < self.random_torque_prob)
            if torque_mask.any():
                torque_envs = torque_mask.nonzero(as_tuple=False).squeeze(-1)
                new_torques = torch.randn(len(torque_envs), 3, device=self.device) * self.cfg.torque_scale * 0.1
                if self.cfg.torque_only_when_lifted:
                    new_torques *= self.lifted_object[torque_envs].float().unsqueeze(-1)
                self.object.set_external_force_and_torque(
                    forces=torch.zeros_like(new_torques).unsqueeze(1),
                    torques=new_torques.unsqueeze(1),
                    env_ids=torque_envs,
                )

    # ══════════════════════════════════════════════════════════════════
    # Depth rendering
    # ══════════════════════════════════════════════════════════════════

    def _render_depth(self):
        """Render depth from TiledCamera and fill self.depth_buf."""
        if not self.cfg.use_depth_camera or not hasattr(self, '_tiled_camera'):
            return

        depth = self._tiled_camera.data.output["depth"]  # (N, H, W, 1)
        self.depth_buf = depth.permute(0, 3, 1, 2)  # (N, 1, H, W)
        # Isaac Lab TiledCamera returns positive depth (unlike IsaacGym which was negative)
        self.depth_buf.clamp_(0.0, self.cfg.depth_far)
        self.depth_buf /= self.cfg.depth_far  # normalize to [0, 1]

    # ══════════════════════════════════════════════════════════════════
    # Utility methods
    # ══════════════════════════════════════════════════════════════════

    def _quat_rotate_xyzw(self, q_xyzw: Tensor, v: Tensor) -> Tensor:
        """Rotate vector v by quaternion q in xyzw convention.

        Args:
            q_xyzw: (N, 4) quaternion in xyzw
            v: (N, 3) vector
        Returns:
            (N, 3) rotated vector
        """
        q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
        return quat_rotate_wxyz(q_wxyz, v)

    def _random_quaternion(self, n: int) -> Tensor:
        """Generate n random unit quaternions in wxyz convention."""
        u = torch.rand(n, 3, device=self.device)
        q = torch.stack([
            torch.sqrt(1 - u[:, 0]) * torch.sin(2 * math.pi * u[:, 1]),
            torch.sqrt(1 - u[:, 0]) * torch.cos(2 * math.pi * u[:, 1]),
            torch.sqrt(u[:, 0]) * torch.sin(2 * math.pi * u[:, 2]),
            torch.sqrt(u[:, 0]) * torch.cos(2 * math.pi * u[:, 2]),
        ], dim=-1)
        # Reorder from xyzw to wxyz
        return quat_xyzw_to_wxyz(q)

    def _random_quaternion_xyzw(self, n: int) -> Tensor:
        """Generate n random unit quaternions in xyzw convention."""
        q_wxyz = self._random_quaternion(n)
        return quat_wxyz_to_xyzw(q_wxyz)

    def _sample_delta_quat_xyzw(self, input_quat_xyzw: Tensor, delta_degrees: float) -> Tensor:
        """Apply a random rotation delta to input quaternions.

        Args:
            input_quat_xyzw: (N, 4) quaternions in xyzw
            delta_degrees: max rotation angle in degrees
        Returns:
            (N, 4) perturbed quaternions in xyzw
        """
        N = input_quat_xyzw.shape[0]
        # Random axis
        axis = torch.randn(N, 3, device=self.device)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        # Random angle
        angle = torch_rand_float(
            -math.radians(delta_degrees), math.radians(delta_degrees),
            (N, 1), self.device
        ).squeeze(-1)
        # Axis-angle to quaternion (wxyz)
        half_angle = angle * 0.5
        delta_w = torch.cos(half_angle)
        delta_xyz = axis * torch.sin(half_angle).unsqueeze(-1)
        delta_wxyz = torch.cat([delta_w.unsqueeze(-1), delta_xyz], dim=-1)

        # Multiply: delta * input
        input_wxyz = quat_xyzw_to_wxyz(input_quat_xyzw)
        from isaaclab_envs.utils import quat_mul_wxyz
        result_wxyz = quat_mul_wxyz(delta_wxyz, input_wxyz)
        return quat_wxyz_to_xyzw(result_wxyz)

    def _update_queue(self, queue: Tensor, current_values: Tensor) -> Tensor:
        """Update FIFO queue: push current values, shift older ones.

        Ports: update_queue()
        """
        N, T, D = queue.shape
        # On episode start, fill queue with current values
        is_start = (self.episode_length_buf == 0)
        queue = torch.where(
            is_start.unsqueeze(1).unsqueeze(2).expand_as(queue),
            current_values.unsqueeze(1).expand(N, T, D),
            queue,
        )
        # Shift and insert
        queue[:, 1:] = queue[:, :-1].clone()
        queue[:, 0] = current_values
        return queue

    def _sample_log_uniform(self, min_val: float, max_val: float, n: int) -> Tensor:
        """Sample n values from log-uniform distribution in [min_val, max_val]."""
        log_min = math.log(min_val)
        log_max = math.log(max_val)
        return torch.exp(
            torch.rand(n, device=self.device) * (log_max - log_min) + log_min
        )

    # ══════════════════════════════════════════════════════════════════
    # Env info interface (for RL agent compatibility)
    # ══════════════════════════════════════════════════════════════════

    def get_env_info(self):
        """Return environment info matching the expected agent interface."""
        import gymnasium.spaces as spaces
        return {
            'action_space': spaces.Box(-1.0, 1.0, (self.cfg.action_space,)),
            'observation_space': spaces.Box(-float('inf'), float('inf'), (self.cfg.observation_space,)),
            'state_space': spaces.Box(-float('inf'), float('inf'), (self.cfg.state_space,)),
        }

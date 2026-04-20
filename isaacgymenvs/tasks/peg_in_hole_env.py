"""PegInHoleEnv: SimToolReal subclass for multi-init, multi-goal,
multi-tolerance peg-in-hole training.

Loads ``assets/urdf/peg_in_hole/scenes/scenes.npz`` produced by
``peg_in_hole/scene_generation/generate_scenes.py``. The scene schema has
three axes:
  - ``N`` hole positions (scenes). Each has a unique table+hole scene URDF
    baked at a specific XY.
  - ``M`` peg start poses per scene. Each has a pre-generated collision-safe
    trajectory to the insertion pose.
  - ``K`` tolerance-slot URDFs per scene. Same hole XY, different slot
    clearance (drawn from a log-uniform pool in [0.1, 10] mm).

Runtime axis assignment:
  - **Static per env** (baked at env creation):
    ``(scene_idx, tol_slot_idx) = (env_idx // K, env_idx % K) mod (N * K)``
    → picks which scene URDF the env loads. With many training envs,
    each of the ``N * K`` URDFs appears roughly equally.
  - **Dynamic per reset**: ``peg_idx`` uniform in ``[0, M)``. The policy
    sees a fresh peg start pose (and the associated cached trajectory)
    each episode.

The policy observes peg + hole poses (via the goal trajectory) but NOT the
tolerance, so it is forced to be robust across the clearance continuum.

Goal modes (same semantics as FabricaEnv's ``finalGoalOnly`` /
``preInsertAndFinal``):
  - ``dense``              — all per-(scene, peg) waypoints.
  - ``preInsertAndFinal``  — keep only ``[goals[tl-2], goals[tl-1]]``.
  - ``finalGoalOnly``      — keep only ``[goals[tl-1]]``.

Retract-phase reward is duplicated verbatim from FabricaEnv: after the peg
reaches the final goal, the hand is rewarded for moving away while the peg
stays in place. Intentionally not factored into a shared helper yet.
"""

from __future__ import annotations

import os
import tempfile
from copy import copy
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from isaacgym import gymapi
from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
from isaacgymenvs.tasks.simtoolreal.utils import populate_dof_properties
from isaacgymenvs.utils.torch_jit_utils import get_axis_params, to_torch, torch_rand_float


VALID_GOAL_MODES = ("dense", "preInsertAndFinal", "finalGoalOnly")


class PegInHoleEnv(SimToolReal):
    # ────────────────────────────────────────────────────────────────
    # __init__ + scene loading
    # ────────────────────────────────────────────────────────────────

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        # Retract params (copied from FabricaEnv)
        self.enable_retract = cfg["env"].get("enableRetract", False)
        self.retract_reward_scale = cfg["env"].get("retractRewardScale", 1.0)
        self.retract_distance_threshold = cfg["env"].get("retractDistanceThreshold", 0.1)
        self.retract_success_bonus = cfg["env"].get("retractSuccessBonus", 1000.0)
        self.retract_success_tolerance = cfg["env"].get("retractSuccessTolerance", 0.005)

        self.goal_mode = cfg["env"].get("goalMode", "dense")
        assert self.goal_mode in VALID_GOAL_MODES, (
            f"goalMode must be one of {VALID_GOAL_MODES}, got {self.goal_mode!r}"
        )

        # Load scenes.npz + apply goal-mode truncation + build per-env indexing.
        # Stores: self._pih_start_poses, self._pih_goals, self._pih_traj_lengths,
        # self._pih_hole_positions (numpy), self._pih_scene_urdfs (list of str),
        # self._pih_env_scene_idx, self._pih_env_tol_slot_idx (numpy).
        self._init_peg_in_hole_config(cfg)

        # Env-0 compat placeholders so parent SimToolReal.__init__ can read
        # a valid fixed object pose / goals / table asset path. Real per-env
        # values are applied during _create_envs / reset_object_pose.
        cfg["env"]["objectName"] = "peg"
        cfg["env"]["useFixedGoalStates"] = True
        cfg["env"]["useFixedInitObjectPose"] = True

        # Pick env-0's first peg as the placeholder start pose + goals.
        s0 = int(self._pih_env_scene_idx[0])
        start0 = self._pih_start_poses[s0, 0].tolist()                # (7,)
        tl0 = int(self._pih_traj_lengths[s0, 0])
        goals0 = self._pih_goals[s0, 0, :tl0].tolist()                # list of [7]
        cfg["env"]["objectStartPose"] = start0
        cfg["env"]["fixedGoalStates"] = goals0
        cfg["env"]["asset"]["table"] = self._pih_scene_urdfs[0]

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        # Move scene tensors to device now that self.device is set by parent.
        self._pih_start_poses_t = torch.tensor(
            self._pih_start_poses, dtype=torch.float32, device=self.device
        )                                                             # (N, M, 7)
        self._pih_goals_t = torch.tensor(
            self._pih_goals, dtype=torch.float32, device=self.device
        )                                                             # (N, M, T, 7)
        self._pih_traj_lengths_t = torch.tensor(
            self._pih_traj_lengths, dtype=torch.long, device=self.device
        )                                                             # (N, M)
        self._pih_env_scene_idx_t = torch.tensor(
            self._pih_env_scene_idx, dtype=torch.long, device=self.device
        )                                                             # (num_envs,)
        self._pih_env_tol_slot_t = torch.tensor(
            self._pih_env_tol_slot_idx, dtype=torch.long, device=self.device
        )                                                             # (num_envs,)
        # Track the peg_idx sampled for each env on the most recent reset
        # (used by _reset_target for goal cycling).
        self.env_peg_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Per-env max goals (depends on scene+peg when goalMode=="dense";
        # constant otherwise). Populated on reset; seed to the peg-0 length
        # for each env's scene so the first reward step has a valid denominator.
        self.env_max_goals = self._pih_traj_lengths_t[
            self._pih_env_scene_idx_t, 0
        ].clone()
        # Snapshot of the just-ended episode's env_max_goals (for success_ratio logging).
        self.prev_episode_env_max_goals = self.env_max_goals.clone()

        # Per-env retract state (copied from FabricaEnv lines 128-130).
        self.retract_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.retract_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Register retract_rew in the episode reward tracker (mirrors FabricaEnv:132).
        self.rewards_episode["retract_rew"] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        # Per-episode XY jitter on the goal position, observed by the actor
        # only (see populate_obs_and_states_buffers override below).
        self.goal_pos_obs_noise = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device,
        )
        # Compute obs_buf slice for keypoints_rel_goal (cached once). See
        # SimToolReal.populate_obs_and_states_buffers (env.py:3153-3291)
        # for the field layout.
        n_dof = self.num_hand_arm_dofs
        nk = self.num_keypoints
        nft = self.num_fingertips
        _sz = {
            "joint_pos": n_dof, "joint_vel": n_dof, "prev_action_targets": n_dof,
            "palm_pos": 3, "palm_rot": 4, "palm_vel": 6,
            "object_rot": 4, "object_vel": 6,
            "fingertip_pos_rel_palm": 3 * nft,
            "keypoints_rel_palm": 3 * nk, "keypoints_rel_goal": 3 * nk,
            "object_scales": 3,
        }
        _off = 0
        for _k in self.obs_list:
            if _k == "keypoints_rel_goal":
                break
            _off += _sz[_k]
        self._goal_kp_obs_slice = slice(_off, _off + 3 * nk)

    def _init_peg_in_hole_config(self, cfg):
        """Load scenes.npz, truncate goals per goal_mode, build per-env indexing."""
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
        )
        scenes_path = cfg["env"]["scenesPath"]
        if not os.path.isabs(scenes_path):
            scenes_path = os.path.join(repo_root, scenes_path)
        assert os.path.exists(scenes_path), (
            f"scenesPath {scenes_path} does not exist. Run "
            f"peg_in_hole/scene_generation/generate_scenes.py first."
        )
        data = np.load(scenes_path)
        start_poses = data["start_poses"].astype(np.float32)            # (N, M, 7)
        goals = data["goals"].astype(np.float32)                        # (N, M, T, 7)
        traj_lengths = data["traj_lengths"].astype(np.int64)            # (N, M)
        hole_positions = data["hole_positions"].astype(np.float32)      # (N, 3)
        tol_pool_m = data["tolerance_pool_m"].astype(np.float32)        # (pool_size,)
        scene_tol_indices = data["scene_tolerance_indices"].astype(np.int64)  # (N, K)

        N, M, _ = start_poses.shape
        _, _, T, _ = goals.shape
        K = scene_tol_indices.shape[1]
        assert scene_tol_indices.shape[0] == N
        assert traj_lengths.shape == (N, M)

        # ── Goal-mode truncation ────────────────────────────────
        # Mirror FabricaEnv::_load_multi_init_states (lines 292-327).
        if self.goal_mode == "finalGoalOnly":
            n_idx = np.arange(N)[:, None].repeat(M, axis=1)            # (N, M)
            m_idx = np.arange(M)[None, :].repeat(N, axis=0)            # (N, M)
            final_idx = traj_lengths - 1                                # (N, M)
            final_goals = goals[n_idx, m_idx, final_idx]                # (N, M, 7)
            goals = final_goals[:, :, None, :]                          # (N, M, 1, 7)
            traj_lengths = np.ones_like(traj_lengths)
            T = 1
            print(f"[PegInHoleEnv] goalMode=finalGoalOnly: goals truncated to {goals.shape}")
        elif self.goal_mode == "preInsertAndFinal":
            n_idx = np.arange(N)[:, None].repeat(M, axis=1)
            m_idx = np.arange(M)[None, :].repeat(N, axis=0)
            final_idx = traj_lengths - 1
            pre_idx = np.clip(traj_lengths - 2, a_min=0, a_max=None)
            pre_goals = goals[n_idx, m_idx, pre_idx]
            final_goals = goals[n_idx, m_idx, final_idx]
            goals = np.stack([pre_goals, final_goals], axis=2)          # (N, M, 2, 7)
            traj_lengths = np.clip(traj_lengths, a_min=None, a_max=2)
            T = 2
            print(f"[PegInHoleEnv] goalMode=preInsertAndFinal: goals truncated to {goals.shape}")
        # else: dense — leave as-is.

        self._pih_num_scenes = N
        self._pih_num_pegs = M
        self._pih_num_tol_slots = K
        self._pih_max_traj_len = T
        self._pih_start_poses = start_poses
        self._pih_goals = goals
        self._pih_traj_lengths = traj_lengths
        self._pih_hole_positions = hole_positions
        self._pih_tol_pool_m = tol_pool_m
        self._pih_scene_tol_indices = scene_tol_indices

        # ── Per-env (scene_idx, tol_slot_idx) round-robin assignment ──
        # Optional override: ``forceSceneTolCombo = [scene_idx, tol_slot_idx]``
        # pins every env to the same combo. Used by the multi-init eval GUI
        # to target a specific (scene, tol) at a time.
        num_envs = cfg["env"]["numEnvs"]
        combo_count = N * K
        force_combo = cfg["env"].get("forceSceneTolCombo", None)
        if force_combo is not None:
            assert len(force_combo) == 2, "forceSceneTolCombo must be [scene_idx, tol_slot_idx]"
            fs, ft = int(force_combo[0]), int(force_combo[1])
            assert 0 <= fs < N and 0 <= ft < K, (
                f"forceSceneTolCombo=({fs},{ft}) out of range (N={N}, K={K})"
            )
            self._pih_env_scene_idx = np.full(num_envs, fs, dtype=np.int64)
            self._pih_env_tol_slot_idx = np.full(num_envs, ft, dtype=np.int64)
            print(f"[PegInHoleEnv] forceSceneTolCombo=({fs}, {ft}) — all {num_envs} envs pinned to this combo.")
        else:
            combo_ids = np.arange(num_envs) % combo_count               # (num_envs,)
            self._pih_env_scene_idx = (combo_ids // K).astype(np.int64)
            self._pih_env_tol_slot_idx = (combo_ids % K).astype(np.int64)

        # Per-env scene URDF paths (relative to repo_root/assets).
        self._pih_scene_urdfs = []
        for env_i in range(num_envs):
            s = int(self._pih_env_scene_idx[env_i])
            ts = int(self._pih_env_tol_slot_idx[env_i])
            self._pih_scene_urdfs.append(
                f"urdf/peg_in_hole/scenes/scene_{s:04d}/scene_tol{ts:02d}.urdf"
            )

        print(
            f"[PegInHoleEnv] Loaded {N} scenes × {M} pegs × {K} tol slots "
            f"(goal_mode={self.goal_mode}, max_traj_len={T})"
        )
        print(
            f"[PegInHoleEnv] {num_envs} envs → {combo_count} unique (scene, tol) "
            f"combos ({num_envs / combo_count:.1f}x coverage)"
        )

    # ────────────────────────────────────────────────────────────────
    # Env creation
    # ────────────────────────────────────────────────────────────────

    def _create_envs(self, num_envs, spacing, num_per_row):
        """Build envs with one peg asset + per-env scene (table+hole) URDFs.

        Follows the SimToolReal _create_envs skeleton but loads N*K unique
        table_assets (one per (scene, tol) combo, deduplicated by path) and
        assigns each env its pre-computed asset by combo index.
        """
        if self.should_load_initial_states:
            self.load_initial_states()

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )
        object_asset_root = asset_root
        tmp_assets_dir = tempfile.TemporaryDirectory()
        self.object_asset_files, self.object_asset_scales, self.object_need_vhacds = (
            self._main_object_assets_and_scales(object_asset_root, tmp_assets_dir.name)
        )

        # ── Robot asset ──
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.0 if self.cfg["env"].get("useSDF", False) else 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        print(f"Loading robot asset {self.robot_asset_file} from {asset_root}")
        robot_asset = self.gym.load_asset(
            self.sim, asset_root, self.robot_asset_file, asset_options
        )

        self.num_hand_arm_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_hand_arm_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        num_hand_arm_dofs = self.gym.get_asset_dof_count(robot_asset)
        assert self.num_hand_arm_dofs == num_hand_arm_dofs

        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        for i in range(self.num_hand_arm_dofs):
            self.arm_hand_dof_lower_limits.append(robot_dof_props["lower"][i])
            self.arm_hand_dof_upper_limits.append(robot_dof_props["upper"][i])
        self.arm_hand_dof_lower_limits = to_torch(
            self.arm_hand_dof_lower_limits, device=self.device
        )
        self.arm_hand_dof_upper_limits = to_torch(
            self.arm_hand_dof_upper_limits, device=self.device
        )

        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx)) + gymapi.Vec3(0.0, 0.8, 0)
        robot_pose.r = gymapi.Quat(0, 0, 0, 1)

        # ── Object (peg) asset ──
        # Single asset — every env shares it.
        object_assets, object_rb_count, object_shapes_count = self._load_main_object_asset()

        # ── Table (scene) assets — one per unique scene URDF path ──
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.collapse_fixed_joints = True
        if self.cfg["env"].get("useSDF", False):
            table_asset_options.thickness = 0.0

        # De-dup: each of the N*K URDFs appears once in env_scene_urdfs per combo.
        # We load each unique URDF once and index by combo.
        unique_urdfs = sorted(set(self._pih_scene_urdfs))
        urdf_to_asset_idx = {u: i for i, u in enumerate(unique_urdfs)}
        table_assets = []
        per_combo_table_rb = []
        per_combo_table_shapes = []
        for urdf in unique_urdfs:
            ta = self.gym.load_asset(self.sim, asset_root, urdf, table_asset_options)
            table_assets.append(ta)
            per_combo_table_rb.append(self.gym.get_asset_rigid_body_count(ta))
            per_combo_table_shapes.append(self.gym.get_asset_rigid_shape_count(ta))
        max_table_rb = max(per_combo_table_rb)
        max_table_shapes = max(per_combo_table_shapes)
        print(
            f"[PegInHoleEnv] Loaded {len(unique_urdfs)} unique scene URDFs "
            f"(max {max_table_rb} bodies, {max_table_shapes} shapes)"
        )

        if self.with_table_force_sensor:
            table_sensor_pose = gymapi.Transform()
            table_sensor_props = gymapi.ForceSensorProperties()
            table_sensor_props.enable_constraint_solver_forces = True
            table_sensor_props.enable_forward_dynamics_forces = False
            table_sensor_props.use_world_frame = True
            self.table_sensor_idx = self.gym.create_asset_force_sensor(
                asset=table_assets[0], body_idx=0,
                local_pose=table_sensor_pose, props=table_sensor_props,
            )
            if self.table_sensor_idx == -1:
                raise ValueError("Failed to create table force sensor")
            for ta in table_assets[1:]:
                self.gym.create_asset_force_sensor(
                    asset=ta, body_idx=0,
                    local_pose=table_sensor_pose, props=table_sensor_props,
                )

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3()
        table_pose.p.x = robot_pose.p.x
        table_pose_dy, table_pose_dz = -0.8, self.cfg["env"]["tableResetZ"]
        table_pose.p.y = robot_pose.p.y + table_pose_dy
        table_pose.p.z = robot_pose.p.z + table_pose_dz

        # ── Goal (marker) asset — shared across envs ──
        additional_rb, additional_shapes = self._load_additional_assets(
            object_asset_root, robot_pose
        )

        # Compute max_agg as per-env-worst-case
        max_agg_bodies = (
            self.num_hand_arm_bodies + object_rb_count + max_table_rb + additional_rb
        )
        max_agg_shapes = (
            self.num_hand_arm_shapes + object_shapes_count + max_table_shapes + additional_shapes
        )

        # ── Per-env start pose (env-0 placeholder, actual start set on reset) ──
        sp0 = self._pih_start_poses[int(self._pih_env_scene_idx[0]), 0]
        obj_start_pose_env0 = gymapi.Transform()
        obj_start_pose_env0.p = gymapi.Vec3(float(sp0[0]), float(sp0[1]), float(sp0[2]))
        obj_start_pose_env0.r = gymapi.Quat(float(sp0[3]), float(sp0[4]), float(sp0[5]), float(sp0[6]))
        self.object_start_pose = obj_start_pose_env0

        # ── Env loop ──
        self.robots = []
        self.envs = []
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robots = []
        self.objects = []

        object_init_state = []
        table_init_state = []
        self.rigid_body_name_to_idx = {}
        self.robot_indices = []
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robot_indices = []
        object_indices = []
        table_indices = []
        object_scales = []
        object_keypoint_offsets = []
        object_keypoint_offsets_fixed_size = []

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        for name in self.fingertips:
            assert name in body_names
        assert "iiwa14_link_7" in body_names

        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(robot_asset, name)
            for name in self.fingertips
        ]

        if self.with_fingertip_force_sensors:
            finger_sensor_pose = gymapi.Transform()
            self.finger_sensor_idxs = [
                self.gym.create_asset_force_sensor(
                    asset=robot_asset, body_idx=ft_handle, local_pose=finger_sensor_pose
                )
                for ft_handle in self.fingertip_handles
            ]

        self.robot_name = "iiwa14"
        self.palm_handle = self.gym.find_asset_rigid_body_index(robot_asset, "iiwa14_link_7")

        self.object_rb_handles = list(
            range(self.num_hand_arm_bodies, self.num_hand_arm_bodies + object_rb_count)
        )
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.object_rb_handles = list(
                range(2 * self.num_hand_arm_bodies, 2 * self.num_hand_arm_bodies + object_rb_count)
            )

        # Asset frictions
        if self.cfg["env"]["modifyAssetFrictions"]:
            self.set_robot_asset_rigid_shape_properties(
                robot_asset=robot_asset,
                friction=self.cfg["env"]["robotFriction"],
                fingertip_friction=self.cfg["env"]["fingerTipFriction"],
            )
            for ta in table_assets:
                self.set_table_asset_rigid_shape_properties(
                    table_asset=ta, friction=self.cfg["env"]["tableFriction"],
                )
            self.set_object_asset_rigid_shape_properties(
                object_asset=object_assets[0], friction=self.cfg["env"]["objectFriction"],
            )
        else:
            self.set_robot_asset_rigid_shape_properties(
                robot_asset=robot_asset, friction=None, fingertip_friction=None,
            )

        print(f"[PegInHoleEnv] Creating {num_envs} envs with "
              f"max_agg_bodies={max_agg_bodies}, max_agg_shapes={max_agg_shapes}")

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Robot
            robot_actor = self.gym.create_actor(
                env_ptr, robot_asset, robot_pose, "robot", i, -1, 0,
            )
            populate_dof_properties(robot_dof_props, self.num_arm_dofs, self.num_hand_dofs)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            robot_idx = self.gym.get_actor_index(env_ptr, robot_actor, gymapi.DOMAIN_SIM)
            self.robot_indices.append(robot_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, robot_actor):
                self.rigid_body_name_to_idx["robot/" + name] = (
                    self.gym.find_actor_rigid_body_index(env_ptr, robot_actor, name, gymapi.DOMAIN_ENV)
                )

            if self.with_dof_force_sensors:
                self.gym.enable_actor_dof_force_sensors(env_ptr, robot_actor)

            if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
                blue_robot_actor = self.gym.create_actor(
                    env_ptr, robot_asset, robot_pose, "blue_robot",
                    i + self.num_envs * 2, -1, 0,
                )
                self.gym.set_actor_dof_properties(env_ptr, blue_robot_actor, robot_dof_props)
                self.blue_robots.append(blue_robot_actor)
                self._set_actor_color(env_ptr, blue_robot_actor, (0, 0, 1))
                self.blue_robot_indices.append(
                    self.gym.get_actor_index(env_ptr, blue_robot_actor, gymapi.DOMAIN_SIM)
                )

            # Object (peg) — same asset every env.
            # The initial start pose we seed here is env-0's; actual per-env
            # starts are written into object_init_state below and applied by
            # reset_object_pose on the first reset.
            object_asset = object_assets[0]
            sp = self._pih_start_poses[int(self._pih_env_scene_idx[i]), 0]  # use peg 0 as seed
            env_obj_pose = gymapi.Transform()
            env_obj_pose.p = gymapi.Vec3(float(sp[0]), float(sp[1]), float(sp[2]))
            env_obj_pose.r = gymapi.Quat(float(sp[3]), float(sp[4]), float(sp[5]), float(sp[6]))
            object_handle = self.gym.create_actor(
                env_ptr, object_asset, env_obj_pose, "object", i, 0, 0,
            )
            object_init_state.append([
                env_obj_pose.p.x, env_obj_pose.p.y, env_obj_pose.p.z,
                env_obj_pose.r.x, env_obj_pose.r.y, env_obj_pose.r.z, env_obj_pose.r.w,
                0, 0, 0, 0, 0, 0,
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            object_indices.append(object_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, object_handle):
                self.rigid_body_name_to_idx["object/" + name] = (
                    self.gym.find_actor_rigid_body_index(env_ptr, object_handle, name, gymapi.DOMAIN_ENV)
                )

            object_scale = self.object_asset_scales[0]
            object_scales.append(object_scale)
            object_offsets = []
            for keypoint in self.keypoints_offsets:
                keypoint = copy(keypoint)
                for coord_idx in range(3):
                    keypoint[coord_idx] *= (
                        object_scale[coord_idx] * self.object_base_size * self.keypoint_scale / 2
                    )
                object_offsets.append(keypoint)
            object_keypoint_offsets.append(object_offsets)

            object_scale_fixed_size = self.cfg["env"]["fixedSize"]
            object_offsets_fixed_size = []
            for keypoint in self.keypoints_offsets:
                kfs = copy(keypoint)
                for coord_idx in range(3):
                    kfs[coord_idx] *= (
                        object_scale_fixed_size[coord_idx] * self.keypoint_scale / 2
                    )
                object_offsets_fixed_size.append(kfs)
            object_keypoint_offsets_fixed_size.append(object_offsets_fixed_size)

            # Table — per-env scene URDF
            urdf = self._pih_scene_urdfs[i]
            table_asset = table_assets[urdf_to_asset_idx[urdf]]
            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table_object", i, 0, 0,
            )
            table_init_state.append([
                table_pose.p.x, table_pose.p.y, table_pose.p.z,
                table_pose.r.x, table_pose.r.y, table_pose.r.z, table_pose.r.w,
                0, 0, 0, 0, 0, 0,
            ])
            table_indices.append(
                self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            )
            for name in self.gym.get_actor_rigid_body_names(env_ptr, table_handle):
                self.rigid_body_name_to_idx["table/" + name] = (
                    self.gym.find_actor_rigid_body_index(env_ptr, table_handle, name, gymapi.DOMAIN_ENV)
                )

            # Goal (marker)
            self._create_additional_objects(env_ptr, env_idx=i, object_asset_idx=0)

            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.objects.append(object_handle)

        # ── Post-env setup ──
        object_rb_props = self.gym.get_actor_rigid_body_properties(self.envs[0], object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(
            object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.table_init_state = to_torch(
            table_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.robot_indices = to_torch(self.robot_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(table_indices, dtype=torch.long, device=self.device)
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robot_indices = to_torch(self.blue_robot_indices, dtype=torch.long, device=self.device)

        self.object_scales = to_torch(object_scales, dtype=torch.float, device=self.device)
        self.object_keypoint_offsets = to_torch(object_keypoint_offsets, dtype=torch.float, device=self.device)
        self.object_keypoint_offsets_fixed_size = to_torch(
            object_keypoint_offsets_fixed_size, dtype=torch.float, device=self.device
        )

        self.joint_names = self.gym.get_actor_joint_names(env_ptr, robot_actor)
        props = self.gym.get_actor_dof_properties(env_ptr, robot_actor)
        self.joint_lower_limits = props["lower"]
        self.joint_upper_limits = props["upper"]

        # max_consecutive_successes defined by goal_mode / goals tensor shape.
        # (Parent's reward loop reads self.max_consecutive_successes.)
        self.max_consecutive_successes = self._pih_max_traj_len

        self._after_envs_created()

        try:
            tmp_assets_dir.cleanup()
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────────
    # Reset — resample peg_idx per episode
    # ────────────────────────────────────────────────────────────────

    def reset_object_pose(self, env_ids, reset_buf_idxs=None, tensor_reset=True):
        """Per reset-env: draw peg_idx ~ U[0, M) (or forced by config),
        load the corresponding start pose + trajectory length from scenes.npz."""
        if tensor_reset and len(env_ids) > 0 and reset_buf_idxs is None:
            # Snapshot prior episode's env_max_goals before overwriting (for logging).
            self.prev_episode_env_max_goals[env_ids] = self.env_max_goals[env_ids]

            # Optional override: ``forcePegIdx = int`` pins peg_idx to a
            # specific value for all resets. Used by the multi-init eval GUI.
            force_peg = self.cfg["env"].get("forcePegIdx", None)
            if force_peg is not None:
                new_peg = torch.full(
                    (len(env_ids),), int(force_peg), dtype=torch.long, device=self.device,
                )
            else:
                new_peg = torch.randint(
                    0, self._pih_num_pegs, (len(env_ids),), device=self.device
                )
            self.env_peg_idx[env_ids] = new_peg
            scene_ids = self._pih_env_scene_idx_t[env_ids]

            # Update per-env traj length (= env_max_goals).
            self.env_max_goals[env_ids] = self._pih_traj_lengths_t[scene_ids, new_peg]

            # Update per-env object_init_state xy + quat. Parent
            # (SimToolReal.reset_object_pose) sets z = table_reset_z +
            # tableObjectZOffset, so the peg starts hovering 10 cm above the
            # table and settles onto it via gravity. This mirrors fabrica's
            # multi_init_states behavior (fabrica_env.py:901-902) — the
            # cached z in scenes.npz is ignored at runtime.
            poses = self._pih_start_poses_t[scene_ids, new_peg]           # (len, 7)
            self.object_init_state[env_ids, 0:2] = poses[:, 0:2]
            self.object_init_state[env_ids, 3:7] = poses[:, 3:7]

            # Per-episode XY jitter on the goal position (observed by actor
            # only). When goalXyObsNoise=0, torch_rand_float(-0, +0, ...)
            # returns zero, so no gate is needed.
            self.goal_pos_obs_noise[env_ids, 0:2] = torch_rand_float(
                -self.cfg["env"]["goalXyObsNoise"],
                self.cfg["env"]["goalXyObsNoise"],
                (len(env_ids), 2), device=self.device,
            )
            # Z stays 0 (buffer is zero-init'd, never written on this axis).

        super().reset_object_pose(env_ids, reset_buf_idxs, tensor_reset)

    def _reset_target(self, env_ids, reset_buf_idxs=None, tensor_reset=True, is_first_goal=True):
        """Per reset-env: set goal_states to the current subgoal of the
        env's (scene_idx, peg_idx) trajectory. Mirrors FabricaEnv's
        multi_init_states path (lines 908-958) but without multi-part."""
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            USE_FIXED_GOAL_STATES = self.cfg["env"]["useFixedGoalStates"]
            if USE_FIXED_GOAL_STATES:
                current_subgoal_idx = (
                    self.successes[env_ids] % self.env_max_goals[env_ids]
                ).long()
                scene_ids = self._pih_env_scene_idx_t[env_ids]
                peg_ids = self.env_peg_idx[env_ids]
                goals = self._pih_goals_t[scene_ids, peg_ids, current_subgoal_idx]  # (len, 7)

                self.goal_states[env_ids, 0:7] = goals
                # Table z randomization delta (inherited from SimToolReal).
                table_base_z = self.cfg["env"]["tableResetZ"]
                delta_z = self.table_init_state[env_ids, 2:3] - table_base_z
                self.goal_states[env_ids, 2:3] += delta_z
            else:
                super()._reset_target(env_ids, reset_buf_idxs, tensor_reset, is_first_goal)
                return

            self.root_state_tensor[self.goal_object_indices[env_ids], 0:7] = (
                self.goal_states[env_ids, 0:7]
            )
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = (
                torch.zeros_like(
                    self.root_state_tensor[self.goal_object_indices[env_ids], 7:13]
                )
            )
            self.deferred_set_actor_root_state_tensor_indexed(
                [self.goal_object_indices[env_ids]]
            )
        if len(env_ids) > 0 and reset_buf_idxs is not None and tensor_reset:
            rs_ofs = self.root_state_resets.shape[1]
            self.root_state_tensor[self.goal_object_indices[env_ids], :] = (
                self.root_state_resets[env_ids, rs_ofs:]
            )

    # ────────────────────────────────────────────────────────────────
    # Retract reward + reset logic — verbatim copy from FabricaEnv
    # (adapted: multi_init_states path always taken, no multi_part loops).
    # ────────────────────────────────────────────────────────────────

    def _compute_resets(self, is_success):
        if not self.enable_retract:
            return super()._compute_resets(is_success)

        ones = torch.ones_like(self.reset_buf)
        zeros = torch.zeros_like(self.reset_buf)

        object_z_low = torch.where(self.object_pos[:, 2] < 0.1, ones, zeros)

        if self.max_consecutive_successes > 0:
            self.progress_buf = torch.where(
                is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf
            )
            max_consecutive_successes_reached = torch.where(
                self.retract_succeeded, ones, zeros,
            )
        else:
            max_consecutive_successes_reached = zeros

        max_episode_length_reached = torch.where(
            self.progress_buf >= self.max_episode_length - 1, ones, zeros
        )

        if self.with_table_force_sensor:
            # Config: tableForceResetThreshold (Newtons). Default 100 matches
            # the previously-hardcoded threshold. Override per run to raise
            # (e.g. for eval where we want to observe high-force inserts) or
            # lower (to train more compliant policies).
            table_force_threshold = float(
                self.cfg["env"].get("tableForceResetThreshold", 100.0)
            )
            table_force_too_high = torch.where(
                self.max_table_sensor_force_norm_smoothed > table_force_threshold,
                ones, zeros,
            )
        else:
            table_force_too_high = zeros

        hand_far_from_object = torch.where(
            (self.curr_fingertip_distances.max(dim=-1).values > 1.5) & ~self.retract_phase,
            ones, zeros,
        )

        if self.cfg["env"]["resetWhenDropped"]:
            dropped_z = self.object_init_state[:, 2]
            dropped = (
                torch.where(self.object_pos[:, 2] < dropped_z, ones, zeros) * self.lifted_object
            )
        else:
            dropped = zeros

        resets = (
            self.reset_buf
            | object_z_low
            | max_consecutive_successes_reached
            | max_episode_length_reached
            | table_force_too_high
            | hand_far_from_object
            | dropped
        )
        resets = self._extra_reset_rules(resets)

        self.retract_phase[resets.bool()] = False
        self.retract_succeeded[resets.bool()] = False

        return resets

    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        """Retract-aware reward, adapted from FabricaEnv::compute_kuka_reward."""
        if not self.enable_retract:
            return super().compute_kuka_reward()

        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew, hand_delta_penalty = self._distance_delta_rewards(lifted_object)
        keypoint_rew, keypoint_rew_fixed_size = self._keypoint_reward(lifted_object)
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            keypoint_rew = keypoint_rew_fixed_size

        if self.cfg["env"].get("finalGoalToleranceCurriculumEnabled", False):
            final_tol = self.final_goal_success_tolerance
        else:
            final_tol = self.cfg["env"].get("finalGoalSuccessTolerance", None)
        if final_tol is not None and self.cfg["env"]["useFixedGoalStates"]:
            is_final_goal = (self.successes == (self.env_max_goals - 1)) | self.retract_phase
            base_tol = self.success_tolerance * self.keypoint_scale
            tight_tol = final_tol * self.keypoint_scale
            keypoint_success_tolerance = torch.where(is_final_goal, tight_tol, base_tol)
        else:
            keypoint_success_tolerance = self.success_tolerance * self.keypoint_scale

        near_goal = self.keypoints_max_dist <= keypoint_success_tolerance
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            near_goal = self.keypoints_max_dist_fixed_size <= keypoint_success_tolerance

        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            self.near_goal_steps = (self.near_goal_steps + near_goal) * near_goal
        else:
            self.near_goal_steps += near_goal

        is_success = self.near_goal_steps >= self.success_steps
        is_success[self.retract_phase] = False
        goal_resets = is_success.clone()
        self.successes += is_success

        just_entered_retract = (self.successes >= self.env_max_goals) & ~self.retract_phase
        self.retract_phase |= just_entered_retract
        self.successes.clamp_(max=self.max_consecutive_successes)

        goal_resets[self.retract_phase] = False
        self.reset_goal_buf[:] = goal_resets

        object_lin_vel_penalty = -torch.sum(torch.square(self.object_linvel), dim=-1)
        object_ang_vel_penalty = -torch.sum(torch.square(self.object_angvel), dim=-1)

        self.rewards_episode["raw_fingertip_delta_rew"] += fingertip_delta_rew
        self.rewards_episode["raw_hand_delta_penalty"] += hand_delta_penalty
        self.rewards_episode["raw_lifting_rew"] += lifting_rew
        self.rewards_episode["raw_keypoint_rew"] += keypoint_rew
        self.rewards_episode["raw_object_lin_vel_penalty"] += object_lin_vel_penalty
        self.rewards_episode["raw_object_ang_vel_penalty"] += object_ang_vel_penalty

        fingertip_delta_rew *= self.distance_delta_rew_scale
        hand_delta_penalty *= self.distance_delta_rew_scale * 0
        lifting_rew *= self.lifting_rew_scale
        keypoint_rew *= self.keypoint_rew_scale
        object_lin_vel_penalty *= self.object_lin_vel_penalty_scale
        object_ang_vel_penalty *= self.object_ang_vel_penalty_scale

        kuka_actions_penalty, hand_actions_penalty = self._action_penalties()

        bonus_rew = near_goal * (self.reach_goal_bonus / self.success_steps)
        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            bonus_rew = is_success * self.reach_goal_bonus
        bonus_rew[self.retract_phase & ~just_entered_retract] = 0.0

        base_reward = (
            fingertip_delta_rew
            + hand_delta_penalty
            + lifting_rew
            + lift_bonus_rew
            + keypoint_rew
            + kuka_actions_penalty
            + hand_actions_penalty
            + bonus_rew
            + object_lin_vel_penalty
            + object_ang_vel_penalty
        )

        retract_rew = torch.zeros_like(base_reward)
        if self.retract_phase.any():
            retract_tol = self.retract_success_tolerance * self.keypoint_scale
            object_at_goal = (self.keypoints_max_dist <= retract_tol).float()
            mean_fingertip_dist = self.curr_fingertip_distances.mean(dim=-1)
            retract_dist_rew = mean_fingertip_dist * self.retract_reward_scale * object_at_goal
            just_retracted = (
                (mean_fingertip_dist > self.retract_distance_threshold)
                & self.retract_phase
                & ~self.retract_succeeded
                & object_at_goal.bool()
            )
            self.retract_succeeded |= just_retracted
            retract_bonus = just_retracted.float() * self.retract_success_bonus
            retract_rew = (retract_dist_rew + retract_bonus) * self.retract_phase.float()

        already_in_retract = self.retract_phase & ~just_entered_retract
        retract_env_reward = kuka_actions_penalty + hand_actions_penalty + retract_rew
        reward = torch.where(already_in_retract, retract_env_reward, base_reward)

        self.rew_buf[:] = reward

        self.extras["retract_phase_ratio"] = self.retract_phase.float().mean().item()
        self.extras["retract_success_ratio"] = self.retract_succeeded.float().mean().item()
        self.extras["retract_success_tolerance"] = self.retract_success_tolerance

        resets = self._compute_resets(is_success)
        self.reset_buf[:] = resets

        if self.cfg["env"]["forceNoReset"]:
            self.reset_buf[:] = False

        self.extras["successes"] = self.prev_episode_successes
        self.extras["success_ratio"] = (
            self.prev_episode_successes / self.prev_episode_env_max_goals.float()
        ).mean().item()
        self.extras["all_goals_hit_ratio"] = (
            self.prev_episode_successes >= self.prev_episode_env_max_goals
        ).float().mean().item()
        self.extras["closest_keypoint_max_dist"] = self.prev_episode_closest_keypoint_max_dist
        self.extras["final_goal_tolerance"] = self.final_goal_success_tolerance
        self.true_objective = self._true_objective()
        self.extras["true_objective"] = self.true_objective

        rewards = [
            (fingertip_delta_rew, "fingertip_delta_rew"),
            (hand_delta_penalty, "hand_delta_penalty"),
            (lifting_rew, "lifting_rew"),
            (lift_bonus_rew, "lift_bonus_rew"),
            (keypoint_rew, "keypoint_rew"),
            (kuka_actions_penalty, "kuka_actions_penalty"),
            (hand_actions_penalty, "hand_actions_penalty"),
            (bonus_rew, "bonus_rew"),
            (object_lin_vel_penalty, "object_lin_vel_penalty"),
            (object_ang_vel_penalty, "object_ang_vel_penalty"),
            (retract_rew, "retract_rew"),
            (reward, "total_reward"),
        ]

        episode_cumulative = dict()
        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
            episode_cumulative[rew_name] = rew_value
        self.extras["rewards_episode"] = self.rewards_episode
        self.extras["episode_cumulative"] = episode_cumulative

        return self.rew_buf, is_success

    # ────────────────────────────────────────────────────────────────
    # Actor-only goal XY noise (states_buf / critic stays clean).
    # ────────────────────────────────────────────────────────────────

    def populate_obs_and_states_buffers(self):
        super().populate_obs_and_states_buffers()
        # Parent built states_buf (L3225) with clean keypoints_rel_goal and
        # then built obs_buf (L3289) also clean. Here we post-hoc subtract
        # the per-episode XY offset from the obs_buf slice that encodes
        # keypoints_rel_goal — goal shifts by +noise => obs shifts by -noise.
        # states_buf is untouched → asymmetric critic stays on clean goal.
        self.obs_buf[:, self._goal_kp_obs_slice].view(
            self.num_envs, self.num_keypoints, 3
        ).sub_(self.goal_pos_obs_noise.unsqueeze(1))

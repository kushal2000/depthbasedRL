"""FabricaEnv: SimToolReal subclass with retract-after-insertion reward
and multi-part training support.

After the robot completes all insertion goals, it enters a retract phase
where it is rewarded for moving its hand away from the object while the
object remains at the final goal pose.

Controlled by config:
  env.enableRetract (bool): Master switch. When False, behaves identically
      to SimToolReal (episode resets after final goal).
  env.retractRewardScale (float): Scale factor for retract distance reward.
  env.retractDistanceThreshold (float): Min hand-object distance to count
      as successfully retracted.
  env.retractSuccessBonus (float): One-time bonus when hand clears threshold.

Multi-part training (env.multiPart=True):
  Trains multiple assembly parts simultaneously. Each env is assigned a part
  via round-robin. Requires per-part object names, scene URDFs, start poses,
  and goal trajectories. See FabricaEnv.yaml for config fields.
"""

import json
import os
import tempfile
from copy import copy
from typing import List, Tuple

import torch
from torch import Tensor

from isaacgym import gymapi
from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
from isaacgymenvs.tasks.simtoolreal.utils import populate_dof_properties
from isaacgymenvs.utils.torch_jit_utils import (
    get_axis_params,
    to_torch,
)


class FabricaEnv(SimToolReal):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        self.enable_retract = cfg["env"].get("enableRetract", False)
        self.retract_reward_scale = cfg["env"].get("retractRewardScale", 1.0)
        self.retract_distance_threshold = cfg["env"].get("retractDistanceThreshold", 0.1)
        self.retract_success_bonus = cfg["env"].get("retractSuccessBonus", 1000.0)
        self.retract_success_tolerance = cfg["env"].get("retractSuccessTolerance", 0.005)

        # Multi-part config — read before super().__init__ which calls _create_envs
        self.multi_part = cfg["env"].get("multiPart", False)
        self.multi_init_states = cfg["env"].get("multiInitStates", False)
        self.final_goal_only = cfg["env"].get("finalGoalOnly", False)
        self.pre_insert_and_final = cfg["env"].get("preInsertAndFinal", False)
        assert not (self.final_goal_only and self.pre_insert_and_final), (
            "finalGoalOnly and preInsertAndFinal are mutually exclusive"
        )
        if self.multi_init_states:
            assert self.multi_part, "multiInitStates requires multiPart=True"
            assert self.enable_retract, "multiInitStates requires enableRetract=True"
        if self.multi_part:
            self._init_multi_part_config(cfg)
            # Set objectName to first part so parent code that references it works
            # (e.g., viser viewer, eval logging)
            cfg["env"]["objectName"] = self._mp_object_names[0]
            # Enable fixed goal states (trajectories are handled per-env in overrides)
            cfg["env"]["useFixedGoalStates"] = True
            cfg["env"]["useFixedInitObjectPose"] = True
            # Set single-part fields to first part's values for parent compatibility
            cfg["env"]["objectStartPose"] = self._mp_start_poses[0]
            cfg["env"]["fixedGoalStates"] = self._mp_goal_states[0]
            cfg["env"]["asset"]["table"] = self._mp_table_urdfs[0]
        else:
            # Single-part: truncate subgoal list in-place so parent SimToolReal
            # builds trajectory_states and max_consecutive_successes from just
            # the final waypoint. Start pose is untouched.
            if (
                self.final_goal_only
                and cfg["env"].get("useFixedGoalStates", False)
                and cfg["env"].get("fixedGoalStates")
            ):
                fgs = cfg["env"]["fixedGoalStates"]
                cfg["env"]["fixedGoalStates"] = [fgs[-1]]
                print(
                    f"[FabricaEnv] finalGoalOnly=True (single-part): "
                    f"truncated fixedGoalStates from {len(fgs)} → 1"
                )
            elif (
                self.pre_insert_and_final
                and cfg["env"].get("useFixedGoalStates", False)
                and cfg["env"].get("fixedGoalStates")
            ):
                fgs = cfg["env"]["fixedGoalStates"]
                cfg["env"]["fixedGoalStates"] = (
                    [fgs[-2], fgs[-1]] if len(fgs) >= 2 else [fgs[-1]]
                )
                print(
                    f"[FabricaEnv] preInsertAndFinal=True (single-part): "
                    f"truncated fixedGoalStates from {len(fgs)} → "
                    f"{len(cfg['env']['fixedGoalStates'])}"
                )

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        # Upgrade viser viewer with multi-part mesh swapping
        if self.multi_part and self.viser_viz_enabled:
            self._upgrade_viser_for_multi_part()

        # Per-env multi-init-states tracking
        if self.multi_init_states:
            self.env_scene_idx = torch.randint(
                0, self._ms_num_scenes, (self.num_envs,), device=self.device
            )
            self.env_max_goals = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            for part_idx in range(self._mp_num_parts):
                mask = self.env_part_idx == part_idx
                if mask.any():
                    self.env_max_goals[mask] = self._ms_traj_lengths[
                        part_idx, self.env_scene_idx[mask]
                    ]
            # Snapshot of env_max_goals from the just-ended episode, captured in
            # reset_object_pose before env_max_goals is overwritten. Used by the
            # success_ratio / all_goals_hit_ratio metrics so the numerator
            # (prev_episode_successes) and denominator come from the same episode.
            self.prev_episode_env_max_goals = self.env_max_goals.clone()

        # Per-env retract state
        self.retract_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.retract_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Register retract_rew in episode reward tracking
        self.rewards_episode["retract_rew"] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    # ── Multi-part initialization ──────────────────────────────────

    @staticmethod
    def _parse_object_name(obj_name: str):
        """Parse object name like 'beam_2_coacd' into (assembly, part_id, method).

        Returns (assembly, part_id, method) where method may be None for vhacd objects.
        """
        KNOWN_METHODS = {"coacd", "sdf"}
        parts = obj_name.split("_")
        if parts[-1] in KNOWN_METHODS:
            method = parts[-1]
            rest = parts[:-1]
        else:
            method = None
            rest = parts
        part_id = rest[-1]
        assembly = "_".join(rest[:-1])
        return assembly, part_id, method

    def _init_multi_part_config(self, cfg):
        """Parse multi-part config. Only objectNames is required — everything
        else (scene URDFs, start poses, trajectories) is auto-derived."""
        env_cfg = cfg["env"]
        self._mp_object_names = env_cfg["objectNames"]
        assert self._mp_object_names is not None and len(self._mp_object_names) > 0, (
            "multiPart requires objectNames to be a non-empty list"
        )
        self._mp_num_parts = len(self._mp_object_names)

        # Resolve repo root for finding trajectory JSONs
        repo_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../.."
        )

        self._mp_table_urdfs = []
        self._mp_start_poses = []
        self._mp_goal_states = []

        for obj_name in self._mp_object_names:
            assembly, part_id, method = self._parse_object_name(obj_name)

            # Scene URDF
            if method is not None:
                scene_urdf = f"urdf/fabrica/{assembly}/environments/{part_id}/scene_{method}.urdf"
            else:
                scene_urdf = f"urdf/fabrica/{assembly}/environments/{part_id}/scene.urdf"
            self._mp_table_urdfs.append(scene_urdf)

            # Trajectory JSON → start pose + goals
            traj_path = os.path.join(
                repo_root, "assets", "urdf", "fabrica", assembly, "trajectories", part_id, "pick_place.json"
            )
            with open(traj_path, "r") as f:
                traj = json.load(f)
            self._mp_start_poses.append(traj["start_pose"])
            self._mp_goal_states.append(traj["goals"])

        # Validate all trajectories have same length
        traj_lens = [len(g) for g in self._mp_goal_states]
        assert len(set(traj_lens)) == 1, (
            f"All parts must have the same number of goals, got {traj_lens}"
        )

        # Truncate to the final waypoint per part before downstream code
        # builds tensors and derives max_consecutive_successes from
        # len(_mp_goal_states[0]). Start poses are untouched.
        if self.final_goal_only:
            orig_len = traj_lens[0]
            self._mp_goal_states = [[goals[-1]] for goals in self._mp_goal_states]
            print(
                f"[FabricaEnv] finalGoalOnly=True (multi-part): "
                f"truncated per-part goals from {orig_len} → 1"
            )
        elif self.pre_insert_and_final:
            orig_len = traj_lens[0]
            self._mp_goal_states = [
                ([goals[-2], goals[-1]] if len(goals) >= 2 else [goals[-1]])
                for goals in self._mp_goal_states
            ]
            print(
                f"[FabricaEnv] preInsertAndFinal=True (multi-part): "
                f"truncated per-part goals from {orig_len} → "
                f"{len(self._mp_goal_states[0])}"
            )

        print(f"[FabricaEnv] Multi-part training with {self._mp_num_parts} parts: {self._mp_object_names}")
        for i, name in enumerate(self._mp_object_names):
            print(f"  Part {i}: {name} → scene={self._mp_table_urdfs[i]}, "
                  f"start={self._mp_start_poses[i][:3]}, goals={len(self._mp_goal_states[i])}")

        if self.multi_init_states:
            self._load_multi_init_states(cfg, repo_root)

    def _load_multi_init_states(self, cfg, repo_root):
        """Load scenes.npz for multi-init-states training."""
        import numpy as np

        # All parts must share the same assembly
        assemblies = set()
        config_part_ids = []
        for obj_name in self._mp_object_names:
            assembly, part_id, _ = self._parse_object_name(obj_name)
            assemblies.add(assembly)
            config_part_ids.append(part_id)
        assert len(assemblies) == 1, (
            f"multiInitStates requires all parts from the same assembly, got {assemblies}"
        )
        assembly = assemblies.pop()

        # Load scenes.npz
        scenes_path = os.path.join(
            repo_root, "assets", "urdf", "fabrica", assembly, "scenes.npz"
        )
        scenes = np.load(scenes_path)
        start_poses = scenes["start_poses"]    # [num_scenes, num_parts_in_order, 7]
        goals = scenes["goals"]                # [num_scenes, num_parts_in_order, max_traj_len, 7]
        traj_lengths = scenes["traj_lengths"]  # [num_scenes, num_parts_in_order]

        # Load assembly_order to map scenes.npz part indices → config part indices
        order_path = os.path.join(
            repo_root, "assets", "urdf", "fabrica", assembly, "assembly_order.json"
        )
        with open(order_path, "r") as f:
            assembly_order = json.load(f)["steps"]  # e.g., ["6", "2", "0", "3", "1"]

        # Build reorder mapping: config_part_idx → scenes.npz part index
        order_to_npz_idx = {pid: i for i, pid in enumerate(assembly_order)}
        reorder = []
        for pid in config_part_ids:
            assert pid in order_to_npz_idx, (
                f"Part {pid} not found in assembly_order {assembly_order}"
            )
            reorder.append(order_to_npz_idx[pid])

        # Reorder to match config ordering and transpose to [num_parts, num_scenes, ...]
        # start_poses: [num_scenes, num_parts, 7] → index by reorder → transpose
        start_poses = start_poses[:, reorder, :]           # [S, P_config, 7]
        goals = goals[:, reorder, :, :]                    # [S, P_config, max_traj, 7]
        traj_lengths = traj_lengths[:, reorder]             # [S, P_config]

        self._ms_num_scenes = start_poses.shape[0]
        self._ms_max_traj_len = int(goals.shape[2])

        # Store as tensors [num_parts, num_scenes, ...] for efficient per-part indexing
        self._ms_start_poses = torch.tensor(
            start_poses.transpose(1, 0, 2), dtype=torch.float32
        )  # [P, S, 7]
        self._ms_goals = torch.tensor(
            goals.transpose(1, 0, 2, 3), dtype=torch.float32
        )  # [P, S, max_traj, 7]
        self._ms_traj_lengths = torch.tensor(
            traj_lengths.T, dtype=torch.long
        )  # [P, S]

        # finalGoalOnly: gather per-(part, scene) final goal — scenes.npz
        # zero-pads past per-(scene, part) traj_length so index -1 is unsafe;
        # index by (traj_length - 1) instead. Start poses untouched.
        if self.final_goal_only:
            P, S, _, _ = self._ms_goals.shape
            final_idx = self._ms_traj_lengths - 1  # [P, S], already long
            p_ar = torch.arange(P).unsqueeze(1).expand(P, S)
            s_ar = torch.arange(S).unsqueeze(0).expand(P, S)
            final_goals = self._ms_goals[p_ar, s_ar, final_idx]  # [P, S, 7]
            self._ms_goals = final_goals.unsqueeze(2)  # [P, S, 1, 7]
            self._ms_max_traj_len = 1
            self._ms_traj_lengths = torch.ones_like(self._ms_traj_lengths)
            print(
                f"[FabricaEnv] finalGoalOnly=True (multi-init): "
                f"truncated _ms_goals to shape {tuple(self._ms_goals.shape)}"
            )
        elif self.pre_insert_and_final:
            # Per-(part, scene): take the last two valid waypoints. Scenes
            # whose traj_length is 1 keep a single final goal (the "pre" slot
            # duplicates the final — but traj_lengths stays 1, so downstream
            # code only reads slot 0).
            P, S, _, _ = self._ms_goals.shape
            final_idx = self._ms_traj_lengths - 1                       # [P, S]
            pre_idx = torch.clamp(self._ms_traj_lengths - 2, min=0)     # [P, S]
            p_ar = torch.arange(P).unsqueeze(1).expand(P, S)
            s_ar = torch.arange(S).unsqueeze(0).expand(P, S)
            pre_goals = self._ms_goals[p_ar, s_ar, pre_idx]             # [P, S, 7]
            final_goals = self._ms_goals[p_ar, s_ar, final_idx]         # [P, S, 7]
            self._ms_goals = torch.stack([pre_goals, final_goals], dim=2)  # [P, S, 2, 7]
            self._ms_max_traj_len = 2
            self._ms_traj_lengths = torch.clamp(self._ms_traj_lengths, max=2)
            print(
                f"[FabricaEnv] preInsertAndFinal=True (multi-init): "
                f"truncated _ms_goals to shape {tuple(self._ms_goals.shape)}"
            )

        print(f"[FabricaEnv] Multi-init-states: {self._ms_num_scenes} scenes, "
              f"max_traj_len={self._ms_max_traj_len}, "
              f"traj_lengths range={int(self._ms_traj_lengths.min())}-{int(self._ms_traj_lengths.max())}")
        print(f"  Part order mapping (config→npz): {list(zip(config_part_ids, reorder))}")

    def _upgrade_viser_for_multi_part(self):
        """Inject multi-part mesh swapping into the existing viser viewer."""
        from pathlib import Path

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )

        # Build per-part URDF paths
        part_object_urdf_paths = [Path(f) for f in self.object_asset_files]
        part_scene_urdf_paths = [Path(asset_root) / u for u in self._mp_table_urdfs]

        # Build env→part map
        env_part_map = [i % self._mp_num_parts for i in range(self.num_envs)]

        # Inject into viewer
        self.viser_viewer._env_part_map = env_part_map
        self.viser_viewer._part_object_urdf_paths = part_object_urdf_paths
        self.viser_viewer._part_scene_urdf_paths = part_scene_urdf_paths
        self.viser_viewer._part_names = list(self._mp_object_names)
        self.viser_viewer._current_displayed_part = 0

        print(f"[FabricaEnv] Viser viewer upgraded for {self._mp_num_parts} parts")

    # ── Override asset loading for multi-part ─────────────────────

    def _main_object_assets_and_scales(self, object_asset_root, tmp_assets_dir):
        """Override to load multiple object assets when multiPart is enabled."""
        if not self.multi_part:
            return super()._main_object_assets_and_scales(object_asset_root, tmp_assets_dir)

        # Load assets for ALL parts
        all_files = []
        all_scales = []
        all_need_vhacds = []
        for obj_name in self._mp_object_names:
            obj = NAME_TO_OBJECT[obj_name]
            all_files.append(obj.urdf_path)
            all_scales.append(obj.scale)
            all_need_vhacds.append(obj.need_vhacd)

        # Set up trajectory states as a list (per-part)
        # The parent's _main_object_assets_and_scales sets self.trajectory_states
        # for single-part. For multi-part, we build per-part tensors.
        self._mp_trajectory_states = []
        for goals in self._mp_goal_states:
            self._mp_trajectory_states.append(
                torch.tensor(goals, device=self.device)
            )

        # Set self.trajectory_states to the first part's trajectory so parent code
        # that reads len(self.trajectory_states) for max_consecutive_successes works.
        self.trajectory_states = self._mp_trajectory_states[0]
        self.max_consecutive_successes = len(self.trajectory_states)

        if self.multi_init_states:
            self.max_consecutive_successes = self._ms_max_traj_len
            # Move scene tensors to device now that self.device is available
            self._ms_start_poses = self._ms_start_poses.to(self.device)
            self._ms_goals = self._ms_goals.to(self.device)
            self._ms_traj_lengths = self._ms_traj_lengths.to(self.device)

        # Store per-part file/scale/vhacd info for _load_main_object_asset
        self.object_asset_files = all_files
        self.object_asset_scales = all_scales
        self.object_need_vhacds = all_need_vhacds

        return all_files, all_scales, all_need_vhacds

    def _load_main_object_asset(self):
        """Override to return per-part shape/body counts and assets."""
        if not self.multi_part:
            return super()._load_main_object_asset()

        object_assets = []
        per_part_rb = []
        per_part_shapes = []
        for object_asset_file, need_vhacd in zip(
            self.object_asset_files, self.object_need_vhacds
        ):
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.vhacd_enabled = need_vhacd
            if self.cfg["env"].get("useSDF", False):
                object_asset_options.thickness = 0.0
            object_asset_options.collapse_fixed_joints = True
            object_asset_options.replace_cylinder_with_capsule = True

            object_asset_dir = os.path.dirname(object_asset_file)
            object_asset_fname = os.path.basename(object_asset_file)
            asset = self.gym.load_asset(
                self.sim, object_asset_dir, object_asset_fname, object_asset_options
            )
            object_assets.append(asset)

            rb = self.gym.get_asset_rigid_body_count(asset)
            shapes = self.gym.get_asset_rigid_shape_count(asset)
            per_part_rb.append(rb)
            per_part_shapes.append(shapes)
            print(f"[FabricaEnv] Object asset {object_asset_fname}: {rb} bodies, {shapes} shapes")

        max_rb_count = max(per_part_rb)
        max_shapes_count = max(per_part_shapes)
        print(f"[FabricaEnv] Max object: {max_rb_count} bodies, {max_shapes_count} shapes")
        return object_assets, max_rb_count, max_shapes_count, per_part_rb, per_part_shapes

    def _load_additional_assets(self, object_asset_root, arm_pose):
        """Override to load goal assets for each part and return per-part counts."""
        if not self.multi_part:
            return super()._load_additional_assets(object_asset_root, arm_pose)

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True
        object_asset_options.collapse_fixed_joints = True
        object_asset_options.replace_cylinder_with_capsule = True

        self.goal_assets = []
        per_part_goal_rb = []
        per_part_goal_shapes = []
        for object_asset_file in self.object_asset_files:
            object_asset_dir = os.path.dirname(object_asset_file)
            object_asset_fname = os.path.basename(object_asset_file)
            goal_asset_ = self.gym.load_asset(
                self.sim, object_asset_dir, object_asset_fname, object_asset_options
            )
            self.goal_assets.append(goal_asset_)
            rb = self.gym.get_asset_rigid_body_count(goal_asset_)
            shapes = self.gym.get_asset_rigid_shape_count(goal_asset_)
            per_part_goal_rb.append(rb)
            per_part_goal_shapes.append(shapes)

        max_goal_rb = max(per_part_goal_rb)
        max_goal_shapes = max(per_part_goal_shapes)
        return max_goal_rb, max_goal_shapes, per_part_goal_rb, per_part_goal_shapes

    # ── Override _create_envs for multi-part ──────────────────────

    def _create_envs(self, num_envs, spacing, num_per_row):
        """Override to support per-env table assets, start poses, and part assignment."""
        if not self.multi_part:
            return super()._create_envs(num_envs, spacing, num_per_row)

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

        print(f"Loading asset {self.robot_asset_file} from {asset_root}")
        robot_asset = self.gym.load_asset(
            self.sim, asset_root, self.robot_asset_file, asset_options
        )
        print(f"Loaded asset {robot_asset}")

        self.num_hand_arm_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_hand_arm_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        num_hand_arm_dofs = self.gym.get_asset_dof_count(robot_asset)
        assert self.num_hand_arm_dofs == num_hand_arm_dofs

        max_agg_bodies = 0
        max_agg_shapes = 0

        robot_rigid_body_names = [
            self.gym.get_asset_rigid_body_name(robot_asset, i)
            for i in range(self.num_hand_arm_bodies)
        ]
        print(f"Robot num rigid bodies: {self.num_hand_arm_bodies}")
        print(f"Robot rigid bodies: {robot_rigid_body_names}")

        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        for i in range(self.num_hand_arm_dofs):
            self.arm_hand_dof_lower_limits.append(robot_dof_props["lower"][i])
            self.arm_hand_dof_upper_limits.append(robot_dof_props["upper"][i])
        self.arm_hand_dof_lower_limits = to_torch(self.arm_hand_dof_lower_limits, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(self.arm_hand_dof_upper_limits, device=self.device)

        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx)) + gymapi.Vec3(0.0, 0.8, 0)
        robot_pose.r = gymapi.Quat(0, 0, 0, 1)

        # ── Object assets (multiple parts) ──
        object_assets, object_rb_count, object_shapes_count, per_part_obj_rb, per_part_obj_shapes = self._load_main_object_asset()

        # ── Table assets (one per part, with collapse_fixed_joints) ──
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.collapse_fixed_joints = True  # Critical: uniform body count
        if self.cfg["env"].get("useSDF", False):
            table_asset_options.thickness = 0.0

        table_assets = []
        per_part_table_rb = []
        per_part_table_shapes = []
        max_table_rb = 0
        max_table_shapes = 0
        for table_urdf in self._mp_table_urdfs:
            table_asset = self.gym.load_asset(
                self.sim, asset_root, table_urdf, table_asset_options
            )
            table_assets.append(table_asset)
            rb = self.gym.get_asset_rigid_body_count(table_asset)
            shapes = self.gym.get_asset_rigid_shape_count(table_asset)
            per_part_table_rb.append(rb)
            per_part_table_shapes.append(shapes)
            max_table_rb = max(max_table_rb, rb)
            max_table_shapes = max(max_table_shapes, shapes)
            print(f"[FabricaEnv] Table asset {table_urdf}: {rb} bodies, {shapes} shapes")

        if self.with_table_force_sensor:
            table_sensor_pose = gymapi.Transform()
            table_sensor_props = gymapi.ForceSensorProperties()
            table_sensor_props.enable_constraint_solver_forces = True
            table_sensor_props.enable_forward_dynamics_forces = False
            table_sensor_props.use_world_frame = True
            # Create sensor on first table asset (all have same collapsed structure)
            self.table_sensor_idx = self.gym.create_asset_force_sensor(
                asset=table_assets[0], body_idx=0,
                local_pose=table_sensor_pose, props=table_sensor_props,
            )
            if self.table_sensor_idx == -1:
                raise ValueError("Failed to create table force sensor")
            # Create sensors on remaining table assets
            for ta in table_assets[1:]:
                self.gym.create_asset_force_sensor(
                    asset=ta, body_idx=0,
                    local_pose=table_sensor_pose, props=table_sensor_props,
                )

        print(f"[FabricaEnv] Max table: {max_table_rb} bodies, {max_table_shapes} shapes")

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3()
        table_pose.p.x = robot_pose.p.x
        table_pose_dy, table_pose_dz = -0.8, self.cfg["env"]["tableResetZ"]
        table_pose.p.y = robot_pose.p.y + table_pose_dy
        table_pose.p.z = robot_pose.p.z + table_pose_dz

        # ── Goal (additional) assets ──
        additional_rb, additional_shapes, per_part_goal_rb, per_part_goal_shapes = self._load_additional_assets(
            object_asset_root, robot_pose
        )

        # ── Compute max_agg as max-of-per-part-sums (not sum-of-maxes) ──
        # Each env only has one (object, table, goal) combination, so we compute
        # the actual worst-case per environment instead of over-allocating.
        for i in range(self._mp_num_parts):
            env_bodies = self.num_hand_arm_bodies + per_part_obj_rb[i] + per_part_table_rb[i] + per_part_goal_rb[i]
            env_shapes = self.num_hand_arm_shapes + per_part_obj_shapes[i] + per_part_table_shapes[i] + per_part_goal_shapes[i]
            max_agg_bodies = max(max_agg_bodies, env_bodies)
            max_agg_shapes = max(max_agg_shapes, env_shapes)

        old_max_shapes = self.num_hand_arm_shapes + object_shapes_count + max_table_shapes + additional_shapes
        print(f"[FabricaEnv] max_agg_shapes: {max_agg_shapes} (was {old_max_shapes} with sum-of-maxes)")

        # ── Per-part start poses ──
        object_start_poses = []
        for sp in self._mp_start_poses:
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(sp[0], sp[1], sp[2])
            pose.r = gymapi.Quat(sp[3], sp[4], sp[5], sp[6])
            object_start_poses.append(pose)

        # Set self.object_start_pose to first part (used by _create_additional_objects)
        self.object_start_pose = object_start_poses[0]

        # ── Per-env part assignment ──
        self.env_part_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        for i in range(num_envs):
            self.env_part_idx[i] = i % self._mp_num_parts

        # ── Create envs ──
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

        # Sanity checks
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

        # Set asset friction properties
        MODIFY_ASSET_FRICTIONS = self.cfg["env"]["modifyAssetFrictions"]
        if MODIFY_ASSET_FRICTIONS:
            self.set_robot_asset_rigid_shape_properties(
                robot_asset=robot_asset,
                friction=self.cfg["env"]["robotFriction"],
                fingertip_friction=self.cfg["env"]["fingerTipFriction"],
            )
            for ta in table_assets:
                self.set_table_asset_rigid_shape_properties(
                    table_asset=ta, friction=self.cfg["env"]["tableFriction"],
                )
            for oa in object_assets:
                self.set_object_asset_rigid_shape_properties(
                    object_asset=oa, friction=self.cfg["env"]["objectFriction"],
                )
        else:
            self.set_robot_asset_rigid_shape_properties(
                robot_asset=robot_asset, friction=None, fingertip_friction=None,
            )

        print(f"[FabricaEnv] Creating {num_envs} envs with max_agg_bodies={max_agg_bodies}, max_agg_shapes={max_agg_shapes}")

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            part_idx = i % self._mp_num_parts

            # Robot
            collision_group = i
            collision_filter = -1
            segmentation_id = 0
            robot_actor = self.gym.create_actor(
                env_ptr, robot_asset, robot_pose, "robot",
                collision_group, collision_filter, segmentation_id,
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
                blue_robot_idx = self.gym.get_actor_index(env_ptr, blue_robot_actor, gymapi.DOMAIN_SIM)
                self.blue_robot_indices.append(blue_robot_idx)

            # Object — per-part asset and start pose
            object_asset_idx = part_idx
            object_asset = object_assets[object_asset_idx]
            start_pose = object_start_poses[part_idx]

            object_handle = self.gym.create_actor(
                env_ptr, object_asset, start_pose, "object", i, 0, 0
            )
            object_init_state.append([
                start_pose.p.x, start_pose.p.y, start_pose.p.z,
                start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w,
                0, 0, 0, 0, 0, 0,
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            object_indices.append(object_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, object_handle):
                self.rigid_body_name_to_idx["object/" + name] = (
                    self.gym.find_actor_rigid_body_index(env_ptr, object_handle, name, gymapi.DOMAIN_ENV)
                )

            object_scale = self.object_asset_scales[object_asset_idx]
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
                keypoint_fixed_size = copy(keypoint)
                for coord_idx in range(3):
                    keypoint_fixed_size[coord_idx] *= (
                        object_scale_fixed_size[coord_idx] * self.keypoint_scale / 2
                    )
                object_offsets_fixed_size.append(keypoint_fixed_size)
            object_keypoint_offsets_fixed_size.append(object_offsets_fixed_size)

            # Table — per-part scene asset
            table_asset = table_assets[part_idx]
            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table_object", i, 0, 0
            )
            table_init_state.append([
                table_pose.p.x, table_pose.p.y, table_pose.p.z,
                table_pose.r.x, table_pose.r.y, table_pose.r.z, table_pose.r.w,
                0, 0, 0, 0, 0, 0,
            ])
            table_object_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            table_indices.append(table_object_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, table_handle):
                self.rigid_body_name_to_idx["table/" + name] = (
                    self.gym.find_actor_rigid_body_index(env_ptr, table_handle, name, gymapi.DOMAIN_ENV)
                )

            # Goal object — per-part
            self._create_additional_objects(env_ptr, env_idx=i, object_asset_idx=object_asset_idx)

            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.objects.append(object_handle)

        # ── Post-creation setup (mirrors parent) ──
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

        print(f"Robot joint names: {self.joint_names}")

        self._after_envs_created()

        try:
            tmp_assets_dir.cleanup()
        except Exception:
            pass

    def _create_additional_objects(self, env_ptr, env_idx, object_asset_idx):
        """Override to use per-env start pose for goal placement."""
        if not self.multi_part:
            return super()._create_additional_objects(env_ptr, env_idx, object_asset_idx)

        self.goal_displacement = gymapi.Vec3(-0.35, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z],
            device=self.device,
        )

        # Use per-part start pose for goal placement
        part_idx = env_idx % self._mp_num_parts
        sp = self._mp_start_poses[part_idx]
        start_p = gymapi.Vec3(sp[0], sp[1], sp[2])

        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = start_p + self.goal_displacement
        goal_start_pose.p.z -= 0.04

        goal_asset = self.goal_assets[object_asset_idx]
        goal_handle = self.gym.create_actor(
            env_ptr, goal_asset, goal_start_pose, "goal_object",
            env_idx + self.num_envs, 0, 0,
        )
        goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
        self.goal_object_indices.append(goal_object_idx)
        for name in self.gym.get_actor_rigid_body_names(env_ptr, goal_handle):
            self.rigid_body_name_to_idx["goal/" + name] = (
                self.gym.find_actor_rigid_body_index(env_ptr, goal_handle, name, gymapi.DOMAIN_ENV)
            )

    # ── Override reset_object_pose for multi-init-states ────────

    def reset_object_pose(self, env_ids, reset_buf_idxs=None, tensor_reset=True):
        """Override to sample new scenes and update start poses."""
        if self.multi_init_states and tensor_reset and len(env_ids) > 0 and reset_buf_idxs is None:
            # Snapshot the just-ended episode's max_goals before overwriting.
            # Pairs with prev_episode_successes (captured in SimToolReal.reset_idx).
            self.prev_episode_env_max_goals[env_ids] = self.env_max_goals[env_ids]

            self.env_scene_idx[env_ids] = torch.randint(
                0, self._ms_num_scenes, (len(env_ids),), device=self.device
            )
            for part_idx in range(self._mp_num_parts):
                mask = self.env_part_idx[env_ids] == part_idx
                if mask.any():
                    scene_ids = self.env_scene_idx[env_ids[mask]]
                    self.env_max_goals[env_ids[mask]] = self._ms_traj_lengths[part_idx, scene_ids]
                    poses = self._ms_start_poses[part_idx, scene_ids]
                    self.object_init_state[env_ids[mask], 0:2] = poses[:, 0:2]
                    self.object_init_state[env_ids[mask], 3:7] = poses[:, 3:7]

        super().reset_object_pose(env_ids, reset_buf_idxs, tensor_reset)

    # ── Override _reset_target for per-env trajectories ───────────

    def _reset_target(self, env_ids, reset_buf_idxs=None, tensor_reset=True, is_first_goal=True):
        """Override to use per-env trajectory states in multi-part mode."""
        if not self.multi_part:
            return super()._reset_target(env_ids, reset_buf_idxs, tensor_reset, is_first_goal)

        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            USE_FIXED_GOAL_STATES = self.cfg["env"]["useFixedGoalStates"]
            if USE_FIXED_GOAL_STATES:
                if self.multi_init_states:
                    current_subgoal_idx = (self.successes[env_ids] % self.env_max_goals[env_ids]).long()
                    goals = torch.zeros(len(env_ids), 7, device=self.device)
                    for part_idx in range(self._mp_num_parts):
                        mask = self.env_part_idx[env_ids] == part_idx
                        if mask.any():
                            scene_ids = self.env_scene_idx[env_ids[mask]]
                            sub_ids = current_subgoal_idx[mask]
                            goals[mask] = self._ms_goals[part_idx, scene_ids, sub_ids]
                else:
                    num_goals = len(self._mp_trajectory_states[0])
                    current_subgoal_idx = (self.successes[env_ids] % num_goals).long()
                    goals = torch.zeros(len(env_ids), 7, device=self.device)
                    for part_idx in range(self._mp_num_parts):
                        mask = self.env_part_idx[env_ids] == part_idx
                        if mask.any():
                            traj = self._mp_trajectory_states[part_idx]
                            sub_ids = current_subgoal_idx[mask]
                            goals[mask] = traj[sub_ids]

                self.goal_states[env_ids, 0:7] = goals

                # Adjust goal z for table height randomization
                table_base_z = self.cfg["env"]["tableResetZ"]
                delta_z = self.table_init_state[env_ids, 2:3] - table_base_z
                self.goal_states[env_ids, 2:3] += delta_z
            else:
                # Fall back to parent's non-fixed goal logic
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

    # ── Retract reward and reset (unchanged from before) ──────────

    def _compute_resets(self, is_success):
        """Override to enter retract phase instead of resetting on max successes."""
        if not self.enable_retract:
            return super()._compute_resets(is_success)

        ones = torch.ones_like(self.reset_buf)
        zeros = torch.zeros_like(self.reset_buf)

        object_z_low = torch.where(self.object_pos[:, 2] < 0.1, ones, zeros)

        if self.max_consecutive_successes > 0:
            self.progress_buf = torch.where(
                is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf
            )
            # Reset on retract success, not max_consecutive_successes
            max_consecutive_successes_reached = torch.where(
                self.retract_succeeded, ones, zeros
            )
        else:
            max_consecutive_successes_reached = zeros

        max_episode_length_reached = torch.where(
            self.progress_buf >= self.max_episode_length - 1, ones, zeros
        )

        if self.with_table_force_sensor:
            TABLE_FORCE_THRESHOLD = 100.0
            table_force_too_high = torch.where(
                self.max_table_sensor_force_norm_smoothed > TABLE_FORCE_THRESHOLD,
                ones, zeros,
            )
        else:
            table_force_too_high = zeros

        # During retract phase, don't reset for hand far from object
        hand_far_from_object = torch.where(
            (self.curr_fingertip_distances.max(dim=-1).values > 1.5) & ~self.retract_phase,
            ones, zeros,
        )

        if self.cfg["env"]["resetWhenDropped"]:
            dropped_z = self.object_init_state[:, 2]
            dropped = (
                torch.where(self.object_pos[:, 2] < dropped_z, ones, zeros)
                * self.lifted_object
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

        # Clear retract state for envs that are resetting
        self.retract_phase[resets.bool()] = False
        self.retract_succeeded[resets.bool()] = False

        return resets

    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        """Override to add retract phase after all insertion goals are reached.

        For envs NOT in retract phase: identical to SimToolReal.
        For envs IN retract phase: action penalties + retract reward only.

        Retract phase is detected at the END of reward computation so the
        transition frame (hitting final goal) gets the full base reward
        including the goal bonus. Retract reward starts next frame.
        """
        if not self.enable_retract:
            return super().compute_kuka_reward()

        # ── Base reward components (identical to SimToolReal) ──
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
            max_goals = self.env_max_goals if self.multi_init_states else self.max_consecutive_successes
            is_final_goal = (self.successes == (max_goals - 1)) | self.retract_phase
            base_tol = self.success_tolerance * self.keypoint_scale
            tight_tol = final_tol * self.keypoint_scale
            keypoint_success_tolerance = torch.where(is_final_goal, tight_tol, base_tol)
        else:
            keypoint_success_tolerance = self.success_tolerance * self.keypoint_scale

        near_goal: Tensor = self.keypoints_max_dist <= keypoint_success_tolerance
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            near_goal = self.keypoints_max_dist_fixed_size <= keypoint_success_tolerance

        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            self.near_goal_steps = (self.near_goal_steps + near_goal) * near_goal
        else:
            self.near_goal_steps += near_goal

        is_success = self.near_goal_steps >= self.success_steps
        # Suppress is_success during retract — prevents progress_buf reset
        # (which would block episode timeout) and spurious success signals
        is_success[self.retract_phase] = False
        goal_resets = is_success.clone()
        self.successes += is_success

        # ── Detect retract phase entry (before setting reset_goal_buf) ──
        max_goals = self.env_max_goals if self.multi_init_states else self.max_consecutive_successes
        just_entered_retract = (
            (self.successes >= max_goals) & ~self.retract_phase
        )
        self.retract_phase |= just_entered_retract
        self.successes.clamp_(max=self.max_consecutive_successes)

        # Suppress goal cycling for all retract envs (including just-entered)
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
        hand_delta_penalty *= self.distance_delta_rew_scale * 0  # currently disabled
        lifting_rew *= self.lifting_rew_scale
        keypoint_rew *= self.keypoint_rew_scale
        object_lin_vel_penalty *= self.object_lin_vel_penalty_scale
        object_ang_vel_penalty *= self.object_ang_vel_penalty_scale

        kuka_actions_penalty, hand_actions_penalty = self._action_penalties()

        bonus_rew = near_goal * (self.reach_goal_bonus / self.success_steps)
        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            bonus_rew = is_success * self.reach_goal_bonus
        # No goal bonus during retract (except transition frame — keep final goal bonus)
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

        # ── Retract reward (for envs already in retract phase) ──
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

        # Log retract metrics BEFORE _compute_resets clears state for resetting envs
        self.extras["retract_phase_ratio"] = self.retract_phase.float().mean().item()
        self.extras["retract_success_ratio"] = self.retract_succeeded.float().mean().item()
        self.extras["retract_success_tolerance"] = self.retract_success_tolerance

        resets = self._compute_resets(is_success)
        self.reset_buf[:] = resets

        if self.cfg["env"]["forceNoReset"]:
            self.reset_buf[:] = False

        # ── Logging ──
        self.extras["successes"] = self.prev_episode_successes
        if self.multi_init_states:
            self.extras["success_ratio"] = (
                self.prev_episode_successes / self.prev_episode_env_max_goals.float()
            ).mean().item()
            self.extras["all_goals_hit_ratio"] = (
                self.prev_episode_successes >= self.prev_episode_env_max_goals
            ).float().mean().item()
        else:
            self.extras["success_ratio"] = (
                self.prev_episode_successes.mean().item() / self.max_consecutive_successes
            )
            self.extras["all_goals_hit_ratio"] = (
                self.prev_episode_successes >= self.max_consecutive_successes
            ).float().mean().item()
        self.extras["closest_keypoint_max_dist"] = self.prev_episode_closest_keypoint_max_dist
        self.extras["final_goal_tolerance"] = self.final_goal_success_tolerance
        self.true_objective = self._true_objective()
        self.extras["true_objective"] = self.true_objective

        # Per-part metrics
        if self.multi_part:
            for part_idx in range(self._mp_num_parts):
                part_mask = self.env_part_idx == part_idx
                part_name = self._mp_object_names[part_idx]
                if part_mask.any():
                    if self.multi_init_states:
                        self.extras[f"success_ratio/{part_name}"] = (
                            self.prev_episode_successes[part_mask]
                            / self.prev_episode_env_max_goals[part_mask].float()
                        ).mean().item()
                        self.extras[f"all_goals_hit_ratio/{part_name}"] = (
                            self.prev_episode_successes[part_mask] >= self.prev_episode_env_max_goals[part_mask]
                        ).float().mean().item()
                    else:
                        self.extras[f"success_ratio/{part_name}"] = (
                            self.prev_episode_successes[part_mask].mean().item()
                            / self.max_consecutive_successes
                        )
                        self.extras[f"all_goals_hit_ratio/{part_name}"] = (
                            self.prev_episode_successes[part_mask] >= self.max_consecutive_successes
                        ).float().mean().item()
                    self.extras[f"retract_success_ratio/{part_name}"] = (
                        self.retract_succeeded[part_mask].float().mean().item()
                    )

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

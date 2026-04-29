"""FabricaEnv: SimToolReal subclass for fabrica single-insertion training.

Loads a pre-baked ``scenes.npz`` (produced by
``fabrica/scene_generation/generate_scenes.py``) and a set of per-(part, scene)
static-fixture URDFs under ``assets/urdf/fabrica/<assemblyName>/insertion_scenes/``.

Per-env axes:
  - **Static** (baked at env creation): ``(part_idx, scene_idx) = (env_idx // N, env_idx % N) mod (P*N)``.
    Each env trains exactly one ``(insertion_part, partial_assembly_position)`` combo.
  - **Dynamic per reset**: ``start_idx`` uniform in ``[0, M)``. The policy sees
    a fresh inserting-part start pose (and the cached trajectory for that start)
    each episode.

Goal modes:
  - ``dense``              — all per-(part, scene, start) waypoints.
  - ``preInsertAndFinal``  — keep only ``[goals[tl-2], goals[tl-1]]``.
  - ``finalGoalOnly``      — keep only ``[goals[tl-1]]``.

Retract phase: after the inserting part reaches the final goal, the hand is
rewarded for moving away while the part stays at the final goal pose.

Goal-XY observation noise: per-episode XY offset (uniform in
``±goalXyObsNoise``) is subtracted from the ``keypoints_rel_goal`` slice of
``obs_buf``. The asymmetric critic (``states_buf``) stays clean. Identical
mechanism to ``peg_in_hole_env.py``.
"""

from __future__ import annotations

import json
import os
import tempfile
from copy import copy
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from isaacgym import gymapi
from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
from isaacgymenvs.tasks.simtoolreal.utils import populate_dof_properties
from isaacgymenvs.utils.torch_jit_utils import (
    get_axis_params,
    to_torch,
    torch_rand_float,
)


VALID_GOAL_MODES = ("dense", "preInsertAndFinal", "finalGoalOnly")


class FabricaEnv(SimToolReal):
    ASSETS_SUBDIR = "fabrica"

    # ────────────────────────────────────────────────────────────────
    # __init__
    # ────────────────────────────────────────────────────────────────

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        # Retract knobs
        self.enable_retract = cfg["env"].get("enableRetract", False)
        self.retract_reward_scale = cfg["env"].get("retractRewardScale", 1.0)
        self.retract_distance_threshold = cfg["env"].get("retractDistanceThreshold", 0.1)
        self.retract_success_bonus = cfg["env"].get("retractSuccessBonus", 1000.0)
        self.retract_success_tolerance = cfg["env"].get("retractSuccessTolerance", 0.005)

        # Goal-mode enum
        self.goal_mode = cfg["env"].get("goalMode", "dense")
        assert self.goal_mode in VALID_GOAL_MODES, (
            f"goalMode must be one of {VALID_GOAL_MODES}, got {self.goal_mode!r}"
        )

        # Load scenes.npz, build per-env round-robin indices, populate
        # self._si_* numpy arrays + self._mp_object_names / _mp_table_urdfs /
        # _mp_env_table_urdfs / per-part placeholder start pose + goal list.
        self._init_single_insertion_config(cfg)

        # Env-0 placeholder fields so parent SimToolReal.__init__ can read a
        # valid fixed object pose / goals / table asset path. Real per-env
        # values are applied during _create_envs / reset_object_pose.
        cfg["env"]["objectName"] = self._mp_object_names[0]
        cfg["env"]["useFixedGoalStates"] = True
        cfg["env"]["useFixedInitObjectPose"] = True
        cfg["env"]["objectStartPose"] = self._mp_start_poses[0]
        cfg["env"]["fixedGoalStates"] = self._mp_goal_states[0]
        cfg["env"]["asset"]["table"] = self._mp_env_table_urdfs[0]

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        # Move scene tensors to device now that self.device is set by parent.
        self._si_start_poses_t = torch.tensor(
            self._si_start_poses, dtype=torch.float32, device=self.device
        )                                                                   # (P, N, M, 7)
        self._si_goals_t = torch.tensor(
            self._si_goals, dtype=torch.float32, device=self.device
        )                                                                   # (P, N, M, T, 7)
        self._si_traj_lengths_t = torch.tensor(
            self._si_traj_lengths, dtype=torch.long, device=self.device
        )                                                                   # (P, N, M)
        self._si_partial_offsets_t = torch.tensor(
            self._si_partial_offsets, dtype=torch.float32, device=self.device
        )                                                                   # (P, N, 3)

        # Per-env static indices (filled in _init_single_insertion_config as numpy).
        self._si_env_part_idx_t = torch.tensor(
            self._si_env_part_idx, dtype=torch.long, device=self.device
        )                                                                   # (num_envs,)
        self._si_env_scene_idx_t = torch.tensor(
            self._si_env_scene_idx, dtype=torch.long, device=self.device
        )                                                                   # (num_envs,)

        # Per-env dynamic index (re-sampled every reset).
        self._si_env_start_idx_t = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Backwards-name aliases used by parent / reward / extras code paths.
        self.env_part_idx = self._si_env_part_idx_t
        self.env_scene_idx = self._si_env_scene_idx_t

        # Per-env max goals (depends on (part, scene, start); resampled every
        # reset). Seed to start_idx=0 for the first reward step's denominator.
        self.env_max_goals = self._si_traj_lengths_t[
            self._si_env_part_idx_t, self._si_env_scene_idx_t, 0
        ].clone()
        self.prev_episode_env_max_goals = self.env_max_goals.clone()

        # Per-env retract state.
        self.retract_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.retract_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rewards_episode["retract_rew"] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        # ── Goal-XY observation noise (actor-only, mirrors peg-in-hole) ──
        self.goal_pos_obs_noise = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device,
        )
        # Cached obs_buf slice for keypoints_rel_goal. See
        # SimToolReal.populate_obs_and_states_buffers for the field layout.
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

        # Viser viewer multi-part hook (per-part mesh swapping).
        if self.viser_viz_enabled:
            self._upgrade_viser_for_multi_part()

    # ────────────────────────────────────────────────────────────────
    # Single-insertion scene loading
    # ────────────────────────────────────────────────────────────────

    def _init_single_insertion_config(self, cfg):
        """Load scenes.npz, apply goalMode truncation, build per-env indexing.

        Sets the following attributes (all numpy / list at this stage —
        torchified after super().__init__):
          - ``_si_insertion_parts``    : list[str]              — part ids, e.g. ["2","0","3","1"]
          - ``_si_n_scenes``           : int                    — N
          - ``_si_m_starts``           : int                    — M
          - ``_si_max_traj_len``       : int                    — T (after truncation)
          - ``_si_start_poses``        : float32 (P, N, M, 7)
          - ``_si_goals``              : float32 (P, N, M, T, 7)
          - ``_si_traj_lengths``       : int64   (P, N, M)
          - ``_si_partial_offsets``    : float32 (P, N, 3)
          - ``_si_scene_urdf_paths``   : object  (P, N) of str (relative to assets/)
          - ``_si_env_part_idx``       : int64   (num_envs,)    — round-robin
          - ``_si_env_scene_idx``      : int64   (num_envs,)    — round-robin
          - ``_mp_object_names``       : list[str]              — registered insertion-part assets
          - ``_mp_num_parts``          : int                    — P
          - ``_mp_env_table_urdfs``    : list[str]              — len = num_envs
          - ``_mp_table_urdfs``        : list[str]              — len = P (one URDF per part, for viser)
          - ``_mp_start_poses``        : list[list[float]]      — per-part placeholder for env-0 init
          - ``_mp_goal_states``        : list[list[list[float]]]— per-part placeholder for env-0 init
        """
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
        )
        assembly = cfg["env"]["assemblyName"]
        scenes_filename = cfg["env"].get("scenesFilename", "scenes.npz")
        scenes_path = os.path.join(
            repo_root, "assets", "urdf", self.ASSETS_SUBDIR, assembly, scenes_filename
        )
        assert os.path.exists(scenes_path), (
            f"{scenes_filename} not found at {scenes_path}. Run "
            f"`python -m fabrica.scene_generation.generate_scenes "
            f"--assembly {assembly} ...` first."
        )
        data = np.load(scenes_path, allow_pickle=True)

        insertion_parts = [str(p) for p in data["insertion_parts"].tolist()]  # list[str], len P
        start_poses = data["start_poses"].astype(np.float32)            # (P, N, M, 7)
        goals = data["goals"].astype(np.float32)                        # (P, N, M, T, 7)
        traj_lengths = data["traj_lengths"].astype(np.int64)            # (P, N, M)
        partial_offsets = data["partial_assembly_offsets"].astype(np.float32)  # (P, N, 3)
        scene_urdf_paths = data["scene_urdf_paths"]                     # object (P, N) of str

        P, N, M, _ = start_poses.shape
        T = goals.shape[3]
        assert goals.shape == (P, N, M, T, 7), goals.shape
        assert traj_lengths.shape == (P, N, M)
        assert partial_offsets.shape == (P, N, 3)
        assert scene_urdf_paths.shape == (P, N)

        # ── Goal-mode truncation ────────────────────────────────
        if self.goal_mode == "finalGoalOnly":
            p_ar, n_ar, m_ar = np.indices((P, N, M))
            final_idx = traj_lengths - 1
            final_goals = goals[p_ar, n_ar, m_ar, final_idx]            # (P, N, M, 7)
            goals = final_goals[:, :, :, None, :]                        # (P, N, M, 1, 7)
            traj_lengths = np.ones_like(traj_lengths)
            T = 1
            print(f"[FabricaEnv] goalMode=finalGoalOnly: goals truncated to {goals.shape}")
        elif self.goal_mode == "preInsertAndFinal":
            p_ar, n_ar, m_ar = np.indices((P, N, M))
            final_idx = traj_lengths - 1
            pre_idx = np.clip(traj_lengths - 2, a_min=0, a_max=None)
            pre_goals = goals[p_ar, n_ar, m_ar, pre_idx]                 # (P, N, M, 7)
            final_goals = goals[p_ar, n_ar, m_ar, final_idx]             # (P, N, M, 7)
            goals = np.stack([pre_goals, final_goals], axis=3)           # (P, N, M, 2, 7)
            traj_lengths = np.clip(traj_lengths, a_min=None, a_max=2)
            T = 2
            print(f"[FabricaEnv] goalMode=preInsertAndFinal: goals truncated to {goals.shape}")
        # else dense — leave as-is.

        # Stash final tensors (still numpy here; torchified after super init).
        self._si_insertion_parts = insertion_parts
        self._si_n_scenes = N
        self._si_m_starts = M
        self._si_max_traj_len = T
        self._si_start_poses = start_poses
        self._si_goals = goals
        self._si_traj_lengths = traj_lengths
        self._si_partial_offsets = partial_offsets
        self._si_scene_urdf_paths = scene_urdf_paths

        # ── Per-env round-robin (part_idx, scene_idx) assignment ──
        num_envs = cfg["env"]["numEnvs"]
        combo_count = P * N
        combo_ids = np.arange(num_envs) % combo_count                    # (num_envs,)
        self._si_env_part_idx = (combo_ids // N).astype(np.int64)
        self._si_env_scene_idx = (combo_ids % N).astype(np.int64)

        # ── Optional override: pin all envs to a specific (part, scene) combo.
        # Used by fabrica_multi_init_eval.py to drive a single (part, scene).
        force_p = int(cfg["env"].get("forcePartIdx", -1))
        force_n = int(cfg["env"].get("forceSceneIdx", -1))
        if force_p >= 0:
            assert 0 <= force_p < P, f"forcePartIdx={force_p} out of range [0, {P})"
            self._si_env_part_idx[:] = force_p
        if force_n >= 0:
            assert 0 <= force_n < N, f"forceSceneIdx={force_n} out of range [0, {N})"
            self._si_env_scene_idx[:] = force_n
        if force_p >= 0 or force_n >= 0:
            print(
                f"[FabricaEnv] pinned (part_idx, scene_idx) = "
                f"({force_p if force_p >= 0 else 'round-robin'}, "
                f"{force_n if force_n >= 0 else 'round-robin'})"
            )

        # Per-env table URDF path (length num_envs).
        self._mp_env_table_urdfs = [
            str(scene_urdf_paths[self._si_env_part_idx[i], self._si_env_scene_idx[i]])
            for i in range(num_envs)
        ]

        # Per-PART URDF list (length P) — used by the viser viewer for per-part
        # mesh swapping. Pick scene_0000 as the canonical fixture for each part.
        self._mp_table_urdfs = [str(scene_urdf_paths[p, 0]) for p in range(P)]

        # Inserting-part object asset names (registered by fabrica.objects via
        # the {assembly}_{pid}_coacd convention).
        self._mp_object_names = [f"{assembly}_{pid}_coacd" for pid in insertion_parts]
        self._mp_num_parts = P
        for name in self._mp_object_names:
            assert name in NAME_TO_OBJECT, (
                f"Object {name!r} not registered in NAME_TO_OBJECT. "
                f"Did you run scale_assembly.py for assembly={assembly!r}?"
            )

        # ── Per-part placeholder start pose + goal list for env-0 init ──
        # The parent SimToolReal needs a single objectStartPose + fixedGoalStates
        # to allocate buffers; the real per-env values come from _si_*_t and are
        # applied in reset_object_pose / _reset_target. Pick scene 0, start 0.
        self._mp_start_poses = []
        self._mp_goal_states = []
        for p_idx in range(P):
            sp = start_poses[p_idx, 0, 0]                                # (7,) xyzw
            self._mp_start_poses.append(sp.tolist())
            tl = int(traj_lengths[p_idx, 0, 0])
            gs = goals[p_idx, 0, 0, :tl].tolist()                        # list of [7]
            self._mp_goal_states.append(gs)

        print(
            f"[FabricaEnv] assembly={assembly!r}, insertion_parts={insertion_parts}, "
            f"P={P}, N={N}, M={M}, T={T} (goal_mode={self.goal_mode})"
        )
        print(
            f"[FabricaEnv] {num_envs} envs → {combo_count} unique (part, scene) "
            f"combos ({num_envs / combo_count:.2f}x coverage)"
        )

    # ────────────────────────────────────────────────────────────────
    # Viser viewer hook
    # ────────────────────────────────────────────────────────────────

    def _upgrade_viser_for_multi_part(self):
        """Inject per-part mesh swapping into the existing viser viewer."""
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )
        part_object_urdf_paths = [Path(f) for f in self.object_asset_files]
        # _mp_table_urdfs is the per-PART list (length P), one canonical scene
        # URDF per insertion part — used only for viser preview, not env loading.
        part_scene_urdf_paths = [Path(asset_root) / u for u in self._mp_table_urdfs]
        env_part_map = self._si_env_part_idx.tolist()

        self.viser_viewer._env_part_map = env_part_map
        self.viser_viewer._part_object_urdf_paths = part_object_urdf_paths
        self.viser_viewer._part_scene_urdf_paths = part_scene_urdf_paths
        self.viser_viewer._part_names = list(self._mp_object_names)
        self.viser_viewer._current_displayed_part = 0

        print(f"[FabricaEnv] Viser viewer upgraded for {self._mp_num_parts} parts")

    # ────────────────────────────────────────────────────────────────
    # Asset loading hooks (always multi-part path)
    # ────────────────────────────────────────────────────────────────

    def _main_object_assets_and_scales(self, object_asset_root, tmp_assets_dir):
        all_files, all_scales, all_need_vhacds = [], [], []
        for obj_name in self._mp_object_names:
            obj = NAME_TO_OBJECT[obj_name]
            all_files.append(obj.urdf_path)
            all_scales.append(obj.scale)
            all_need_vhacds.append(obj.need_vhacd)

        # Build per-part trajectory tensors (the env-0 placeholder slice is
        # used for parent buffer allocation; per-env trajectories come from
        # _si_goals_t at runtime via _reset_target).
        self._mp_trajectory_states = [
            torch.tensor(g, device=self.device) for g in self._mp_goal_states
        ]
        self.trajectory_states = self._mp_trajectory_states[0]
        self.max_consecutive_successes = self._si_max_traj_len

        self.object_asset_files = all_files
        self.object_asset_scales = all_scales
        self.object_need_vhacds = all_need_vhacds
        return all_files, all_scales, all_need_vhacds

    def _load_main_object_asset(self):
        object_assets = []
        per_part_rb = []
        per_part_shapes = []
        for object_asset_file, need_vhacd in zip(
            self.object_asset_files, self.object_need_vhacds
        ):
            opts = gymapi.AssetOptions()
            opts.vhacd_enabled = need_vhacd
            if self.cfg["env"].get("useSDF", False):
                opts.thickness = 0.0
            opts.collapse_fixed_joints = True
            opts.replace_cylinder_with_capsule = True

            asset_dir = os.path.dirname(object_asset_file)
            asset_fname = os.path.basename(object_asset_file)
            asset = self.gym.load_asset(self.sim, asset_dir, asset_fname, opts)
            object_assets.append(asset)
            per_part_rb.append(self.gym.get_asset_rigid_body_count(asset))
            per_part_shapes.append(self.gym.get_asset_rigid_shape_count(asset))
            print(
                f"[FabricaEnv] Object asset {asset_fname}: "
                f"{per_part_rb[-1]} bodies, {per_part_shapes[-1]} shapes"
            )

        max_rb = max(per_part_rb)
        max_shapes = max(per_part_shapes)
        print(f"[FabricaEnv] Max object: {max_rb} bodies, {max_shapes} shapes")
        return object_assets, max_rb, max_shapes, per_part_rb, per_part_shapes

    def _load_additional_assets(self, object_asset_root, arm_pose):
        opts = gymapi.AssetOptions()
        opts.disable_gravity = True
        opts.collapse_fixed_joints = True
        opts.replace_cylinder_with_capsule = True

        self.goal_assets = []
        per_part_goal_rb = []
        per_part_goal_shapes = []
        for object_asset_file in self.object_asset_files:
            asset_dir = os.path.dirname(object_asset_file)
            asset_fname = os.path.basename(object_asset_file)
            goal_asset = self.gym.load_asset(self.sim, asset_dir, asset_fname, opts)
            self.goal_assets.append(goal_asset)
            per_part_goal_rb.append(self.gym.get_asset_rigid_body_count(goal_asset))
            per_part_goal_shapes.append(self.gym.get_asset_rigid_shape_count(goal_asset))

        return max(per_part_goal_rb), max(per_part_goal_shapes), per_part_goal_rb, per_part_goal_shapes

    # ────────────────────────────────────────────────────────────────
    # _create_envs (per-env table URDFs, deduped by path)
    # ────────────────────────────────────────────────────────────────

    def _create_envs(self, num_envs, spacing, num_per_row):
        if self.should_load_initial_states:
            self.load_initial_states()

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )
        tmp_assets_dir = tempfile.TemporaryDirectory()
        self.object_asset_files, self.object_asset_scales, self.object_need_vhacds = (
            self._main_object_assets_and_scales(asset_root, tmp_assets_dir.name)
        )

        # ── Robot asset ──
        ao = gymapi.AssetOptions()
        ao.fix_base_link = True
        ao.flip_visual_attachments = False
        ao.collapse_fixed_joints = True
        ao.disable_gravity = True
        ao.thickness = 0.0 if self.cfg["env"].get("useSDF", False) else 0.001
        ao.angular_damping = 0.01
        ao.linear_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            ao.use_physx_armature = True
        ao.default_dof_drive_mode = gymapi.DOF_MODE_POS

        print(f"Loading asset {self.robot_asset_file} from {asset_root}")
        robot_asset = self.gym.load_asset(self.sim, asset_root, self.robot_asset_file, ao)
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

        # ── Object assets (per insertion part) ──
        (object_assets, object_rb_count, object_shapes_count,
         per_part_obj_rb, per_part_obj_shapes) = self._load_main_object_asset()

        # ── Table assets: dedupe by URDF path. With P*N unique scenes we load
        #    each URDF exactly once. Per-part body/shape counts come from one
        #    representative scene per insertion part (scenes within a part share
        #    the same skeletal structure — only the joint origins differ).
        table_ao = gymapi.AssetOptions()
        table_ao.disable_gravity = True
        table_ao.fix_base_link = True
        table_ao.collapse_fixed_joints = True
        table_ao.vhacd_enabled = False  # rest-of-fixture meshes use canonical OBJs as-is
        if self.cfg["env"].get("useSDF", False):
            table_ao.thickness = 0.0

        # Union of (a) per-env URDFs actually used and (b) per-part rep URDFs
        # (needed for body/shape counts even if no env uses that part — can
        # happen when num_envs < P*N and round-robin doesn't cover every part).
        unique_paths = sorted(set(self._mp_env_table_urdfs) | set(self._mp_table_urdfs))
        table_asset_by_path: dict = {}
        for path in unique_paths:
            table_asset_by_path[path] = self.gym.load_asset(
                self.sim, asset_root, path, table_ao
            )

        per_part_table_rb = [0] * self._mp_num_parts
        per_part_table_shapes = [0] * self._mp_num_parts
        for p_idx in range(self._mp_num_parts):
            rep_path = self._mp_table_urdfs[p_idx]
            rep_asset = table_asset_by_path[rep_path]
            per_part_table_rb[p_idx] = self.gym.get_asset_rigid_body_count(rep_asset)
            per_part_table_shapes[p_idx] = self.gym.get_asset_rigid_shape_count(rep_asset)

        max_table_rb = max(per_part_table_rb)
        max_table_shapes = max(per_part_table_shapes)
        print(
            f"[FabricaEnv] {len(table_asset_by_path)} unique table URDFs loaded; "
            f"per-part body/shape counts = "
            f"{list(zip(per_part_table_rb, per_part_table_shapes))}; "
            f"max table {max_table_rb} bodies / {max_table_shapes} shapes"
        )

        if self.with_table_force_sensor:
            table_sensor_pose = gymapi.Transform()
            table_sensor_props = gymapi.ForceSensorProperties()
            table_sensor_props.enable_constraint_solver_forces = True
            table_sensor_props.enable_forward_dynamics_forces = False
            table_sensor_props.use_world_frame = True
            first_path = unique_paths[0]
            self.table_sensor_idx = self.gym.create_asset_force_sensor(
                asset=table_asset_by_path[first_path], body_idx=0,
                local_pose=table_sensor_pose, props=table_sensor_props,
            )
            if self.table_sensor_idx == -1:
                raise ValueError("Failed to create table force sensor")
            for path in unique_paths[1:]:
                self.gym.create_asset_force_sensor(
                    asset=table_asset_by_path[path], body_idx=0,
                    local_pose=table_sensor_pose, props=table_sensor_props,
                )

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3()
        table_pose.p.x = robot_pose.p.x
        table_pose_dy, table_pose_dz = -0.8, self.cfg["env"]["tableResetZ"]
        table_pose.p.y = robot_pose.p.y + table_pose_dy
        table_pose.p.z = robot_pose.p.z + table_pose_dz

        # ── Goal (additional) assets per part ──
        (additional_rb, additional_shapes,
         per_part_goal_rb, per_part_goal_shapes) = self._load_additional_assets(
            asset_root, robot_pose
        )

        # ── Compute max_agg as max-of-per-part-sums ──
        for p_idx in range(self._mp_num_parts):
            env_bodies = (self.num_hand_arm_bodies + per_part_obj_rb[p_idx]
                          + per_part_table_rb[p_idx] + per_part_goal_rb[p_idx])
            env_shapes = (self.num_hand_arm_shapes + per_part_obj_shapes[p_idx]
                          + per_part_table_shapes[p_idx] + per_part_goal_shapes[p_idx])
            max_agg_bodies = max(max_agg_bodies, env_bodies)
            max_agg_shapes = max(max_agg_shapes, env_shapes)

        # ── Per-part placeholder start poses (used for actor creation; the
        #    real per-env start is written by reset_object_pose on first reset). ──
        object_start_poses = []
        for sp in self._mp_start_poses:
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(sp[0], sp[1], sp[2])
            pose.r = gymapi.Quat(sp[3], sp[4], sp[5], sp[6])
            object_start_poses.append(pose)
        self.object_start_pose = object_start_poses[0]

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

        MODIFY_ASSET_FRICTIONS = self.cfg["env"]["modifyAssetFrictions"]
        if MODIFY_ASSET_FRICTIONS:
            self.set_robot_asset_rigid_shape_properties(
                robot_asset=robot_asset,
                friction=self.cfg["env"]["robotFriction"],
                fingertip_friction=self.cfg["env"]["fingerTipFriction"],
            )
            for ta in table_asset_by_path.values():
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

        print(f"[FabricaEnv] Creating {num_envs} envs with "
              f"max_agg_bodies={max_agg_bodies}, max_agg_shapes={max_agg_shapes}")

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            part_idx = int(self._si_env_part_idx[i])
            scene_idx = int(self._si_env_scene_idx[i])

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
                blue_robot_idx = self.gym.get_actor_index(env_ptr, blue_robot_actor, gymapi.DOMAIN_SIM)
                self.blue_robot_indices.append(blue_robot_idx)

            # Object — per-part asset
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

            # Table — per-env scene asset (looked up by URDF path)
            env_table_path = self._mp_env_table_urdfs[i]
            table_asset = table_asset_by_path[env_table_path]
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

        # ── Post-creation setup ──
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
        self.goal_displacement = gymapi.Vec3(-0.35, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z],
            device=self.device,
        )
        part_idx = int(self._si_env_part_idx[env_idx])
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

    # ────────────────────────────────────────────────────────────────
    # reset_object_pose / _reset_target
    # ────────────────────────────────────────────────────────────────

    def reset_object_pose(self, env_ids, reset_buf_idxs=None, tensor_reset=True):
        """Per-reset: sample new start_idx, write start pose, sample goal noise."""
        if tensor_reset and len(env_ids) > 0 and reset_buf_idxs is None:
            # Snapshot last episode's max_goals before overwriting (used by
            # success_ratio metric pairing in compute_kuka_reward).
            self.prev_episode_env_max_goals[env_ids] = self.env_max_goals[env_ids]

            # Sample a fresh start_idx ∈ [0, M) per env, unless pinned by
            # forceStartIdx (used for eval). forceStartIdx is read every
            # reset so the eval GUI can change it without rebuilding the env.
            force_start = int(self.cfg["env"].get("forceStartIdx", -1))
            if force_start >= 0:
                new_start = torch.full(
                    (len(env_ids),), force_start, dtype=torch.long, device=self.device
                )
            else:
                new_start = torch.randint(
                    0, self._si_m_starts, (len(env_ids),), device=self.device
                )
            self._si_env_start_idx_t[env_ids] = new_start

            part_ids = self._si_env_part_idx_t[env_ids]
            scene_ids = self._si_env_scene_idx_t[env_ids]

            # Per-env max goals = traj length for this (part, scene, start).
            self.env_max_goals[env_ids] = self._si_traj_lengths_t[part_ids, scene_ids, new_start]

            # Per-env start pose. Z is overridden by SimToolReal (table_reset_z
            # + tableObjectZOffset), so the part hovers above the table and
            # settles via gravity; the cached z in scenes.npz is ignored at
            # runtime (mirrors PegInHoleEnv reset_object_pose).
            poses = self._si_start_poses_t[part_ids, scene_ids, new_start]   # (len, 7)
            self.object_init_state[env_ids, 0:2] = poses[:, 0:2]
            self.object_init_state[env_ids, 3:7] = poses[:, 3:7]

            # Per-episode goal-XY observation noise (actor-only; subtracted
            # from the keypoints_rel_goal slice in populate_obs_and_states_buffers).
            self.goal_pos_obs_noise[env_ids, 0:2] = torch_rand_float(
                -self.cfg["env"]["goalXyObsNoise"],
                self.cfg["env"]["goalXyObsNoise"],
                (len(env_ids), 2), device=self.device,
            )
            # Z stays 0 (buffer is zero-init'd, never written on this axis).

        super().reset_object_pose(env_ids, reset_buf_idxs, tensor_reset)

    def _reset_target(self, env_ids, reset_buf_idxs=None, tensor_reset=True, is_first_goal=True):
        """Per reset-env: write goal_states from _si_goals_t for the env's
        (part_idx, scene_idx, start_idx) and current subgoal index."""
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            current_subgoal_idx = (
                self.successes[env_ids] % self.env_max_goals[env_ids]
            ).long()
            part_ids = self._si_env_part_idx_t[env_ids]
            scene_ids = self._si_env_scene_idx_t[env_ids]
            start_ids = self._si_env_start_idx_t[env_ids]
            goals = self._si_goals_t[part_ids, scene_ids, start_ids, current_subgoal_idx]
            self.goal_states[env_ids, 0:7] = goals

            # Apply table-z randomization delta (same as parent path).
            table_base_z = self.cfg["env"]["tableResetZ"]
            delta_z = self.table_init_state[env_ids, 2:3] - table_base_z
            self.goal_states[env_ids, 2:3] += delta_z

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
    # Retract reward + reset (preserved from the previous FabricaEnv)
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
                self.retract_succeeded, ones, zeros
            )
        else:
            max_consecutive_successes_reached = zeros

        max_episode_length_reached = torch.where(
            self.progress_buf >= self.max_episode_length - 1, ones, zeros
        )

        if self.with_table_force_sensor:
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
        self.retract_phase[resets.bool()] = False
        self.retract_succeeded[resets.bool()] = False
        return resets

    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
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

        near_goal: Tensor = self.keypoints_max_dist <= keypoint_success_tolerance
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

        just_entered_retract = (
            (self.successes >= self.env_max_goals) & ~self.retract_phase
        )
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

        # ── Logging ──
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

        # Per-part metrics.
        for part_idx in range(self._mp_num_parts):
            part_mask = self._si_env_part_idx_t == part_idx
            part_name = self._mp_object_names[part_idx]
            if part_mask.any():
                self.extras[f"success_ratio/{part_name}"] = (
                    self.prev_episode_successes[part_mask]
                    / self.prev_episode_env_max_goals[part_mask].float()
                ).mean().item()
                self.extras[f"all_goals_hit_ratio/{part_name}"] = (
                    self.prev_episode_successes[part_mask] >= self.prev_episode_env_max_goals[part_mask]
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

    # ────────────────────────────────────────────────────────────────
    # Actor-only goal-XY noise (states_buf / critic stays clean).
    # ────────────────────────────────────────────────────────────────

    def populate_obs_and_states_buffers(self):
        super().populate_obs_and_states_buffers()
        # Parent built states_buf and obs_buf with clean keypoints_rel_goal.
        # Subtract the per-episode goal XY offset from the obs_buf slice only
        # — goal shifts by +noise, so observed keypoints_rel_goal shifts by
        # -noise. states_buf is untouched → asymmetric critic stays clean.
        self.obs_buf[:, self._goal_kp_obs_slice].view(
            self.num_envs, self.num_keypoints, 3
        ).sub_(self.goal_pos_obs_noise.unsqueeze(1))

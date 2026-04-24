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

import json
import math
import os
import random
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
from envs.sim_tool_real_cfg import (
    NUM_ARM_DOFS,
    NUM_FINGERTIPS,
    NUM_HAND_ARM_DOFS,
    NUM_HAND_DOFS,
    FINGERTIP_NAMES,
    PALM_LINK,
    DEPTH_MOUNT_LINK,
    SimToolRealEnvCfg,
)
from envs.utils import (
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

    def __del__(self):
        # Best-effort final flush of AABB corner log
        if hasattr(self, '_bbox_corner_log') and self._bbox_corner_log:
            self._flush_aabb_corners()

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

        # ── Apply friction settings (must be after super().__init__ so physx views exist) ──
        if cfg.modify_asset_frictions:
            self._apply_friction_settings()

        # ── Allocate persistent buffers ──
        self._allocate_buffers()

        # ── TiledCamera depth/rgb buffer ──
        # Determine channel count from camera data_types config
        self._wrist_cam_type = cfg.tiled_camera.data_types[0] if cfg.tiled_camera.data_types else "depth"
        wrist_channels = 3 if self._wrist_cam_type == "rgb" else 1
        if cfg.use_depth_camera:
            self.depth_buf = torch.zeros(
                self.num_envs, wrist_channels, cfg.depth_height, cfg.depth_width,
                device=self.device,
            )
        else:
            self.depth_buf = None

        # ── Third-person depth/rgb buffer ──
        self._tp_cam_type = cfg.third_person_camera.data_types[0] if cfg.third_person_camera.data_types else "depth"
        tp_channels = 3 if self._tp_cam_type == "rgb" else 1
        if cfg.use_third_person_camera:
            self.tp_depth_buf = torch.zeros(
                self.num_envs, tp_channels, cfg.depth_height, cfg.depth_width,
                device=self.device,
            )
        else:
            self.tp_depth_buf = None

        # ── Video capture state (mirrors IsaacGym _capture_video_if_needed) ──
        self._video_frames: list | None = None
        self._video_annotator = None
        self._control_steps = 0
        if cfg.capture_video:
            self._init_video_camera()

        # ── Third-person RGB camera (env 0 only, same pose as depth TP cam) ──
        self._tp_rgb_annotator = None
        self._birds_eye_rgb_annotator = None
        self._side_view_rgb_annotator = None
        
        self._side_view_env_idx = 0
        if cfg.use_tp_rgb_camera:
            self._init_tp_rgb_camera(env_idx=self._side_view_env_idx)
        if cfg.use_birds_eye_camera:
            self._init_birds_eye_camera(env_idx=self._side_view_env_idx)
        if cfg.use_side_view_camera:
            self._init_side_view_camera(env_idx=self._side_view_env_idx)
        if cfg.draw_collision_bbox:
            self._draw_collision_bbox(env_idx=self._side_view_env_idx)
        if cfg.show_grasping_bbox and self._uses_ra:
            # Draw grasp bbox only for envs using real_assets
            ra_env_ids = [i for i, mode in enumerate(self._env_asset_mode_list) if mode == 1]
            if ra_env_ids:
                self._draw_grasp_bbox(env_idx=ra_env_ids)

        # ── Compute table spawn bounds (dynamic measurement from USD) ──
        self._compute_table_spawn_bounds()
        if cfg.debug_multi_object:
            self._draw_spawn_zone_wireframe(env_idx=self._side_view_env_idx)
            self._init_handle_wireframes(env_idx=self._side_view_env_idx)

        # ── Goal keypoint markers (VisualizationMarkers spheres, updated each step) ──
        self._goal_kp_wireframe_enabled = cfg.draw_goal_keypoint_wireframe
        if self._goal_kp_wireframe_enabled:
            self._init_goal_keypoint_markers()

        # ── Eval RGB video recording state (all envs) ──
        self._eval_rgb_recording = False
        self._eval_rgb_frames: dict[int, list] = {}  # env_idx -> list of (H,W,3) numpy arrays

        # ── Eval side-view video recording state (all envs) ──
        self._eval_side_view_recording = False
        self._eval_side_view_frames: dict[int, list] = {}

    # ══════════════════════════════════════════════════════════════════
    # Scene setup
    # ══════════════════════════════════════════════════════════════════

    def _compute_asset_mode_assignments(self):
        """Compute per-env asset mode assignments based on config percentages.

        Returns:
            Tuple of (uses_hhp, uses_ra, env_asset_mode_list) where:
            - uses_hhp: bool, whether any env uses handle_head_primitives
            - uses_ra: bool, whether any env uses real_assets
            - env_asset_mode_list: list[int], mode for each env (0=HHP, 1=RA)
        """
        N = self.cfg.scene.num_envs

        # Handle backward compatibility: if object_name is set, use 100% of that mode
        if self.cfg.object_name is not None:
            if self.cfg.object_name == "handle_head_primitives":
                return True, False, [0] * N
            elif self.cfg.object_name == "real_assets":
                return False, True, [1] * N
            else:
                raise ValueError(f"Unknown object_name: {self.cfg.object_name}")

        # Validate percentages
        total_pct = self.cfg.real_assets_pct + self.cfg.handle_head_primitives_pct
        if abs(total_pct - 1.0) > 1e-6:
            raise ValueError(
                f"real_assets_pct ({self.cfg.real_assets_pct}) + "
                f"handle_head_primitives_pct ({self.cfg.handle_head_primitives_pct}) "
                f"must equal 1.0, got {total_pct}"
            )

        paired = getattr(self.cfg, 'object_assignment_teacher_student_paired', False)

        if paired:
            # When teacher-student paired, we need matching modes for paired envs:
            # Teacher env i and Student env i+N/2 must have the same asset mode.
            # Split is applied within each half (teacher/student) separately.
            num_teacher = N // 2
            num_student = N - num_teacher

            # Compute HHP count for teacher half
            n_hhp_teacher = int(num_teacher * self.cfg.handle_head_primitives_pct)
            n_ra_teacher = num_teacher - n_hhp_teacher

            # Student half mirrors teacher half exactly
            n_hhp_student = n_hhp_teacher
            n_ra_student = num_student - n_hhp_student

            # Build mode list: [teacher_modes..., student_modes...]
            # Teacher: first n_hhp_teacher get HHP, rest get RA
            teacher_modes = [0] * n_hhp_teacher + [1] * n_ra_teacher
            # Student: same pattern so env i and env i+num_teacher have same mode
            student_modes = [0] * n_hhp_student + [1] * n_ra_student

            env_asset_mode_list = teacher_modes + student_modes

            n_hhp = n_hhp_teacher + n_hhp_student
            n_ra = n_ra_teacher + n_ra_student

            print(f"[SimToolRealEnv] Asset mode assignment (teacher-student paired):")
            print(f"  Teachers: {n_hhp_teacher} HHP, {n_ra_teacher} RA")
            print(f"  Students: {n_hhp_student} HHP, {n_ra_student} RA")
        else:
            # Standard sequential assignment
            n_hhp = int(N * self.cfg.handle_head_primitives_pct)
            n_ra = N - n_hhp  # Assign remainder to real_assets to ensure total = N

            # Assign modes: first n_hhp envs get HHP, rest get RA
            env_asset_mode_list = [0] * n_hhp + [1] * n_ra

            print(f"[SimToolRealEnv] Asset mode assignment: {n_hhp} HHP envs, {n_ra} RA envs")

        uses_hhp = n_hhp > 0
        uses_ra = n_ra > 0

        return uses_hhp, uses_ra, env_asset_mode_list

    def _setup_scene(self):
        """Create the scene: robot, object, table, camera, ground plane."""
        # Compute per-env asset mode assignments
        self._uses_hhp, self._uses_ra, self._env_asset_mode_list = self._compute_asset_mode_assignments()

        # Generate diverse procedural objects for each mode that's used
        if self._uses_hhp:
            self._generate_handle_head_primitives()
        if self._uses_ra:
            self._generate_real_assets()

        # Spawn assets
        self.robot = Articulation(self.cfg.robot_cfg)
        import copy
        self.objects = []
        for j in range(self.cfg.n_objects_max):
            cfg_j = copy.deepcopy(self.cfg.object_cfg)
            cfg_j.prim_path = f"/World/envs/env_.*/Object_{j}"
            self.objects.append(RigidObject(cfg_j))
        # Note: self.object alias removed — use self.objects[slot] directly
        self.table = RigidObject(self.cfg.table_cfg)

        # TiledCamera for depth (wrist)
        if self.cfg.use_depth_camera:
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Third-person camera
        if self.cfg.use_third_person_camera:
            self._tp_camera = TiledCamera(self.cfg.third_person_camera)
            self.scene.sensors["tp_camera"] = self._tp_camera

        # Eval RGB camera (third-person RGB on ALL envs, for recording eval videos)
        if self.cfg.use_eval_rgb_cameras:
            self._eval_rgb_camera = TiledCamera(self.cfg.eval_rgb_camera)
            self.scene.sensors["eval_rgb_camera"] = self._eval_rgb_camera

        # Eval side-view RGB camera (all envs, same pose as single-env SideViewRGBCam)
        if self.cfg.use_eval_side_view_cameras:
            self._create_eval_side_view_prims()
            self._eval_side_view_camera = TiledCamera(self.cfg.eval_side_view_camera)
            self.scene.sensors["eval_side_view_camera"] = self._eval_side_view_camera

        # Ground plane
        cfg_ground = sim_utils.GroundPlaneCfg()
        cfg_ground.func("/World/ground", cfg_ground)

        # Disable self-collisions between adjacent links (matches IsaacGym behavior).
        # Must be done BEFORE clone_environments() so filters propagate to all clones.
        self._apply_adjacent_link_collision_filters()

        # Clone environments
        # When using procedural per-env geometry (real_assets or handle_head_primitives),
        # we MUST use copy_from_source=True. With copy_from_source=False, the Cloner
        # uses USD scene-graph instancing where all envs share env_0's prototype.
        # Per-env modifications (adding/removing children) on instance prims are
        # invisible to PhysX, so only the placeholder 0.04m cube would be used as
        # the collision shape, causing objects to penetrate the table.
        needs_per_env_geometry = self._uses_hhp or self._uses_ra
        # When using per-env geometry, disable replicate_physics. The PhysX
        # replicator only parses env_0's collision shapes and copies them to
        # all other envs, which ignores per-env collision boxes added after
        # cloning. Each env needs its own collision shapes parsed individually.
        if needs_per_env_geometry:
            self.scene.cfg.replicate_physics = False
        self.scene.clone_environments(copy_from_source=needs_per_env_geometry)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # Replace placeholder geometry with correct per-env shapes based on mode
        if self._uses_hhp and hasattr(self, '_hhp_handle_scales'):
            self._apply_per_env_object_geometry()
        if self._uses_ra and hasattr(self, '_ra_scales'):
            self._apply_per_env_real_asset_geometry()

        # Register with scene
        self.scene.articulations["robot"] = self.robot
        for j, obj in enumerate(self.objects):
            self.scene.rigid_objects[f"object_{j}"] = obj
        self.scene.rigid_objects["table"] = self.table

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_adjacent_link_collision_filters(self):
        """Disable collisions between adjacent robot links using UsdPhysics.FilteredPairsAPI.

        This mimics the IsaacGym bitwise collision filter approach: self-collisions
        are enabled on the articulation, but adjacent link pairs (defined in
        adjacent_links.py) are explicitly filtered out so they don't collide.

        Applied on the source env (env_0) before cloning so filters propagate.
        """
        from pxr import UsdPhysics, Sdf

        from envs.adjacent_links import (
            LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS,
            RIGHT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS,
        )

        # Determine handedness from the robot config
        use_left = any("left_" in name for name in self.cfg.robot_cfg.init_state.joint_pos)
        link_to_adjacent = (
            LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS
            if use_left
            else RIGHT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS
        )

        # Build unique no-collision pairs
        no_collision_pairs = set()
        for link, adjacent_links in link_to_adjacent.items():
            for adj in adjacent_links:
                no_collision_pairs.add(tuple(sorted((link, adj))))

        # Apply FilteredPairsAPI on the source env (env_0) robot prim
        stage = self.scene.stage
        source_robot_path = "/World/envs/env_0/Robot"

        for link_a, link_b in sorted(no_collision_pairs):
            prim_a_path = f"{source_robot_path}/{link_a}"
            prim_b_path = f"{source_robot_path}/{link_b}"

            prim_a = stage.GetPrimAtPath(prim_a_path)
            prim_b = stage.GetPrimAtPath(prim_b_path)

            if not prim_a.IsValid() or not prim_b.IsValid():
                print(f"[WARN] Adjacent link prim not found: {prim_a_path} ({prim_a.IsValid()}) "
                      f"or {prim_b_path} ({prim_b.IsValid()}), skipping filter pair")
                continue

            # Apply FilteredPairsAPI to prim_a and add prim_b as a filtered target
            api_a = UsdPhysics.FilteredPairsAPI.Apply(prim_a)
            rel_a = api_a.CreateFilteredPairsRel()
            rel_a.AddTarget(Sdf.Path(prim_b_path))

            # Apply FilteredPairsAPI to prim_b and add prim_a as a filtered target
            api_b = UsdPhysics.FilteredPairsAPI.Apply(prim_b)
            rel_b = api_b.CreateFilteredPairsRel()
            rel_b.AddTarget(Sdf.Path(prim_a_path))

        print(f"[INFO] Applied {len(no_collision_pairs)} adjacent-link collision filter pairs "
              f"on {source_robot_path}")

    def _generate_handle_head_primitives(self):
        """Load handle_head_primitives metadata and prepare for per-env geometry.

        Reads metadata.json from assets/usd/handle_head_primitives/ and stores
        the handle/head scale parameters. The actual per-env geometry is created
        after scene cloning in _apply_per_env_object_geometry().

        Uses a simple CuboidCfg as placeholder spawn — the real geometry
        (cylinders, composite handle+head) is built per-env after cloning.
        """
        asset_dir = Path(__file__).resolve().parent.parent / "assets" / "usd" / "handle_head_primitives"
        meta_path = asset_dir / "metadata.json"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"handle_head_primitives metadata not found at {meta_path}.\n"
                f"Run: python scripts/generate_handle_head_metadata.py"
            )

        with open(meta_path) as f:
            metadata = json.load(f)

        self._hhp_handle_scales = [entry["handle_scale"] for entry in metadata]
        self._hhp_head_scales = [entry.get("head_scale") for entry in metadata]

        # Compute full bounding box (handle + head union) for each object.
        # Handle is centered at origin along X; head is offset at +X end.
        all_scales = []
        for entry in metadata:
            hs = entry["handle_scale"]
            head = entry.get("head_scale")
            handle_x = hs[0]
            handle_y = hs[1] if len(hs) >= 2 else hs[1]
            handle_z = hs[2] if len(hs) >= 3 else handle_y  # cylinder: y=z=diameter
            if head is not None:
                if len(head) == 3:
                    hx, hy, hz = head
                    # Head is offset: center at handle_x/2 + hx/2
                    # Full X extent: from -handle_x/2 to handle_x/2 + hx
                    total_x = handle_x + hx
                elif len(head) == 2:
                    head_height, head_diameter = head
                    total_x = handle_x + head_diameter
                    hy = max(head_height, head_diameter)
                    hz = max(head_height, head_diameter)
                total_y = max(handle_y, hy)
                total_z = max(handle_z, hz)
            else:
                total_x = handle_x
                total_y = handle_y
                total_z = handle_z
            all_scales.append((total_x, total_y, total_z))
        self._hhp_scales = all_scales

        print(f"[SimToolRealEnv] Loaded {len(metadata)} handle_head_primitives (runtime geometry)")

        # Use a small placeholder cuboid — geometry will be replaced per-env after cloning.
        # IMPORTANT: Do NOT pass collision_props to the placeholder! The CuboidCfg would
        # apply CollisionAPI to the Object prim itself, creating a phantom 0.04m collision
        # cube. The real collision shapes are added per-env in _apply_per_env_object_geometry().
        existing_spawn = self.cfg.object_cfg.spawn
        rigid_props = existing_spawn.rigid_props if hasattr(existing_spawn, 'rigid_props') else None

        self.cfg.object_cfg.spawn = sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=rigid_props,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.27, 0.07),
            ),
        )

    def _generate_real_assets(self):
        """Load real_assets metadata and prepare for per-env USD mesh spawning.

        Scans assets/usd/real_assets/ for asset subdirectories, loads each
        JSON for bounding box scale/offset, and stores USD paths. Uses a
        placeholder CuboidCfg spawn — actual per-env USD meshes are
        referenced after scene cloning in _apply_per_env_real_asset_geometry().

        If metadata.json exists for an asset, reads:
        - base_real_asset_bbox: custom bounding box for initial scaling
        - scale: additional scale factor applied after bbox scaling
        """
        # Use config override if provided, else default path
        if self.cfg.real_assets_path:
            asset_dir = Path(self.cfg.real_assets_path)
        else:
            asset_dir = Path(__file__).resolve().parent.parent / "assets" / "usd" / "real_assets_train"

        if not asset_dir.exists():
            raise FileNotFoundError(
                f"real_assets directory not found at {asset_dir}."
            )

        # Discover all asset subdirectories (sorted for determinism)
        asset_ids = sorted(
            d.name for d in asset_dir.iterdir()
            if d.is_dir() and (d / f"{d.name}.json").exists()
        )

        # Filter to specific asset IDs if configured
        if self.cfg.real_asset_ids:
            filter_set = set(self.cfg.real_asset_ids)
            asset_ids = [aid for aid in asset_ids if aid in filter_set]

        if not asset_ids:
            raise FileNotFoundError(f"No valid real_assets found in {asset_dir}")

        scales = []
        offsets = []
        rotations = []
        usd_paths = []
        pa_centers = []
        pa_scale_factors = []
        # Per-asset scale metadata (from metadata.json)
        asset_base_bboxes = []  # base_real_asset_bbox per asset
        asset_extra_scales = []  # scale multiplier per asset

        import trimesh

        for aid in asset_ids:
            json_path = asset_dir / aid / f"{aid}.json"
            usd_path = asset_dir / aid / f"{aid}.usd"
            mesh_path = asset_dir / aid / f"{aid}.obj"
            metadata_path = asset_dir / aid / "metadata.json"

            with open(json_path) as f:
                meta = json.load(f)

            scales.append(tuple(meta["scale"]))
            offsets.append(tuple(meta["offset"]))
            # rotation is xyzw quaternion
            rotations.append(tuple(meta.get("rotation", [0.0, 0.0, 0.0, 1.0])))
            usd_paths.append(usd_path)

            # Load per-asset scale metadata if available
            if metadata_path.exists():
                with open(metadata_path) as f:
                    scale_meta = json.load(f)
                base_bbox = tuple(scale_meta.get("base_real_asset_bbox", [0.1, 0.1, 0.1]))
                extra_scale = float(scale_meta.get("scale", 1.0))
            else:
                # Default values when no metadata.json exists
                base_bbox = (0.1, 0.1, 0.1)
                extra_scale = 1.0
            asset_base_bboxes.append(base_bbox)
            asset_extra_scales.append(extra_scale)

            # PrimitiveAnything normalization: centers the raw mesh and scales
            # the longest axis to 1.6 units.  The .obj is the original SAM3D
            # mesh whose coordinate system the grasp bbox JSON is defined in.
            # (The .glb is a grasp visualization mesh in a different frame.)
            mesh = trimesh.load(str(mesh_path), force="mesh")
            raw_min, raw_max = mesh.bounds[0], mesh.bounds[1]
            center = (raw_min + raw_max) / 2.0
            scale_factor = 1.6 / (raw_max - raw_min).max()
            pa_centers.append(tuple(center.tolist()))
            pa_scale_factors.append(float(scale_factor))


        # Grasp bbox in raw mesh coordinates (will be converted to sim space
        # after per-asset scale factors are computed in _apply_per_env_real_asset_geometry)
        self._ra_scales_raw = scales
        self._ra_offsets_raw = offsets
        self._ra_rotations = rotations       # xyzw quaternions (unchanged by isotropic scaling)
        self._ra_scales = list(scales)       # will be overwritten with sim-space values
        self._ra_offsets = list(offsets)      # will be overwritten with sim-space values
        self._ra_usd_paths = usd_paths
        self._ra_pa_centers = pa_centers
        self._ra_pa_scale_factors = pa_scale_factors
        # Store per-asset scale metadata for use in _apply_per_env_real_asset_geometry
        self._ra_base_bboxes = asset_base_bboxes
        self._ra_extra_scales = asset_extra_scales

        print(f"[SimToolRealEnv] Loaded {len(asset_ids)} real_assets (USD meshes)")

        # Use a small placeholder cuboid — geometry will be replaced per-env after cloning.
        # IMPORTANT: Do NOT pass collision_props to the placeholder! The CuboidCfg would
        # apply CollisionAPI to the Object prim itself, creating a phantom 0.04m collision
        # cube. The real collision shapes are added per-env in _apply_per_env_real_asset_geometry().
        existing_spawn = self.cfg.object_cfg.spawn
        rigid_props = existing_spawn.rigid_props if hasattr(existing_spawn, 'rigid_props') else None

        self.cfg.object_cfg.spawn = sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=rigid_props,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.27, 0.07),
            ),
        )

    def _get_obj_idx_for_env(self, ei, num_objects):
        """Object index for env ei. When teacher_student_paired, env i and env i+N//2 share object."""
        N = self.num_envs
        if (getattr(self.cfg, 'object_assignment_teacher_student_paired', False) and
                (self._uses_hhp or self._uses_ra)):
            num_teacher = N // 2
            if ei < num_teacher:
                return ei % num_objects
            return (ei - num_teacher) % num_objects
        return ei % num_objects

    def _get_obj_idx_for_env_and_slot(self, ei, slot_j, num_objects):
        """Object geometry index for env ei, object slot j. Ensures heterogeneous objects per env."""
        N = self.num_envs
        if (getattr(self.cfg, 'object_assignment_teacher_student_paired', False) and
                (self._uses_hhp or self._uses_ra)):
            num_teacher = N // 2
            base_ei = ei if ei < num_teacher else (ei - num_teacher)
            return (base_ei * self.cfg.n_objects_max + slot_j) % num_objects
        return (ei * self.cfg.n_objects_max + slot_j) % num_objects

    def _get_hhp_obj_idx_for_env(self, ei, num_hhp_objects):
        """Object index for HHP envs. Maps env index to HHP asset index.

        With teacher-student pairing, paired envs (i and i+N/2) get the same object.
        """
        paired = getattr(self.cfg, 'object_assignment_teacher_student_paired', False)
        if paired:
            # Map to teacher-space first, then to local HHP index
            N = len(self._env_asset_mode_list)
            num_teacher = N // 2
            base_ei = ei if ei < num_teacher else (ei - num_teacher)
            # Now base_ei is in [0, num_teacher), find local HHP index for teacher env
            # HHP teacher envs are at indices 0..n_hhp_teacher-1
            return base_ei % num_hhp_objects
        else:
            # Use cached mapping for O(1) lookup
            if not hasattr(self, '_hhp_env_to_local'):
                self._hhp_env_to_local = {env_id: i for i, env_id in enumerate(
                    [j for j, mode in enumerate(self._env_asset_mode_list) if mode == 0]
                )}
            local_idx = self._hhp_env_to_local.get(ei, ei)
            return local_idx % num_hhp_objects

    def _get_hhp_obj_idx_for_env_and_slot(self, ei, slot_j, num_hhp_objects):
        """Object geometry index for HHP envs.

        With teacher-student pairing, paired envs (i and i+N/2) get the same object.
        """
        paired = getattr(self.cfg, 'object_assignment_teacher_student_paired', False)
        if paired:
            N = len(self._env_asset_mode_list)
            num_teacher = N // 2
            base_ei = ei if ei < num_teacher else (ei - num_teacher)
            return (base_ei * self.cfg.n_objects_max + slot_j) % num_hhp_objects
        else:
            if not hasattr(self, '_hhp_env_to_local'):
                self._hhp_env_to_local = {env_id: i for i, env_id in enumerate(
                    [j for j, mode in enumerate(self._env_asset_mode_list) if mode == 0]
                )}
            local_idx = self._hhp_env_to_local.get(ei, ei)
            return (local_idx * self.cfg.n_objects_max + slot_j) % num_hhp_objects

    def _get_ra_obj_idx_for_env(self, ei, num_ra_objects):
        """Object index for RA envs. Maps env index to RA asset index.

        With teacher-student pairing, paired envs (i and i+N/2) get the same object.
        """
        paired = getattr(self.cfg, 'object_assignment_teacher_student_paired', False)
        if paired:
            N = len(self._env_asset_mode_list)
            num_teacher = N // 2
            base_ei = ei if ei < num_teacher else (ei - num_teacher)
            # RA teacher envs start at index n_hhp_teacher in the teacher half
            # But for object indexing, we just need consistent mapping
            return base_ei % num_ra_objects
        else:
            if not hasattr(self, '_ra_env_to_local'):
                self._ra_env_to_local = {env_id: i for i, env_id in enumerate(
                    [j for j, mode in enumerate(self._env_asset_mode_list) if mode == 1]
                )}
            local_idx = self._ra_env_to_local.get(ei, ei)
            return local_idx % num_ra_objects

    def _get_ra_obj_idx_for_env_and_slot(self, ei, slot_j, num_ra_objects):
        """Object geometry index for RA envs.

        With teacher-student pairing, paired envs (i and i+N/2) get the same object.
        """
        paired = getattr(self.cfg, 'object_assignment_teacher_student_paired', False)
        if paired:
            N = len(self._env_asset_mode_list)
            num_teacher = N // 2
            base_ei = ei if ei < num_teacher else (ei - num_teacher)
            return (base_ei * self.cfg.n_objects_max + slot_j) % num_ra_objects
        else:
            if not hasattr(self, '_ra_env_to_local'):
                self._ra_env_to_local = {env_id: i for i, env_id in enumerate(
                    [j for j, mode in enumerate(self._env_asset_mode_list) if mode == 1]
                )}
            local_idx = self._ra_env_to_local.get(ei, ei)
            return (local_idx * self.cfg.n_objects_max + slot_j) % num_ra_objects

    def _apply_per_env_object_geometry(self):
        """Replace placeholder cuboid geometry with correct handle+head shapes per env.

        Called after clone_environments() so each env can have unique geometry.
        Uses the pxr USD API to create Cylinder/Cube prims under each env's
        Object rigid body prim.
        """
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        num_objects = len(self._hhp_handle_scales)
        N = self.num_envs

        paired = getattr(self.cfg, 'object_assignment_teacher_student_paired', False)
        if paired:
            print(f"[SimToolRealEnv] Object assignment: teacher-student paired (env i & env i+{N//2} share object)")

        # Only apply HHP geometry to envs that use HHP mode
        hhp_env_ids = [i for i, mode in enumerate(self._env_asset_mode_list) if mode == 0]
        print(f"[SimToolRealEnv] Creating per-env HHP geometry for {len(hhp_env_ids)} envs ({num_objects} unique objects)")

        def _make_cube(path, sx, sy, sz, collision=False):
            """Create or overwrite a Cube prim with given extents."""
            cube = UsdGeom.Cube.Define(stage, path)
            cube.GetSizeAttr().Set(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddScaleOp().Set(Gf.Vec3d(float(sx), float(sy), float(sz)))
            if collision:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                cube.GetPrim().CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set(UsdGeom.Tokens.guide)

        def _make_cylinder(path, h, r, axis="X", collision=False):
            """Create or overwrite a Cylinder prim."""
            cyl = UsdGeom.Cylinder.Define(stage, path)
            cyl.GetHeightAttr().Set(float(h))
            cyl.GetRadiusAttr().Set(float(r))
            cyl.GetAxisAttr().Set(axis)
            xf = UsdGeom.Xformable(cyl.GetPrim())
            xf.ClearXformOpOrder()
            if collision:
                UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
                cyl.GetPrim().CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set(UsdGeom.Tokens.guide)

        def _make_offset_xform(path, x_offset):
            """Create or overwrite an Xform with a translate offset."""
            xf = UsdGeom.Xform.Define(stage, path)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(float(x_offset), 0.0, 0.0))

        for ei in hhp_env_ids:
            for j in range(self.cfg.n_objects_max):
                obj_idx = self._get_hhp_obj_idx_for_env_and_slot(ei, j, num_objects)
                handle_scale = self._hhp_handle_scales[obj_idx]
                head_scale = self._hhp_head_scales[obj_idx]

                obj_prim_path = f"/World/envs/env_{ei}/Object_{j}"
                obj_prim = stage.GetPrimAtPath(obj_prim_path)
                if not obj_prim.IsValid():
                    print(f"  [WARN] Object prim not found at {obj_prim_path}")
                    continue

                # Remove ALL children of the Object prim (placeholder + any cloned geometry)
                children_to_remove = [c.GetPath().pathString for c in obj_prim.GetAllChildren()]
                for child_path in children_to_remove:
                    stage.RemovePrim(child_path)

                # Strip CollisionAPI from the Object prim itself if the placeholder
                # CuboidCfg applied it.  Only child geometry prims should have
                # collision shapes; leaving it on the Object prim would create a
                # phantom 0.04m collision cube that interferes with the real geometry.
                if obj_prim.HasAPI(UsdPhysics.CollisionAPI):
                    for attr_name in ("physics:collisionEnabled",):
                        attr = obj_prim.GetAttribute(attr_name)
                        if attr and attr.IsValid():
                            attr.Set(False)

                # Create handle geometry
                if len(handle_scale) == 3:
                    lx, ly, lz = handle_scale
                    _make_cube(f"{obj_prim_path}/vis_handle", lx, ly, lz)
                    _make_cube(f"{obj_prim_path}/col_handle", lx, ly, lz, collision=True)
                elif len(handle_scale) == 2:
                    height, diameter = handle_scale
                    _make_cylinder(f"{obj_prim_path}/vis_handle", height, diameter / 2, "X")
                    _make_cylinder(f"{obj_prim_path}/col_handle", height, diameter / 2, "X", collision=True)

                # Create head geometry (if present)
                if head_scale is not None:
                    if len(head_scale) == 3:
                        hx, hy, hz = head_scale
                        x_offset = handle_scale[0] / 2 + hx / 2
                        _make_offset_xform(f"{obj_prim_path}/vis_head", x_offset)
                        _make_cube(f"{obj_prim_path}/vis_head/box", hx, hy, hz)
                        _make_offset_xform(f"{obj_prim_path}/col_head", x_offset)
                        _make_cube(f"{obj_prim_path}/col_head/box", hx, hy, hz, collision=True)
                    elif len(head_scale) == 2:
                        head_height, head_diameter = head_scale
                        x_offset = handle_scale[0] / 2 + head_diameter / 2
                        _make_offset_xform(f"{obj_prim_path}/vis_head", x_offset)
                        _make_cylinder(f"{obj_prim_path}/vis_head/cyl", head_height, head_diameter / 2, "Y")
                        _make_offset_xform(f"{obj_prim_path}/col_head", x_offset)
                        _make_cylinder(f"{obj_prim_path}/col_head/cyl", head_height, head_diameter / 2, "Y", collision=True)

        print(f"[SimToolRealEnv] Per-env object geometry created")

    def _apply_per_env_real_asset_geometry(self):
        """Replace placeholder cuboid with real asset USD references per env.

        Called after clone_environments() so each env can have unique geometry.
        Directly references the USD files via Sdf.Reference instead of
        reconstructing individual primitives. Each env's Object prim gets a
        child Xform that references the USD's /Root, with a uniform scale
        applied to fit within real_asset_bbox.
        """
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        num_objects = len(self._ra_usd_paths)
        N = self.num_envs

        # Only apply RA geometry to envs that use RA mode
        ra_env_ids = [i for i, mode in enumerate(self._env_asset_mode_list) if mode == 1]
        print(f"[SimToolRealEnv] Creating per-env real asset geometry for {len(ra_env_ids)} envs ({num_objects} unique assets)")

        # ── Parse each unique USD file once and cache primitive data ──
        # Needed for AABB computation and viser visualization cache.
        usd_prim_cache: dict[str, list[dict]] = {}
        for usd_path in self._ra_usd_paths:
            usd_str = str(usd_path)
            if usd_str in usd_prim_cache:
                continue

            ref_stage = Usd.Stage.Open(usd_str)
            prims_data = []
            for prim in ref_stage.Traverse():
                type_name = prim.GetTypeName()
                if type_name not in ("Cube", "Sphere", "Cylinder"):
                    continue
                xf = UsdGeom.Xformable(prim)
                translate = Gf.Vec3d(0, 0, 0)
                orient = Gf.Quatf(1, 0, 0, 0)
                prim_scale = Gf.Vec3d(1, 1, 1)
                for op in xf.GetOrderedXformOps():
                    op_name = op.GetOpName()
                    if op_name == "xformOp:translate":
                        translate = op.Get()
                    elif op_name == "xformOp:orient":
                        orient = op.Get()
                    elif op_name == "xformOp:scale":
                        prim_scale = op.Get()
                prims_data.append({
                    "type": type_name,
                    "translate": translate,
                    "orient": orient,
                    "scale": prim_scale,
                })
            usd_prim_cache[usd_str] = prims_data

            # Validate that source USD geometric prims have CollisionAPI
            missing_collision = []
            for prim in ref_stage.Traverse():
                if prim.GetTypeName() in ("Cube", "Sphere", "Cylinder"):
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        missing_collision.append(prim.GetPath().pathString)
            if missing_collision:
                print(f"  [INFO] {usd_str}: {len(missing_collision)} primitives missing "
                      f"CollisionAPI — will be applied in layer conversion: {missing_collision[:3]}")

            print(f"  [USD] Parsed {usd_str}: {len(prims_data)} primitives")

        # ── Compute per-asset AABB extents (for scaling calculation) ──
        # We cache the PA-space AABB extent per USD path for reuse.
        usd_pa_extents: dict[str, list] = {}
        for usd_str, prims_data in usd_prim_cache.items():
            # Compute rotation-aware AABB of all primitives in PA normalized space.
            # USD defaults: Cube size=2, Sphere radius=1, Cylinder h=2 r=1,
            # so xform scale gives half-extents in local frame.  For rotated
            # primitives the world-space AABB half-extents are |R| @ local_half.
            mins = [float('inf')] * 3
            maxs = [float('-inf')] * 3
            for pd in prims_data:
                t, sc, orient = pd["translate"], pd["scale"], pd["orient"]
                w = float(orient.GetReal())
                im = orient.GetImaginary()
                x, y, z = float(im[0]), float(im[1]), float(im[2])
                sx, sy, sz = float(sc[0]), float(sc[1]), float(sc[2])
                r00 = 1-2*(y*y+z*z); r01 = 2*(x*y-w*z);   r02 = 2*(x*z+w*y)
                r10 = 2*(x*y+w*z);   r11 = 1-2*(x*x+z*z); r12 = 2*(y*z-w*x)
                r20 = 2*(x*z-w*y);   r21 = 2*(y*z+w*x);   r22 = 1-2*(x*x+y*y)
                hx = abs(r00)*sx + abs(r01)*sy + abs(r02)*sz
                hy = abs(r10)*sx + abs(r11)*sy + abs(r12)*sz
                hz = abs(r20)*sx + abs(r21)*sy + abs(r22)*sz
                for ax, h in enumerate((hx, hy, hz)):
                    lo = float(t[ax]) - h
                    hi = float(t[ax]) + h
                    mins[ax] = min(mins[ax], lo)
                    maxs[ax] = max(maxs[ax], hi)
            pa_extent = [maxs[ax] - mins[ax] for ax in range(3)]
            usd_pa_extents[usd_str] = pa_extent
            print(f"  [USD] {usd_str}: PA extent=({pa_extent[0]:.3f}, {pa_extent[1]:.3f}, {pa_extent[2]:.3f})")

        # ── Compute per-asset isotropic scale using metadata (base_bbox + extra scale) ──
        # Each asset can have its own base_real_asset_bbox and scale multiplier from metadata.json.
        # Final scale = (scale to fit base_bbox) * extra_scale
        asset_scales: list[float] = []
        for obj_idx in range(num_objects):
            usd_str = str(self._ra_usd_paths[obj_idx])
            pa_extent = usd_pa_extents[usd_str]
            # Use per-asset base_bbox from metadata.json (or default)
            base_bbox = self._ra_base_bboxes[obj_idx]
            extra_scale = self._ra_extra_scales[obj_idx]
            # Compute isotropic scale to fit within base_bbox
            bbox_scale = min(base_bbox[ax] / pa_extent[ax] for ax in range(3) if pa_extent[ax] > 0)
            # Apply extra scale multiplier from metadata
            final_scale = bbox_scale * extra_scale
            asset_scales.append(final_scale)
            print(f"  [USD] obj {obj_idx}: base_bbox={base_bbox}, extra_scale={extra_scale:.2f} → final_scale={final_scale:.4f}")

        # ── Convert grasp bounding box from raw mesh coords to sim coords ──
        # The grasp bbox (offset/scale) is in the raw mesh coordinate system.
        # PrimitiveAnything internally swaps Y↔Z: raw [x,y,z] → model [x,-z,y].
        for obj_idx in range(num_objects):
            s = asset_scales[obj_idx]
            pf = self._ra_pa_scale_factors[obj_idx]
            pc = self._ra_pa_centers[obj_idx]
            raw_scale = self._ra_scales_raw[obj_idx]
            raw_offset = self._ra_offsets_raw[obj_idx]
            pa_off = tuple((raw_offset[i] - pc[i]) * pf for i in range(3))
            pa_sc  = tuple(raw_scale[i] * pf for i in range(3))
            # Half-extents (unsigned): swap Y↔Z
            self._ra_scales[obj_idx] = (pa_sc[0] * s, pa_sc[2] * s, pa_sc[1] * s)
            # Center offset: swap Y↔Z with sign flip on new-Y
            self._ra_offsets[obj_idx] = (pa_off[0] * s, -pa_off[2] * s, pa_off[1] * s)
            print(f"  [Grasp] obj {obj_idx}: raw_scale={raw_scale} → sim_scale={self._ra_scales[obj_idx]}, "
                  f"raw_offset={raw_offset} → sim_offset={self._ra_offsets[obj_idx]}")

        # ── Create /PhysicsMaterial in the scene ──
        # USD files bind collision shapes to </PhysicsMaterial> via absolute path.
        # Create it so the binding resolves. Actual friction values are overridden
        # by _apply_friction_settings() via root_physx_view.
        phys_mat_path = "/PhysicsMaterial"
        if not stage.GetPrimAtPath(phys_mat_path).IsValid():
            mat_prim = stage.DefinePrim(phys_mat_path, "Material")
            UsdPhysics.MaterialAPI.Apply(mat_prim)
            mat_prim.CreateAttribute("physics:staticFriction", Sdf.ValueTypeNames.Float).Set(0.5)
            mat_prim.CreateAttribute("physics:dynamicFriction", Sdf.ValueTypeNames.Float).Set(0.5)
            mat_prim.CreateAttribute("physics:restitution", Sdf.ValueTypeNames.Float).Set(0.0)

        # ── Open each unique USD layer, normalize collision geometry ──
        # We use CopySpec (not AddReference) because clone_environments with
        # copy_from_source=False uses scene-graph instancing; composition arcs
        # added after cloning get shared across instances. CopySpec creates
        # independent local prims per env, matching how the old manual
        # reconstruction approach worked.
        #
        # IMPORTANT: Non-uniformly-scaled Sphere/Cylinder must be converted
        # to convex-hull Mesh BEFORE CopySpec.  PhysX's native sphere /
        # capsule geometries only support a single radius so they can't
        # represent ellipsoids or elliptic cylinders, and modifying prims
        # on the stage after CopySpec (via RemovePrim+Define or in-place
        # typeName change) breaks PhysX collision registration.  We
        # convert in an anonymous layer copy.  CollisionAPI is ensured on
        # all geometric prims during layer conversion so PhysX registers
        # them as collision shapes.
        usd_layers: dict[str, "Sdf.Layer"] = {}
        for usd_path in self._ra_usd_paths:
            usd_str = str(usd_path)
            if usd_str not in usd_layers:
                orig_layer = Sdf.Layer.FindOrOpen(usd_str)
                mod_layer = Sdf.Layer.CreateAnonymous()
                Sdf.CopySpec(orig_layer, Sdf.Path("/Root"), mod_layer, Sdf.Path("/Root"))
                self._normalize_collision_prims_in_layer(mod_layer)
                usd_layers[usd_str] = mod_layer

        dst_layer = stage.GetRootLayer()

        # ── Copy USD prims directly into each env that uses RA mode ──
        for ei in ra_env_ids:
            for j in range(self.cfg.n_objects_max):
                obj_idx = self._get_ra_obj_idx_for_env_and_slot(ei, j, num_objects)
                usd_path = self._ra_usd_paths[obj_idx]
                usd_str = str(usd_path)
                s = asset_scales[obj_idx]
                src_layer = usd_layers[usd_str]

                obj_prim_path = f"/World/envs/env_{ei}/Object_{j}"
                obj_prim = stage.GetPrimAtPath(obj_prim_path)
                if not obj_prim.IsValid():
                    print(f"  [WARN] Object prim not found at {obj_prim_path}")
                    continue

                # Remove placeholder children
                children_to_remove = [c.GetPath().pathString for c in obj_prim.GetAllChildren()]
                for child_path in children_to_remove:
                    stage.RemovePrim(child_path)

                # Disable collision on the Object prim itself if the placeholder
                # CuboidCfg applied it — only child geometry should collide.
                if obj_prim.HasAPI(UsdPhysics.CollisionAPI):
                    for attr_name in ("physics:collisionEnabled",):
                        attr = obj_prim.GetAttribute(attr_name)
                        if attr and attr.IsValid():
                            attr.Set(False)

                # Create a scaled container Xform under the Object rigid body.
                # Child prims have CollisionAPI (ensured during layer conversion)
                # so PhysX uses the collision geometry (converted to Cubes in the layer).
                asset_path = f"{obj_prim_path}/asset"
                asset_xform = UsdGeom.Xform.Define(stage, asset_path)
                xf = UsdGeom.Xformable(asset_xform.GetPrim())
                xf.ClearXformOpOrder()
                xf.AddScaleOp().Set(Gf.Vec3d(s, s, s))

                # Copy each primitive from the USD's /Root into the stage.
                root_spec = src_layer.GetPrimAtPath("/Root")
                for child_spec in root_spec.nameChildren:
                    src_child_path = Sdf.Path(f"/Root/{child_spec.name}")
                    dst_child_path = Sdf.Path(f"{asset_path}/{child_spec.name}")
                    Sdf.CopySpec(src_layer, src_child_path, dst_layer, dst_child_path)

        # Cache scaled primitive data per-env for viser visualization (only RA envs)
        self._ra_env_prims = {}  # Changed from list to dict for sparse indexing
        for ei in ra_env_ids:
            obj_idx = self._get_ra_obj_idx_for_env(ei, num_objects)
            usd_str = str(self._ra_usd_paths[obj_idx])
            s = asset_scales[obj_idx]
            scaled = []
            for pd in usd_prim_cache[usd_str]:
                t = pd["translate"]
                sc = pd["scale"]
                orient = pd["orient"]
                scaled.append({
                    "type": pd["type"],
                    "translate": (float(t[0] * s), float(t[1] * s), float(t[2] * s)),
                    "orient_wxyz": (float(orient.GetReal()),
                                    float(orient.GetImaginary()[0]),
                                    float(orient.GetImaginary()[1]),
                                    float(orient.GetImaginary()[2])),
                    "scale": (float(sc[0] * s), float(sc[1] * s), float(sc[2] * s)),
                    "dimensions": (float(sc[0] * s * 2), float(sc[1] * s * 2), float(sc[2] * s * 2)),
                })
            self._ra_env_prims[ei] = scaled

        # Build a (num_objects, 3) tensor of grasp-center offsets for vectorized
        # lookup during _populate_sim_buffers. These are in object-local frame.
        self._ra_offsets_tensor = torch.tensor(
            self._ra_offsets, dtype=torch.float32, device=self.device
        )  # (num_objects, 3)

        print(f"[SimToolRealEnv] Per-env real asset geometry created")

    @staticmethod
    def _normalize_collision_prims_in_layer(layer):
        """Normalize collision prims in an Sdf.Layer so PhysX registers them correctly.

        Must be called on an anonymous layer copy BEFORE Sdf.CopySpec into
        the stage.  Modifying prim types after CopySpec breaks PhysX
        collision registration.

        Normalization steps:
          - Apply PhysicsCollisionAPI to every Cube/Sphere/Cylinder prim.
            Source USD files (e.g. from PrimitiveAnything) may lack it;
            without CollisionAPI, PhysX treats the geometry as visual-only
            and objects fall through the table.
          - For Cube prims, bake the `size` attribute into xformOp:scale
            and set size=1.0 so collision half-extents = scale/2.
          - For non-uniformly-scaled Sphere/Cylinder, replace the prim
            with a tessellated Mesh and apply PhysicsMeshCollisionAPI
            with physics:approximation="convexHull".  PhysX's native
            sphere and capsule geometries only support a single radius so
            they can't represent ellipsoids or elliptic cylinders; a
            convex-mesh collider does support non-uniform scaling, and
            the convex hull of a tessellated ellipsoid / elliptic
            cylinder matches the true shape to tessellation precision
            for both visuals and collision.
          - Uniformly-scaled Sphere/Cylinder are left as-is (PhysX
            handles them natively with a single radius).
        """
        from pxr import Sdf, Gf, Vt

        root_spec = layer.GetPrimAtPath("/Root")
        if not root_spec:
            return

        for child_spec in list(root_spec.nameChildren):
            ptype = child_spec.typeName

            # ── Ensure CollisionAPI on all geometric prims ──
            if ptype in ("Cube", "Sphere", "Cylinder"):
                schemas = child_spec.GetInfo("apiSchemas")
                if schemas is None:
                    schemas = Sdf.TokenListOp()
                prepended = list(schemas.prependedItems) if schemas.prependedItems else []
                if "PhysicsCollisionAPI" not in prepended:
                    prepended.append("PhysicsCollisionAPI")
                    schemas.prependedItems = prepended
                    child_spec.SetInfo("apiSchemas", schemas)

            # ── Existing Cube prims ──
            # Bake size into scale: half-extent = (size/2)*scale → set size=1, scale *= old_size
            if ptype == "Cube":
                size_prop = child_spec.properties.get("size")
                old_size = float(size_prop.default) if size_prop and size_prop.default is not None else 2.0
                # Set size=1.0
                if size_prop is None:
                    size_prop = Sdf.AttributeSpec(child_spec, "size", Sdf.ValueTypeNames.Double)
                size_prop.default = 1.0
                # Multiply scale by old_size so extent is preserved
                scale_prop = child_spec.properties.get("xformOp:scale")
                if scale_prop and scale_prop.default is not None:
                    sv = scale_prop.default
                    scale_prop.default = type(sv)(sv[0] * old_size, sv[1] * old_size, sv[2] * old_size)
                print(f"    [LayerConvert] {child_spec.name}: Cube size={old_size}→1.0 (baked into scale)")
                continue

            if ptype not in ("Sphere", "Cylinder"):
                continue

            # Read scale
            scale_prop = child_spec.properties.get("xformOp:scale")
            if not scale_prop or scale_prop.default is None:
                continue
            sv = scale_prop.default

            # Uniform scale — PhysX handles native Sphere / Cylinder here.
            if (abs(float(sv[0]) - float(sv[1])) < 1e-6
                    and abs(float(sv[1]) - float(sv[2])) < 1e-6):
                continue

            # ── Non-uniform Sphere / Cylinder → convex-hull Mesh ──
            # Tessellate a *unit* primitive (radius=1, half_height=1 for
            # cylinders) so the mesh's native half-extents are all 1.0,
            # then bake the shape's own radius / height into xformOp:scale
            # so the final half-extents match the original prim exactly.
            if ptype == "Sphere":
                radius_prop = child_spec.properties.get("radius")
                radius = float(radius_prop.default) if radius_prop and radius_prop.default is not None else 1.0
                points, face_counts, face_indices = SimToolRealEnv._unit_icosphere_mesh(subdivisions=2)
                # Unit icosphere → native half-extents = 1.  Final extent = radius * scale.
                new_scale = Gf.Vec3f(radius * sv[0], radius * sv[1], radius * sv[2])
            else:  # Cylinder
                radius_prop = child_spec.properties.get("radius")
                height_prop = child_spec.properties.get("height")
                axis_prop = child_spec.properties.get("axis")
                radius = float(radius_prop.default) if radius_prop and radius_prop.default is not None else 1.0
                height = float(height_prop.default) if height_prop and height_prop.default is not None else 2.0
                axis = str(axis_prop.default) if axis_prop and axis_prop.default is not None else "Z"
                half_h = height / 2.0
                points, face_counts, face_indices = SimToolRealEnv._unit_cylinder_mesh(axis=axis, n_sides=32)
                # Unit cylinder along `axis` → native half-extents = 1 everywhere.
                if axis == "X":
                    new_scale = Gf.Vec3f(half_h * sv[0], radius * sv[1], radius * sv[2])
                elif axis == "Y":
                    new_scale = Gf.Vec3f(radius * sv[0], half_h * sv[1], radius * sv[2])
                else:  # Z
                    new_scale = Gf.Vec3f(radius * sv[0], radius * sv[1], half_h * sv[2])

            # Change type to Mesh
            child_spec.typeName = "Mesh"

            # Remove Sphere / Cylinder–specific attributes
            for attr_name in ("radius", "height", "axis", "size", "extent"):
                prop = child_spec.properties.get(attr_name)
                if prop is not None:
                    child_spec.RemoveProperty(prop)

            # Mesh topology attributes
            pts_attr = Sdf.AttributeSpec(child_spec, "points", Sdf.ValueTypeNames.Point3fArray)
            pts_attr.default = Vt.Vec3fArray([Gf.Vec3f(*p) for p in points])

            fvc_attr = Sdf.AttributeSpec(child_spec, "faceVertexCounts", Sdf.ValueTypeNames.IntArray)
            fvc_attr.default = Vt.IntArray(face_counts)

            fvi_attr = Sdf.AttributeSpec(child_spec, "faceVertexIndices", Sdf.ValueTypeNames.IntArray)
            fvi_attr.default = Vt.IntArray(face_indices)

            # "none" → raw polygon mesh, no Catmull-Clark subdivision applied at render time
            subdiv_attr = Sdf.AttributeSpec(child_spec, "subdivisionScheme", Sdf.ValueTypeNames.Token)
            subdiv_attr.default = "none"

            # Apply PhysicsMeshCollisionAPI and set physics:approximation = "convexHull"
            schemas = child_spec.GetInfo("apiSchemas")
            if schemas is None:
                schemas = Sdf.TokenListOp()
            prepended = list(schemas.prependedItems) if schemas.prependedItems else []
            if "PhysicsMeshCollisionAPI" not in prepended:
                prepended.append("PhysicsMeshCollisionAPI")
                schemas.prependedItems = prepended
                child_spec.SetInfo("apiSchemas", schemas)

            approx_attr = Sdf.AttributeSpec(child_spec, "physics:approximation", Sdf.ValueTypeNames.Token)
            approx_attr.default = "convexHull"

            # Bake shape dimensions into xformOp:scale
            scale_prop.default = new_scale

            print(f"    [LayerConvert] {child_spec.name}: {ptype} → Mesh(convexHull, "
                  f"verts={len(points)}, faces={len(face_counts)})")

    @staticmethod
    def _unit_icosphere_mesh(subdivisions: int = 2):
        """Return (points, face_vertex_counts, face_vertex_indices) for a unit icosphere.

        The icosphere is a subdivided icosahedron with every vertex projected to
        the unit sphere.  `subdivisions=2` yields 162 vertices / 320 triangles,
        which is plenty for a convex-hull approximation of an ellipsoid.
        """
        phi = (1.0 + math.sqrt(5.0)) / 2.0

        def _norm(v):
            L = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            return (v[0] / L, v[1] / L, v[2] / L)

        verts = [_norm(v) for v in [
            (-1.0,  phi,  0.0), ( 1.0,  phi,  0.0), (-1.0, -phi,  0.0), ( 1.0, -phi,  0.0),
            ( 0.0, -1.0,  phi), ( 0.0,  1.0,  phi), ( 0.0, -1.0, -phi), ( 0.0,  1.0, -phi),
            ( phi,  0.0, -1.0), ( phi,  0.0,  1.0), (-phi,  0.0, -1.0), (-phi,  0.0,  1.0),
        ]]
        faces = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]

        for _ in range(subdivisions):
            midpoint_cache: dict = {}

            def _mid(i, j, _verts=verts, _cache=midpoint_cache):
                key = (i, j) if i < j else (j, i)
                if key in _cache:
                    return _cache[key]
                a, b = _verts[i], _verts[j]
                m = _norm(((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5))
                idx = len(_verts)
                _verts.append(m)
                _cache[key] = idx
                return idx

            new_faces = []
            for a, b, c in faces:
                ab = _mid(a, b)
                bc = _mid(b, c)
                ca = _mid(c, a)
                new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
            faces = new_faces

        face_counts = [3] * len(faces)
        face_indices = [i for tri in faces for i in tri]
        return verts, face_counts, face_indices

    @staticmethod
    def _unit_cylinder_mesh(axis: str = "Z", n_sides: int = 32):
        """Return (points, face_vertex_counts, face_vertex_indices) for a unit cylinder.

        Unit cylinder: radius=1, half_height=1 (spans -1..+1 along `axis`).
        `axis` must be one of "X", "Y", "Z".  The mesh is built with its axis
        along +Z and then rotated so that +Z maps onto the requested axis.
        Using a proper rotation (det=+1) preserves the right-handed winding
        of all faces, so outward normals remain outward after re-orientation.
        """
        # Build along +Z
        top_z: list[tuple[float, float, float]] = []
        bot_z: list[tuple[float, float, float]] = []
        for i in range(n_sides):
            angle = 2.0 * math.pi * i / n_sides
            c = math.cos(angle)
            s = math.sin(angle)
            top_z.append((c, s,  1.0))
            bot_z.append((c, s, -1.0))
        top_c_z = (0.0, 0.0,  1.0)
        bot_c_z = (0.0, 0.0, -1.0)

        # Rotation mapping +Z → `axis`.  Both non-identity maps have det=+1.
        if axis == "X":
            # Rotate +90° around Y: (x, y, z) → (z, y, -x)
            rot = lambda p: (p[2], p[1], -p[0])
        elif axis == "Y":
            # Rotate -90° around X: (x, y, z) → (x, z, -y)
            rot = lambda p: (p[0], p[2], -p[1])
        else:  # "Z"
            rot = lambda p: p

        top = [rot(p) for p in top_z]
        bot = [rot(p) for p in bot_z]
        top_center = rot(top_c_z)
        bot_center = rot(bot_c_z)

        # Index layout:
        #   top[i]     = i
        #   bot[i]     = n_sides + i
        #   top_center = 2*n_sides
        #   bot_center = 2*n_sides + 1
        points: list[tuple[float, float, float]] = []
        points.extend(top)
        points.extend(bot)
        points.append(top_center)
        points.append(bot_center)

        top_ci = 2 * n_sides
        bot_ci = 2 * n_sides + 1

        face_counts: list[int] = []
        face_indices: list[int] = []

        # Side quads: (top[i], bot[i], bot[i+1], top[i+1]) → outward radial normal
        for i in range(n_sides):
            j = (i + 1) % n_sides
            face_indices.extend([i, n_sides + i, n_sides + j, j])
            face_counts.append(4)

        # Top cap triangles: (top_center, top[i], top[i+1]) → normal along +axis
        for i in range(n_sides):
            j = (i + 1) % n_sides
            face_indices.extend([top_ci, i, j])
            face_counts.append(3)

        # Bottom cap triangles: (bot_center, bot[i+1], bot[i]) → normal along -axis
        for i in range(n_sides):
            j = (i + 1) % n_sides
            face_indices.extend([bot_ci, n_sides + j, n_sides + i])
            face_counts.append(3)

        return points, face_counts, face_indices

    # ══════════════════════════════════════════════════════════════════
    # Friction settings
    # ══════════════════════════════════════════════════════════════════

    def _apply_friction_settings(self):
        """Apply friction values to match IsaacGym's set_*_asset_rigid_shape_properties().

        In IsaacGym, friction is set on the ASSET before env creation via
        set_robot/table/object_asset_rigid_shape_properties(). In Isaac Lab,
        we modify material properties after scene setup via root_physx_view.

        Material tensor shape: (num_envs, max_shapes, 3)
          dim 2: [static_friction, dynamic_friction, restitution]
        """
        from isaacsim.core.simulation_manager import SimulationManager

        physics_sim_view = SimulationManager.get_physics_sim_view()
        all_env_ids = torch.arange(self.num_envs, device="cpu")

        robot_friction = self.cfg.robot_friction
        fingertip_friction = self.cfg.fingertip_friction
        object_friction = self.cfg.object_friction
        table_friction = self.cfg.table_friction

        # ── Robot: base friction + higher fingertip friction ──
        robot_materials = self.robot.root_physx_view.get_material_properties()
        # (num_envs, max_shapes, 3)

        # Set all robot shapes to base friction
        robot_materials[:, :, 0] = robot_friction   # static_friction
        robot_materials[:, :, 1] = robot_friction   # dynamic_friction

        # Build per-body shape counts to find fingertip shape indices
        num_shapes_per_body = []
        for link_path in self.robot.root_physx_view.link_paths[0]:
            link_view = physics_sim_view.create_rigid_body_view(link_path)
            num_shapes_per_body.append(link_view.max_shapes)

        # Find fingertip body indices and override their shapes
        body_names = self.robot.data.body_names
        fingertip_body_ids = [body_names.index(name) for name in FINGERTIP_NAMES]
        for body_id in fingertip_body_ids:
            start_idx = sum(num_shapes_per_body[:body_id])
            end_idx = start_idx + num_shapes_per_body[body_id]
            robot_materials[:, start_idx:end_idx, 0] = fingertip_friction
            robot_materials[:, start_idx:end_idx, 1] = fingertip_friction

        self.robot.root_physx_view.set_material_properties(robot_materials, all_env_ids)

        # ── Objects (all slots) ──
        obj_shapes_per_slot = []
        for j, obj in enumerate(self.objects):
            obj_num_shapes = obj.root_physx_view.max_shapes
            obj_shapes_per_slot.append(obj_num_shapes)
            if obj_num_shapes > 0:
                object_materials = obj.root_physx_view.get_material_properties()
                object_materials[:, :, 0] = object_friction
                object_materials[:, :, 1] = object_friction
                obj.root_physx_view.set_material_properties(object_materials, all_env_ids)
            else:
                print(f"[WARN] Object_{j} has 0 collision shapes — skipping friction. "
                      f"Friction was already set to {object_friction} in the USD physics material.")

        # ── Table ──
        table_materials = self.table.root_physx_view.get_material_properties()
        table_materials[:, :, 0] = table_friction
        table_materials[:, :, 1] = table_friction
        self.table.root_physx_view.set_material_properties(table_materials, all_env_ids)

        print(f"[SimToolRealEnv] Applied friction settings:")
        print(f"  Robot body friction: {robot_friction}")
        print(f"  Fingertip friction:  {fingertip_friction} (bodies: {FINGERTIP_NAMES})")
        print(f"  Object friction:     {object_friction}")
        print(f"  Table friction:      {table_friction}")
        print(f"  Robot shapes: {robot_materials.shape[1]}, "
              f"Object shapes per slot: {obj_shapes_per_slot}, "
              f"Table shapes: {table_materials.shape[1]}")

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
        self.lifted_object = torch.zeros(N, self.cfg.n_objects_max, dtype=torch.bool, device=device)
        self.lift_bonus_consumed = torch.zeros(N, self.cfg.n_objects_max, dtype=torch.bool, device=device)
        self.closest_fingertip_dist = torch.full((N, self.num_fingertips), -1.0, device=device)
        self.furthest_hand_dist = torch.full((N,), -1.0, device=device)
        self.closest_keypoint_max_dist = torch.full((N,), -1.0, device=device)
        self.closest_keypoint_max_dist_fixed_size = torch.full((N,), -1.0, device=device)
        self.total_episode_closest_keypoint_max_dist = torch.zeros(N, device=device)
        self.prev_total_episode_closest_keypoint_max_dist = torch.zeros(N, device=device)
        self.prev_episode_closest_keypoint_max_dist = 1000 * torch.ones(N, device=device)
        self.prev_episode_true_objective = torch.zeros(N, device=device)
        self.true_objective = torch.zeros(N, device=device)

        # Object mass for random force scaling (matches IsaacGym: force * mass)
        # Populated after first scene.update() in _populate_sim_buffers
        self._object_rb_masses = None

        # Goal states: [pos(3) + quat_wxyz(4) + linvel(3) + angvel(3)] = 13
        self.goal_states = torch.zeros(N, 13, device=device)

        # Object init state (recorded at reset for lifting reward reference)
        self.object_init_state = torch.zeros(N, self.cfg.n_objects_max, 13, device=device)

        # Reward tracking
        self.rewards_episode = {}
        reward_names = [
            "raw_fingertip_delta_rew", "raw_hand_delta_penalty", "raw_lifting_rew",
            "raw_keypoint_rew", "raw_object_lin_vel_penalty", "raw_object_ang_vel_penalty",
            "fingertip_delta_rew", "hand_delta_penalty", "lifting_rew", "lift_bonus_rew",
            "keypoint_rew", "kuka_actions_penalty", "hand_actions_penalty", "joint_power_penalty",
            "bonus_rew", "object_lin_vel_penalty", "object_ang_vel_penalty", "drop_penalty", "total_reward",
        ]
        for name in reward_names:
            self.rewards_episode[name] = torch.zeros(N, device=device)

        # Reset goal buffer
        self.reset_goal_buf = torch.zeros(N, dtype=torch.bool, device=device)

        # Multi-object tracking
        self.active_object_idx = torch.zeros(N, dtype=torch.long, device=device)
        self.num_switches_this_episode = torch.zeros(N, dtype=torch.long, device=device)
        self.consecutive_successes_current_object = torch.zeros(N, dtype=torch.long, device=device)
        self.drop_penalty_applied = torch.zeros(N, self.cfg.n_objects_max, dtype=torch.bool, device=device)
        self.num_drops = torch.zeros(N, device=device)
        # Per-env object count (sampled from [n_objects_min, n_objects_max] at reset)
        self.n_objects_per_env = torch.ones(N, dtype=torch.long, device=device) * self.cfg.n_objects_min
        # Multi-object curriculum
        self.curriculum_max_objects = torch.ones(N, dtype=torch.long, device=device) * self.cfg.n_objects_min
        self.curriculum_cumulative_successes = torch.zeros(N, device=device)

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
        self.object_keypoint_offsets = (offsets * base_size * kp_scale / 2).unsqueeze(0).unsqueeze(0).expand(N, self.cfg.n_objects_max, -1, -1).clone()

        fixed_size = torch.tensor(self.cfg.fixed_size, device=device, dtype=torch.float)
        self.object_keypoint_offsets_fixed_size = (
            (offsets * fixed_size.unsqueeze(0) * kp_scale / 2).unsqueeze(0).unsqueeze(0).expand(N, self.cfg.n_objects_max, -1, -1).clone()
        )

        # Object scales (for procedural objects; set to 1.0 initially)
        self.object_scales = torch.ones(N, self.cfg.n_objects_max, 3, device=device)
        # Full bounding box scales (handle + head) for spawn clamping only
        self.object_bbox_scales = torch.ones(N, self.cfg.n_objects_max, 3, device=device)
        self.object_scale_noise_multiplier = torch.ones(N, self.cfg.n_objects_max, 3, device=device)

        # Per-env asset mode tensor (0=HHP, 1=RA)
        self._env_asset_mode = torch.tensor(self._env_asset_mode_list, dtype=torch.long, device=device)

        # Per-env grasp center offsets (zeros for HHP, actual offset for RA)
        # This unified tensor simplifies runtime offset application
        self._env_grasp_offsets = torch.zeros(N, self.cfg.n_objects_max, 3, device=device)

        # Override per-env scales and keypoint offsets for diverse procedural objects
        # Handle both HHP and RA modes based on per-env assignment
        num_hhp = len(self._hhp_scales) if hasattr(self, '_hhp_scales') else 0
        num_ra = len(self._ra_scales) if hasattr(self, '_ra_scales') else 0
        kp_scale = self.cfg.keypoint_scale

        for ei in range(N):
            mode = self._env_asset_mode[ei].item()
            for j in range(self.cfg.n_objects_max):
                if mode == 0 and num_hhp > 0:
                    # Handle-head primitives mode
                    obj_idx = self._get_hhp_obj_idx_for_env_and_slot(ei, j, num_hhp)
                    # Handle-only scale for object_scales (observations) and keypoints
                    hs = self._hhp_handle_scales[obj_idx]
                    if len(hs) == 3:
                        handle_xyz = hs
                    else:
                        # Cylinder: [height, diameter] → [height, diameter, diameter]
                        handle_xyz = [hs[0], hs[1], hs[1]]
                    self.object_scales[ei, j] = torch.tensor(
                        [h / base_size for h in handle_xyz], device=device
                    )
                    # Full bbox (handle + head) for spawn clamping only
                    raw = self._hhp_scales[obj_idx]  # (x, y, z) full bbox in meters
                    self.object_bbox_scales[ei, j] = torch.tensor(
                        [r / base_size for r in raw], device=device
                    )
                    for ki in range(self.num_keypoints):
                        for ci in range(3):
                            self.object_keypoint_offsets[ei, j, ki, ci] = (
                                self.keypoint_offsets_np[ki, ci] * handle_xyz[ci] * kp_scale / 2
                            )
                    # HHP has no grasp offset (centered at origin)
                    # _env_grasp_offsets[ei, j] stays zeros

                elif mode == 1 and num_ra > 0:
                    # Real assets mode
                    obj_idx = self._get_ra_obj_idx_for_env_and_slot(ei, j, num_ra)
                    raw = self._ra_scales[obj_idx]      # (x, y, z) bounding box HALF-extents in meters
                    offset = self._ra_offsets[obj_idx]   # (x, y, z) BB center offset
                    scale_tensor = torch.tensor(
                        [r * 2 / base_size for r in raw], device=device
                    )
                    self.object_scales[ei, j] = scale_tensor
                    self.object_bbox_scales[ei, j] = scale_tensor
                    for ki in range(self.num_keypoints):
                        for ci in range(3):
                            # Corner offset + BB center offset so keypoints are
                            # centered on the grasp region, not the mesh origin.
                            # raw is a half-extent, so raw * kp_scale (no /2) matches
                            # the HHP path where full_extent * kp_scale / 2 is used.
                            self.object_keypoint_offsets[ei, j, ki, ci] = (
                                self.keypoint_offsets_np[ki, ci] * raw[ci] * kp_scale
                                + offset[ci]
                            )
                    # Store grasp offset for RA envs
                    self._env_grasp_offsets[ei, j] = torch.tensor(offset, device=device)

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
        self.above_threshold_since = None

        # Tyler curriculum (obs dropout)
        self._tyler_curriculum_scale = self.cfg.init_tyler_curriculum_scale if hasattr(self.cfg, 'init_tyler_curriculum_scale') else 0.0
        self._last_tyler_curriculum_update = time.time()

        # First env index considered a "student" env. Set to num_teacher_envs by
        # TeacherDepthDaggerAgent so that curriculum and scalar stats only reflect
        # the student policy (teacher envs run the frozen expert, inflating success counts).
        self.student_env_start = 0

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
        # ── Goal-only resets (consecutive success chaining) ──
        # Mirrors IsaacGym pre_physics_step lines 3798-3808.
        # reset_goal_buf was set in _compute_kuka_reward for envs that achieved success.
        # We sample a new goal WITHOUT resetting the episode, allowing up to
        # max_consecutive_successes goals per episode.
        reset_goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_goal_env_ids) > 0:
            # Filter out envs that also need a full reset (those are handled by _reset_idx)
            # In Isaac Lab, _reset_idx already ran for terminated/truncated envs.
            # But reset_goal_buf may still be set for envs that got fully reset.
            # We only process goal-only resets for envs that are still active.
            # Use episode_length_buf > 0 as proxy: after _reset_idx, buf is 0.
            active_mask = self.episode_length_buf[reset_goal_env_ids] > 0
            goal_only_ids = reset_goal_env_ids[active_mask]
            if len(goal_only_ids) > 0:
                self._reset_target_pose(goal_only_ids, is_first_goal=False)
                self.near_goal_steps[goal_only_ids] = 0
                # Accumulate closest keypoint dist for this goal before resetting
                self.prev_total_episode_closest_keypoint_max_dist[goal_only_ids] = (
                    self.total_episode_closest_keypoint_max_dist[goal_only_ids]
                )
                self.total_episode_closest_keypoint_max_dist[goal_only_ids] += torch.where(
                    self.closest_keypoint_max_dist[goal_only_ids] > 0,
                    self.closest_keypoint_max_dist[goal_only_ids],
                    torch.zeros_like(self.closest_keypoint_max_dist[goal_only_ids]),
                )
                self.closest_keypoint_max_dist[goal_only_ids] = -1.0
                self.closest_keypoint_max_dist_fixed_size[goal_only_ids] = -1.0

                # ── Multi-object switching ──
                # Track consecutive successes for the current object
                self.consecutive_successes_current_object[goal_only_ids] += 1

                if self.cfg.n_objects_max > 1 and self.cfg.switch_object_prob > 0:
                    # Only envs with >1 object can switch
                    can_switch = self.n_objects_per_env[goal_only_ids] > 1
                    switchable_ids = goal_only_ids[can_switch]

                    if len(switchable_ids) > 0:
                        # Exponential scaling: p = base_prob * 2^(consecutive_successes - 1)
                        # First success: base_prob, second: 2*base, third: 4*base, ...
                        exponent = (self.consecutive_successes_current_object[switchable_ids] - 1).float()
                        p_switch = torch.clamp(
                            self.cfg.switch_object_prob * (2.0 ** exponent),
                            max=1.0,
                        )
                        switch_mask = torch.rand(len(switchable_ids), device=self.device) < p_switch
                        switch_ids = switchable_ids[switch_mask]

                        if len(switch_ids) > 0:
                            # Reset consecutive success counter for switched envs
                            self.consecutive_successes_current_object[switch_ids] = 0

                            # Sample new active object (uniform over OTHER objects within per-env count)
                            current = self.active_object_idx[switch_ids]
                            n_per_env = self.n_objects_per_env[switch_ids]  # (S,)
                            new_idx = torch.zeros(len(switch_ids), dtype=torch.long, device=self.device)
                            for ii in range(len(switch_ids)):
                                n_i = n_per_env[ii].item()
                                # Sample from [0, n_i-2] then skip current
                                raw = torch.randint(0, n_i - 1, (1,), device=self.device).item()
                                new_idx[ii] = raw if raw < current[ii].item() else raw + 1
                            self.active_object_idx[switch_ids] = new_idx
                            self.num_switches_this_episode[switch_ids] += 1

                            # Re-sample goal for the NEW active object
                            self._reset_target_pose(switch_ids, is_first_goal=False)

                # Reset episode length to extend the episode (matches IsaacGym progress_buf reset)
                self.episode_length_buf[goal_only_ids] = 0
            self.reset_goal_buf[:] = False

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

        # ── Log world-space AABB corners per timestep ──
        self._log_aabb_corners_step()

        # ── Collision drop-test: log object Z for first 200 steps (only if RA envs exist) ──
        if (self._uses_ra
                and not getattr(self, '_collision_test_done', False)):
            step = getattr(self, '_collision_test_step', 0)
            if step == 0:
                self._collision_z_log = []
            all_obj_z = torch.stack([obj.data.root_pos_w[:, 2] for obj in self.objects], dim=1)
            arange_N = torch.arange(self.num_envs, device=self.device)
            obj_z = all_obj_z[arange_N, self.active_object_idx].cpu().numpy().copy()
            env_origins_z = self.scene.env_origins[:, 2].cpu().numpy()
            local_z = obj_z - env_origins_z
            self._collision_z_log.append(local_z.copy())
            self._collision_test_step = step + 1

            if step == 199:
                import numpy as _np
                z_arr = _np.array(self._collision_z_log)  # (200, num_envs)
                _np.save("/tmp/collision_drop_test.npy", z_arr)
                table_z = 0.53  # table surface approx
                final_z = z_arr[-1]
                n_on_table = ((final_z > table_z - 0.05) & (final_z < table_z + 0.15)).sum()
                n_fell = (final_z < table_z - 0.05).sum()
                n_float = (final_z > table_z + 0.15).sum()

                # Compute effective collision half-height
                settled_z_mean = final_z.mean()
                effective_half_z = settled_z_mean - table_z

                print(f"\n[CollisionTest] Drop test (200 steps, zero actions):")
                print(f"  On table: {n_on_table}/{len(final_z)}, "
                      f"Fell through: {n_fell}, Still floating: {n_float}")
                print(f"  Z range: [{final_z.min():.4f}, {final_z.max():.4f}]")
                print(f"  Per-env final Z (local): {final_z[:min(10, len(final_z))]}")
                print(f"  Mean settled Z: {settled_z_mean:.4f}")
                print(f"  Effective collision half-height: {effective_half_z:.4f}m")
                print(f"    (≈0.02 means 0.04m placeholder cube is being used)")
                print(f"  Saved to /tmp/collision_drop_test.npy")

                self._collision_test_done = True

        # Video capture (after physics, matching IsaacGym post_physics_step location)
        self._control_steps += 1
        self._capture_video_if_needed()

        if self.cfg.use_depth_camera or self.cfg.use_third_person_camera:
            self._render_depth()

        obs_buf, states_buf = self._compute_obs_and_states()

        # Clamp observations
        obs_buf = torch.clamp(obs_buf, -self.cfg.clamp_abs_observations, self.cfg.clamp_abs_observations)

        result = {"policy": obs_buf}
        if states_buf is not None:
            result["critic"] = states_buf
        if self.cfg.use_depth_camera and self.depth_buf is not None:
            result["depth"] = self.depth_buf
        if self.cfg.use_third_person_camera and self.tp_depth_buf is not None:
            result["tp_depth"] = self.tp_depth_buf

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
        # Populate derived state from fresh physics data BEFORE dones & rewards.
        # Isaac Lab step order: _get_dones → _get_rewards → _reset_idx → _get_observations.
        # Without this call, derived state (curr_fingertip_distances, keypoints, etc.)
        # would be stale from the PREVIOUS step's _get_observations() call.
        # In IsaacGym, populate_sim_buffers() is called BEFORE compute_kuka_reward().
        self._populate_sim_buffers()

        # Update goal keypoint markers (must happen before next sim.render() for camera visibility)
        if self._goal_kp_wireframe_enabled:
            self._update_goal_keypoint_markers()

        # Show handle wireframe only on the active object
        if self.cfg.debug_multi_object and hasattr(self, '_handle_wf_paths'):
            self._update_handle_wireframe_visibility()

        ones = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Object fell below threshold (active object)
        object_pos = self.object_pos  # (N, 3) - active object, set by _populate_sim_buffers
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

        # Reset when dropped: object was lifted but dropped back below init z
        # (matches IsaacGym env.py:2476-2489)
        if self.cfg.reset_when_dropped:
            arange_N = torch.arange(self.num_envs, device=self.device)
            idx = self.active_object_idx
            dropped_z = self.object_init_state[arange_N, idx, 2]
            dropped = (object_pos[:, 2] < dropped_z) & self.lifted_object[arange_N, idx]
        else:
            dropped = zeros

        terminated = object_z_low | max_succ_reached | hand_far | dropped

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

        # Save episode lengths BEFORE super() resets them
        _saved_ep_lens = self.episode_length_buf[env_ids].clone()

        super()._reset_idx(env_ids)

        # Reset episode tracking
        self.prev_episode_successes[env_ids] = self.successes[env_ids]
        self.successes[env_ids] = 0
        self.near_goal_steps[env_ids] = 0
        self.lifted_object[env_ids] = False  # resets all n_objects_max slots for these envs
        self.lift_bonus_consumed[env_ids] = False  # allow one lift bonus per slot each new episode
        self.active_object_idx[env_ids] = 0
        self.num_switches_this_episode[env_ids] = 0
        self.consecutive_successes_current_object[env_ids] = 0
        self.drop_penalty_applied[env_ids] = False
        self.num_drops[env_ids] = 0
        # Sample per-env object count
        if self.cfg.use_multi_object_curriculum:
            # Curriculum: use each env's current curriculum level
            self.n_objects_per_env[env_ids] = self.curriculum_max_objects[env_ids]
        elif self.cfg.n_objects_min < self.cfg.n_objects_max:
            self.n_objects_per_env[env_ids] = torch.randint(
                self.cfg.n_objects_min, self.cfg.n_objects_max + 1,
                (len(env_ids),), device=self.device,
            )
        else:
            self.n_objects_per_env[env_ids] = self.cfg.n_objects_max
        self.closest_fingertip_dist[env_ids] = -1.0
        self.furthest_hand_dist[env_ids] = -1.0

        # Accumulate final goal's closest_keypoint_max_dist into total
        # (matches IsaacGym reset_target_pose lines 3403-3410, called from reset_idx)
        self.prev_total_episode_closest_keypoint_max_dist[env_ids] = (
            self.total_episode_closest_keypoint_max_dist[env_ids]
        )
        self.total_episode_closest_keypoint_max_dist[env_ids] += torch.where(
            self.closest_keypoint_max_dist[env_ids] > 0,
            self.closest_keypoint_max_dist[env_ids],
            torch.zeros_like(self.closest_keypoint_max_dist[env_ids]),
        )
        self.closest_keypoint_max_dist[env_ids] = -1.0
        self.closest_keypoint_max_dist_fixed_size[env_ids] = -1.0

        # Compute prev_episode_closest_keypoint_max_dist (matches IsaacGym reset_idx)
        self.prev_episode_closest_keypoint_max_dist[env_ids] = torch.where(
            self.prev_episode_successes[env_ids] > 0,
            self.prev_total_episode_closest_keypoint_max_dist[env_ids]
            / self.prev_episode_successes[env_ids],
            self.total_episode_closest_keypoint_max_dist[env_ids],
        )
        self.total_episode_closest_keypoint_max_dist[env_ids] = 0
        self.prev_total_episode_closest_keypoint_max_dist[env_ids] = 0

        self.prev_episode_true_objective[env_ids] = self.true_objective[env_ids]
        self.true_objective[env_ids] = 0

        # DEBUG: Log per-component reward breakdown at reset
        if not hasattr(self, '_debug_reset_count'):
            self._debug_reset_count = 0
        self._debug_reset_count += 1
        # Only log resets where episodes actually ran (ep_len > 0)
        non_zero_mask = _saved_ep_lens > 0
        if non_zero_mask.any() and self._debug_reset_count % 10 == 0:
            ids_nz = env_ids[non_zero_mask]
            ep_lens = _saved_ep_lens[non_zero_mask].float()
            mean_ep_len = ep_lens.mean().item()
            parts = []
            for key in self.rewards_episode:
                val = self.rewards_episode[key][ids_nz].mean().item()
                parts.append(f"{key}={val:.2f}")
            # print(f"[DEBUG REWARD] n={len(ids_nz)} ep_len={mean_ep_len:.1f} | {' | '.join(parts)}", flush=True)

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

        # Reset object pose first (matches IsaacGym order)
        self._reset_object_pose(env_ids)

        # Reset goal
        self._reset_target_pose(env_ids, is_first_goal=True)

        # Reset robot joint positions with noise (after object, matching IsaacGym)
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

        # Object scale noise multiplier
        noise_min, noise_max = self.cfg.object_scale_noise_multiplier_range
        self.object_scale_noise_multiplier[env_ids] = torch_rand_float(
            noise_min, noise_max, (len(env_ids), self.cfg.n_objects_max, 3), device=self.device
        )

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
        # Lazy-init object mass for all slots (available after first scene.update())
        if self._object_rb_masses is None:
            # (N, n_objects_max) — one mass per object slot per env
            self._object_rb_masses = torch.stack(
                [obj.data.default_mass.squeeze(-1).to(self.device) for obj in self.objects], dim=1
            )

        # Robot state
        self.arm_hand_dof_pos = self._get_joint_pos_in_our_order()
        self.arm_hand_dof_vel = self._get_joint_vel_in_our_order()

        # Object state (wxyz quaternion in Isaac Lab)
        N = self.num_envs
        arange_N = torch.arange(N, device=self.device)
        idx = self.active_object_idx  # (N,)

        # Stack all objects
        self.all_obj_pos = torch.stack([obj.data.root_pos_w for obj in self.objects], dim=1)  # (N, n_obj, 3)
        self.all_obj_quat_wxyz = torch.stack([obj.data.root_quat_w for obj in self.objects], dim=1)
        self.all_obj_linvel = torch.stack([obj.data.root_lin_vel_w for obj in self.objects], dim=1)
        self.all_obj_angvel = torch.stack([obj.data.root_ang_vel_w for obj in self.objects], dim=1)

        # Gather active object
        self.object_pos = self.all_obj_pos[arange_N, idx]  # (N, 3)
        self.object_quat_wxyz = self.all_obj_quat_wxyz[arange_N, idx]
        self.object_rot = quat_wxyz_to_xyzw(self.object_quat_wxyz)  # convert to xyzw for reward math compatibility
        self.object_linvel = self.all_obj_linvel[arange_N, idx]
        self.object_angvel = self.all_obj_angvel[arange_N, idx]

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

        # Compute fingertip distances relative to grasp region center.
        # For RA envs, this uses the grasp bbox offset; for HHP envs, offset is zero.
        # Using the unified _env_grasp_offsets tensor simplifies this to a single code path.
        env_offsets = self._env_grasp_offsets[arange_N, idx]  # (N, 3)
        grasp_anchor_pos = self.object_pos + self._quat_rotate_xyzw(
            self.object_rot, env_offsets
        )

        obj_pos_repeat = grasp_anchor_pos.unsqueeze(1).expand(-1, self.num_fingertips, -1)
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

        # Gather active object's keypoint offsets and scale noise
        active_kp_offsets = self.object_keypoint_offsets[arange_N, idx]  # (N, K, 3)
        active_kp_offsets_fixed = self.object_keypoint_offsets_fixed_size[arange_N, idx]  # (N, K, 3)
        active_scale_noise = self.object_scale_noise_multiplier[arange_N, idx]  # (N, 3)

        # Keypoints
        for i in range(self.num_keypoints):
            self.obj_keypoint_pos[:, i] = self.object_pos + self._quat_rotate_xyzw(
                self.object_rot, active_kp_offsets[:, i] * active_scale_noise
            )
            self.goal_keypoint_pos[:, i] = self.goal_pos + self._quat_rotate_xyzw(
                self.goal_rot, active_kp_offsets[:, i] * active_scale_noise
            )
            self.observed_obj_keypoint_pos[:, i] = self.observed_object_pos + self._quat_rotate_xyzw(
                self.observed_object_rot, active_kp_offsets[:, i] * active_scale_noise
            )
            self.obj_keypoint_pos_fixed_size[:, i] = self.object_pos + self._quat_rotate_xyzw(
                self.object_rot, active_kp_offsets_fixed[:, i]
            )
            self.goal_keypoint_pos_fixed_size[:, i] = self.goal_pos + self._quat_rotate_xyzw(
                self.goal_rot, active_kp_offsets_fixed[:, i]
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
        # IsaacGym rigid_body_state_tensor returns env-local positions;
        # Isaac Lab body_pos_w is world-frame. Subtract env_origins to match.
        obs_dict["palm_pos"] = self.palm_center_pos - self.scene.env_origins
        obs_dict["palm_rot"] = self._palm_state[:, 3:7]
        obs_dict["object_rot"] = self.object_state[:, 3:7]
        obs_dict["keypoints_rel_palm"] = self.keypoints_rel_palm.reshape(N, -1)
        obs_dict["keypoints_rel_goal"] = self.keypoints_rel_goal.reshape(N, -1)
        obs_dict["fingertip_pos_rel_palm"] = self.fingertip_pos_rel_palm.reshape(N, -1)
        arange_N = torch.arange(N, device=self.device)
        idx = self.active_object_idx
        obs_dict["object_scales"] = self.object_scales[arange_N, idx] * self.object_scale_noise_multiplier[arange_N, idx]

        # Critic-only observations
        obs_dict["palm_vel"] = self._palm_state[:, 7:13]
        obs_dict["object_vel"] = self.object_state[:, 7:13]
        obs_dict["closest_keypoint_max_dist"] = self.closest_keypoint_max_dist.unsqueeze(-1)
        if self.cfg.fixed_size_keypoint_reward:
            obs_dict["closest_keypoint_max_dist"] = self.closest_keypoint_max_dist_fixed_size.unsqueeze(-1)
        obs_dict["closest_fingertip_dist"] = self.closest_fingertip_dist
        obs_dict["lifted_object"] = self.lifted_object[arange_N, self.active_object_idx].float().unsqueeze(-1)
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

        # Multi-object curriculum advancement
        if self.cfg.use_multi_object_curriculum:
            self.curriculum_cumulative_successes += is_success.float()
            advance_mask = (
                (self.curriculum_cumulative_successes >= self.cfg.multi_object_success_threshold)
                & (self.curriculum_max_objects < self.cfg.n_objects_max)
            )
            if advance_mask.any():
                self.curriculum_max_objects[advance_mask] += 1
                self.curriculum_cumulative_successes[advance_mask] = 0.0
                n_advanced = advance_mask.sum().item()
                new_levels = self.curriculum_max_objects[advance_mask]
                print(f"[MultiObjCurriculum] {n_advanced} envs advanced: "
                      f"levels={new_levels[:5].tolist()}...", flush=True)

        # Scale rewards
        object_lin_vel_penalty = -torch.sum(torch.square(self.object_linvel), dim=-1)
        object_ang_vel_penalty = -torch.sum(torch.square(self.object_angvel), dim=-1)

        fingertip_delta_rew_scaled = fingertip_delta_rew * self.cfg.distance_delta_rew_scale
        hand_delta_penalty_scaled = hand_delta_penalty * self.cfg.distance_delta_rew_scale * 0  # disabled (matches IsaacGym)
        lifting_rew_scaled = lifting_rew * self.cfg.lifting_rew_scale
        keypoint_rew_scaled = keypoint_rew * self.cfg.keypoint_rew_scale
        object_lin_vel_penalty_scaled = object_lin_vel_penalty * self.cfg.object_lin_vel_penalty_scale
        object_ang_vel_penalty_scaled = object_ang_vel_penalty * self.cfg.object_ang_vel_penalty_scale

        kuka_actions_penalty, hand_actions_penalty = self._action_penalties()

        # Joint power penalty: penalize |velocity * torque| across all joints
        joint_torques = self.robot.data.applied_torque[:, self._all_joint_ids_t]
        joint_power_penalty = (
            -torch.sum(torch.abs(self.arm_hand_dof_vel * joint_torques), dim=-1)
            * self.cfg.joint_power_penalty
        )

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
            + joint_power_penalty
            + bonus_rew
            + object_lin_vel_penalty_scaled
            + object_ang_vel_penalty_scaled
        )

        # ── Drop penalty ──
        # When a lifted object's Z reaches init_z + threshold (back at table),
        # check if the robot's hand is near the object:
        #   Near (< drop_held_distance) → place, no penalty
        #   Far  (>= drop_held_distance) → drop, apply penalty
        # Either way, clear lifted flag for that slot (resolved).
        drop_penalty_rew = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.drop_penalty > 0:
            for j in range(self.cfg.n_objects_max):
                slot_in_use = j < self.n_objects_per_env  # (N,) bool
                obj_pos_j = self.all_obj_pos[:, j]  # (N, 3)
                init_z = self.object_init_state[:, j, 2]
                was_lifted = self.lifted_object[:, j]
                at_table = obj_pos_j[:, 2] <= init_z + self.cfg.drop_penalty_threshold
                resolve = slot_in_use & was_lifted & at_table

                if resolve.any():
                    # Compute min fingertip distance to this slot's object
                    obj_expanded = obj_pos_j[resolve].unsqueeze(1).expand(-1, self.num_fingertips, -1)
                    ft_dist = (self.fingertip_pos_offset[resolve] - obj_expanded).norm(dim=-1)  # (R, F)
                    min_ft_dist = ft_dist.min(dim=-1).values  # (R,)
                    not_held = min_ft_dist >= self.cfg.drop_held_distance

                    # Apply penalty only to drops (not held), and only once
                    not_yet_penalized = ~self.drop_penalty_applied[resolve, j]
                    do_penalize = not_held & not_yet_penalized
                    # Scatter penalty back to full (N,) tensor
                    resolve_ids = resolve.nonzero(as_tuple=False).squeeze(-1)
                    penalize_ids = resolve_ids[do_penalize]
                    drop_penalty_rew[penalize_ids] -= self.cfg.drop_penalty
                    self.drop_penalty_applied[penalize_ids, j] = True
                    self.num_drops[penalize_ids] += 1

                    # Clear lifted flag — this slot is resolved (placed or dropped)
                    self.lifted_object[resolve, j] = False

        reward = reward + drop_penalty_rew

        # Per-step reward values for EnvInfoObserver episode tracking
        episode_cumulative = {
            "fingertip_delta_rew": fingertip_delta_rew_scaled,
            "hand_delta_penalty": hand_delta_penalty_scaled,
            "lifting_rew": lifting_rew_scaled,
            "lift_bonus_rew": lift_bonus_rew,
            "keypoint_rew": keypoint_rew_scaled,
            "kuka_actions_penalty": kuka_actions_penalty,
            "hand_actions_penalty": hand_actions_penalty,
            "joint_power_penalty": joint_power_penalty,
            "bonus_rew": bonus_rew,
            "object_lin_vel_penalty": object_lin_vel_penalty_scaled,
            "object_ang_vel_penalty": object_ang_vel_penalty_scaled,
            "drop_penalty": drop_penalty_rew,
            "total_reward": reward,
        }

        # Track cumulative per-episode rewards
        for key, val in episode_cumulative.items():
            self.rewards_episode[key] += val

        # Compute true_objective (matches IsaacGym _true_objective)
        self.true_objective = self._true_objective()

        # Extras for logging (student envs only — teacher envs run the frozen expert
        # and would pollute all metrics).  All per-env tensors are sliced to
        # [student_env_start:] here so EnvInfoObserver receives student data directly.
        s = self.student_env_start
        self.extras["episode_cumulative"] = {k: v[s:] for k, v in episode_cumulative.items()}
        self.extras["rewards_episode"] = {k: v[s:] for k, v in self.rewards_episode.items()}
        self.extras["successes"] = self.prev_episode_successes[s:]
        self.extras["success_ratio"] = (
            self.prev_episode_successes[s:].mean().item()
            / self.cfg.max_consecutive_successes
        )
        self.extras["closest_keypoint_max_dist"] = self.prev_episode_closest_keypoint_max_dist[s:]
        self.extras["true_objective"] = self.prev_episode_true_objective[s:]
        self.extras["num_drops"] = self.num_drops[s:]
        self.extras["scalars"] = {"success_tolerance": self.success_tolerance}

        # Multi-object logging
        if self.cfg.n_objects_max > 1:
            # Per-object-count cumulative successes
            for n_obj in range(1, self.cfg.n_objects_max + 1):
                obj_count_mask = self.n_objects_per_env[s:] == n_obj
                if obj_count_mask.any():
                    self.extras[f"multi_object/{n_obj}_obj_cumulative"] = (
                        self.successes[s:][obj_count_mask].sum().item()
                    )
                else:
                    self.extras[f"multi_object/{n_obj}_obj_cumulative"] = 0.0
            # Curriculum level distribution
            if self.cfg.use_multi_object_curriculum:
                for n_obj in range(self.cfg.n_objects_min, self.cfg.n_objects_max + 1):
                    frac = (self.curriculum_max_objects[s:] == n_obj).float().mean().item()
                    self.extras[f"multi_object/curriculum_frac_{n_obj}_obj"] = frac
                self.extras["multi_object/curriculum_mean_level"] = (
                    self.curriculum_max_objects[s:].float().mean().item()
                )

        # Reset reason tracking
        self._track_reset_reasons()

        # Tolerance curriculum
        self._update_tolerance_curriculum()

        # Tyler curriculum (obs dropout scaling)
        self._update_tyler_curriculum()

        # Control step counter
        self.control_steps += 1
        self.frame_since_restart += 1

        return reward

    # ── Reward sub-functions ──

    def _lifting_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting object off table."""
        arange_N = torch.arange(self.num_envs, device=self.device)
        idx = self.active_object_idx

        active_init_z = self.object_init_state[arange_N, idx, 2]
        z_lift = 0.05 + self.object_pos[:, 2] - active_init_z
        lifting_rew = torch.clip(z_lift, 0, 0.5)

        active_lifted = self.lifted_object[arange_N, idx]
        active_bonus_consumed = self.lift_bonus_consumed[arange_N, idx]
        lifted_object = (z_lift > self.cfg.lifting_bonus_threshold) | active_lifted
        just_lifted = lifted_object & ~active_lifted
        bonus_eligible = just_lifted & ~active_bonus_consumed
        lift_bonus_rew = self.cfg.lifting_bonus * bonus_eligible.float()

        # DEBUG: Log when lifting bonus fires
        if just_lifted.any():
            n_just = just_lifted.sum().item()
            i_idx = just_lifted.nonzero(as_tuple=True)[0][:3]  # first 3
            for i in i_idx:
                ii = i.item()
                print(f"[DEBUG LIFT] env={ii} step={self.episode_length_buf[ii].item()} "
                      f"obj_z={self.object_pos[ii, 2].item():.4f} "
                      f"init_z={active_init_z[ii].item():.4f} "
                      f"z_lift={z_lift[ii].item():.4f} "
                      f"threshold={self.cfg.lifting_bonus_threshold}", flush=True)

        lifting_rew *= (~lifted_object).float()
        self.lifted_object[arange_N, idx] = lifted_object
        self.lift_bonus_consumed[arange_N, idx] = active_bonus_consumed | bonus_eligible

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
        mean_succ = self.prev_episode_successes[self.student_env_start:].mean().item()
        if mean_succ < 3.0:
            self.above_threshold_since = None
            return
        if self.above_threshold_since is None:
            self.above_threshold_since = self.frame_since_restart
        if self.frame_since_restart - self.last_curriculum_update < self.cfg.tolerance_curriculum_interval:
            return
        if self.frame_since_restart - self.above_threshold_since < self.cfg.tolerance_curriculum_interval:
            return
        self.success_tolerance *= self.cfg.tolerance_curriculum_increment
        self.success_tolerance = min(self.success_tolerance, self.initial_tolerance)
        self.success_tolerance = max(self.success_tolerance, self.target_tolerance)
        self.last_curriculum_update = self.frame_since_restart
        self.above_threshold_since = None
        print(f"[Curriculum] tolerance -> {self.success_tolerance:.4f} (mean_succ={mean_succ:.1f})")

    def _true_objective(self) -> Tensor:
        """Compute PBT objective (matches IsaacGym _true_objective)."""
        if self.initial_tolerance > self.target_tolerance:
            span = self.initial_tolerance - self.target_tolerance
            tolerance_objective = (self.initial_tolerance - self.success_tolerance) / span
        else:
            tolerance_objective = 1.0

        if self.success_tolerance > self.target_tolerance:
            return (self.successes * 0.01) + tolerance_objective
        else:
            return self.successes + tolerance_objective

    def _track_reset_reasons(self):
        """Track and log reset reason fractions (student envs only)."""
        s = self.student_env_start
        object_pos = self.object_pos[s:]  # active object pos, set by _populate_sim_buffers
        object_z_low = (object_pos[:, 2] < 0.1).sum().item()

        if self.cfg.max_consecutive_successes > 0:
            max_succ = (self.successes[s:] >= self.cfg.max_consecutive_successes).sum().item()
        else:
            max_succ = 0

        max_ep_len = (self.episode_length_buf[s:] >= self.max_episode_length - 1).sum().item()

        if hasattr(self, 'curr_fingertip_distances'):
            hand_far = (self.curr_fingertip_distances[s:].max(dim=-1).values > 1.5).sum().item()
        else:
            hand_far = 0

        if self.cfg.reset_when_dropped:
            arange_s = torch.arange(self.num_envs - s, device=self.device)
            idx_s = self.active_object_idx[s:]
            dropped_z = self.object_init_state[s:][arange_s, idx_s, 2]
            dropped = ((object_pos[:, 2] < dropped_z) & self.lifted_object[s:][arange_s, idx_s]).sum().item()
        else:
            dropped = 0

        current_counts = {
            "object_z_low": int(object_z_low),
            "max_consecutive_successes_reached": int(max_succ),
            "max_episode_length_reached": int(max_ep_len),
            "hand_far_from_object": int(hand_far),
            "dropped": int(dropped),
        }

        for reason, count in current_counts.items():
            self.recent_reset_reason_history.extend([reason] * count)
        recent_counts = Counter(self.recent_reset_reason_history)
        recent_total = sum(recent_counts.values())
        for reason, count in recent_counts.items():
            self.extras[f"reset/{reason}"] = (
                count / recent_total if recent_total > 0 else 0
            )

    def _update_tyler_curriculum(self):
        """Tyler curriculum: scale obs dropout from 0 (easy) to 1 (hard)."""
        mean_successes = self.prev_episode_successes[self.student_env_start:].mean().item()
        minutes_elapsed = (time.time() - self._last_tyler_curriculum_update) / 60
        success_ratio = mean_successes / self.cfg.max_consecutive_successes
        doing_well = success_ratio > self.cfg.curriculum_success_ratio

        time_to_update = getattr(self.cfg, 'time_to_update_tyler_curriculum', 5.0)
        update_step_size = getattr(self.cfg, 'update_step_size_tyler_curriculum', 0.01)

        if doing_well and minutes_elapsed > time_to_update:
            self._tyler_curriculum_scale += update_step_size
            if self._tyler_curriculum_scale > 1.0:
                self._tyler_curriculum_scale = 1.0
            self._last_tyler_curriculum_update = time.time()

        self.extras["tyler_curriculum_scale"] = self._tyler_curriculum_scale
        self.extras["mean_successes"] = mean_successes
        self.extras["mean_success_ratio"] = (
            mean_successes / self.cfg.max_consecutive_successes
        )
        self.extras["minutes_elapsed_since_last_update"] = minutes_elapsed

    @property
    def turn_off_palm_vel_obs_scale(self) -> float:
        if self.cfg.turn_off_palm_vel_obs:
            scale = 0.0
        elif self.cfg.turn_off_palm_vel_obs_slowly:
            if self.cfg.use_obs_dropout:
                scale = 0.0 if random.random() < self._tyler_curriculum_scale else 1.0
            else:
                scale = 1.0 - self._tyler_curriculum_scale
        else:
            scale = 1.0
        self.extras["turn_off_palm_vel_obs_scale"] = scale
        return scale

    @property
    def turn_off_object_vel_obs_scale(self) -> float:
        if self.cfg.turn_off_object_vel_obs:
            scale = 0.0
        elif self.cfg.turn_off_object_vel_obs_slowly:
            if self.cfg.use_obs_dropout:
                scale = 0.0 if random.random() < self._tyler_curriculum_scale else 1.0
            else:
                scale = 1.0 - self._tyler_curriculum_scale
        else:
            scale = 1.0
        self.extras["turn_off_object_vel_obs_scale"] = scale
        return scale

    @property
    def turn_off_extra_obs_scale(self) -> float:
        if self.cfg.turn_off_extra_obs:
            scale = 0.0
        elif self.cfg.turn_off_extra_obs_slowly:
            if self.cfg.use_obs_dropout:
                scale = 0.0 if random.random() < self._tyler_curriculum_scale else 1.0
            else:
                scale = 1.0 - self._tyler_curriculum_scale
        else:
            scale = 1.0
        self.extras["turn_off_extra_obs_scale"] = scale
        return scale

    # ══════════════════════════════════════════════════════════════════
    # Object reset helpers
    # ══════════════════════════════════════════════════════════════════

    def _reset_object_pose(self, env_ids: Tensor):
        """Reset all objects to start poses with noise, clamped to table bounds."""
        n_obj = self.cfg.n_objects_max
        N = len(env_ids)

        # Per-env object counts (already sampled in _reset_idx)
        env_n_objs = self.n_objects_per_env[env_ids]  # (N,)

        # Fixed slot positions across the table (n_objects_max evenly-spaced)
        if n_obj > 1:
            slot_positions = torch.linspace(
                -self.cfg.multi_object_x_spread * (n_obj - 1) / 2,
                self.cfg.multi_object_x_spread * (n_obj - 1) / 2,
                n_obj, device=self.device,
            )
        else:
            slot_positions = torch.zeros(1, device=self.device)

        # Random slot assignment: each env gets a random permutation of slot indices
        # so any object can spawn at any position (1 object max per slot).
        # slot_perm[i, j] = which slot position object j uses in env i
        slot_perm = torch.stack([torch.randperm(n_obj, device=self.device) for _ in range(N)])

        # ── Phase 1: Compute positions for all slots ──
        all_pos = []           # list of (N, 3) tensors per slot
        all_canonical = []     # list of (N, 3) tensors per slot
        all_quat_wxyz = []     # list of (N, 4) tensors per slot
        all_active_mask = []   # list of (N,) bool tensors per slot
        all_clamp_bounds = []  # list of (clamp_x_min, clamp_x_max, clamp_y_min, clamp_y_max) per slot

        for j in range(n_obj):
            active_mask = j < env_n_objs  # (N,) bool

            pos = torch.zeros(N, 3, device=self.device)
            pos[:, 0] = slot_positions[slot_perm[:, j]]
            pos[:, 1] = 0.0
            pos[:, 2] = self.cfg.table_reset_z + self.cfg.table_object_z_offset

            canonical_pos = pos.clone()

            if self.cfg.randomize_object_rotation:
                quat_wxyz = self._random_quaternion(N)
            else:
                quat_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(N, -1)

            # Apply grasp center offset (zeros for HHP, actual offset for RA)
            # Using the unified _env_grasp_offsets tensor handles both modes
            quat_xyzw_for_offset = quat_wxyz_to_xyzw(quat_wxyz)
            for i, eid in enumerate(env_ids):
                if not active_mask[i]:
                    continue
                offset_local = self._env_grasp_offsets[eid, j]
                if offset_local.abs().sum() > 1e-6:  # Only apply if non-zero
                    rotated_offset = self._quat_rotate_xyzw(
                        quat_xyzw_for_offset[i:i+1], offset_local.unsqueeze(0)
                    ).squeeze(0)
                    pos[i] -= rotated_offset
                    canonical_pos[i] -= rotated_offset

            # Position noise
            pos[:, 0] += torch_rand_float(
                -self.cfg.reset_position_noise_x, self.cfg.reset_position_noise_x, (N, 1), self.device
            ).squeeze(-1)
            pos[:, 1] += torch_rand_float(
                -self.cfg.reset_position_noise_y, self.cfg.reset_position_noise_y, (N, 1), self.device
            ).squeeze(-1)
            pos[:, 2] += torch_rand_float(
                -self.cfg.reset_position_noise_z, self.cfg.reset_position_noise_z, (N, 1), self.device
            ).squeeze(-1)

            # Compute per-env rotation-aware AABB half-extents for clamping.
            # object_bbox_scales stores the full bounding box (handle + head) where
            # actual_size_meters = object_bbox_scales * object_base_size.
            obj_half = self.object_bbox_scales[env_ids, j] * self.cfg.object_base_size / 2.0  # (N, 3)
            # Rotation matrix from wxyz quaternion
            w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
            # Absolute rotation matrix rows for X and Y axes (for AABB computation)
            abs_r00 = (1 - 2*(y*y + z*z)).abs()
            abs_r01 = (2*(x*y - w*z)).abs()
            abs_r02 = (2*(x*z + w*y)).abs()
            abs_r10 = (2*(x*y + w*z)).abs()
            abs_r11 = (1 - 2*(x*x + z*z)).abs()
            abs_r12 = (2*(y*z - w*x)).abs()
            # Rotated AABB half-extent in X and Y
            aabb_half_x = abs_r00 * obj_half[:, 0] + abs_r01 * obj_half[:, 1] + abs_r02 * obj_half[:, 2]
            aabb_half_y = abs_r10 * obj_half[:, 0] + abs_r11 * obj_half[:, 1] + abs_r12 * obj_half[:, 2]
            # Add a small extra margin on top of the object AABB
            extra = self.cfg.spawn_object_margin
            margin_x = aabb_half_x + extra  # (N,)
            margin_y = aabb_half_y + extra  # (N,)
            # Per-env clamp bounds
            clamp_x_min = -self._table_half_x + margin_x
            clamp_x_max = self._table_half_x - margin_x
            clamp_y_min = -self._table_half_y + margin_y
            clamp_y_max = self._table_half_y - margin_y
            # Ensure min <= max (if object is wider than table, center it)
            clamp_x_min = torch.minimum(clamp_x_min, torch.zeros_like(clamp_x_min))
            clamp_x_max = torch.maximum(clamp_x_max, torch.zeros_like(clamp_x_max))
            clamp_y_min = torch.minimum(clamp_y_min, torch.zeros_like(clamp_y_min))
            clamp_y_max = torch.maximum(clamp_y_max, torch.zeros_like(clamp_y_max))

            pos[:, 0] = torch.max(torch.min(pos[:, 0], clamp_x_max), clamp_x_min)
            pos[:, 1] = torch.max(torch.min(pos[:, 1], clamp_y_max), clamp_y_min)
            canonical_pos[:, 0] = torch.max(torch.min(canonical_pos[:, 0], clamp_x_max), clamp_x_min)
            canonical_pos[:, 1] = torch.max(torch.min(canonical_pos[:, 1], clamp_y_max), clamp_y_min)

            # Hide unused object slots underground so they don't interfere
            pos[~active_mask, 2] = -10.0
            canonical_pos[~active_mask, 2] = -10.0

            all_pos.append(pos)
            all_canonical.append(canonical_pos)
            all_quat_wxyz.append(quat_wxyz)
            all_active_mask.append(active_mask)
            all_clamp_bounds.append((clamp_x_min, clamp_x_max, clamp_y_min, clamp_y_max))

        # ── Phase 2: Enforce minimum separation between active objects ──
        if n_obj > 1 and self.cfg.spawn_min_separation > 0:
            min_sep = self.cfg.spawn_min_separation
            for _iteration in range(2):  # 2 passes for cascading pushes
                for i in range(n_obj):
                    for j_idx in range(i + 1, n_obj):
                        both_active = all_active_mask[i] & all_active_mask[j_idx]
                        if not both_active.any():
                            continue
                        dx = all_pos[i][both_active, 0] - all_pos[j_idx][both_active, 0]
                        dy = all_pos[i][both_active, 1] - all_pos[j_idx][both_active, 1]
                        dist_xy = (dx * dx + dy * dy).sqrt()
                        too_close = dist_xy < min_sep
                        if not too_close.any():
                            continue
                        # Indices into the both_active subset that are too close
                        ba_indices = both_active.nonzero(as_tuple=True)[0]
                        tc_indices = ba_indices[too_close]

                        dx_tc = all_pos[i][tc_indices, 0] - all_pos[j_idx][tc_indices, 0]
                        dy_tc = all_pos[i][tc_indices, 1] - all_pos[j_idx][tc_indices, 1]
                        dist_tc = dist_xy[too_close].clamp(min=1e-6)
                        deficit = (min_sep - dist_tc) / 2.0
                        # Normalize direction and push each object half the deficit
                        nx = dx_tc / dist_tc
                        ny = dy_tc / dist_tc
                        all_pos[i][tc_indices, 0] += deficit * nx
                        all_pos[i][tc_indices, 1] += deficit * ny
                        all_pos[j_idx][tc_indices, 0] -= deficit * nx
                        all_pos[j_idx][tc_indices, 1] -= deficit * ny
                        all_canonical[i][tc_indices, 0] += deficit * nx
                        all_canonical[i][tc_indices, 1] += deficit * ny
                        all_canonical[j_idx][tc_indices, 0] -= deficit * nx
                        all_canonical[j_idx][tc_indices, 1] -= deficit * ny

                # Re-clamp after pushing (using per-slot rotation-aware bounds)
                for k in range(n_obj):
                    cxmin, cxmax, cymin, cymax = all_clamp_bounds[k]
                    all_pos[k][:, 0] = torch.max(torch.min(all_pos[k][:, 0], cxmax), cxmin)
                    all_pos[k][:, 1] = torch.max(torch.min(all_pos[k][:, 1], cymax), cymin)
                    all_canonical[k][:, 0] = torch.max(torch.min(all_canonical[k][:, 0], cxmax), cxmin)
                    all_canonical[k][:, 1] = torch.max(torch.min(all_canonical[k][:, 1], cymax), cymin)

        # ── Phase 3: Write final positions to sim ──
        for j in range(n_obj):
            pos = all_pos[j]
            canonical_pos = all_canonical[j]
            quat_wxyz = all_quat_wxyz[j]

            vel = torch.zeros(N, 6, device=self.device)

            # Convert to world frame
            pos = pos + self.scene.env_origins[env_ids]
            canonical_pos = canonical_pos + self.scene.env_origins[env_ids]

            # Store per-object init state
            quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
            self.object_init_state[env_ids, j, :3] = canonical_pos
            self.object_init_state[env_ids, j, 3:7] = quat_xyzw

            self.objects[j].write_root_pose_to_sim(
                torch.cat([pos, quat_wxyz], dim=-1), env_ids=env_ids
            )
            self.objects[j].write_root_velocity_to_sim(vel, env_ids=env_ids)

    def _reset_target_pose(self, env_ids: Tensor, is_first_goal: bool = True):
        """Sample a new goal pose for the given environments."""
        N = len(env_ids)
        if N == 0:
            return

        # All positions stored in world frame (matching root_pos_w convention)
        env_origins = self.scene.env_origins[env_ids]

        if not is_first_goal and self.cfg.goal_sampling_type == "delta":
            # Delta from previous goal (matching IsaacGym _sample_delta_goal)
            last_goal_pos = self.goal_states[env_ids, :3]  # previous goal, world frame
            last_goal_rot_xyzw = self.goal_states[env_ids, 3:7]

            # Per-axis uniform delta (matching IsaacGym's cube distribution)
            goal_pos = last_goal_pos + torch_rand_float(
                -self.cfg.delta_goal_distance,
                self.cfg.delta_goal_distance,
                (N, 3), self.device,
            )

            # Clamp to target volume (bounds are local, so convert to world and back)
            mins = torch.tensor(self.cfg.target_volume_mins, device=self.device)
            maxs = torch.tensor(self.cfg.target_volume_maxs, device=self.device)
            goal_pos_local = goal_pos - env_origins
            goal_pos_local = torch.max(torch.min(goal_pos_local, maxs), mins)
            goal_pos = goal_pos_local + env_origins

            # Random rotation delta from previous goal rotation
            goal_rot_xyzw = self._sample_delta_quat_xyzw(
                last_goal_rot_xyzw, self.cfg.delta_rotation_degrees
            )
        else:
            # Absolute sampling in target volume (local frame), then convert to world
            goal_pos = torch.zeros(N, 3, device=self.device)
            for d in range(3):
                goal_pos[:, d] = torch_rand_float(
                    self.cfg.target_volume_mins[d],
                    self.cfg.target_volume_maxs[d],
                    (N, 1), self.device,
                ).squeeze(-1)
            goal_pos += env_origins
            goal_rot_xyzw = self._random_quaternion_xyzw(N)

        self.goal_states[env_ids, :3] = goal_pos
        self.goal_states[env_ids, 3:7] = goal_rot_xyzw

        # Clip goal z so all goals require lifting above the bonus threshold
        # IsaacGym skips this for delta goals ("don't clip goal z for delta poses")
        if is_first_goal or self.cfg.goal_sampling_type != "delta":
            active_init = self.object_init_state[env_ids, self.active_object_idx[env_ids]]  # (N_ids, 13)
            min_z = active_init[:, 2:3] - 0.05 + self.cfg.lifting_bonus_threshold
            self.goal_states[env_ids, 2:3] = torch.max(
                min_z, self.goal_states[env_ids, 2:3]
            )

        # Shift goal by -R_goal@offset so it represents the mesh origin,
        # matching the convention used for object_pos after _reset_object_pose.
        # The offset is in the object's local frame and must be rotated by the goal orientation.
        # For HHP envs, offset is zero so this is a no-op.
        for i, eid in enumerate(env_ids):
            active_slot = self.active_object_idx[eid].item()
            offset_local = self._env_grasp_offsets[eid, active_slot]
            if offset_local.abs().sum() > 1e-6:  # Only apply if non-zero
                rotated_offset = self._quat_rotate_xyzw(
                    goal_rot_xyzw[i:i+1], offset_local.unsqueeze(0)
                ).squeeze(0)
                self.goal_states[eid, :3] -= rotated_offset

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
                obj_mass = (self._object_rb_masses[force_envs, self.active_object_idx[force_envs]].unsqueeze(-1)
                            if self._object_rb_masses is not None else 1.0)
                new_forces = torch.randn(len(force_envs), 3, device=self.device) * obj_mass * self.cfg.force_scale
                if self.cfg.force_only_when_lifted:
                    active_lifted = self.lifted_object[force_envs, self.active_object_idx[force_envs]]
                    new_forces *= active_lifted.float().unsqueeze(-1)
                # Route to active object per env
                for j in range(self.cfg.n_objects_max):
                    mask_j = self.active_object_idx[force_envs] == j
                    if mask_j.any():
                        j_envs = force_envs[mask_j]
                        j_forces = new_forces[mask_j]
                        self.objects[j].set_external_force_and_torque(
                            forces=j_forces.unsqueeze(1),
                            torques=torch.zeros_like(j_forces).unsqueeze(1),
                            env_ids=j_envs,
                        )

        if self.cfg.torque_scale > 0.0:
            torque_mask = (torch.rand(self.num_envs, device=self.device) < self.random_torque_prob)
            if torque_mask.any():
                torque_envs = torque_mask.nonzero(as_tuple=False).squeeze(-1)
                obj_mass = (self._object_rb_masses[torque_envs, self.active_object_idx[torque_envs]].unsqueeze(-1)
                            if self._object_rb_masses is not None else 1.0)
                new_torques = torch.randn(len(torque_envs), 3, device=self.device) * obj_mass * self.cfg.torque_scale
                if self.cfg.torque_only_when_lifted:
                    active_lifted = self.lifted_object[torque_envs, self.active_object_idx[torque_envs]]
                    new_torques *= active_lifted.float().unsqueeze(-1)
                # Route to active object per env
                for j in range(self.cfg.n_objects_max):
                    mask_j = self.active_object_idx[torque_envs] == j
                    if mask_j.any():
                        j_envs = torque_envs[mask_j]
                        j_torques = new_torques[mask_j]
                        self.objects[j].set_external_force_and_torque(
                            forces=torch.zeros_like(j_torques).unsqueeze(1),
                            torques=j_torques.unsqueeze(1),
                            env_ids=j_envs,
                        )

    # ══════════════════════════════════════════════════════════════════
    # Depth rendering
    # ══════════════════════════════════════════════════════════════════

    def _render_depth(self):
        """Render depth/rgb from TiledCamera(s) and fill buffers."""
        # Wrist camera
        if self.cfg.use_depth_camera and hasattr(self, '_tiled_camera'):
            if self._wrist_cam_type == "rgb":
                rgb = self._tiled_camera.data.output["rgb"]  # (N, H, W, 4) RGBA
                self.depth_buf = rgb[..., :3].permute(0, 3, 1, 2).float() / 255.0  # (N, 3, H, W) [0,1]
            else:
                depth = self._tiled_camera.data.output["depth"]  # (N, H, W, 1)
                self.depth_buf = depth.permute(0, 3, 1, 2)  # (N, 1, H, W)
                self.depth_buf.clamp_(0.0, self.cfg.depth_far)
                self.depth_buf /= self.cfg.depth_far  # normalize to [0, 1]

        # Third-person camera
        if self.cfg.use_third_person_camera and hasattr(self, '_tp_camera'):
            if self._tp_cam_type == "rgb":
                tp_rgb = self._tp_camera.data.output["rgb"]  # (N, H, W, 4) RGBA
                self.tp_depth_buf = tp_rgb[..., :3].permute(0, 3, 1, 2).float() / 255.0
            else:
                tp_depth = self._tp_camera.data.output["depth"]  # (N, H, W, 1)
                self.tp_depth_buf = tp_depth.permute(0, 3, 1, 2)  # (N, 1, H, W)
                self.tp_depth_buf.clamp_(0.0, self.cfg.third_person_depth_far)
                self.tp_depth_buf /= self.cfg.third_person_depth_far

    # ══════════════════════════════════════════════════════════════════
    # Video capture (mirrors IsaacGym _capture_video_if_needed)
    # ══════════════════════════════════════════════════════════════════

    def _init_video_camera(self):
        """Create an Omni Replicator camera on env 0 for video capture."""
        try:
            import omni.replicator.core as rep
            from pxr import UsdGeom, Gf
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            cam_path = "/World/envs/env_0/VideoCam"
            cam_prim = stage.DefinePrim(cam_path, "Camera")
            UsdGeom.Xformable(cam_prim).ClearXformOpOrder()
            xform = UsdGeom.Xformable(cam_prim)
            # Camera position: side view looking at robot + table from front
            xform.AddTranslateOp().Set(Gf.Vec3d(0.8, -0.8, 1.0))
            xform.AddRotateXYZOp().Set(Gf.Vec3d(35, 0, -40))
            cam = UsdGeom.Camera(cam_prim)
            cam.GetFocalLengthAttr().Set(18.0)

            rp = rep.create.render_product(cam_path, (320, 240))
            self._video_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._video_annotator.attach([rp])
            print("[Video] Camera sensor created on env 0 for video capture")
        except Exception as e:
            print(f"[Video] Failed to create video camera: {e}")
            self._video_annotator = None

    def _init_birds_eye_camera(self, env_idx=0):
        """Create Omni Replicator RGB camera looking straight down at the workspace."""
        import omni.replicator.core as rep
        from pxr import UsdGeom, Gf
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        cam_path = f"/World/envs/env_{env_idx}/BirdsEyeRGBCam"
        cam_prim = stage.DefinePrim(cam_path, "Camera")
        UsdGeom.Xformable(cam_prim).ClearXformOpOrder()
        xform = UsdGeom.Xformable(cam_prim)
        # Same pose as eval_rgb_camera: pos=(0, -0.2, 0.95), 35° pitch, focal=9
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.2, 0.95))
        xform.AddRotateXYZOp().Set(Gf.Vec3d(35.0, 0.0, 0.0))
        cam = UsdGeom.Camera(cam_prim)
        cam.GetFocalLengthAttr().Set(9.0)
        cam.GetHorizontalApertureAttr().Set(20.955)  # match PinholeCameraCfg default
        cam.GetVerticalApertureAttr().Set(20.955 * 240.0 / 320.0)  # correct for 4:3 aspect
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 4.0))  # match eval_rgb_camera

        rp = rep.create.render_product(cam_path, (320, 240))
        self._birds_eye_rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._birds_eye_rgb_annotator.attach([rp])
        import carb
        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", True)
        print(f"[BirdsEyeCam] Birds-eye RGB camera created on env_{env_idx}", flush=True)

    def _create_eval_side_view_prims(self):
        """Create USD camera prims for eval side-view on ALL envs.

        Uses the exact same transform as _init_side_view_camera so TiledCamera
        can batch-read from them (spawn=None in TiledCameraCfg).
        """
        from pxr import UsdGeom, Gf
        import omni.usd
        from isaaclab.sim.utils import standardize_xform_ops

        stage = omni.usd.get_context().get_stage()
        for ei in range(self.cfg.scene.num_envs):
            cam_path = f"/World/envs/env_{ei}/EvalSideViewCamera"
            cam_prim = stage.DefinePrim(cam_path, "Camera")
            UsdGeom.Xformable(cam_prim).ClearXformOpOrder()
            xform = UsdGeom.Xformable(cam_prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(0.45, 0.0, 0.7))
            xform.AddRotateXYZOp().Set(Gf.Vec3d(90.0, 0.0, 90.0))
            cam = UsdGeom.Camera(cam_prim)
            cam.GetFocalLengthAttr().Set(12.0)
            cam.GetHorizontalApertureAttr().Set(20.955)
            cam.GetVerticalApertureAttr().Set(20.955)
            cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 4.0))
            # Convert to standard [translate, orient, scale] ops for TiledCamera
            standardize_xform_ops(cam_prim)
        print(f"[EvalSideView] Created {self.cfg.scene.num_envs} side-view camera prims")

    def _init_side_view_camera(self, env_idx=0):
        """Create Omni Replicator RGB camera viewing from the side."""
        import omni.replicator.core as rep
        from pxr import UsdGeom, Gf
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        cam_path = f"/World/envs/env_{env_idx}/SideViewRGBCam"
        cam_prim = stage.DefinePrim(cam_path, "Camera")
        UsdGeom.Xformable(cam_prim).ClearXformOpOrder()
        xform = UsdGeom.Xformable(cam_prim)
        # Position close to the table side, looking in -X at the workspace
        xform.AddTranslateOp().Set(Gf.Vec3d(0.45, 0.0, 0.7))
        xform.AddRotateXYZOp().Set(Gf.Vec3d(90.0, 0.0, 90.0))
        cam = UsdGeom.Camera(cam_prim)
        cam.GetFocalLengthAttr().Set(12.0)
        cam.GetHorizontalApertureAttr().Set(20.955)
        cam.GetVerticalApertureAttr().Set(20.955)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 4.0))

        rp = rep.create.render_product(cam_path, (512, 512))
        self._side_view_rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._side_view_rgb_annotator.attach([rp])
        import carb
        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", True)
        print(f"[SideViewCam] Side-view RGB camera created on env_{env_idx}", flush=True)

    def _compute_prim_aabb(self, prim_path: str):
        """Compute the AABB of all collision geometry under a USD prim.

        Returns (bmin, bmax) as numpy arrays of shape (3,) in the prim's
        local coordinate frame, or None if no collision geometry is found.
        """
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        import omni.usd
        import numpy as np

        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            print(f"[CollisionBBox] Prim not found at {prim_path}")
            return None

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        root_world_inv = xform_cache.GetLocalToWorldTransform(root_prim).GetInverse()

        def _local_extent(prim):
            """Return (lo, hi) corners of a prim's local AABB (before xform ops)."""
            t = prim.GetTypeName()
            if t == "Cube":
                s = prim.GetAttribute("size").Get() or 1.0
                h = s / 2.0
                return [(-h, -h, -h), (h, h, h)]
            elif t == "Sphere":
                r = prim.GetAttribute("radius").Get() or 1.0
                return [(-r, -r, -r), (r, r, r)]
            elif t == "Cylinder":
                r = prim.GetAttribute("radius").Get() or 1.0
                ht = (prim.GetAttribute("height").Get() or 2.0) / 2.0
                axis = prim.GetAttribute("axis").Get() or "Z"
                if axis == "X":
                    return [(-ht, -r, -r), (ht, r, r)]
                elif axis == "Y":
                    return [(-r, -ht, -r), (r, ht, r)]
                else:
                    return [(-r, -r, -ht), (r, r, ht)]
            elif prim.IsA(UsdGeom.Mesh):
                pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if pts:
                    arr = np.array(pts)
                    return [tuple(arr.min(0)), tuple(arr.max(0))]
            return None

        all_pts = []
        n_coll_prims = 0
        # Use TraverseInstanceProxies to see through instanceable Xforms
        # (e.g. the table's collisions Xform is marked instanceable=true).
        predicate = Usd.TraverseInstanceProxies()
        for desc in Usd.PrimRange(root_prim, predicate):
            if not desc.HasAPI(UsdPhysics.CollisionAPI):
                continue
            # Skip prims with collision explicitly disabled
            col_enabled = desc.GetAttribute("physics:collisionEnabled")
            if col_enabled.IsValid() and col_enabled.Get() is False:
                continue
            shape_prims = []
            ext = _local_extent(desc)
            if ext is not None:
                shape_prims.append((desc, ext))
            for child in Usd.PrimRange(desc, predicate):
                if child == desc:
                    continue
                ext = _local_extent(child)
                if ext is not None:
                    shape_prims.append((child, ext))

            for shape_prim, (lo, hi) in shape_prims:
                n_coll_prims += 1
                shape_world = xform_cache.GetLocalToWorldTransform(shape_prim)
                for x in [lo[0], hi[0]]:
                    for y in [lo[1], hi[1]]:
                        for z in [lo[2], hi[2]]:
                            pw = shape_world.Transform(Gf.Vec3d(float(x), float(y), float(z)))
                            pl = root_world_inv.Transform(pw)
                            all_pts.append([pl[0], pl[1], pl[2]])

        if not all_pts:
            print(f"[CollisionBBox] No collision geometry found under {prim_path}")
            return None

        pts = np.array(all_pts)
        bmin = pts.min(axis=0)
        bmax = pts.max(axis=0)
        print(f"[CollisionBBox] {prim_path}: {n_coll_prims} collision shapes, "
              f"AABB min={np.round(bmin, 4).tolist()} max={np.round(bmax, 4).tolist()} "
              f"size={np.round(bmax - bmin, 4).tolist()}")
        return bmin, bmax

    def _draw_wireframe_aabb(self, parent_prim_path, bmin, bmax, color, viz_name="CollisionBBoxViz", radius=0.003):
        """Draw 12 thin cylinder edges as a wireframe box under the given prim.

        Args:
            parent_prim_path: USD path to parent the wireframe under.
            bmin: AABB minimum corner in parent's local frame, shape (3,).
            bmax: AABB maximum corner in parent's local frame, shape (3,).
            color: RGB color tuple for the emissive material.
            viz_name: Name of the Xform child prim that holds all edge cylinders.
            radius: Cylinder radius for the wireframe edges.
        """
        from pxr import UsdGeom, UsdShade, Gf, Sdf
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        viz_path = f"{parent_prim_path}/{viz_name}"
        stage.DefinePrim(viz_path, "Xform")

        # Emissive material
        mat_path = f"{viz_path}/Mat"
        stage.DefinePrim(mat_path, "Material")
        shader_prim = stage.DefinePrim(f"{mat_path}/Shader", "Shader")
        shader = UsdShade.Shader(shader_prim)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color))
        mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        edge_id = 0

        def _add_edge(axis, cx, cy, cz, length):
            nonlocal edge_id
            cp = stage.DefinePrim(f"{viz_path}/e{edge_id}", "Cylinder")
            cyl = UsdGeom.Cylinder(cp)
            cyl.GetRadiusAttr().Set(radius)
            cyl.GetHeightAttr().Set(float(length))
            cyl.GetAxisAttr().Set(axis)
            UsdGeom.Xformable(cp).AddTranslateOp().Set(
                Gf.Vec3d(float(cx), float(cy), float(cz)))
            UsdShade.MaterialBindingAPI.Apply(cp)
            UsdShade.MaterialBindingAPI(cp).Bind(mat)
            edge_id += 1

        mx_c = (bmin + bmax) / 2
        for y in [bmin[1], bmax[1]]:
            for z in [bmin[2], bmax[2]]:
                _add_edge("X", mx_c[0], y, z, bmax[0] - bmin[0])
        for x in [bmin[0], bmax[0]]:
            for z in [bmin[2], bmax[2]]:
                _add_edge("Y", x, mx_c[1], z, bmax[1] - bmin[1])
        for x in [bmin[0], bmax[0]]:
            for y in [bmin[1], bmax[1]]:
                _add_edge("Z", x, y, mx_c[2], bmax[2] - bmin[2])

        print(f"[CollisionBBox] Created {edge_id} wireframe edges at {viz_path}")

    @staticmethod
    def _aabb_corners(bmin, bmax):
        """Generate all 8 corners of an AABB from (bmin, bmax)."""
        import numpy as np
        return np.array([
            [bmin[0], bmin[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]],
            [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmax[1], bmax[2]],
            [bmax[0], bmin[1], bmin[2]],
            [bmax[0], bmin[1], bmax[2]],
            [bmax[0], bmax[1], bmin[2]],
            [bmax[0], bmax[1], bmax[2]],
        ])

    def _log_aabb_corners_step(self):
        """Log world-space AABB corners for the current timestep.

        Called each step from _get_observations. Transforms local-frame AABBs
        to world space using current object/table poses. Flushes to
        /tmp/collision_aabb_corners.npy every 500 steps.
        """
        import numpy as np

        if not hasattr(self, '_bbox_corner_log') or not self.cfg.draw_collision_bbox:
            return

        ei = self._bbox_env_idx

        # Object: transform local AABB corners to world space using active slot's geometry
        obj_world = None
        active_slot = self.active_object_idx[ei].item()
        obj_aabb_local = self._bbox_obj_aabb_per_slot.get(active_slot) if hasattr(self, '_bbox_obj_aabb_per_slot') else None
        if obj_aabb_local is not None:
            local_corners = self._aabb_corners(*obj_aabb_local)  # (8, 3)
            obj_pos = self.objects[active_slot].data.root_pos_w[ei].cpu().numpy()
            obj_quat = self.objects[active_slot].data.root_quat_w[ei].cpu().numpy()  # wxyz
            w, qx, qy, qz = [float(v) for v in obj_quat]
            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - w*qz),   2*(qx*qz + w*qy)],
                [2*(qx*qy + w*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - w*qx)],
                [2*(qx*qz - w*qy),     2*(qy*qz + w*qx),     1 - 2*(qx*qx + qy*qy)],
            ])
            obj_world = (local_corners @ R.T) + obj_pos[None, :]

        # Table: transform local AABB corners to world space
        table_world = None
        if self._bbox_table_aabb_local is not None:
            local_corners = self._aabb_corners(*self._bbox_table_aabb_local)  # (8, 3)
            table_pos = self.table.data.root_pos_w[ei].cpu().numpy()
            table_quat = self.table.data.root_quat_w[ei].cpu().numpy()
            w, qx, qy, qz = [float(v) for v in table_quat]
            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - w*qz),   2*(qx*qz + w*qy)],
                [2*(qx*qy + w*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - w*qx)],
                [2*(qx*qz - w*qy),     2*(qy*qz + w*qx),     1 - 2*(qx*qx + qy*qy)],
            ])
            table_world = (local_corners @ R.T) + table_pos[None, :]

        self._bbox_corner_log.append({"object_corners": obj_world,
                                       "table_corners": table_world})

        # Flush to disk every 500 steps
        if len(self._bbox_corner_log) % 50 == 0:
            self._flush_aabb_corners()

    def _flush_aabb_corners(self):
        """Write accumulated AABB corner log to /tmp/collision_aabb_corners.npy."""
        import numpy as np

        if not hasattr(self, '_bbox_corner_log') or not self._bbox_corner_log:
            return

        save_data = {}
        if self._bbox_obj_aabb_local is not None:
            save_data["object_corners"] = np.stack(
                [s["object_corners"] for s in self._bbox_corner_log])  # (T, 8, 3)
        if self._bbox_table_aabb_local is not None:
            save_data["table_corners"] = np.stack(
                [s["table_corners"] for s in self._bbox_corner_log])  # (T, 8, 3)

        save_path = "/tmp/collision_aabb_corners.npy"
        np.save(save_path, save_data)
        n = len(self._bbox_corner_log)
        print(f"[CollisionBBox] Flushed {n}-step AABB corners to {save_path}")

    def _init_goal_keypoint_markers(self):
        """Create VisualizationMarkers for goal keypoints and edges (bright pink)."""
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

        pink = (1.0, 0.0, 0.6)

        # Sphere markers at each keypoint
        sphere_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/GoalKeypoints",
            markers={
                "goal_kp": sim_utils.SphereCfg(
                    radius=0.008,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=pink,
                        emissive_color=pink,
                    ),
                ),
            },
        )
        self._goal_kp_markers = VisualizationMarkers(sphere_cfg)

        # Cylinder markers for edges between keypoints (all 6 edges of K4)
        self._goal_kp_edge_pairs = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        ]
        edge_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/GoalEdges",
            markers={
                "goal_edge": sim_utils.CylinderCfg(
                    radius=0.003,
                    height=1.0,
                    axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=pink,
                        emissive_color=pink,
                    ),
                ),
            },
        )
        self._goal_edge_markers = VisualizationMarkers(edge_cfg)

        N = self.num_envs
        K = self.cfg.num_keypoints
        E = len(self._goal_kp_edge_pairs)
        print(f"[GoalKPMarkers] Created {N*K} keypoint + {N*E} edge markers")

    def _update_goal_keypoint_markers(self):
        """Update goal keypoint sphere + edge cylinder positions each step."""
        # -- Sphere markers at keypoint positions --
        # goal_keypoint_pos: (N, K, 3) world frame
        positions = self.goal_keypoint_pos.reshape(-1, 3)
        self._goal_kp_markers.visualize(translations=positions)

        # -- Cylinder edge markers --
        kp = self.goal_keypoint_pos  # (N, 4, 3)
        N = kp.shape[0]
        E = len(self._goal_kp_edge_pairs)

        edge_pos = torch.zeros(N * E, 3, device=self.device)
        edge_orient = torch.zeros(N * E, 4, device=self.device)  # wxyz
        edge_scale = torch.zeros(N * E, 3, device=self.device)

        for e_idx, (ki, kj) in enumerate(self._goal_kp_edge_pairs):
            p0 = kp[:, ki]  # (N, 3)
            p1 = kp[:, kj]  # (N, 3)
            mid = (p0 + p1) * 0.5
            diff = p1 - p0
            length = torch.norm(diff, dim=-1, keepdim=True)  # (N, 1)
            d = diff / (length + 1e-8)  # (N, 3) unit direction

            # Quaternion to rotate Z axis to d: q = [cos(a/2), sin(a/2) * axis]
            # cross(Z, d) and dot(Z, d)
            z = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            cross = torch.cross(z.expand(N, -1), d, dim=-1)  # (N, 3)
            dot = d[:, 2:3]  # (N, 1) = dot(z, d)
            cross_norm = torch.norm(cross, dim=-1, keepdim=True)  # (N, 1)

            # General case: half-angle formula
            half_angle = torch.atan2(cross_norm, dot) * 0.5  # (N, 1)
            axis = cross / (cross_norm + 1e-8)  # (N, 3)
            sin_ha = torch.sin(half_angle)
            cos_ha = torch.cos(half_angle)
            quat = torch.cat([cos_ha, axis * sin_ha], dim=-1)  # (N, 4) wxyz

            # Handle near-parallel cases (dot ~ 1 or dot ~ -1)
            parallel = (cross_norm.squeeze(-1) < 1e-6)
            if parallel.any():
                # Same direction: identity quat
                quat[parallel & (dot.squeeze(-1) > 0)] = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=self.device)
                # Opposite direction: 180 deg around X
                quat[parallel & (dot.squeeze(-1) <= 0)] = torch.tensor(
                    [0.0, 1.0, 0.0, 0.0], device=self.device)

            sl = slice(e_idx * N, (e_idx + 1) * N)
            edge_pos[sl] = mid
            edge_orient[sl] = quat
            edge_scale[sl, 0] = 1.0
            edge_scale[sl, 1] = 1.0
            edge_scale[sl, 2] = length.squeeze(-1)

        self._goal_edge_markers.visualize(
            translations=edge_pos, orientations=edge_orient, scales=edge_scale,
        )

    def _compute_table_spawn_bounds(self):
        """Dynamically measure table AABB and store raw half-extents.

        The actual spawn bounds used during reset are computed per-object
        by subtracting each object's rotation-aware AABB half-extent from
        the table half-extents.
        """
        import numpy as np

        table_path = "/World/envs/env_0/Table"
        table_aabb = self._compute_prim_aabb(table_path)

        if table_aabb is not None:
            bmin, bmax = table_aabb
            self._table_half_x = float((bmax[0] - bmin[0]) / 2.0)
            self._table_half_y = float((bmax[1] - bmin[1]) / 2.0)
            self._table_top_z_local = float(bmax[2])
        else:
            print("[SpawnBounds] WARNING: Could not measure table AABB, using conservative defaults")
            self._table_half_x = 0.10
            self._table_half_y = 0.07
            self._table_top_z_local = 0.045

        # Fixed bounds (used for spawn zone wireframe and as fallback)
        margin = self.cfg.spawn_object_margin
        self._spawn_x_min = float(-self._table_half_x + margin)
        self._spawn_x_max = float(self._table_half_x - margin)
        self._spawn_y_min = float(-self._table_half_y + margin)
        self._spawn_y_max = float(self._table_half_y - margin)

        print(f"[SpawnBounds] Table half: x={self._table_half_x:.4f} y={self._table_half_y:.4f} | "
              f"Fixed spawn bounds: x=[{self._spawn_x_min:.4f}, {self._spawn_x_max:.4f}] "
              f"y=[{self._spawn_y_min:.4f}, {self._spawn_y_max:.4f}] "
              f"(static margin={margin}, per-object rotation-aware margins applied at reset)")

    def _draw_spawn_zone_wireframe(self, env_idx=0):
        """Draw a purple wireframe showing the valid object spawn zone on the table."""
        import numpy as np

        table_path = f"/World/envs/env_{env_idx}/Table"
        z_top = self._table_top_z_local
        bmin = np.array([self._spawn_x_min, self._spawn_y_min, z_top])
        bmax = np.array([self._spawn_x_max, self._spawn_y_max, z_top + 0.01])
        self._draw_wireframe_aabb(table_path, bmin, bmax,
                                  color=(0.6, 0.0, 0.8), viz_name="SpawnZoneViz",
                                  radius=0.002)
        print(f"[SpawnZone] Drew purple spawn zone wireframe on env_{env_idx}")

    def _init_handle_wireframes(self, env_idx=0):
        """Create green wireframe boxes around the handle/grasp region for each object slot.

        Only the active object's wireframe is shown (visibility toggled per step).
        Wireframes are parented to Object prims so they track movement/rotation.
        """
        import numpy as np
        from pxr import UsdGeom
        import omni.usd

        self._handle_wf_env_idx = env_idx
        self._handle_wf_paths = []  # USD paths of wireframe Xforms per slot

        for j in range(self.cfg.n_objects_max):
            obj_path = f"/World/envs/env_{env_idx}/Object_{j}"

            # Compute handle AABB in object-local frame based on env's asset mode
            env_mode = self._env_asset_mode_list[env_idx]
            if env_mode == 0 and hasattr(self, '_hhp_handle_scales'):
                # HHP mode: use handle scales
                num_objects = len(self._hhp_handle_scales)
                obj_idx = self._get_hhp_obj_idx_for_env_and_slot(env_idx, j, num_objects)
                hs = self._hhp_handle_scales[obj_idx]
                if len(hs) == 3:
                    hx, hy, hz = hs
                else:
                    # Cylinder: [height, diameter]
                    hx, hy, hz = hs[0], hs[1], hs[1]
                bmin = np.array([-hx / 2, -hy / 2, -hz / 2])
                bmax = np.array([hx / 2, hy / 2, hz / 2])
            elif env_mode == 1 and hasattr(self, '_ra_scales'):
                # RA mode: use grasp bbox scales (half-extents)
                num_objects = len(self._ra_scales)
                obj_idx = self._get_ra_obj_idx_for_env_and_slot(env_idx, j, num_objects)
                raw = self._ra_scales[obj_idx]
                offset = self._ra_offsets[obj_idx]
                # raw is half-extents, so bbox is offset +/- raw
                bmin = np.array([offset[0] - raw[0], offset[1] - raw[1], offset[2] - raw[2]])
                bmax = np.array([offset[0] + raw[0], offset[1] + raw[1], offset[2] + raw[2]])
            else:
                # Fallback: use _compute_prim_aabb for unknown modes
                aabb = self._compute_prim_aabb(obj_path)
                if aabb is not None:
                    bmin, bmax = aabb
                else:
                    self._handle_wf_paths.append(None)
                    continue

            viz_name = "HandleWireframe"
            self._draw_wireframe_aabb(obj_path, bmin, bmax,
                                      color=(0.0, 1.0, 0.0), viz_name=viz_name,
                                      radius=0.001)
            wf_path = f"{obj_path}/{viz_name}"
            self._handle_wf_paths.append(wf_path)

            # Start all hidden; the per-step update will show the active one
            stage = omni.usd.get_context().get_stage()
            wf_prim = stage.GetPrimAtPath(wf_path)
            if wf_prim.IsValid():
                UsdGeom.Imageable(wf_prim).MakeInvisible()

        print(f"[HandleWireframe] Created handle wireframes on {len(self._handle_wf_paths)} "
              f"object slots for env_{env_idx}")

    def _update_handle_wireframe_visibility(self):
        """Show handle wireframe only on the active object slot."""
        from pxr import UsdGeom
        import omni.usd

        ei = self._handle_wf_env_idx
        active_slot = self.active_object_idx[ei].item()
        stage = omni.usd.get_context().get_stage()

        for j, wf_path in enumerate(self._handle_wf_paths):
            if wf_path is None:
                continue
            prim = stage.GetPrimAtPath(wf_path)
            if not prim.IsValid():
                continue
            img = UsdGeom.Imageable(prim)
            if j == active_slot:
                img.MakeVisible()
            else:
                img.MakeInvisible()

    def _draw_collision_bbox(self, env_idx=0):
        """Draw collision AABB wireframes: red for object, blue for table.

        Both are parented under their respective prims so they track movement.
        Stores local-frame AABBs for per-timestep world-space corner logging.
        """
        # Compute and cache local AABBs for all object slots
        self._bbox_obj_aabb_per_slot = {}
        for j in range(self.cfg.n_objects_max):
            slot_path = f"/World/envs/env_{env_idx}/Object_{j}"
            slot_aabb = self._compute_prim_aabb(slot_path)
            if slot_aabb is not None:
                self._bbox_obj_aabb_per_slot[j] = slot_aabb

        # Draw wireframe on current active slot
        active_slot = self.active_object_idx[env_idx].item()
        obj_aabb = self._bbox_obj_aabb_per_slot.get(active_slot)
        if obj_aabb is not None:
            obj_path = f"/World/envs/env_{env_idx}/Object_{active_slot}"
            self._draw_wireframe_aabb(obj_path, *obj_aabb, color=(1.0, 0.0, 0.0),
                                      viz_name="CollisionBBoxViz")

        table_path = f"/World/envs/env_{env_idx}/Table"
        table_aabb = self._compute_prim_aabb(table_path)
        if table_aabb is not None:
            self._draw_wireframe_aabb(table_path, *table_aabb, color=(0.0, 0.0, 1.0),
                                      viz_name="TableBBoxViz")

        # Store for per-timestep world-space logging
        self._bbox_table_aabb_local = table_aabb
        self._bbox_env_idx = env_idx
        self._bbox_active_slot = active_slot
        self._bbox_corner_log = []  # list of per-step dicts, flushed every 500 steps

    def _draw_grasp_bbox(self, env_idx=0):
        """Draw a green wireframe box showing the grasp bounding region.

        For RA (real_assets) envs: Uses _ra_offsets (center), _ra_scales
        (half-extents), and _ra_rotations (orientation) in the Object's
        local frame.

        For HHP (handle_head_primitives) envs: Uses _hhp_handle_scales
        with zero offset (grasp region centered at mesh origin).

        Args:
            env_idx: single int or list of env indices to draw on.
        """
        from pxr import UsdGeom, Gf
        import omni.usd
        import numpy as np

        env_ids = env_idx if isinstance(env_idx, (list, tuple)) else [env_idx]
        stage = omni.usd.get_context().get_stage()

        num_ra = len(self._ra_scales) if hasattr(self, '_ra_scales') else 0
        num_hhp = len(self._hhp_handle_scales) if hasattr(self, '_hhp_handle_scales') else 0

        for eidx in env_ids:
            active_slot = self.active_object_idx[eidx].item()
            obj_path = f"/World/envs/env_{eidx}/Object_{active_slot}"

            # Determine env's asset mode
            env_mode = self._env_asset_mode_list[eidx]

            if env_mode == 1 and num_ra > 0:
                # RA mode: use grasp bbox with offset and rotation
                obj_idx = self._get_ra_obj_idx_for_env_and_slot(eidx, active_slot, num_ra)
                half = self._ra_scales[obj_idx]       # (hx, hy, hz) half-extents
                offset = self._ra_offsets[obj_idx]     # (ox, oy, oz) center offset
                rot_xyzw = self._ra_rotations[obj_idx] # (qx, qy, qz, qw) quaternion

                # Create a positioned+rotated Xform so the wireframe is drawn
                # axis-aligned in the grasp box's own frame.
                grasp_xf_path = f"{obj_path}/GraspBBoxXform"
                stage.DefinePrim(grasp_xf_path, "Xform")
                xf = UsdGeom.Xformable(stage.GetPrimAtPath(grasp_xf_path))
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(*offset))
                qx, qy, qz, qw = rot_xyzw
                xf.AddOrientOp().Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))

                bmin = np.array([-half[0], -half[1], -half[2]])
                bmax = np.array([half[0], half[1], half[2]])
                self._draw_wireframe_aabb(grasp_xf_path, bmin, bmax,
                                          color=(0.0, 1.0, 0.0), viz_name="GraspBBoxViz")
                print(f"[GraspBBox] RA env_{eidx}: half={half}, offset={offset}")

            elif env_mode == 0 and num_hhp > 0:
                # HHP mode: grasp region centered at origin, no rotation
                obj_idx = self._get_hhp_obj_idx_for_env_and_slot(eidx, active_slot, num_hhp)
                hs = self._hhp_handle_scales[obj_idx]
                if len(hs) == 3:
                    hx, hy, hz = hs
                else:
                    # Cylinder: [height, diameter]
                    hx, hy, hz = hs[0], hs[1], hs[1]

                # Create Xform at origin (no offset for HHP)
                grasp_xf_path = f"{obj_path}/GraspBBoxXform"
                stage.DefinePrim(grasp_xf_path, "Xform")
                xf = UsdGeom.Xformable(stage.GetPrimAtPath(grasp_xf_path))
                xf.ClearXformOpOrder()

                bmin = np.array([-hx / 2, -hy / 2, -hz / 2])
                bmax = np.array([hx / 2, hy / 2, hz / 2])
                self._draw_wireframe_aabb(grasp_xf_path, bmin, bmax,
                                          color=(0.0, 1.0, 0.0), viz_name="GraspBBoxViz")
                print(f"[GraspBBox] HHP env_{eidx}: half=({hx/2:.3f}, {hy/2:.3f}, {hz/2:.3f})")

            else:
                print(f"[GraspBBox] Skipping env_{eidx}: no asset data for mode={env_mode}")

    def _init_tp_rgb_camera(self, env_idx=0):
        """Create Omni Replicator RGB camera on the given env at the same pose as the TP depth camera."""
        import omni.replicator.core as rep
        from pxr import UsdGeom, Gf
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        cam_path = f"/World/envs/env_{env_idx}/ThirdPersonRGBCam"
        cam_prim = stage.DefinePrim(cam_path, "Camera")
        UsdGeom.Xformable(cam_prim).ClearXformOpOrder()
        xform = UsdGeom.Xformable(cam_prim)
        # Match third_person_camera cfg exactly:
        #   pos=(0.0, -0.2, 0.95), rot=(0.9537, 0.3007, 0, 0) wxyz → 35° pitch
        #   focal_length=9.0, clipping=(0.1, 4.0)
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.2, 0.95))
        xform.AddRotateXYZOp().Set(Gf.Vec3d(35.0, 0.0, 0.0))
        cam = UsdGeom.Camera(cam_prim)
        cam.GetFocalLengthAttr().Set(9.0)
        cam.GetHorizontalApertureAttr().Set(20.955)
        cam.GetVerticalApertureAttr().Set(20.955 * 240.0 / 320.0)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 4.0))

        rp = rep.create.render_product(cam_path, (320, 240))
        self._tp_rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._tp_rgb_annotator.attach([rp])
        # Signal Isaac Lab that RTX sensors exist so sim.render() is called
        # during env stepping (required for Replicator render products to update)
        import carb
        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", True)
        print(f"[TPRGBCam] Third-person RGB camera created on env_{env_idx}, "
              f"rtx_sensors={carb.settings.get_settings().get_as_bool('/isaaclab/render/rtx_sensors')}", flush=True)

    def reinit_tp_rgb_camera(self, env_idx):
        """Recreate the TP RGB camera on a different environment (e.g. first student env)."""
        if self._tp_rgb_annotator is not None:
            self._tp_rgb_annotator.detach()
            self._tp_rgb_annotator = None
        self._init_tp_rgb_camera(env_idx=env_idx)

    def reinit_birds_eye_camera(self, env_idx):
        """Recreate the birds-eye camera on a different environment."""
        if self._birds_eye_rgb_annotator is not None:
            self._birds_eye_rgb_annotator.detach()
            self._birds_eye_rgb_annotator = None
        self._init_birds_eye_camera(env_idx=env_idx)

    def reinit_side_view_camera(self, env_idx):
        """Recreate the side-view camera on a different environment."""
        if self._side_view_rgb_annotator is not None:
            self._side_view_rgb_annotator.detach()
            self._side_view_rgb_annotator = None
        self._init_side_view_camera(env_idx=env_idx)

    def _capture_video_if_needed(self):
        """Check if we should start/continue/finish recording a video clip."""
        if not self.cfg.capture_video or self._video_annotator is None:
            return

        freq = self.cfg.capture_video_freq
        vlen = self.cfg.capture_video_len

        # Should we start a new recording?
        if self._video_frames is None and self._control_steps % freq == 0:
            self._video_frames = []  # signal: start at next reset of env 0

        # Waiting for env 0 to reset before starting
        if self._video_frames is not None and len(self._video_frames) == 0:
            if self.reset_buf[0].item() == 1 or self._control_steps % freq == 0:
                # Start now
                pass
            else:
                return

        # Record a frame
        if self._video_frames is not None:
            try:
                data = self._video_annotator.get_data()
                if data is not None:
                    import numpy as _np
                    rgb = _np.array(data)
                    if rgb.ndim == 3 and rgb.shape[-1] == 4:
                        rgb = rgb[..., :3]
                    self._video_frames.append(rgb)
            except Exception:
                pass

            # Done recording?
            if len(self._video_frames) >= vlen:
                self._save_video()

    def _save_video(self):
        """Save accumulated frames as mp4 and log to wandb."""
        if not self._video_frames:
            self._video_frames = None
            return
        try:
            import imageio
            from pathlib import Path

            videos_dir = Path("videos")
            videos_dir.mkdir(parents=True, exist_ok=True)
            video_path = videos_dir / f"video_step{self._control_steps}.mp4"
            fps = int(1.0 / (self.cfg.sim.dt * self.cfg.decimation))
            imageio.mimsave(str(video_path), self._video_frames, fps=fps)

            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"video": wandb.Video(str(video_path), fps=fps)})
                    print(f"[Video] Logged {len(self._video_frames)} frames to wandb "
                          f"(step={self._control_steps})")
            except Exception:
                pass
        except Exception as e:
            print(f"[Video] Failed to save video: {e}")
        self._video_frames = None

    # ══════════════════════════════════════════════════════════════════
    # Eval RGB video recording (all envs via TiledCamera)
    # ══════════════════════════════════════════════════════════════════

    def start_eval_rgb_recording(self, env_ids: list[int] | None = None):
        """Begin recording RGB frames for the given env IDs (default: all)."""
        if not self.cfg.use_eval_rgb_cameras or not hasattr(self, '_eval_rgb_camera'):
            print("[EvalRGB] WARNING: eval RGB cameras not enabled, cannot record")
            return
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        self._eval_rgb_frames = {i: [] for i in env_ids}
        self._eval_rgb_recording = True
        print(f"[EvalRGB] Started recording for {len(env_ids)} envs")

    def capture_eval_rgb_frame(self):
        """Capture one RGB frame from all recording envs. Call once per step."""
        if not self._eval_rgb_recording or not hasattr(self, '_eval_rgb_camera'):
            return
        # TiledCamera rgb output: (num_envs, H, W, 3) uint8
        rgb = self._eval_rgb_camera.data.output["rgb"]  # (N, H, W, 3)
        rgb_cpu = rgb.cpu().numpy()
        for env_idx in self._eval_rgb_frames:
            self._eval_rgb_frames[env_idx].append(rgb_cpu[env_idx].copy())

    def stop_eval_rgb_recording(self, save_dir: str = "videos", filename_prefix: str = ""):
        """Stop recording and save per-env MP4 videos."""
        if not self._eval_rgb_recording:
            return {}
        self._eval_rgb_recording = False

        import imageio
        from pathlib import Path
        from datetime import datetime

        videos_dir = Path(save_dir)
        videos_dir.mkdir(parents=True, exist_ok=True)
        fps = int(1.0 / (self.cfg.sim.dt * self.cfg.decimation))
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        saved_paths = {}
        for env_idx, frames in self._eval_rgb_frames.items():
            if len(frames) == 0:
                continue
            fname = f"{filename_prefix}{timestamp}_env{env_idx}.mp4"
            video_path = videos_dir / fname
            imageio.mimsave(str(video_path), frames, fps=fps)
            saved_paths[env_idx] = str(video_path)

        print(f"[EvalRGB] Saved {len(saved_paths)} videos to {save_dir}/ "
              f"({len(frames) if saved_paths else 0} frames each, {fps} fps)")
        self._eval_rgb_frames = {}
        return saved_paths

    # ══════════════════════════════════════════════════════════════════
    # Eval side-view video recording (all envs via TiledCamera)
    # ══════════════════════════════════════════════════════════════════

    def start_eval_side_view_recording(self, env_ids: list[int] | None = None):
        """Begin recording side-view RGB frames for the given env IDs (default: all)."""
        if not self.cfg.use_eval_side_view_cameras or not hasattr(self, '_eval_side_view_camera'):
            print("[EvalSideView] WARNING: eval side-view cameras not enabled, cannot record")
            return
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        self._eval_side_view_frames = {i: [] for i in env_ids}
        self._eval_side_view_recording = True
        print(f"[EvalSideView] Started recording for {len(env_ids)} envs")

    def capture_eval_side_view_frame(self):
        """Capture one side-view RGB frame from all recording envs. Call once per step."""
        if not self._eval_side_view_recording or not hasattr(self, '_eval_side_view_camera'):
            return
        rgb = self._eval_side_view_camera.data.output["rgb"]  # (N, H, W, 3)
        rgb_cpu = rgb.cpu().numpy()
        for env_idx in self._eval_side_view_frames:
            self._eval_side_view_frames[env_idx].append(rgb_cpu[env_idx].copy())

    def stop_eval_side_view_recording(self, save_dir: str = "videos", filename_prefix: str = ""):
        """Stop recording and save per-env MP4 videos."""
        if not self._eval_side_view_recording:
            return {}
        self._eval_side_view_recording = False

        import imageio
        from pathlib import Path
        from datetime import datetime

        videos_dir = Path(save_dir)
        videos_dir.mkdir(parents=True, exist_ok=True)
        fps = int(1.0 / (self.cfg.sim.dt * self.cfg.decimation))
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        saved_paths = {}
        for env_idx, frames in self._eval_side_view_frames.items():
            if len(frames) == 0:
                continue
            fname = f"{filename_prefix}{timestamp}_env{env_idx}_sideview.mp4"
            video_path = videos_dir / fname
            imageio.mimsave(str(video_path), frames, fps=fps)
            saved_paths[env_idx] = str(video_path)

        print(f"[EvalSideView] Saved {len(saved_paths)} videos to {save_dir}/ "
              f"({len(frames) if saved_paths else 0} frames each, {fps} fps)")
        self._eval_side_view_frames = {}
        return saved_paths

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
        from envs.utils import quat_mul_wxyz
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
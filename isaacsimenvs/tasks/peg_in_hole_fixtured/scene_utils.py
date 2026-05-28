"""Peg-in-hole-fixtured scene setup.

Identical to ``peg_in_hole/scene_utils.setup_scene`` except for spawning a
second hole-fixture body named ``peg_fixture`` (kinematic, gravity-disabled,
same baked USD as ``hole``). The reset routine in
``PegInHoleFixturedEnv._reset_peg_episode`` writes the peg_fixture pose at
``cfg.peg_in_hole.peg_fixture_xy``.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage

from isaacsimenvs.tasks.peg_in_hole.scene_utils import _asset_path, _pih_rigid_object_cfg
from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import (
    _bake_usd,
    _convert_urdf_to_usd,
    _log_scene_step,
    _materialize_env_prims,
    _robot_joint_drive_cfg,
    build_robot_articulation_usd_cfg,
    hide_goal_viz_for_student_camera,
    setup_student_camera,
)


def _hide_goal_viz_unconditional() -> None:
    """Hide every per-env GoalViz prim from the render product.

    The parent ``hide_goal_viz_for_student_camera`` is gated on
    ``cfg.student_obs.enabled``; for the fixtured env (used purely for video
    capture) we want the goal_viz hidden regardless of student_obs state.
    """
    from pxr import UsdGeom

    stage = get_current_stage()
    for prim_path in find_matching_prim_paths("/World/envs/env_.*/GoalViz"):
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()


def _override_prim_color(prim_path_regex: str, rgb: tuple[float, float, float]) -> None:
    """Recolor every Gprim under ``prim_path_regex`` to ``rgb``.

    Removes any bound shading material first (else USD material wins over
    displayColor), then sets the displayColor primvar on every UsdGeom.Gprim
    in the subtree. RGB components in [0, 1].
    """
    from pxr import Gf, Usd, UsdGeom, UsdShade

    stage = get_current_stage()
    color_vec = Gf.Vec3f(*rgb)
    for root_path in find_matching_prim_paths(prim_path_regex):
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            continue
        for prim in Usd.PrimRange(root_prim):
            try:
                UsdShade.MaterialBindingAPI(prim).UnbindAllBindings()
            except Exception:
                pass
            if prim.IsA(UsdGeom.Gprim):
                UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color_vec])


def setup_scene(env) -> None:
    """Build robot, table, goal hole, peg fixture, object, goal viz, and light."""
    assets_cfg = env.cfg.assets
    setup_t0 = time.perf_counter()
    _log_scene_step(setup_t0, f"peg-in-hole-fixtured setup start num_envs={env.num_envs}")

    env._tmp_asset_dir = tempfile.mkdtemp(prefix="peg_in_hole_fixtured_assets_")
    env._object_urdf_paths = [_asset_path(env._pih_object_urdf_abs)]
    env._hole_urdf_paths = [_asset_path(env._pih_receptive_urdf_abs)]
    env._table_urdf_paths = [_asset_path(assets_cfg.table_urdf)]

    usd_work_dir = Path(env._tmp_asset_dir) / "usd"
    bake_root = Path(env._tmp_asset_dir) / "baked_usd"
    usd_work_dir.mkdir(parents=True, exist_ok=True)

    object_raw_usd = _convert_urdf_to_usd(
        _asset_path(env._pih_object_urdf_abs), usd_work_dir, fix_base=False
    )
    object_usd_path = _bake_usd(
        object_raw_usd,
        bake_root,
        "object",
        props=dict(
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=1000.0,
            rb_solver_position_iterations=4,
            rb_solver_velocity_iterations=0,
            articulation_enabled=False,
        ),
    )
    goalviz_usd_path = _bake_usd(
        object_raw_usd,
        bake_root,
        "goalviz",
        props=dict(
            kinematic_enabled=True,
            disable_gravity=True,
            articulation_enabled=False,
            rb_solver_position_iterations=4,
            rb_solver_velocity_iterations=0,
        ),
        collision_enabled=False,
    )

    hole_usd_path = _bake_usd(
        _convert_urdf_to_usd(
            _asset_path(env._pih_receptive_urdf_abs),
            usd_work_dir / "hole",
            fix_base=False,
        ),
        bake_root,
        "hole",
        props=dict(
            kinematic_enabled=True,
            disable_gravity=True,
            articulation_enabled=False,
            rb_solver_position_iterations=4,
            rb_solver_velocity_iterations=0,
        ),
    )

    robot_usd_path = _bake_usd(
        _convert_urdf_to_usd(
            _asset_path(assets_cfg.robot_urdf),
            usd_work_dir,
            fix_base=True,
            self_collision=False,
            joint_drive=_robot_joint_drive_cfg(),
        ),
        bake_root,
        "robot",
        props=dict(
            disable_gravity=True,
            max_depenetration_velocity=1000.0,
            enabled_self_collisions=False,
            solver_position_iterations=8,
            solver_velocity_iterations=0,
        ),
        apply_physx_articulation=True,
    )
    table_usd_path = _bake_usd(
        _convert_urdf_to_usd(
            _asset_path(assets_cfg.table_urdf),
            usd_work_dir,
            fix_base=False,
        ),
        bake_root,
        "table",
        props=dict(
            kinematic_enabled=True,
            disable_gravity=True,
            articulation_enabled=False,
        ),
    )
    _log_scene_step(setup_t0, "converted object/hole/table URDFs")

    _materialize_env_prims(env)

    env.robot = Articulation(build_robot_articulation_usd_cfg(robot_usd_path))
    env.table = RigidObject(_pih_rigid_object_cfg("/World/envs/env_.*/Table", table_usd_path))
    env.hole = RigidObject(_pih_rigid_object_cfg("/World/envs/env_.*/Hole", hole_usd_path))
    # Second hole-fixture body — reuses the same baked USD as `hole`. Acts as
    # the passive peg fixture (peg sits in this one upright).
    env.peg_fixture = RigidObject(
        _pih_rigid_object_cfg("/World/envs/env_.*/PegFixture", hole_usd_path)
    )
    env.object = RigidObject(_pih_rigid_object_cfg("/World/envs/env_.*/Object", object_usd_path))
    env.goal_viz = RigidObject(_pih_rigid_object_cfg("/World/envs/env_.*/GoalViz", goalviz_usd_path))
    _log_scene_step(setup_t0, "spawned robot/table/hole/peg_fixture/object/goalviz")

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    env._object_scale_per_env = torch.tensor(
        env._pih_object_scale,
        device=env.device,
        dtype=torch.float32,
    ).expand(env.num_envs, -1).contiguous()
    env._object_asset_index_per_env = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.long
    )

    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["hole"] = env.hole
    env.scene.rigid_objects["peg_fixture"] = env.peg_fixture
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    hide_goal_viz_for_student_camera(env)
    # Fixtured env is for rendering / video — always hide GoalViz from the
    # record camera, regardless of cfg.student_obs.enabled.
    _hide_goal_viz_unconditional()
    _log_scene_step(setup_t0, "registered assets with scene")

    if env.scene._default_env_origins is None:
        env.scene.clone_environments(copy_from_source=False)
        _log_scene_step(setup_t0, "cloned environments (replicate_physics=True path)")

    # Recolor: start fixture green, goal fixture red, peg light blue.
    # Applied AFTER clone so every per-env replica is recolored.
    _override_prim_color("/World/envs/env_.*/PegFixture", (0.27, 0.72, 0.45))
    _override_prim_color("/World/envs/env_.*/Hole", (0.85, 0.21, 0.23))
    _override_prim_color("/World/envs/env_.*/Object", (0.49, 0.77, 0.88))
    _log_scene_step(setup_t0, "recolored start/goal/peg prims")

    setup_student_camera(env)


__all__ = ["setup_scene"]

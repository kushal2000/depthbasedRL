"""Peg-in-hole scene setup."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane

from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import (
    _bake_usd,
    _convert_urdf_to_usd,
    _log_scene_step,
    _materialize_env_prims,
    _robot_joint_drive_cfg,
    build_rigid_object_cfg,
    build_robot_articulation_usd_cfg,
    hide_goal_viz_for_student_camera,
    recover_asset_index_per_env,
    setup_student_camera,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _asset_path(path: str | Path) -> str:
    asset_path = Path(path)
    if not asset_path.is_absolute():
        asset_path = REPO_ROOT / asset_path
    return str(asset_path)


def _pih_rigid_object_cfg(prim_path: str, usd_path: str) -> RigidObjectCfg:
    """Single-USD RigidObject spawn using `UsdFileCfg` (not `MultiUsdFileCfg`).

    Why bypass `MultiUsdFileCfg` even though our list has just one entry:
    `MultiUsdFileCfg` sets the global carb flag
    `/isaaclab/spawn/multi_assets=True`, which triggers IsaacLab's
    "varying assets" warning under `replicate_physics=True` and may take a
    slower clone path under the hood. `UsdFileCfg` keeps the regex
    prim_path (so the Articulation/RigidObject discovers all envs) but
    avoids the multi-asset carb branch.
    """
    return RigidObjectCfg(prim_path=prim_path, spawn=UsdFileCfg(usd_path=usd_path))


def setup_scene(env) -> None:
    """Build robot, narrow table, dynamic hole fixture, object, goal, and light."""
    assets_cfg = env.cfg.assets
    setup_t0 = time.perf_counter()
    _log_scene_step(setup_t0, f"peg-in-hole setup start num_envs={env.num_envs}")

    env._tmp_asset_dir = tempfile.mkdtemp(prefix="peg_in_hole_assets_")
    # P problems -> P object/receptive URDFs. These lists are PROBLEM-indexed;
    # _problem_idx_per_env maps envs onto them (see below).
    object_urdfs = [_asset_path(u) for u in env._pih_object_urdfs]
    receptive_urdfs = [_asset_path(u) for u in env._pih_receptive_urdfs]
    n_problems = len(object_urdfs)
    env._object_urdf_paths = list(object_urdfs)
    env._hole_urdf_paths = list(receptive_urdfs)
    env._table_urdf_paths = [_asset_path(assets_cfg.table_urdf)]

    usd_work_dir = Path(env._tmp_asset_dir) / "usd"
    bake_root = Path(env._tmp_asset_dir) / "baked_usd"
    usd_work_dir.mkdir(parents=True, exist_ok=True)

    # Per-problem work/bake dirs. _prepare_urdf_for_isaacsim, _convert_urdf_to_usd
    # and _bake_usd all key on the URDF *stem*, so two problems sharing one
    # (entirely plausible: part_0_sdf_hybrid.urdf exists under both beam_2x/ and
    # beam_3x/) would silently collide and load identical geometry. At P == 1 the
    # names are unchanged, keeping the single-problem bake byte-for-byte.
    def _role(name: str, i: int) -> str:
        return name if n_problems == 1 else f"{name}_p{i:02d}"

    def _work(i: int) -> Path:
        return usd_work_dir if n_problems == 1 else usd_work_dir / f"p{i:02d}"

    _obj_props = dict(
        kinematic_enabled=False,
        disable_gravity=False,
        max_depenetration_velocity=1000.0,
        rb_solver_position_iterations=4,
        rb_solver_velocity_iterations=0,
        articulation_enabled=False,
    )
    _gv_props = dict(
        kinematic_enabled=True,
        disable_gravity=True,
        articulation_enabled=False,
        rb_solver_position_iterations=4,
        rb_solver_velocity_iterations=0,
    )
    object_raw_usds = [
        _convert_urdf_to_usd(u, _work(i), fix_base=False)
        for i, u in enumerate(object_urdfs)
    ]
    object_usd_paths = [
        _bake_usd(raw, bake_root, _role("object", i), props=_obj_props)
        for i, raw in enumerate(object_raw_usds)
    ]
    goalviz_usd_paths = [
        _bake_usd(raw, bake_root, _role("goalviz", i), props=_gv_props,
                  collision_enabled=False)
        for i, raw in enumerate(object_raw_usds)
    ]
    object_raw_usd = object_raw_usds[0]
    object_usd_path = object_usd_paths[0]
    goalviz_usd_path = goalviz_usd_paths[0]

    # A bolted fixture is kinematic: infinite effective mass, ignores contact.
    # Unbolted, it becomes a dynamic body resting on the table under gravity, so
    # the peg can push it. The bolted branch below is byte-for-byte the original
    # prop set -- max_depenetration_velocity is added only in the dynamic case
    # (it is meaningless for a kinematic body, which never depenetrates), so
    # fixture_bolted=True leaves the baked USD exactly as it was.
    _fixture_bolted = bool(getattr(env.cfg.peg_in_hole, "fixture_bolted", True))
    _hole_props = dict(
        kinematic_enabled=_fixture_bolted,
        disable_gravity=_fixture_bolted,
        articulation_enabled=False,
        rb_solver_position_iterations=4,
        rb_solver_velocity_iterations=0,
    )
    if not _fixture_bolted:
        _hole_props["max_depenetration_velocity"] = 1000.0
    hole_usd_paths = [
        _bake_usd(
            _convert_urdf_to_usd(u, _work(i) / "hole", fix_base=False),
            bake_root,
            _role("hole", i),
            props=_hole_props,
        )
        for i, u in enumerate(receptive_urdfs)
    ]
    hole_usd_path = hole_usd_paths[0]

    # Tripwire for the stem-collision failure mode: if two problems collapsed
    # onto one baked USD they would load identical geometry with no error.
    for _label, _paths in (("object", object_usd_paths), ("hole", hole_usd_paths),
                           ("goalviz", goalviz_usd_paths)):
        if len(set(_paths)) != n_problems:
            raise RuntimeError(
                f"{_label}: {n_problems} problems baked to only "
                f"{len(set(_paths))} distinct USDs -- URDF stems collided. "
                f"paths={_paths}"
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

    # Use UsdFileCfg (single USD path) instead of MultiUsdFileCfg.
    # peg_in_hole is a single-geometry task (every env uses the same peg + hole +
    # table), so the multi-asset carb path is pure overhead — and it triggers
    # the noisy "Varying assets ... however replicate_physics is enabled"
    # warning under `replicate_physics=True`. The regex prim_path is kept so
    # Articulation / RigidObject discover all envs after clone.
    env.robot = Articulation(build_robot_articulation_usd_cfg(robot_usd_path))
    env.table = RigidObject(_pih_rigid_object_cfg("/World/envs/env_.*/Table", table_usd_path))

    # slot_problem_idx has one entry per env (a shuffled balanced multiset), so
    # MultiUsdFileCfg's round-robin hands each env its own entry and the
    # assignment IS that shuffle. At P == 1 keep the UsdFileCfg path verbatim:
    # it avoids the multi_assets carb flag and the slower clone branch that
    # _pih_rigid_object_cfg's docstring documents.
    if n_problems == 1:
        env.hole = RigidObject(_pih_rigid_object_cfg("/World/envs/env_.*/Hole", hole_usd_path))
        env.object = RigidObject(_pih_rigid_object_cfg("/World/envs/env_.*/Object", object_usd_path))
        env.goal_viz = RigidObject(_pih_rigid_object_cfg("/World/envs/env_.*/GoalViz", goalviz_usd_path))
    else:
        slots = list(env._pih_slot_problem_idx)
        # Either one entry per env (shuffled) or one per mix slot (periodic).
        if env.num_envs % len(slots) != 0:
            raise RuntimeError(
                f"slot_problem_idx has {len(slots)} entries, which does not "
                f"divide num_envs ({env.num_envs}); the mix would be skewed."
            )
        env.hole = RigidObject(build_rigid_object_cfg(
            "/World/envs/env_.*/Hole", [hole_usd_paths[i] for i in slots]))
        env.object = RigidObject(build_rigid_object_cfg(
            "/World/envs/env_.*/Object", [object_usd_paths[i] for i in slots]))
        env.goal_viz = RigidObject(build_rigid_object_cfg(
            "/World/envs/env_.*/GoalViz", [goalviz_usd_paths[i] for i in slots]))
    _log_scene_step(setup_t0, "spawned robot/table/hole/object/goalviz")

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Phase B: recover which spawn slot (hence which problem) each env got.
    # The spawner walks prims in LEXICOGRAPHIC order, so source_idx != env_id --
    # the mapping must be read back, never assumed.
    if n_problems == 1:
        env._problem_idx_per_env = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )
    else:
        n_slots = len(slots)  # length of the MultiUsdFileCfg list
        slot_obj = recover_asset_index_per_env(env, "/World/envs/env_.*/Object", n_slots)
        # Object/Hole/GoalViz are three independent MultiUsdFileCfg spawns. If
        # their orderings ever disagreed, an env would get problem A's peg with
        # problem B's hole -- geometrically nonsense, but nothing would error.
        for glob, label in (("/World/envs/env_.*/Hole", "Hole"),
                            ("/World/envs/env_.*/GoalViz", "GoalViz")):
            other = recover_asset_index_per_env(env, glob, n_slots)
            if not torch.equal(slot_obj, other):
                bad = (slot_obj != other).nonzero(as_tuple=False).squeeze(-1)[:10]
                raise RuntimeError(
                    f"Object/{label} spawn-slot assignment disagrees for envs "
                    f"{bad.tolist()}; env geometry would be mismatched."
                )
        slot_tensor = torch.as_tensor(slots, device=env.device, dtype=torch.long)
        env._problem_idx_per_env = slot_tensor[slot_obj]
        counts = torch.bincount(env._problem_idx_per_env, minlength=n_problems)
        print(f"[scene_utils] envs per problem: {counts.tolist()} "
              f"({env._pih_problem_names})", flush=True)

    scales = torch.as_tensor(
        env._pih_object_scales, device=env.device, dtype=torch.float32
    )
    env._object_scale_per_env = scales[env._problem_idx_per_env].contiguous()
    env._object_asset_index_per_env = env._problem_idx_per_env.clone()

    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["hole"] = env.hole
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    hide_goal_viz_for_student_camera(env)
    _log_scene_step(setup_t0, "registered assets with scene")

    # When replicate_physics=True, InteractiveScene.__init__ leaves
    # `_default_env_origins=None` and expects `clone_environments()` to
    # populate it later. That auto-call only fires inside
    # `_add_entities_from_cfg` (config-driven scenes). Our scene is
    # manually built, so we call it ourselves here. With
    # replicate_physics=False this is a no-op for env_origins (already
    # set in __init__).
    if env.scene._default_env_origins is None:
        env.scene.clone_environments(copy_from_source=False)
        _log_scene_step(setup_t0, "cloned environments (replicate_physics=True path)")

    # Student camera is set up AFTER the clone so every env path exists at
    # sensor-construction time. The TiledCamera/Camera spawn-cfg regex still
    # creates per-env prims correctly post-clone, and the RayCaster branch
    # can pre-create its Xform parent on each env explicitly.
    setup_student_camera(env)


__all__ = ["setup_scene"]

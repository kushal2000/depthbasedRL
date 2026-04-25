"""Benchmark phase-by-phase wall-clock cost of SimToolRealEnv setup.

Wraps every helper invoked from ``utils.scene_utils.setup_scene`` with a
``time.perf_counter()`` decorator, then constructs the env once and prints a
breakdown. Also runs a small correctness probe afterwards (counts authored
contact_offset / restOffset / collisionEnabled / bound material on env_0
leaves) so subsequent fixes can be diff'd against the baseline without
regressing commits e51aeda and a926a86.

    .venv_isaacsim/bin/python debug_differences/bench_env_creation.py \
        --num_envs 256

Args:
    --num_envs        envs to spawn (default 256; lower = faster iteration)
    --num_per_type    procedural URDF count per handle-head type
                      (default 25; total URDFs = num_per_type * 6 types)
    --probe_envs      how many envs to sample for the correctness probe
                      (default min(num_envs, 4))

The breakdown distinguishes:
    A. mkdtemp + URDF generation
    B. URDF→USD conversion (force_usd_conversion=True bottleneck)
    C. env Xform pre-creation
    D. Robot spawn
    E. Table spawn
    F. Object spawn (MultiUsdFileCfg)
    G. GoalViz spawn (MultiUsdFileCfg)
    H. ground + light
    I. per-env scale tensor build
    J. apply_physx_material_properties (per-shape friction via PhysX
       tensor view, called from SimToolRealEnv.__init__ after sim init)
    M. DirectRLEnv post-setup (sim.reset, view init, allocate_state_buffers)
"""

from __future__ import annotations

import argparse
import functools
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_per_type", type=int, default=25)
    parser.add_argument("--probe_envs", type=int, default=4)
    parser.add_argument(
        "--gpu_max_rigid_contact_count", type=int, default=None,
        help="Override env.sim.physx.gpu_max_rigid_contact_count. Lab default "
        "= 2^23 = 8M; SAPG slurm uses 2^24 = 16777216. Pass when benching at "
        "the slurm scale (24576 envs) so PhysX scene init doesn't OOM on "
        "contact buffer.",
    )
    return parser


def _launch_app():
    """AppLauncher must run BEFORE any isaaclab.* import."""
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    return AppLauncher(args).app, args


_app, _args = _launch_app()


# --- After AppLauncher boot it's safe to import the rest. ---


_PHASES: dict[str, float] = defaultdict(float)
_COUNTS: dict[str, int] = defaultdict(int)


def _timed(label: str):
    """Decorator that accumulates wall-clock into _PHASES[label]."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _PHASES[label] += time.perf_counter() - t0
                _COUNTS[label] += 1

        return wrapped

    return deco


def _instrument_scene_utils() -> None:
    """Wrap helpers inside ``utils.scene_utils`` with timed proxies.

    Done in-place on the module so ``setup_scene`` (which calls these
    by name) picks up the instrumented versions.
    """
    from isaacsimenvs.tasks.simtoolreal.utils import scene_utils as su

    # A. URDF generation
    su.generate_handle_head_urdfs = _timed("A_generate_urdfs")(
        su.generate_handle_head_urdfs
    )

    # B. URDF→USD conversion: instrument UrdfConverter.__init__ + the
    #    converter object's class-level conversion done eagerly in __init__.
    from isaaclab.sim.converters import UrdfConverter as _UrdfConverter
    _orig_urdf_init = _UrdfConverter.__init__

    @functools.wraps(_orig_urdf_init)
    def _timed_urdf_init(self, cfg, *a, **k):
        t0 = time.perf_counter()
        try:
            return _orig_urdf_init(self, cfg, *a, **k)
        finally:
            _PHASES["B_urdf_to_usd"] += time.perf_counter() - t0
            _COUNTS["B_urdf_to_usd"] += 1

    _UrdfConverter.__init__ = _timed_urdf_init
    su.UrdfConverter = _UrdfConverter

    # D. Robot spawn   E. Table spawn   F. Object spawn   G. GoalViz spawn
    #    Wrap the cfg-builders so we can time the immediately-following
    #    Articulation / RigidObject ctor calls. Easier path: monkey-patch
    #    Articulation.__init__ + RigidObject.__init__ with a per-call label
    #    that defaults to a generic bucket. setup_scene happens to construct
    #    them in a fixed order: Robot → Table → Object → GoalViz.
    from isaaclab.assets import Articulation as _Articulation
    from isaaclab.assets import RigidObject as _RigidObject
    _orig_art_init = _Articulation.__init__
    _orig_rb_init = _RigidObject.__init__

    _RB_ORDER = iter(["E_table_spawn", "F_object_spawn", "G_goalviz_spawn"])

    @functools.wraps(_orig_art_init)
    def _timed_art_init(self, cfg, *a, **k):
        t0 = time.perf_counter()
        try:
            return _orig_art_init(self, cfg, *a, **k)
        finally:
            _PHASES["D_robot_spawn"] += time.perf_counter() - t0
            _COUNTS["D_robot_spawn"] += 1

    @functools.wraps(_orig_rb_init)
    def _timed_rb_init(self, cfg, *a, **k):
        try:
            label = next(_RB_ORDER)
        except StopIteration:
            label = "X_extra_rigid_object"
        t0 = time.perf_counter()
        try:
            return _orig_rb_init(self, cfg, *a, **k)
        finally:
            _PHASES[label] += time.perf_counter() - t0
            _COUNTS[label] += 1

    _Articulation.__init__ = _timed_art_init
    _RigidObject.__init__ = _timed_rb_init

    # J. PhysX tensor-view friction assignment (replaces the prior per-env
    #    USD MaterialBindingAPI loop). Called from SimToolRealEnv.__init__
    #    after super().__init__() returns; accumulated in M_residual.
    su.apply_physx_material_properties = _timed("J_set_material_props")(
        su.apply_physx_material_properties
    )

    # H. ground + light: spawn_ground_plane + DomeLight.func.
    from isaaclab.sim.spawners.from_files import spawn_ground_plane as _spawn_ground

    @functools.wraps(_spawn_ground)
    def _timed_ground(*a, **k):
        t0 = time.perf_counter()
        try:
            return _spawn_ground(*a, **k)
        finally:
            _PHASES["H_ground_plus_light"] += time.perf_counter() - t0
            _COUNTS["H_ground_plus_light"] += 1

    su.spawn_ground_plane = _timed_ground


def _build_cfg(num_envs: int, num_per_type: int, gpu_max_rigid_contact_count: int | None = None):
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.assets.num_assets_per_type = num_per_type
    if gpu_max_rigid_contact_count is not None:
        cfg.sim.physx.gpu_max_rigid_contact_count = gpu_max_rigid_contact_count
    return cfg


def _correctness_probe(env, probe_envs: int) -> dict:
    """Sample ``probe_envs`` envs and count whether each invariant holds."""
    from pxr import PhysxSchema, UsdPhysics, UsdShade
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    res = {
        "envs_probed": min(probe_envs, env.num_envs),
        "Robot": {
            "leaves_total": 0,
            "contact_offset_002": 0,
            "rest_offset_0": 0,
            "fingertip_leaves_total": 0,
            "fingertip_with_finger_material": 0,
        },
        "Table": {"leaves_total": 0, "contact_offset_002": 0, "rest_offset_0": 0},
        "Object": {"leaves_total": 0, "contact_offset_002": 0, "rest_offset_0": 0},
        "GoalViz": {
            "leaves_total": 0,
            "contact_offset_002": 0,
            "rest_offset_0": 0,
            "collision_disabled": 0,
        },
    }

    fingertip_links = {"left_index_DP", "left_middle_DP", "left_ring_DP",
                       "left_thumb_DP", "left_pinky_DP"}

    def _walk_collision_leaves(root_path: str):
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return
        stack = [root]
        while stack:
            prim = stack.pop()
            stack.extend(list(prim.GetChildren()))
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                yield prim

    for env_id in range(min(probe_envs, env.num_envs)):
        for role in ("Robot", "Table", "Object", "GoalViz"):
            root = f"/World/envs/env_{env_id}/{role}"
            for prim in _walk_collision_leaves(root):
                res[role]["leaves_total"] += 1
                phys = PhysxSchema.PhysxCollisionAPI(prim)
                if phys:
                    co = phys.GetContactOffsetAttr().Get()
                    ro = phys.GetRestOffsetAttr().Get()
                    if co is not None and abs(co - 0.002) < 1e-9:
                        res[role]["contact_offset_002"] += 1
                    if ro is not None and abs(ro - 0.0) < 1e-9:
                        res[role]["rest_offset_0"] += 1
                if role == "GoalViz":
                    ce = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                    if ce is False:
                        res[role]["collision_disabled"] += 1
                if role == "Robot":
                    rel = str(prim.GetPath())
                    parts = rel[len(f"/World/envs/env_{env_id}/Robot/"):].split("/")
                    link = parts[0] if parts else ""
                    if link in fingertip_links:
                        res["Robot"]["fingertip_leaves_total"] += 1
    try:
        view = env.robot.root_physx_view
        materials = view.get_material_properties()
        shape_start = 0
        fingertip_shape_ids = []
        for link_name, link_path in zip(view.shared_metatype.link_names, view.link_paths[0]):
            link_view = env.robot._physics_sim_view.create_rigid_body_view(link_path)
            shape_end = shape_start + link_view.max_shapes
            if link_name in fingertip_links:
                fingertip_shape_ids.extend(range(shape_start, shape_end))
            shape_start = shape_end
        n_envs = min(probe_envs, env.num_envs)
        fingertip_static = materials[:n_envs, fingertip_shape_ids, 0]
        res["Robot"]["fingertip_with_finger_material"] = int(
            ((fingertip_static - 1.5).abs() < 1e-6).sum().item()
        )
    except Exception as exc:
        res["Robot"]["fingertip_material_probe_error"] = repr(exc)
    return res


def _print_phases() -> None:
    total = sum(_PHASES.values())
    print()
    print("=" * 72)
    print(f"{'phase':<32}{'calls':>8}{'sec':>12}{'%':>10}")
    print("-" * 72)
    for label in sorted(_PHASES.keys()):
        sec = _PHASES[label]
        pct = 100.0 * sec / total if total > 0 else 0.0
        print(f"{label:<32}{_COUNTS[label]:>8d}{sec:>12.2f}{pct:>9.1f}%")
    print("-" * 72)
    print(f"{'TOTAL (instrumented phases)':<32}{'':>8}{total:>12.2f}")
    print("=" * 72)


def main() -> None:
    args = _args
    print(f"[bench] num_envs={args.num_envs} num_per_type={args.num_per_type}")

    _instrument_scene_utils()

    # Trigger gym.register side effects (so the Env class is importable).
    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv

    cfg = _build_cfg(
        args.num_envs, args.num_per_type, args.gpu_max_rigid_contact_count
    )

    t_total0 = time.perf_counter()
    t_env0 = time.perf_counter()
    env = SimToolRealEnv(cfg=cfg)
    t_env_end = time.perf_counter() - t_env0

    # Anything inside SimToolRealEnv.__init__ that wasn't itself instrumented
    # (sim.reset, scene clone wiring, allocate_state_buffers) lands here.
    _PHASES["M_env_post_setup_residual"] = max(
        0.0, t_env_end - sum(_PHASES.values())
    )

    print(f"\n[bench] SimToolRealEnv.__init__ wall-clock: {t_env_end:.2f}s")

    _print_phases()

    print("\n[probe] correctness invariants on env_0..env_{}".format(
        min(args.probe_envs, args.num_envs) - 1
    ))
    res = _correctness_probe(env, args.probe_envs)
    for role, stats in res.items():
        if isinstance(stats, dict):
            print(f"  {role}: {stats}")

    t_total = time.perf_counter() - t_total0
    print(f"\n[bench] full script wall-clock: {t_total:.2f}s")
    env.close()


if __name__ == "__main__":
    main()

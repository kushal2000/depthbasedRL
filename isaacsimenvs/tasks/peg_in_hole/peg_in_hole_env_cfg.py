"""Config for the peg-in-hole SimToolReal variant."""

from __future__ import annotations

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import (
    AssetsCfg,
    SimToolRealEnvCfg,
    _default_sim_cfg,
)


VALID_GOAL_MODES = (
    "preInsertAndFinal",
    "finalGoalOnly",
    "transportPreInsertFinal",
    # Fixtured-env-only mode: trajectory waypoints come from scenes.npz
    # (per-scene precomputed dense paths). Validation happens via the
    # fixtured env's _configure_problem override.
    "dense",
)


@configclass
class PegInHoleAssetsCfg(AssetsCfg):
    object_name: str = "peg"
    object_urdf: str = "assets/urdf/peg_in_hole/peg/peg.urdf"
    receptive_urdf: str = "assets/urdf/peg_in_hole/holes/hole_tol0p5mm/hole_tol0p5mm.urdf"
    table_urdf: str = "assets/urdf/table_narrow.urdf"
    object_scale: tuple[float, float, float] = (6.25, 0.75, 0.5)
    num_assets_per_type: int = 1


@configclass
class PegInHoleCfg:
    problem: str = "peg.tol0p5mm"
    # Multi-problem co-training. When `problems` is non-empty it OVERRIDES
    # `problem`, and different parallel envs run different problems in one run.
    # `problem_mix` gives integer slot counts (one per entry, empty => uniform):
    # integers rather than float weights because the spawner round-robins over a
    # slot list, so integers are exactly what the mechanism expresses and there
    # is no rounding gap between the requested and realized mix.
    #   problems=[A, B, C], problem_mix=[3, 1, 1] -> slots [A,A,A,B,C], L=5
    # Requires num_envs % L == 0, else the trailing envs skew the mix.
    problems: tuple[str, ...] = ()
    problem_mix: tuple[int, ...] = ()
    problem_assignment_seed: int = 0
    # False (default) -> the MultiUsdFileCfg list has one entry per mix slot and
    # the spawner cycles it. True -> one entry per env, drawn from a shuffled
    # balanced multiset.
    #
    # Measured at 512 envs, P=2: the spawn cost is driven entirely by LIST
    # LENGTH, not by multi-asset mode --
    #     P=1  UsdFileCfg            7.15 s
    #     P=2  512-entry (shuffled) 172.00 s
    #     P=2  2-entry   (periodic)   6.93 s
    # and the periodic form is also BETTER stratified across SAPG blocks
    # (worst deviation 0.10% vs 1.66% at 12288 envs / P=4), because the
    # spawner's lexicographic prim ordering already decorrelates env index from
    # problem. So shuffling costs 25x setup and buys nothing; it is kept only as
    # an escape hatch. Totals are exact either way.
    problem_assignment_shuffle: bool = False
    goal_mode: str = "preInsertAndFinal"

    # When True the receptive/fixture is a kinematic body: infinite effective
    # mass, unaffected by contact -- i.e. bolted to the table. Set False to make
    # it a dynamic body that rests under gravity and can be pushed by the peg.
    # With a free fixture the insertion goal is re-derived from the fixture's
    # live pose every step (see PegInHoleEnv._refresh_free_fixture_goal), so the
    # policy's goal observation tracks the fixture and no retraining is needed.
    # NOTE: the fixture's sliding friction against the table comes from the
    # URDF/default physics material -- _bake_usd's prop map has no friction key,
    # so tuning it needs a PhysicsMaterial on the hole prim (not done here).
    fixture_bolted: bool = True

    hole_x_range: tuple[float, float] = (-0.1875, 0.1875)
    hole_y_range: tuple[float, float] = (-0.1, 0.1)
    # Per-episode yaw randomization about +Z applied to the hole and to the
    # derived insertion goals. Range is symmetric: yaw ~ U(-range, +range).
    # Default 0.0 keeps legacy behavior (identity hole quat).
    hole_yaw_range_deg: float = 0.0

    goal_xy_obs_noise: float = 0.002
    # Per-episode yaw obs noise about world +Z applied to the observed goal
    # (insertion targets only). Range is symmetric: yaw_noise ~ U(-deg, +deg).
    # Default 0.0 keeps legacy behavior.
    goal_yaw_obs_noise_deg: float = 0.0
    random_goal_fraction: float = 0.0
    random_goal_max_successes: int = 5
    random_goal_curriculum_success_threshold: float = 2.0
    lift_bonus_fade_threshold: float = 2.0

    insertion_success_tolerance: float = 0.01

    enable_retract: bool = True
    retract_reward_scale: float = 1.0
    retract_distance_threshold: float = 0.1
    retract_success_bonus: float = 1000.0
    retract_success_tolerance: float = 0.005

    # When True, the lift_rew / lift_bonus terms apply to insertion-only envs
    # (not only random-goal envs). Used by dense-trajectory experiments where
    # the policy needs explicit Z-progress shaping during the lift-in-place
    # prelude stage.
    force_lift_reward_active: bool = False

    # Terminate when the peg has slipped out of the gripper and is sitting on
    # the table. The check fires when (1) object z is within
    # ``dropped_on_table_z_margin`` of the table top AND (2) the mean
    # fingertip-to-object distance exceeds ``dropped_on_table_ft_distance``,
    # AND (3) the env is not in retract_phase (the deliberate finger-pull-away
    # would otherwise false-trigger this). Off by default — opt in per .sub.
    enable_dropped_on_table_term: bool = False
    dropped_on_table_z_margin: float = 0.05
    dropped_on_table_ft_distance: float = 0.15


def _default_peg_in_hole_sim_cfg() -> SimulationCfg:
    """Return a render-overlay-compatible sim cfg for teacher and student tasks."""
    sim_cfg = _default_sim_cfg()
    # UWLab partial-assembly defaults for tight insertion/contact-rich problems.
    # Keep this task's 60 Hz policy control cadence; these are only PhysX
    # solver/contact/buffer settings.
    sim_cfg.physx.solver_type = 1
    sim_cfg.physx.min_position_iteration_count = 1
    sim_cfg.physx.max_position_iteration_count = 192
    sim_cfg.physx.min_velocity_iteration_count = 0
    sim_cfg.physx.max_velocity_iteration_count = 1
    sim_cfg.physx.bounce_threshold_velocity = 0.02
    sim_cfg.physx.friction_offset_threshold = 0.01
    sim_cfg.physx.friction_correlation_distance = 0.0005
    sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    sim_cfg.physx.gpu_total_aggregate_pairs_capacity = 2**23
    sim_cfg.physx.gpu_max_rigid_contact_count = 2**23
    sim_cfg.physx.gpu_max_rigid_patch_count = 2**23
    sim_cfg.physx.gpu_collision_stack_size = 2**31
    sim_cfg.render.rendering_mode = "performance"
    sim_cfg.render.antialiasing_mode = "Off"
    return sim_cfg


@configclass
class PegInHoleEnvCfg(SimToolRealEnvCfg):
    sim: SimulationCfg = _default_peg_in_hole_sim_cfg()
    assets: PegInHoleAssetsCfg = PegInHoleAssetsCfg()
    peg_in_hole: PegInHoleCfg = PegInHoleCfg()


__all__ = [
    "PegInHoleEnvCfg",
    "PegInHoleAssetsCfg",
    "PegInHoleCfg",
    "VALID_GOAL_MODES",
]

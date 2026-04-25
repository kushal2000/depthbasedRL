"""Scene-setup helpers for SimToolRealEnv._setup_scene.

Isolates:
  - Per-joint PD gain tables (arm + hand, 29 DOFs total), default arm pose.
  - Body/joint name constants used for downstream lookups.
  - ArticulationCfg / RigidObjectCfg builders for robot and table.
  - Per-env distinct-USD spawning via MultiUsdFileCfg (one of the procedural
    handle-head USDs is bound into each env's Object + GoalViz).
  - Physics-material attachment (default friction + fingertip override).

Gains are taken from isaacsim_conversion/isaacsim_env.py (lines 24-80) — they
originate in isaacgymenvs/tasks/simtoolreal/utils.py and were verified to work
with the pretrained checkpoint via the bit-exact rollout path. ImplicitActuatorCfg
runs its own implicit PD on top of the USD, so we pass RAW values here (no
180/π compensation — that compensation is only needed when routing gains
through UrdfConverter's joint_drive, which we skip in favor of explicit actuator
configs).
"""

from __future__ import annotations

import tempfile

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.spawners import UrdfFileCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
from isaaclab.sim.schemas import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sim.utils import (
    find_matching_prim_paths,
    get_current_stage,
)

from .generate_objects import generate_handle_head_urdfs


# ----------------------------------------------------------------------------
# Joint names / regexes / body names
# ----------------------------------------------------------------------------


ARM_JOINT_REGEX = "iiwa14_joint_.*"
HAND_JOINT_REGEX = "left_.*"


# Canonical 29-DOF joint order from the legacy isaacgymenvs training pipeline
# (isaacgymenvs/utils/observation_action_utils_sharpa.py:51). The pretrained
# policy and any external code (deployment/, isaacsim_conversion/) consume
# joint tensors in THIS order — depth-first per finger, with `left_1`..`left_5`
# prefixes inserted into the URDF so isaacgym's alphabetical sort lands on it.
#
# Isaac Lab's articulation parser uses a different order (breadth-first across
# fingers: all MCP_FE → all MCP_AA → all PIP → all DIP). The env reads/writes
# joint tensors in Lab order internally and converts to/from this canonical
# order at the policy I/O boundary — see action_utils.apply_action_pipeline
# and obs_utils.build_observations.
JOINT_NAMES_CANONICAL: tuple[str, ...] = (
    "iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
    "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7",
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP",
    "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP",
    "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP",
    "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
)
assert len(JOINT_NAMES_CANONICAL) == 29
PALM_BODY_NAME = "iiwa14_link_7"
# Fingertip body = the merged DP link. Both Isaac Lab and isaacgym
# collapse the DP / elastomer / fingertip chain when merge=True (the
# default we keep), so DP holds the merged body. There is a small (~2mm)
# parser drift between the two sims on the merged body's anchor — see
# dextoolbench/diff_simtoolreal_obs.py and STATUS.md. Switching merge=False
# bypasses that drift but creates a worse problem (different inertia
# aggregation between sims).
FINGERTIP_BODY_REGEX = "left_(index|middle|ring|thumb|pinky)_DP"
FINGERTIP_LINK_NAMES: tuple[str, ...] = (
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_thumb_DP",
    "left_pinky_DP",
)


# ----------------------------------------------------------------------------
# Per-joint PD gains (verified with pretrained checkpoint)
# ----------------------------------------------------------------------------


ARM_JOINT_STIFFNESS: dict[str, float] = {
    "iiwa14_joint_1": 600.0,
    "iiwa14_joint_2": 600.0,
    "iiwa14_joint_3": 500.0,
    "iiwa14_joint_4": 400.0,
    "iiwa14_joint_5": 200.0,
    "iiwa14_joint_6": 200.0,
    "iiwa14_joint_7": 200.0,
}

ARM_JOINT_DAMPING: dict[str, float] = {
    "iiwa14_joint_1": 27.027026473513512,
    "iiwa14_joint_2": 27.027026473513512,
    "iiwa14_joint_3": 24.672186769721083,
    "iiwa14_joint_4": 22.067474708266914,
    "iiwa14_joint_5": 9.752538131173853,
    "iiwa14_joint_6": 9.147747263670984,
    "iiwa14_joint_7": 9.147747263670984,
}

HAND_JOINT_STIFFNESS: dict[str, float] = {
    # Thumb (5 DOF)
    "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2,
    "left_thumb_MCP_FE": 4.76, "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
    # Index (4 DOF)
    "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
    "left_index_PIP": 0.9, "left_index_DIP": 0.9,
    # Middle (4 DOF)
    "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
    "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
    # Ring (4 DOF)
    "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
    "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
    # Pinky (5 DOF)
    "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76,
    "left_pinky_MCP_AA": 6.62, "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
}

HAND_JOINT_DAMPING: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.28676845, "left_thumb_CMC_AA": 0.40845109,
    "left_thumb_MCP_FE": 0.20394083, "left_thumb_MCP_AA": 0.24044435,
    "left_thumb_IP": 0.04190723,
    "left_2_index_MCP_FE": 0.20859232, "left_index_MCP_AA": 0.24595532,
    "left_index_PIP": 0.04243185, "left_index_DIP": 0.03504461,
    "left_3_middle_MCP_FE": 0.2085923, "left_middle_MCP_AA": 0.24595532,
    "left_middle_PIP": 0.04243185, "left_middle_DIP": 0.03504461,
    "left_4_ring_MCP_FE": 0.20859226, "left_ring_MCP_AA": 0.24595528,
    "left_ring_PIP": 0.04243183, "left_ring_DIP": 0.0350446,
    "left_5_pinky_CMC": 0.02782345, "left_pinky_MCP_FE": 0.20859229,
    "left_pinky_MCP_AA": 0.24595528, "left_pinky_PIP": 0.04243183,
    "left_pinky_DIP": 0.0350446,
}

assert len(ARM_JOINT_STIFFNESS) == 7 and len(ARM_JOINT_DAMPING) == 7
assert len(HAND_JOINT_STIFFNESS) == 22 and len(HAND_JOINT_DAMPING) == 22


# Note: if you ever stop using ImplicitActuatorCfg and need the USD-level
# JointDriveCfg gains to be the source of truth, multiply them by 180/π
# before passing into PDGainsCfg. UrdfConverter applies a π/180 conversion
# for revolute joints (urdf_converter.py:292-320), since UsdPhysics.DriveAPI
# convention is per-degree while PhysX is per-radian. Skipped only for
# JOINT_PRISMATIC. Production path doesn't need this — JointDriveCfg gains
# below are 0.0 (placeholder), overwritten by ImplicitActuatorCfg at sim init.


# Per-joint armature (reflected motor inertia) and static friction.
# Legacy values from `isaacgymenvs/tasks/simtoolreal/utils.py:158-205`. Without
# armature, hand DIP/IP joints have only their link mass as inertia, which
# inflates the effective damping ratio (D / (2 sqrt(K·M))) and makes the
# response look overdamped. Without friction, joints move under any tiny
# load. Both must be set to match legacy hand dynamics.
HAND_JOINT_ARMATURE: dict[str, float] = {
    # Thumb
    "left_1_thumb_CMC_FE": 0.0032, "left_thumb_CMC_AA": 0.0032,
    "left_thumb_MCP_FE": 0.00265, "left_thumb_MCP_AA": 0.00265,
    "left_thumb_IP": 0.0006,
    # Index
    "left_2_index_MCP_FE": 0.00265, "left_index_MCP_AA": 0.00265,
    "left_index_PIP": 0.0006, "left_index_DIP": 0.00042,
    # Middle
    "left_3_middle_MCP_FE": 0.00265, "left_middle_MCP_AA": 0.00265,
    "left_middle_PIP": 0.0006, "left_middle_DIP": 0.00042,
    # Ring
    "left_4_ring_MCP_FE": 0.00265, "left_ring_MCP_AA": 0.00265,
    "left_ring_PIP": 0.0006, "left_ring_DIP": 0.00042,
    # Pinky
    "left_5_pinky_CMC": 0.00012, "left_pinky_MCP_FE": 0.00265,
    "left_pinky_MCP_AA": 0.00265, "left_pinky_PIP": 0.0006,
    "left_pinky_DIP": 0.00042,
}

HAND_JOINT_FRICTION: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.132, "left_thumb_CMC_AA": 0.132,
    "left_thumb_MCP_FE": 0.07456, "left_thumb_MCP_AA": 0.07456,
    "left_thumb_IP": 0.01276,
    "left_2_index_MCP_FE": 0.07456, "left_index_MCP_AA": 0.07456,
    "left_index_PIP": 0.01276, "left_index_DIP": 0.00378738,
    "left_3_middle_MCP_FE": 0.07456, "left_middle_MCP_AA": 0.07456,
    "left_middle_PIP": 0.01276, "left_middle_DIP": 0.00378738,
    "left_4_ring_MCP_FE": 0.07456, "left_ring_MCP_AA": 0.07456,
    "left_ring_PIP": 0.01276, "left_ring_DIP": 0.00378738,
    "left_5_pinky_CMC": 0.012, "left_pinky_MCP_FE": 0.07456,
    "left_pinky_MCP_AA": 0.07456, "left_pinky_PIP": 0.01276,
    "left_pinky_DIP": 0.00378738,
}

assert len(HAND_JOINT_ARMATURE) == 22 and len(HAND_JOINT_FRICTION) == 22


# Proven-working default arm pose (isaacsim_conversion/isaacsim_env.py:101-109).
ARM_DEFAULT_JOINT_POS: dict[str, float] = {
    "iiwa14_joint_1": -1.571,
    "iiwa14_joint_2": 1.571,
    "iiwa14_joint_3": 0.0,
    "iiwa14_joint_4": 1.376,
    "iiwa14_joint_5": 0.0,
    "iiwa14_joint_6": 1.485,
    "iiwa14_joint_7": 1.308,
}


# ----------------------------------------------------------------------------
# Cfg builders
# ----------------------------------------------------------------------------


def build_robot_articulation_cfg(assets_cfg, usd_dir: str | None = None) -> ArticulationCfg:
    """Assemble the IIWA14 + SHARPA hand `ArticulationCfg`.

    Robot is fixed-base, gravity disabled, self-collisions off. Joint drives use
    explicit `ImplicitActuatorCfg` groups together with a USD-level
    JointDriveCfg (see two-layer comment below).

    `usd_dir` overrides Isaac Lab's default `/tmp/IsaacLab/usd_<ts>_<id>`
    cache — required on shared filesystems where the default parent dir is
    owned by whichever user ran first and others hit EACCES.

    Open-loop sine-wave parity vs the legacy isaacgym SimToolReal env (per
    ``debug_differences/sine_hand_*.py`` ablation, branch
    ``2026_04_24_debug_isaac_gym_sim_differences``):

      - 180/π pre-compensation on USD joint drive: required (without it
        the hand overshoots ~1 rad on a 0.07-rad command).
      - ``HAND_JOINT_ARMATURE`` on the hand actuator: 18× residual
        reduction (0.32 mrad max vs 5.79 mrad without).
      - ``HAND_JOINT_FRICTION``: NOT applied — Lab's ``friction`` is a
        unitless coefficient on transmitted force, the legacy gymapi
        values are Coulomb torques in Nm. Dropping legacy numbers in
        directly locks the hand. The dict is left in scope for a future
        proper port (e.g. via `dynamic_friction` after unit conversion).
    """
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UrdfFileCfg(
            asset_path=assets_cfg.robot_urdf,
            usd_dir=usd_dir,
            fix_base=True,
            merge_fixed_joints=True,
            self_collision=False,
            # make_instanceable=False so per-link material binding (fingertip
            # friction override) can reach individual collider prims under
            # /World/envs/env_*/Robot/<link>/collisions.
            make_instanceable=False,
            # Two-layer drive setup, mirroring the proven-working
            # `isaacsim_conversion/isaacsim_env.py:228-235`:
            #   - USD-level JointDriveCfg sets drive_type=force / target=position
            #     plus the (180/π)-compensated gains so the USD joint drive is
            #     consistent with PhysX units.
            #   - The ImplicitActuatorCfg below applies the *uncompensated*
            #     gains at runtime; ImplicitActuator pushes them straight into
            #     PhysX without the rad→deg conversion.
            # Setting joint_drive=None left the USD drive in a stub state and
            # produced an over-damped/sluggish hand response.
            # The JointDriveCfg authors a USD `UsdPhysics.DriveAPI` prim on
            # each joint with the configured drive_type / target_type. PhysX
            # only registers a drive on a joint when this prim is present —
            # without it, ImplicitActuator's runtime gain writes have no slot
            # to land in (verified empirically: joint_drive=None gives 0.94 rad
            # max parity error while production gives 1.6 mrad).
            #
            # The numeric stiffness/damping values inside PDGainsCfg are
            # OVERWRITTEN at simulation start by the ImplicitActuatorCfg below
            # (Lab actuator_base.py:175-200 documents this precedence: cfg-set
            # values win over USD-parsed values). They are also subject to a
            # rad→deg multiplication inside UrdfConverter for revolute joints
            # (urdf_converter.py:304-305). We pass 0.0 here to make it explicit
            # that the runtime layer is the source of truth — no need to dual-
            # maintain compensated copies.
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                ),
            ),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.8, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                **ARM_DEFAULT_JOINT_POS,
                **{name: 0.0 for name in HAND_JOINT_STIFFNESS},
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[ARM_JOINT_REGEX],
                stiffness=ARM_JOINT_STIFFNESS,
                damping=ARM_JOINT_DAMPING,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[HAND_JOINT_REGEX],
                stiffness=HAND_JOINT_STIFFNESS,
                damping=HAND_JOINT_DAMPING,
                armature=HAND_JOINT_ARMATURE,
            ),
        },
    )


def build_table_rigid_object_cfg(
    assets_cfg, z: float, usd_dir: str | None = None
) -> RigidObjectCfg:
    """Table as a static (kinematic) rigid body at world height `z`.

    fix_base=False on purpose: with fix_base=True, UrdfConverter inserts a
    fixed PhysicsJoint between world and the base link, which PhysX rejects
    as a "joint between static bodies" once each env owns its own copy of
    the prim (the legacy instance-proxy flow swallowed the error). The
    kinematic_enabled rigid prop is what actually pins the table in place
    — kinematic bodies don't fall under gravity and only move when their
    pose is explicitly written, so we don't need the URDF root joint.
    """
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=UrdfFileCfg(
            asset_path=assets_cfg.table_urdf,
            usd_dir=usd_dir,
            fix_base=False,
            merge_fixed_joints=True,
            make_instanceable=False,
            joint_drive=None,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# ----------------------------------------------------------------------------
# Per-env distinct-USD object cfg builders (MultiUsdFileCfg cycling)
# ----------------------------------------------------------------------------


def build_object_rigid_object_cfg(usd_paths: list[str]) -> RigidObjectCfg:
    """Dynamic Object cfg — MultiUsdFileCfg cycles `usd_paths` across envs.

    With InteractiveSceneCfg.replicate_physics=False, the env xforms exist
    before _setup_scene runs. spawn_multi_usd_file then resolves the regex
    prim_path to all env_* prims and copies a different proto USD into each
    via deterministic cycling (env at source-prim-path index `i` receives
    `usd_paths[i % len(usd_paths)]`).

    articulation_props=articulation_enabled=False is required: UrdfConverter
    authors an ArticulationRootAPI on single-link URDFs even with
    fix_base=False, and when each env owns an independent copy of the prim
    PhysX treats them all as top-level articulations — the resulting
    per-env articulation count collides with the RigidObject view's
    expectation of num_envs plain rigid bodies and triggers a device-side
    assert inside GpuRigidBodyView during set_transforms. The legacy
    instance-proxy flow dodged this because articulation parsing was
    deferred on inherited prims.
    """
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=MultiUsdFileCfg(
            usd_path=list(usd_paths),
            random_choice=False,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
    )


def build_goal_viz_rigid_object_cfg(usd_paths: list[str]) -> RigidObjectCfg:
    """Kinematic visual twin of Object — same cycling, gravity + collisions
    disabled.

    The legacy isaacgym side puts goal_object in its own collision group
    (env.py:1369 — `collision_filter_idx = env_idx + num_envs`) so it never
    collides with the real object/robot/table. Isaac Lab has no equivalent
    per-actor group field, so we disable collisions outright on the visual
    twin (it's kinematic + gravity-off anyway, no need for contacts).
    """
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalViz",
        spawn=MultiUsdFileCfg(
            usd_path=list(usd_paths),
            random_choice=False,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=False,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
    )


# ----------------------------------------------------------------------------
# Physics materials (default friction + fingertip override)
# ----------------------------------------------------------------------------


def _uninstance_subtree(root_prim_path: str) -> None:
    """Promote any instanced prim under ``root_prim_path`` to non-instanced.

    Required before walking for leaf colliders or material-bound prims:
    ``MultiUsdFileCfg`` re-instances the per-env USDs, which makes the
    actual collider Mesh leaves live under the shared prototype (read-only
    via instance proxies). After this call, the leaves are local-stage
    writable and discoverable by a regular ``GetChildren()`` walk.
    Idempotent — re-calling on an already-uninstanced subtree is a no-op.
    """
    from pxr import Usd
    stage = get_current_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        return
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)


def _iter_collision_prim_paths(root_prim_path: str) -> list[str]:
    """Collect every *leaf* collider prim (has ``UsdPhysics.CollisionAPI``
    applied) under ``root_prim_path``.

    Auto un-instances the subtree first — required so that USDs spawned via
    ``MultiUsdFileCfg`` (per-env Object/GoalViz) actually expose their
    collider Mesh leaves on the local stage. The legacy "child named
    'collisions'" filter returned the *grouping Xform*, not the leaf shape
    that PhysX evaluates the API on; material bindings to the grouping
    prim relied on USD inheritance and were a no-op for instanced assets.
    """
    from pxr import UsdPhysics
    _uninstance_subtree(root_prim_path)
    stage = get_current_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        return []
    paths: list[str] = []
    stack = [root]
    while stack:
        prim = stack.pop()
        stack.extend(list(prim.GetChildren()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            paths.append(str(prim.GetPath()))
    return paths


_DEFAULT_MATERIAL_PATH = "/World/PhysicsMaterials/simtoolreal_default"
_FINGERTIP_MATERIAL_PATH = "/World/PhysicsMaterials/simtoolreal_fingertip"


def author_physics_materials(assets_cfg) -> None:
    """Author the two shared material prims once (idempotent)."""
    if not assets_cfg.modify_asset_frictions:
        return
    default_cfg = RigidBodyMaterialCfg(
        static_friction=assets_cfg.robot_friction,
        dynamic_friction=assets_cfg.robot_friction,
        restitution=0.0,
    )
    default_cfg.func(_DEFAULT_MATERIAL_PATH, default_cfg)
    fingertip_cfg = RigidBodyMaterialCfg(
        static_friction=assets_cfg.finger_tip_friction,
        dynamic_friction=assets_cfg.finger_tip_friction,
        restitution=0.0,
    )
    fingertip_cfg.func(_FINGERTIP_MATERIAL_PATH, fingertip_cfg)


def _split_fingertip_colliders(
    robot_colliders: list[str], robot_prim_root: str
) -> tuple[list[str], list[str]]:
    """Partition robot collider paths into (fingertip, non-fingertip)."""
    prefix = robot_prim_root.rstrip("/") + "/"
    fingertip, non_fingertip = [], []
    for path in robot_colliders:
        if not path.startswith(prefix):
            non_fingertip.append(path)
            continue
        # path = {robot_prim_root}/<link>/<sub...>/collisions
        rel = path[len(prefix):]
        link_name = rel.split("/", 1)[0]
        if link_name in FINGERTIP_LINK_NAMES:
            fingertip.append(path)
        else:
            non_fingertip.append(path)
    return fingertip, non_fingertip


def apply_physics_props_for_env(env_id: int, assets_cfg) -> None:
    """Author all per-env physics properties for one env in a single pass.

    Walks each role's collider subtree exactly once (Robot, Table, Object,
    GoalViz) and authors:

      - ``MaterialBindingAPI`` to the default friction material on every
        non-fingertip leaf, and to the fingertip-friction material on every
        fingertip leaf (gated on ``assets_cfg.modify_asset_frictions``).
      - ``physxCollision:contactOffset = 2 mm`` and ``restOffset = 0`` on
        every leaf — undoing the spawn-cfg's silent default-to-2-cm caused
        by ``apply_nested``'s instance-proxy bail (see commit a926a86).
      - ``physics:collisionEnabled = False`` on every GoalViz leaf — the
        spawn-cfg ``collision_enabled=False`` is a no-op on nested colliders
        (see commit e51aeda); without this the kinematic visual twin still
        collides with the real object and the trained hammer wedges at
        ~4 cm from the goal.

    Replaces the previous three sequential per-env loops (steps 10/11/12 in
    setup_scene), which together walked each subtree 9× per env. Author
    materials via ``author_physics_materials`` must have already run.

    Layering w.r.t. ``Sdf.ChangeBlock``:
      - Walks (``_iter_collision_prim_paths``) run *outside* the block —
        they include ``SetInstanceable(False)`` writes whose effects must
        be visible to the subsequent ``GetChildren()`` walk.
      - ``MaterialBindingAPI.Apply`` runs *outside* — empirically, applying
        it inside a ChangeBlock causes the subsequent ``Bind`` to write a
        relationship that ``GetDirectBindingRel`` can't see (apiSchemas
        metadata write is queued; relationship name doesn't resolve).
      - ``PhysxCollisionAPI.Apply`` and all ``Set``/``Bind`` calls go
        *inside* the block — those are pure attribute / relationship writes
        and benefit from the notification batching.
    """
    from pxr import PhysxSchema, Sdf, UsdPhysics, UsdShade

    stage = get_current_stage()

    role_roots = {
        "Robot":   f"/World/envs/env_{env_id}/Robot",
        "Table":   f"/World/envs/env_{env_id}/Table",
        "Object":  f"/World/envs/env_{env_id}/Object",
        "GoalViz": f"/World/envs/env_{env_id}/GoalViz",
    }
    leaves_by_role = {
        role: _iter_collision_prim_paths(root) for role, root in role_roots.items()
    }
    fingertip_leaves, robot_non_fingertip = _split_fingertip_colliders(
        leaves_by_role["Robot"], role_roots["Robot"]
    )

    bind_materials = assets_cfg.modify_asset_frictions
    if bind_materials:
        default_material = UsdShade.Material(stage.GetPrimAtPath(_DEFAULT_MATERIAL_PATH))
        fingertip_material = UsdShade.Material(stage.GetPrimAtPath(_FINGERTIP_MATERIAL_PATH))
        default_paths = (
            robot_non_fingertip
            + leaves_by_role["Table"]
            + leaves_by_role["Object"]
            + leaves_by_role["GoalViz"]
        )
        default_apis = [
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(p))
            for p in default_paths
        ]
        fingertip_apis = [
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(p))
            for p in fingertip_leaves
        ]
        bind_strength = UsdShade.Tokens.strongerThanDescendants

    with Sdf.ChangeBlock():
        if bind_materials:
            for api in default_apis:
                api.Bind(default_material, bindingStrength=bind_strength, materialPurpose="physics")
            for api in fingertip_apis:
                api.Bind(fingertip_material, bindingStrength=bind_strength, materialPurpose="physics")
        # Contact / rest offsets on every leaf.
        for leaves in leaves_by_role.values():
            for path in leaves:
                prim = stage.GetPrimAtPath(path)
                px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                (px.GetContactOffsetAttr() or px.CreateContactOffsetAttr()).Set(0.002)
                (px.GetRestOffsetAttr() or px.CreateRestOffsetAttr()).Set(0.0)
        # Disable collisions on every GoalViz leaf (kinematic visual twin
        # must not collide with the real object).
        for path in leaves_by_role["GoalViz"]:
            prim = stage.GetPrimAtPath(path)
            ce = UsdPhysics.CollisionAPI(prim)
            (ce.GetCollisionEnabledAttr() or ce.CreateCollisionEnabledAttr()).Set(False)


# ----------------------------------------------------------------------------
# Scene orchestrator (Phase B)
# ----------------------------------------------------------------------------


def setup_scene(env) -> None:
    """Top-level ``_setup_scene`` body — spawns robot, table, per-env
    distinct-USD object + goal-viz, ground plane + lighting, and binds
    physics materials.

    Mutates env: sets ``env.robot``, ``env.table``, ``env.object``,
    ``env.goal_viz``, ``env._object_scale_per_env``, ``env._tmp_asset_dir``.
    Also registers the assets with ``env.scene`` so the framework refreshes
    their tensors each step.

    Cloning model: ``InteractiveSceneCfg.replicate_physics=False`` is set
    so PhysX parses each env independently (required for MultiUsdFileCfg's
    distinct per-env meshes). InteractiveScene only creates env_0 by
    default and its built-in clone_environments() is a no-op against an
    empty source, so we materialize env_1..env_{N-1} as empty Xform prims
    ourselves. Each spawner below (Robot/Table via ``@clone``,
    Object/GoalViz via ``MultiUsdFileCfg``) then resolves its regex
    prim_path against all env_* and fills them in.
    """
    assets_cfg = env.cfg.assets

    # 1. Generate procedural URDFs (100 per handle-head type by default).
    env._tmp_asset_dir = tempfile.mkdtemp(prefix="simtoolreal_assets_")
    urdf_paths, object_scales_normalized = generate_handle_head_urdfs(
        handle_head_types=tuple(assets_cfg.handle_head_types),
        num_per_type=assets_cfg.num_assets_per_type,
        out_dir=env._tmp_asset_dir,
        shuffle=assets_cfg.shuffle_assets,
    )

    # 2. Force-convert URDFs → USDs. force_usd_conversion=True regenerates
    #    every launch for determinism. joint_drive=None since procedural
    #    objects have no joints. usd_dir routes the per-converter cache to
    #    the per-process tmpdir instead of Isaac Lab's default /tmp/IsaacLab,
    #    which collides on shared hosts (the default parent dir is
    #    created by whichever user runs first and other users hit EACCES).
    usd_paths: list[str] = []
    for urdf in urdf_paths:
        converter = UrdfConverter(
            UrdfConverterCfg(
                asset_path=urdf,
                usd_dir=env._tmp_asset_dir,
                fix_base=False,
                merge_fixed_joints=True,
                force_usd_conversion=True,
                make_instanceable=False,
                joint_drive=None,
            )
        )
        usd_paths.append(converter.usd_path)

    # 3. Pre-create empty env_1..env_{N-1} Xform prims so subsequent regex
    #    spawns (Robot/Table via @clone, Object/GoalViz via MultiUsdFileCfg)
    #    resolve to all envs. InteractiveScene only creates env_0 by
    #    default, and its clone_environments() with replicate_physics=False
    #    is a no-op against an empty source — so we materialize the prims
    #    here ourselves and let each spawner fill them in.
    stage = get_current_stage()
    for env_path in env.scene.env_prim_paths:
        if not stage.GetPrimAtPath(env_path).IsValid():
            stage.DefinePrim(env_path, "Xform")

    # 4. Robot + table — homogeneous regex spawn. The @clone-decorated
    #    UrdfFileCfg spawner now resolves the regex to all env_* prims and
    #    writes a copy of the converted USD into each. usd_dir routes the
    #    converter's USD cache to the per-process tmpdir (avoids the
    #    /tmp/IsaacLab shared-host EACCES).
    env.robot = Articulation(
        build_robot_articulation_cfg(assets_cfg, usd_dir=env._tmp_asset_dir)
    )
    env.table = RigidObject(
        build_table_rigid_object_cfg(
            assets_cfg, z=env.cfg.reset.table_reset_z, usd_dir=env._tmp_asset_dir,
        )
    )

    # 5. Object + GoalViz — per-env distinct USDs via MultiUsdFileCfg.
    #    Deterministic cycling: env at source-prim-path index i receives
    #    usd_paths[i % len(usd_paths)]. Per-env size variability still gets
    #    further multiplied at runtime by the `_object_scale_multiplier` DR
    #    knob (applied to keypoint offsets and object_scales obs).
    env.object = RigidObject(build_object_rigid_object_cfg(usd_paths))
    env.goal_viz = RigidObject(build_goal_viz_rigid_object_cfg(usd_paths))

    # 6. Ground plane + dome light (global, outside env_*).
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 7. Cross-env collision filtering against /World/ground: only required
    #    on the CPU PhysX pipeline. The GPU pipeline filters automatically
    #    via collision groups, and SimToolReal is GPU-only (MultiUsdFileCfg
    #    + large num_envs aren't supported on CPU). Removed the CPU branch.

    # 8. Build per-env scale tensor by parsing env_K from each spawned
    #    Object prim path. spawn_multi_asset iterates the regex-matched
    #    source_prim_paths with `index % len(usd_paths)` cycling, so the
    #    source_idx → usd_idx mapping is direct; we then store the scale
    #    indexed by env_id (parsed from ".../env_K/Object").
    num_envs = env.num_envs
    object_prim_paths = find_matching_prim_paths("/World/envs/env_.*/Object")
    if len(object_prim_paths) != num_envs:
        raise RuntimeError(
            f"Expected {num_envs} Object prims after MultiUsdFileCfg spawn, "
            f"got {len(object_prim_paths)}. Cloner-drop bug may have returned."
        )
    # object_scales_normalized[i] is a (sx, sy, sz) 3-tuple, so the
    # per-env tensor is shape (N, 3) — matches the legacy `torch.tensor(
    # [scales[chosen]] * N)` allocation that downstream obs/reward code
    # expects.
    env._object_scale_per_env = torch.zeros(
        num_envs, 3, device=env.device, dtype=torch.float32
    )
    for source_idx, obj_path in enumerate(object_prim_paths):
        env_segment = obj_path.rsplit("/", 2)[-2]  # ".../env_K/Object" → "env_K"
        env_id = int(env_segment.removeprefix("env_"))
        env._object_scale_per_env[env_id] = torch.tensor(
            object_scales_normalized[source_idx % len(usd_paths)],
            device=env.device,
            dtype=torch.float32,
        )

    # 9. Register with scene so DirectRLEnv refreshes their tensors each step.
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz

    # 10. Per-env physics-prop authoring — material binding, contact/rest
    #     offsets, and GoalViz collision-disable, fused into one walk per
    #     role per env (was three separate loops in pre-fusion code).
    author_physics_materials(assets_cfg)
    for env_id in range(num_envs):
        apply_physics_props_for_env(env_id, assets_cfg)


__all__ = [
    "ARM_JOINT_REGEX",
    "HAND_JOINT_REGEX",
    "JOINT_NAMES_CANONICAL",
    "PALM_BODY_NAME",
    "FINGERTIP_BODY_REGEX",
    "FINGERTIP_LINK_NAMES",
    "ARM_JOINT_STIFFNESS",
    "ARM_JOINT_DAMPING",
    "HAND_JOINT_STIFFNESS",
    "HAND_JOINT_DAMPING",
    "HAND_JOINT_ARMATURE",
    "HAND_JOINT_FRICTION",
    "ARM_DEFAULT_JOINT_POS",
    "build_robot_articulation_cfg",
    "build_table_rigid_object_cfg",
    "build_object_rigid_object_cfg",
    "build_goal_viz_rigid_object_cfg",
    "author_physics_materials",
    "apply_physics_props_for_env",
    "setup_scene",
]

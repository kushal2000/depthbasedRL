"""Scene-setup helpers for SimToolRealEnv._setup_scene.

Isolates:
  - Per-joint PD gain tables (arm + hand, 29 DOFs total), default arm pose.
  - Body/joint name constants used for downstream lookups.
  - ArticulationCfg / RigidObjectCfg builders for robot and table.
  - Per-env distinct-USD spawning via MultiUsdFileCfg (one of the procedural
    handle-head USDs is bound into each env's Object + GoalViz).
  - Contact-material assignment through PhysX tensor views after sim startup.

Gains are taken from isaacsim_conversion/isaacsim_env.py (lines 24-80) — they
originate in isaacgymenvs/tasks/simtoolreal/utils.py and were verified to work
with the pretrained checkpoint via the bit-exact rollout path. ImplicitActuatorCfg
runs its own implicit PD on top of the USD, so we pass RAW values here (no
180/π compensation — that compensation is only needed when routing gains
through UrdfConverter's joint_drive, which we skip in favor of explicit actuator
configs).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg
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
# parser drift between the two sims on the merged body's anchor. Switching
# merge=False bypasses that drift but creates a worse problem: different
# inertia aggregation between sims.
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


def build_robot_articulation_usd_cfg(usd_path: str) -> ArticulationCfg:
    """Robot cfg from a pre-converted, postprocessed USD."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(usd_path=usd_path),
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


def build_table_rigid_object_usd_cfg(usd_path: str, z: float) -> RigidObjectCfg:
    """Table cfg from a pre-converted, postprocessed USD."""
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=UsdFileCfg(usd_path=usd_path),
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
            # collision/articulation props are baked into the role-specific
            # USDs. Passing them here would send Isaac Lab's apply_nested
            # walker over every template again.
        ),
    )


def build_goal_viz_rigid_object_cfg(usd_paths: list[str]) -> RigidObjectCfg:
    """Kinematic visual twin of Object — gravity + collisions disabled.

    Caller can pass either ``usd_paths[:1]`` (single shared shape, fast
    spawn — preferred for training, where downstream code only reads
    ``goal_viz.data.root_pos_w / root_quat_w``) or the full pool (per-env
    distinct meshes matching Object — useful for debug viz). Spawn cost
    is dominated by the proto-load loop in spawn_multi_asset
    (wrappers.py:73-98), which is O(unique USDs); a single shared shape
    drops GoalViz spawn from ~35 s → ~1 s at 256 envs.

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
            # collision/articulation props are baked into the role-specific
            # USDs. Passing them here would send Isaac Lab's apply_nested
            # walker over every template again.
        ),
    )


# ----------------------------------------------------------------------------
# USD baking and contact materials
# ----------------------------------------------------------------------------


_USD_BAKE_VERSION = "simtoolreal_usd_bake_v6_schema_only"
_CONTACT_OFFSET = 0.002
_REST_OFFSET = 0.0


def _scene_debug_enabled() -> bool:
    return os.environ.get("SIMTOOLREAL_SCENE_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _scene_debug(message: str, *, start_time: float | None = None) -> None:
    if not _scene_debug_enabled():
        return
    prefix = "[scene_utils]"
    if start_time is not None:
        prefix += f"[+{time.perf_counter() - start_time:.2f}s]"
    print(f"{prefix} {message}", flush=True)


def _raw_usd_fingerprint(usd_path: str) -> dict[str, int | str]:
    path = Path(usd_path)
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _bake_spec(raw_usd_path: str, role: str) -> dict:
    """Small sidecar spec for invalidating role-specific baked USDs."""
    return {
        "version": _USD_BAKE_VERSION,
        "role": role,
        "raw_usd": _raw_usd_fingerprint(raw_usd_path),
        "contact_offset": _CONTACT_OFFSET,
        "rest_offset": _REST_OFFSET,
        "materials_set_via_physx_view": True,
    }


def _spec_digest(spec: dict) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _baked_usd_path(raw_usd_path: str, bake_root: Path, role: str) -> Path:
    raw = Path(raw_usd_path)
    asset_key = raw.parent.name
    return bake_root / role / asset_key / raw.name


def _copy_raw_usd_asset(raw_usd_path: str, baked_usd_path: Path) -> None:
    """Copy the converter output USD and any adjacent asset folders."""
    raw = Path(raw_usd_path)
    baked_usd_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw, baked_usd_path)
    # Procedural primitives are self-contained, but copy standard converter
    # side folders if they exist so this helper also works for mesh-backed USDs.
    for child in raw.parent.iterdir():
        if child.name.startswith(".") or child.name == raw.name or child.name == "config.yaml":
            continue
        dst = baked_usd_path.parent / child.name
        if child.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(child, dst)
        elif child.is_file():
            shutil.copy2(child, dst)


def _asset_root_prim(stage):
    root = stage.GetDefaultPrim()
    if root and root.IsValid():
        return root
    for prim in stage.GetPseudoRoot().GetChildren():
        if prim.IsValid():
            return prim
    return None


def _collision_leaf_prims(root_prim) -> list:
    from pxr import Usd, UsdPhysics

    leaves = []
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            leaves.append(prim)
    return leaves


def _uninstance_stage_subtree(root_prim) -> None:
    from pxr import Usd

    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)


def _set_usd_attr(prim, name: str, value, value_type) -> None:
    attr = prim.GetAttribute(name)
    if attr and (not attr.GetTypeName() or not str(attr.GetTypeName())):
        prim.RemoveProperty(name)
        attr = None
    if not attr:
        attr = prim.CreateAttribute(name, value_type, False)
    attr.Set(value)


def _bake_role_schema_props(root_prim, role: str) -> None:
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if role == "robot":
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                _set_usd_attr(prim, "physxRigidBody:disableGravity", True, Sdf.ValueTypeNames.Bool)
                _set_usd_attr(
                    prim,
                    "physxRigidBody:maxDepenetrationVelocity",
                    1000.0,
                    Sdf.ValueTypeNames.Float,
                )
            elif role == "table":
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                _set_usd_attr(prim, "physics:kinematicEnabled", True, Sdf.ValueTypeNames.Bool)
                _set_usd_attr(prim, "physxRigidBody:disableGravity", True, Sdf.ValueTypeNames.Bool)
            elif role == "object":
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                _set_usd_attr(prim, "physics:kinematicEnabled", False, Sdf.ValueTypeNames.Bool)
                _set_usd_attr(prim, "physxRigidBody:disableGravity", False, Sdf.ValueTypeNames.Bool)
                _set_usd_attr(
                    prim,
                    "physxRigidBody:maxDepenetrationVelocity",
                    1000.0,
                    Sdf.ValueTypeNames.Float,
                )
            elif role == "goalviz":
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                _set_usd_attr(prim, "physics:kinematicEnabled", True, Sdf.ValueTypeNames.Bool)
                _set_usd_attr(prim, "physxRigidBody:disableGravity", True, Sdf.ValueTypeNames.Bool)

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            if role == "robot":
                PhysxSchema.PhysxArticulationAPI.Apply(prim)
                _set_usd_attr(
                    prim,
                    "physxArticulation:enabledSelfCollisions",
                    False,
                    Sdf.ValueTypeNames.Bool,
                )
                _set_usd_attr(
                    prim,
                    "physxArticulation:solverPositionIterationCount",
                    8,
                    Sdf.ValueTypeNames.Int,
                )
                _set_usd_attr(
                    prim,
                    "physxArticulation:solverVelocityIterationCount",
                    0,
                    Sdf.ValueTypeNames.Int,
                )
            else:
                _set_usd_attr(prim, "physics:articulationEnabled", False, Sdf.ValueTypeNames.Bool)


def bake_physics_props_into_usd(
    raw_usd_path: str, bake_root: Path, role: str, force: bool = False
) -> str:
    """Create a role-specific USD with static collider props pre-authored.

    This moves leaf-schema writes out of the per-env setup loop. The baked USD
    is invalidated by a small sidecar hash that includes the raw converter
    output fingerprint, role, contact offsets, and bake behavior version.
    """
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    baked_usd_path = _baked_usd_path(raw_usd_path, bake_root, role)
    spec = _bake_spec(raw_usd_path, role)
    digest = _spec_digest(spec)
    sidecar = baked_usd_path.with_suffix(baked_usd_path.suffix + ".simtoolreal_post_hash")
    if (
        not force
        and baked_usd_path.exists()
        and sidecar.exists()
        and sidecar.read_text().strip() == digest
    ):
        return str(baked_usd_path)

    _copy_raw_usd_asset(raw_usd_path, baked_usd_path)
    stage = Usd.Stage.Open(str(baked_usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open baked USD for postprocess: {baked_usd_path}")
    root = _asset_root_prim(stage)
    if root is None:
        raise RuntimeError(f"No root prim found in USD: {baked_usd_path}")

    _uninstance_stage_subtree(root)
    leaves = _collision_leaf_prims(root)
    _bake_role_schema_props(root, role)

    for prim in leaves:
        px = PhysxSchema.PhysxCollisionAPI(prim)
        if not px:
            px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr().Set(_CONTACT_OFFSET)
        px.CreateRestOffsetAttr().Set(_REST_OFFSET)
        if role == "goalviz":
            ce = UsdPhysics.CollisionAPI(prim)
            (ce.GetCollisionEnabledAttr() or ce.CreateCollisionEnabledAttr()).Set(False)

    stage.GetRootLayer().Save()
    sidecar.write_text(digest + "\n")
    return str(baked_usd_path)


def apply_physx_material_properties(env) -> None:
    """Set contact material properties through PhysX tensor views.

    This follows Isaac Lab's large-scale material-randomization path: avoid USD
    relationship authoring after spawn, and avoid creating one material prim per
    cloned asset. It must run after ``DirectRLEnv`` starts the simulator, when
    asset ``root_physx_view`` handles are initialized.
    """
    assets_cfg = env.cfg.assets
    if not assets_cfg.modify_asset_frictions:
        return

    t0 = time.perf_counter()
    default = torch.tensor(
        [float(assets_cfg.robot_friction), float(assets_cfg.robot_friction), 0.0],
        dtype=torch.float32,
        device="cpu",
    )
    fingertip = torch.tensor(
        [float(assets_cfg.finger_tip_friction), float(assets_cfg.finger_tip_friction), 0.0],
        dtype=torch.float32,
        device="cpu",
    )
    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device="cpu")

    robot_view = env.robot.root_physx_view
    robot_materials = robot_view.get_material_properties()
    robot_materials[:] = default

    shape_start = 0
    for link_name, link_path in zip(robot_view.shared_metatype.link_names, robot_view.link_paths[0]):
        link_view = env.robot._physics_sim_view.create_rigid_body_view(link_path)
        shape_end = shape_start + link_view.max_shapes
        if link_name in FINGERTIP_LINK_NAMES:
            robot_materials[:, shape_start:shape_end] = fingertip
        shape_start = shape_end
    if shape_start != robot_view.max_shapes:
        raise RuntimeError(
            f"Robot shape count mismatch while assigning materials: "
            f"computed {shape_start}, view reports {robot_view.max_shapes}."
        )
    robot_view.set_material_properties(robot_materials, env_ids)

    for name in ("table", "object", "goal_viz"):
        asset = getattr(env, name)
        view = asset.root_physx_view
        materials = view.get_material_properties()
        materials[:] = default
        view.set_material_properties(materials, env_ids)

    _scene_debug(
        f"applied PhysX material properties in {time.perf_counter() - t0:.2f}s",
    )


def _resolve_usd_cache_root() -> Path:
    cache_root_env = os.environ.get("SIMTOOLREAL_CACHE_ROOT")
    if cache_root_env:
        return Path(cache_root_env)
    if Path("/scratch").is_dir() and os.access("/scratch", os.W_OK):
        return Path("/scratch") / "simtoolreal_assets" / "v1"
    return Path.home() / ".cache" / "simtoolreal_assets" / "v1"


def _convert_urdf_to_usd(
    asset_path: str,
    usd_cache_root: Path,
    *,
    force_rebuild: bool,
    fix_base: bool,
    merge_fixed_joints: bool = True,
    make_instanceable: bool = False,
    self_collision: bool | None = None,
    joint_drive=None,
) -> str:
    cfg_kwargs = {
        "asset_path": asset_path,
        "usd_dir": str(usd_cache_root / Path(asset_path).stem),
        "force_usd_conversion": force_rebuild,
        "fix_base": fix_base,
        "merge_fixed_joints": merge_fixed_joints,
        "make_instanceable": make_instanceable,
        "joint_drive": joint_drive,
    }
    if self_collision is not None:
        cfg_kwargs["self_collision"] = self_collision
    return UrdfConverter(UrdfConverterCfg(**cfg_kwargs)).usd_path


def _robot_joint_drive_cfg():
    # The DriveAPI prims must exist for ImplicitActuator runtime gains to land.
    return UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0.0,
            damping=0.0,
        ),
    )


def _convert_object_urdfs_to_usds(
    urdf_paths: list[str],
    usd_cache_root: Path,
    *,
    force_rebuild: bool,
) -> list[str]:
    return [
        _convert_urdf_to_usd(
            urdf,
            usd_cache_root,
            force_rebuild=force_rebuild,
            fix_base=False,
            joint_drive=None,
        )
        for urdf in urdf_paths
    ]


def _materialize_env_prims(env) -> None:
    stage = get_current_stage()
    for env_path in env.scene.env_prim_paths:
        if not stage.GetPrimAtPath(env_path).IsValid():
            stage.DefinePrim(env_path, "Xform")


def _build_object_scale_tensor(env, object_scales_normalized, num_object_usds: int) -> None:
    num_envs = env.num_envs
    object_prim_paths = find_matching_prim_paths("/World/envs/env_.*/Object")
    if len(object_prim_paths) != num_envs:
        raise RuntimeError(
            f"Expected {num_envs} Object prims after MultiUsdFileCfg spawn, "
            f"got {len(object_prim_paths)}. Cloner-drop bug may have returned."
        )

    env._object_scale_per_env = torch.zeros(
        num_envs, 3, device=env.device, dtype=torch.float32
    )
    env._object_asset_index_per_env = torch.zeros(
        num_envs, device=env.device, dtype=torch.long
    )
    for source_idx, obj_path in enumerate(object_prim_paths):
        env_segment = obj_path.rsplit("/", 2)[-2]  # ".../env_K/Object" -> "env_K"
        env_id = int(env_segment.removeprefix("env_"))
        asset_index = source_idx % num_object_usds
        env._object_scale_per_env[env_id] = torch.tensor(
            object_scales_normalized[asset_index],
            device=env.device,
            dtype=torch.float32,
        )
        env._object_asset_index_per_env[env_id] = asset_index


# ----------------------------------------------------------------------------
# Scene orchestrator (Phase B)
# ----------------------------------------------------------------------------


def setup_scene(env) -> None:
    """Top-level ``_setup_scene`` body — spawns robot, table, per-env
    distinct-USD object + goal-viz, ground plane + lighting.

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
    force_rebuild = bool(getattr(assets_cfg, "rebuild_assets", False))
    setup_t0 = time.perf_counter()
    _scene_debug(
        f"setup start num_envs={env.num_envs} num_assets_per_type={assets_cfg.num_assets_per_type} "
        f"force_rebuild={force_rebuild}",
        start_time=setup_t0,
    )

    # 1. Generate procedural URDFs in a per-launch temp dir.
    env._tmp_asset_dir = tempfile.mkdtemp(prefix="simtoolreal_assets_")
    urdf_paths, object_scales_normalized = generate_handle_head_urdfs(
        handle_head_types=tuple(assets_cfg.handle_head_types),
        num_per_type=assets_cfg.num_assets_per_type,
        out_dir=env._tmp_asset_dir,
        shuffle=assets_cfg.shuffle_assets,
    )
    env._object_urdf_paths = [str(path) for path in urdf_paths]
    _scene_debug(
        f"generated {len(urdf_paths)} object URDFs in {env._tmp_asset_dir}",
        start_time=setup_t0,
    )
    if not urdf_paths:
        raise ValueError(
            "No procedural object URDFs were generated. "
            "Check cfg.assets.handle_head_types and num_assets_per_type."
        )

    # 2. Convert/cache USDs, then bake static schema props once per asset.
    usd_cache_root = _resolve_usd_cache_root()
    usd_cache_root.mkdir(parents=True, exist_ok=True)
    env._usd_cache_root = str(usd_cache_root)
    _scene_debug(f"using USD cache root {usd_cache_root}", start_time=setup_t0)
    usd_paths = _convert_object_urdfs_to_usds(
        urdf_paths,
        usd_cache_root,
        force_rebuild=force_rebuild,
    )
    _scene_debug(f"resolved {len(usd_paths)} object USDs", start_time=setup_t0)
    baked_cache_root = usd_cache_root / "_baked"
    object_usd_paths = [
        bake_physics_props_into_usd(
            raw_usd_path=usd,
            bake_root=baked_cache_root,
            role="object",
            force=force_rebuild,
        )
        for usd in usd_paths
    ]
    _scene_debug(f"resolved {len(object_usd_paths)} baked object USDs", start_time=setup_t0)
    goalviz_usd_paths = [
        bake_physics_props_into_usd(
            raw_usd_path=usd_paths[0],
            bake_root=baked_cache_root,
            role="goalviz",
            force=force_rebuild,
        )
    ]
    _scene_debug(f"resolved {len(goalviz_usd_paths)} baked goalviz USDs", start_time=setup_t0)

    # 3. Pre-create env roots so regex spawns resolve to every env.
    _materialize_env_prims(env)
    _scene_debug(
        f"materialized {len(env.scene.env_prim_paths)} env prims",
        start_time=setup_t0,
    )

    # 4. Robot/table use the same convert -> bake -> spawn path.
    robot_raw_usd = _convert_urdf_to_usd(
        assets_cfg.robot_urdf,
        usd_cache_root,
        force_rebuild=force_rebuild,
        fix_base=True,
        self_collision=False,
        joint_drive=_robot_joint_drive_cfg(),
    )
    table_raw_usd = _convert_urdf_to_usd(
        assets_cfg.table_urdf,
        usd_cache_root,
        force_rebuild=force_rebuild,
        fix_base=False,
        joint_drive=None,
    )
    _scene_debug("resolved robot/table raw USDs", start_time=setup_t0)
    robot_usd_path = bake_physics_props_into_usd(
        raw_usd_path=robot_raw_usd,
        bake_root=baked_cache_root,
        role="robot",
        force=force_rebuild,
    )
    table_usd_path = bake_physics_props_into_usd(
        raw_usd_path=table_raw_usd,
        bake_root=baked_cache_root,
        role="table",
        force=force_rebuild,
    )
    _scene_debug("resolved robot/table baked USDs", start_time=setup_t0)
    env.robot = Articulation(
        build_robot_articulation_usd_cfg(robot_usd_path)
    )
    env.table = RigidObject(
        build_table_rigid_object_usd_cfg(table_usd_path, z=env.cfg.reset.table_reset_z)
    )
    _scene_debug("spawned robot and table", start_time=setup_t0)

    # 5. Objects cycle through all baked USDs; GoalViz uses one shared shape.
    env.object = RigidObject(build_object_rigid_object_cfg(object_usd_paths))
    env.goal_viz = RigidObject(build_goal_viz_rigid_object_cfg(goalviz_usd_paths))
    _scene_debug("spawned object and goalviz", start_time=setup_t0)

    # 6. Ground plane + dome light (global, outside env_*).
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    _scene_debug("spawned ground and light", start_time=setup_t0)

    # 7. Store the procedural scale attached to each spawned Object.
    _build_object_scale_tensor(env, object_scales_normalized, len(usd_paths))
    _scene_debug("built per-env object scale tensor", start_time=setup_t0)

    # 8. Register with scene so DirectRLEnv refreshes their tensors each step.
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    _scene_debug("registered assets with scene", start_time=setup_t0)

    # Static leaf props are baked above; friction is set through PhysX views
    # after DirectRLEnv starts the simulator.


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
    "build_robot_articulation_usd_cfg",
    "build_table_rigid_object_usd_cfg",
    "build_object_rigid_object_cfg",
    "build_goal_viz_rigid_object_cfg",
    "apply_physx_material_properties",
    "setup_scene",
]

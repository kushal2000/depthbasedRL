"""
Isaac Sim environment for policy rollout.
Handles scene setup (robot, table, object), physics config, state extraction.

Phases 4-6 of the plan:
  - URDF → USD conversion
  - Scene setup with correct physics params
  - State extraction with quaternion conversion
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


# DOF drive properties from isaacgymenvs/tasks/simtoolreal/utils.py
KUKA_STIFFNESSES = [600, 600, 500, 400, 200, 200, 200]
KUKA_DAMPINGS = [
    27.027026473513512, 27.027026473513512, 24.672186769721083,
    22.067474708266914, 9.752538131173853, 9.147747263670984, 9.147747263670984,
]
KUKA_EFFORTS = [300, 300, 300, 300, 300, 300, 300]

HAND_STIFFNESSES = [
    6.95, 13.2, 4.76, 6.62, 0.9,
    4.76, 6.62, 0.9, 0.9,
    4.76, 6.62, 0.9, 0.9,
    4.76, 6.62, 0.9, 0.9,
    1.38, 4.76, 6.62, 0.9, 0.9,
]
HAND_DAMPINGS = [
    0.28676845, 0.40845109, 0.20394083, 0.24044435, 0.04190723,
    0.20859232, 0.24595532, 0.04243185, 0.03504461,
    0.2085923, 0.24595532, 0.04243185, 0.03504461,
    0.20859226, 0.24595528, 0.04243183, 0.0350446,
    0.02782345, 0.20859229, 0.24595528, 0.04243183, 0.0350446,
]

ALL_STIFFNESSES = KUKA_STIFFNESSES + HAND_STIFFNESSES
ALL_DAMPINGS = KUKA_DAMPINGS + HAND_DAMPINGS
assert len(ALL_STIFFNESSES) == 29
assert len(ALL_DAMPINGS) == 29


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert Isaac Sim quaternion (w,x,y,z) to Isaac Gym convention (x,y,z,w)."""
    return quat_wxyz[[1, 2, 3, 0]]


def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert Isaac Gym quaternion (x,y,z,w) to Isaac Sim convention (w,x,y,z)."""
    return quat_xyzw[[3, 0, 1, 2]]


class IsaacSimEnv:
    def __init__(
        self,
        robot_urdf: str,
        table_urdf: str,
        object_urdf: str,
        headless: bool = True,
    ):
        # Must import after SimulationApp is created
        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": headless})

        from omni.isaac.core import World
        from omni.isaac.core.utils.extensions import enable_extension

        # Physics params matching SimToolReal.yaml
        self.world = World(
            physics_dt=1 / 60,
            rendering_dt=1 / 60,
            stage_units_in_meters=1.0,
        )
        print("World created", flush=True)

        # URDF import
        enable_extension("omni.importer.urdf")
        from omni.importer.urdf import _urdf
        self._urdf_interface = _urdf
        print("URDF importer ready")

        # Import assets
        self.robot_prim_path = self._import_robot(robot_urdf)
        self.table_prim_path = self._import_table(table_urdf)
        self.object_prim_path = self._import_object(object_urdf)

        # Set physics params after scene is populated
        self._set_physics_params()

        # Set up articulations after world reset
        self._setup_robot()
        self._set_friction_properties()

        # Joint ordering validation
        self.permutation = self._validate_joint_ordering()

    def _set_physics_params(self):
        """Set PhysX scene params to match SimToolReal.yaml."""
        from pxr import UsdPhysics, PhysxSchema, Gf
        stage = self.world.stage

        # Find the physics scene
        for prim in stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                physx_scene = PhysxSchema.PhysxSceneAPI(prim)
                physx_scene.GetSolverTypeAttr().Set("TGS")
                physx_scene.GetEnableStabilizationAttr().Set(True)
                physx_scene.GetBounceThresholdAttr().Set(0.2)

                scene_api = UsdPhysics.Scene(prim)
                scene_api.GetGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
                scene_api.GetGravityMagnitudeAttr().Set(9.81)
                print(f"Physics scene configured: TGS solver, gravity=-9.81")
                break

    def _import_robot(self, urdf_path: str) -> str:
        """Import robot URDF, place at (0, 0.8, 0)."""
        _urdf = self._urdf_interface

        config = _urdf.ImportConfig()
        config.merge_fixed_joints = False
        config.fix_base = True
        config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        config.self_collision = False

        prim_path = "/World/Robot"
        result = _urdf.import_robot(urdf_path, prim_path, config)
        print(f"Robot imported to {prim_path}")

        # Set position to (0, 0.8, 0)
        from pxr import UsdGeom, Gf
        stage = self.world.stage
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.8, 0.0))

        return prim_path

    def _import_table(self, urdf_path: str) -> str:
        """Import table/scene URDF at (0, 0, 0.38)."""
        _urdf = self._urdf_interface

        config = _urdf.ImportConfig()
        config.merge_fixed_joints = True
        config.fix_base = True
        config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_NONE

        prim_path = "/World/Table"
        result = _urdf.import_robot(urdf_path, prim_path, config)
        print(f"Table imported to {prim_path}")

        from pxr import UsdGeom, Gf
        stage = self.world.stage
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.38))

        return prim_path

    def _import_object(self, urdf_path: str) -> str:
        """Import manipulation object (dynamic, not fixed)."""
        _urdf = self._urdf_interface

        config = _urdf.ImportConfig()
        config.merge_fixed_joints = True
        config.fix_base = False
        config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_NONE

        prim_path = "/World/Object"
        result = _urdf.import_robot(urdf_path, prim_path, config)
        print(f"Object imported to {prim_path}")
        return prim_path

    def _setup_robot(self):
        """Set up robot articulation and DOF drive properties."""
        from omni.isaac.core.articulations import Articulation

        self.robot = self.world.scene.add(
            Articulation(prim_path=self.robot_prim_path, name="robot")
        )
        self.world.reset()  # Initialize articulation

        # Set drive properties
        num_dofs = self.robot.num_dof
        print(f"Robot has {num_dofs} DOFs")
        assert num_dofs == 29, f"Expected 29 DOFs, got {num_dofs}"

        stiffnesses = np.array(ALL_STIFFNESSES, dtype=np.float32)
        dampings = np.array(ALL_DAMPINGS, dtype=np.float32)

        self.robot.set_gains(
            kps=stiffnesses,
            kds=dampings,
        )

        # Verify by reading back
        kps, kds = self.robot.get_gains()
        print(f"Drive stiffness set: {kps[:7]} (arm), {kps[7:12]} (thumb)")
        print(f"Drive damping set:   {kds[:7]} (arm), {kds[7:12]} (thumb)")

    def _set_friction_properties(self):
        """Set friction: robot=0.5, fingertips=1.5, object=0.5, table=0.5."""
        # TODO: Set per-link friction via PhysxMaterialAPI
        # For now, rely on URDF defaults
        print("WARNING: Per-link friction not yet set (using URDF defaults)")

    def _validate_joint_ordering(self) -> Optional[np.ndarray]:
        """Compare Isaac Sim joint names to JOINT_NAMES_ISAACGYM. Return permutation if needed."""
        from isaacgymenvs.utils.observation_action_utils_sharpa import JOINT_NAMES_ISAACGYM

        sim_names = list(self.robot.dof_names)
        print(f"\nJoint ordering validation:")
        print(f"  Isaac Sim:  {sim_names}")
        print(f"  Isaac Gym:  {JOINT_NAMES_ISAACGYM}")

        if sim_names == JOINT_NAMES_ISAACGYM:
            print("  -> MATCH! No permutation needed.")
            return None
        else:
            print("  -> MISMATCH! Computing permutation...")
            permutation = np.array(
                [sim_names.index(name) for name in JOINT_NAMES_ISAACGYM]
            )
            print(f"  -> Permutation: {permutation}")
            return permutation

    def set_object_pose(self, pos_xyz: np.ndarray, quat_xyzw: np.ndarray):
        """Set object world pose. Takes xyzw quaternion (Isaac Gym convention)."""
        from omni.isaac.core.prims import RigidPrim
        obj = RigidPrim(prim_path=self.object_prim_path)
        quat_wxyz = xyzw_to_wxyz(quat_xyzw)
        obj.set_world_pose(position=pos_xyz, orientation=quat_wxyz)

    def get_robot_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get joint positions and velocities in JOINT_NAMES_ISAACGYM order."""
        q = self.robot.get_joint_positions()
        qd = self.robot.get_joint_velocities()
        if self.permutation is not None:
            q = q[self.permutation]
            qd = qd[self.permutation]
        return q, qd

    def get_object_pose_xyzw(self) -> np.ndarray:
        """Get object world pose as (7,) array: [x, y, z, qx, qy, qz, qw]."""
        from omni.isaac.core.prims import RigidPrim
        obj = RigidPrim(prim_path=self.object_prim_path)
        pos, quat_wxyz = obj.get_world_pose()
        quat_xyzw = wxyz_to_xyzw(quat_wxyz)
        return np.concatenate([pos, quat_xyzw])

    def set_joint_position_targets(self, targets: np.ndarray):
        """Set joint position targets. Takes targets in JOINT_NAMES_ISAACGYM order."""
        if self.permutation is not None:
            reordered = np.zeros_like(targets)
            for i, p in enumerate(self.permutation):
                reordered[p] = targets[i]
            targets = reordered
        self.robot.set_joint_position_targets(targets)

    def step(self, render: bool = True):
        self.world.step(render=render)

    def reset(self):
        self.world.reset()

    def close(self):
        self.app.close()


def run_phase5_test():
    """Phase 5: Robot-only scene stability test."""
    repo_root = Path(__file__).parent.parent
    robot_urdf = str(repo_root / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf")
    table_urdf = str(repo_root / "assets/urdf/fabrica/environments/beam_2/scene_coacd.urdf")

    # Find object URDF
    from dextoolbench.objects import NAME_TO_OBJECT
    import fabrica.objects  # noqa: registers fabrica parts
    obj_info = NAME_TO_OBJECT.get("beam_2_coacd")
    if obj_info:
        object_urdf = obj_info.urdf_path
    else:
        # Fallback: use a simple cube
        object_urdf = str(repo_root / "assets/urdf/objects/cube_multicolor.urdf")

    print(f"Robot URDF:  {robot_urdf}")
    print(f"Table URDF:  {table_urdf}")
    print(f"Object URDF: {object_urdf}")

    env = IsaacSimEnv(
        robot_urdf=robot_urdf,
        table_urdf=table_urdf,
        object_urdf=object_urdf,
        headless=True,
    )

    # Stability test (Phase 5e)
    print("\n--- Stability test: robot should hold default pose for 5s ---")
    q_initial = env.get_robot_state()[0].copy()
    for i in range(300):  # 5 seconds at 60Hz
        env.step(render=False)
    q_final = env.get_robot_state()[0]
    drift = np.abs(q_final - q_initial).max()
    print(f"Max joint drift after 5s: {drift:.6f} rad")
    if drift < 0.01:
        print("PASS: Robot holds pose stably")
    else:
        print(f"WARN: Drift {drift:.4f} > 0.01 — check drive properties")

    # Object stability test (Phase 6d)
    print("\n--- Object stability test ---")
    obj_pose = env.get_object_pose_xyzw()
    print(f"Object pose: {obj_pose}")
    for i in range(120):  # 2 seconds
        env.step(render=False)
    obj_pose_after = env.get_object_pose_xyzw()
    print(f"Object pose after 2s: {obj_pose_after}")
    if obj_pose_after[2] > 0.3:
        print(f"PASS: Object stable at z={obj_pose_after[2]:.3f}")
    else:
        print(f"FAIL: Object fell! z={obj_pose_after[2]:.3f}")

    # Manual action test (Phase 8d)
    print("\n--- Manual action test: move arm joint 1 by 0.1 rad ---")
    q0 = env.get_robot_state()[0].copy()
    test_targets = q0.copy()
    test_targets[0] += 0.1
    env.set_joint_position_targets(test_targets)
    for _ in range(60):
        env.step(render=False)
    q_after = env.get_robot_state()[0]
    error = abs(q_after[0] - test_targets[0])
    print(f"Commanded: {test_targets[0]:.3f}, Achieved: {q_after[0]:.3f}, Error: {error:.4f}")
    if error < 0.01:
        print("PASS: Joint tracking OK")
    else:
        print(f"WARN: Tracking error {error:.4f} > 0.01")

    env.close()
    print("\n=== Phase 5-6 tests complete ===")


if __name__ == "__main__":
    run_phase5_test()

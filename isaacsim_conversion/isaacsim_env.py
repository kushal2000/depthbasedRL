"""
Isaac Sim environment for policy rollout, using Isaac Lab APIs.
Handles URDF→USD conversion, scene setup, physics config, state extraction.

Uses isaaclab.sim.converters.UrdfConverter for URDF import with per-joint PD gains,
and isaaclab.sim.SimulationContext for physics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from isaacgymenvs.utils.observation_action_utils_sharpa import JOINT_NAMES_ISAACGYM


# DOF drive properties from isaacgymenvs/tasks/simtoolreal/utils.py
# These are the exact values used during training in Isaac Gym.
JOINT_STIFFNESSES = {
    # Arm (7 DOF)
    "iiwa14_joint_1": 600, "iiwa14_joint_2": 600, "iiwa14_joint_3": 500,
    "iiwa14_joint_4": 400, "iiwa14_joint_5": 200, "iiwa14_joint_6": 200,
    "iiwa14_joint_7": 200,
    # Hand — thumb (5 DOF)
    "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2,
    "left_thumb_MCP_FE": 4.76, "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
    # Hand — index (4 DOF)
    "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
    "left_index_PIP": 0.9, "left_index_DIP": 0.9,
    # Hand — middle (4 DOF)
    "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
    "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
    # Hand — ring (4 DOF)
    "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
    "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
    # Hand — pinky (5 DOF)
    "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76,
    "left_pinky_MCP_AA": 6.62, "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
}

JOINT_DAMPINGS = {
    # Arm
    "iiwa14_joint_1": 27.027026473513512, "iiwa14_joint_2": 27.027026473513512,
    "iiwa14_joint_3": 24.672186769721083, "iiwa14_joint_4": 22.067474708266914,
    "iiwa14_joint_5": 9.752538131173853, "iiwa14_joint_6": 9.147747263670984,
    "iiwa14_joint_7": 9.147747263670984,
    # Hand — thumb
    "left_1_thumb_CMC_FE": 0.28676845, "left_thumb_CMC_AA": 0.40845109,
    "left_thumb_MCP_FE": 0.20394083, "left_thumb_MCP_AA": 0.24044435,
    "left_thumb_IP": 0.04190723,
    # Hand — index
    "left_2_index_MCP_FE": 0.20859232, "left_index_MCP_AA": 0.24595532,
    "left_index_PIP": 0.04243185, "left_index_DIP": 0.03504461,
    # Hand — middle
    "left_3_middle_MCP_FE": 0.2085923, "left_middle_MCP_AA": 0.24595532,
    "left_middle_PIP": 0.04243185, "left_middle_DIP": 0.03504461,
    # Hand — ring
    "left_4_ring_MCP_FE": 0.20859226, "left_ring_MCP_AA": 0.24595528,
    "left_ring_PIP": 0.04243183, "left_ring_DIP": 0.0350446,
    # Hand — pinky
    "left_5_pinky_CMC": 0.02782345, "left_pinky_MCP_FE": 0.20859229,
    "left_pinky_MCP_AA": 0.24595528, "left_pinky_PIP": 0.04243183,
    "left_pinky_DIP": 0.0350446,
}

assert len(JOINT_STIFFNESSES) == 29
assert len(JOINT_DAMPINGS) == 29


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert Isaac Sim quaternion (w,x,y,z) to Isaac Gym convention (x,y,z,w)."""
    return quat_wxyz[[1, 2, 3, 0]]


def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert Isaac Gym quaternion (x,y,z,w) to Isaac Sim convention (w,x,y,z)."""
    return quat_xyzw[[3, 0, 1, 2]]


def _log(msg: str):
    print(msg, flush=True)


class IsaacSimEnv:
    def __init__(
        self,
        robot_urdf: str,
        table_urdf: str,
        object_urdf: str,
        headless: bool = True,
        usd_cache_dir: str | None = None,
    ):
        """Create Isaac Sim environment with robot, table, and object.

        Args:
            robot_urdf: Path to robot URDF.
            table_urdf: Path to table/scene URDF.
            object_urdf: Path to manipulation object URDF.
            headless: Run without display.
            usd_cache_dir: Directory to cache converted USD files. If None, uses /tmp.
        """
        # Launch Isaac Sim
        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": headless})
        _log("SimulationApp created")

        # Isaac Lab imports (must be after SimulationApp)
        import isaaclab.sim as sim_utils
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        self._usd_cache_dir = usd_cache_dir or "/tmp/isaaclab_usd_cache"

        # Create simulation context with physics params matching SimToolReal.yaml
        sim_cfg = SimulationCfg(
            dt=1 / 60,
            render_interval=1,
            gravity=(0.0, 0.0, -9.81),
            physx=SimulationCfg.PhysxCfg(
                solver_type=1,  # TGS
                bounce_threshold_velocity=0.2,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        _log("SimulationContext created")

        # Convert and spawn assets
        self._robot_usd = self._convert_robot_urdf(robot_urdf, UrdfConverterCfg, UrdfConverter)
        self._table_usd = self._convert_table_urdf(table_urdf, UrdfConverterCfg, UrdfConverter)
        self._object_usd = self._convert_object_urdf(object_urdf, UrdfConverterCfg, UrdfConverter)

        # Spawn into scene
        self._spawn_scene(sim_utils)

        # Reset to initialize physics
        self.sim.reset()
        _log("Simulation reset complete")

        # Set up articulation handle
        self._setup_articulations()

        # Validate joint ordering
        self.permutation = self._validate_joint_ordering()

    def _convert_robot_urdf(self, urdf_path, UrdfConverterCfg, UrdfConverter):
        """Convert robot URDF to USD with per-joint PD gains."""
        _log(f"Converting robot URDF: {urdf_path}")
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=f"{self._usd_cache_dir}/robot",
            fix_base=True,
            merge_fixed_joints=False,
            self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=JOINT_STIFFNESSES,
                    damping=JOINT_DAMPINGS,
                ),
            ),
        )
        converter = UrdfConverter(cfg)
        _log(f"Robot USD: {converter.usd_path}")
        return converter.usd_path

    def _convert_table_urdf(self, urdf_path, UrdfConverterCfg, UrdfConverter):
        """Convert table URDF to USD (static, fixed base)."""
        _log(f"Converting table URDF: {urdf_path}")
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=f"{self._usd_cache_dir}/table",
            fix_base=True,
            merge_fixed_joints=True,
            joint_drive=None,
        )
        converter = UrdfConverter(cfg)
        _log(f"Table USD: {converter.usd_path}")
        return converter.usd_path

    def _convert_object_urdf(self, urdf_path, UrdfConverterCfg, UrdfConverter):
        """Convert object URDF to USD (dynamic, not fixed)."""
        _log(f"Converting object URDF: {urdf_path}")
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=f"{self._usd_cache_dir}/object",
            fix_base=False,
            merge_fixed_joints=True,
            joint_drive=None,
        )
        converter = UrdfConverter(cfg)
        _log(f"Object USD: {converter.usd_path}")
        return converter.usd_path

    def _spawn_scene(self, sim_utils):
        """Spawn robot, table, and object into the USD stage."""
        from pxr import UsdGeom, Gf
        import omni.usd

        stage = omni.usd.get_context().get_stage()

        # Spawn robot at (0, 0.8, 0)
        robot_prim_path = "/World/Robot"
        sim_utils.spawners.from_files.spawn_from_usd(
            robot_prim_path, sim_utils.spawners.from_files.UsdFileCfg(usd_path=self._robot_usd)
        )
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(robot_prim_path))
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.8, 0.0))
        self.robot_prim_path = robot_prim_path
        _log(f"Robot spawned at {robot_prim_path}")

        # Spawn table at (0, 0, 0.38)
        table_prim_path = "/World/Table"
        sim_utils.spawners.from_files.spawn_from_usd(
            table_prim_path, sim_utils.spawners.from_files.UsdFileCfg(usd_path=self._table_usd)
        )
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(table_prim_path))
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.38))
        self.table_prim_path = table_prim_path
        _log(f"Table spawned at {table_prim_path}")

        # Spawn object (position set later via set_object_pose)
        object_prim_path = "/World/Object"
        sim_utils.spawners.from_files.spawn_from_usd(
            object_prim_path, sim_utils.spawners.from_files.UsdFileCfg(usd_path=self._object_usd)
        )
        self.object_prim_path = object_prim_path
        _log(f"Object spawned at {object_prim_path}")

    def _setup_articulations(self):
        """Set up articulation views for robot and object."""
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.prims import RigidPrim

        self.robot = Articulation(prim_path=self.robot_prim_path)
        self.robot.initialize()

        num_dofs = self.robot.num_dof
        _log(f"Robot has {num_dofs} DOFs")
        assert num_dofs == 29, f"Expected 29 DOFs, got {num_dofs}"

        self.object_prim = RigidPrim(prim_path=self.object_prim_path)
        self.object_prim.initialize()

    def _validate_joint_ordering(self) -> Optional[np.ndarray]:
        """Compare Isaac Sim joint names to JOINT_NAMES_ISAACGYM. Return permutation if needed."""
        sim_names = list(self.robot.dof_names)
        _log(f"Joint ordering validation:")
        _log(f"  Isaac Sim:  {sim_names}")
        _log(f"  Isaac Gym:  {JOINT_NAMES_ISAACGYM}")

        if sim_names == JOINT_NAMES_ISAACGYM:
            _log("  -> MATCH! No permutation needed.")
            return None
        else:
            _log("  -> MISMATCH! Computing permutation...")
            permutation = np.array(
                [sim_names.index(name) for name in JOINT_NAMES_ISAACGYM]
            )
            _log(f"  -> Permutation: {permutation}")
            return permutation

    def set_object_pose(self, pos_xyz: np.ndarray, quat_xyzw: np.ndarray):
        """Set object world pose. Takes xyzw quaternion (Isaac Gym convention)."""
        quat_wxyz = xyzw_to_wxyz(quat_xyzw)
        self.object_prim.set_world_pose(position=pos_xyz, orientation=quat_wxyz)

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
        pos, quat_wxyz = self.object_prim.get_world_pose()
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
        self.sim.step(render=render)

    def close(self):
        self.sim.stop()
        self.app.close()


def run_phase5_test():
    """Phase 5-6: Scene setup and stability tests."""
    repo_root = Path(__file__).parent.parent
    robot_urdf = str(repo_root / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf")
    table_urdf = str(repo_root / "assets/urdf/fabrica/environments/beam_2/scene_coacd.urdf")

    # Find object URDF
    sys.path.insert(0, str(repo_root))
    import fabrica.objects  # noqa: registers fabrica parts
    from dextoolbench.objects import NAME_TO_OBJECT
    obj_info = NAME_TO_OBJECT.get("beam_2_coacd")
    if obj_info:
        object_urdf = obj_info.urdf_path
    else:
        object_urdf = str(repo_root / "assets/urdf/objects/cube_multicolor.urdf")

    _log(f"Robot URDF:  {robot_urdf}")
    _log(f"Table URDF:  {table_urdf}")
    _log(f"Object URDF: {object_urdf}")

    env = IsaacSimEnv(
        robot_urdf=robot_urdf,
        table_urdf=table_urdf,
        object_urdf=object_urdf,
        headless=True,
    )

    # Stability test (Phase 5e)
    _log("\n--- Stability test: robot should hold default pose for 5s ---")
    q_initial = env.get_robot_state()[0].copy()
    for i in range(300):
        env.step(render=False)
    q_final = env.get_robot_state()[0]
    drift = np.abs(q_final - q_initial).max()
    _log(f"Max joint drift after 5s: {drift:.6f} rad")
    _log("PASS: Robot holds pose stably" if drift < 0.01 else f"WARN: Drift {drift:.4f} > 0.01")

    # Object stability test (Phase 6d)
    _log("\n--- Object stability test ---")
    obj_pose = env.get_object_pose_xyzw()
    _log(f"Object pose: {obj_pose}")
    for i in range(120):
        env.step(render=False)
    obj_pose_after = env.get_object_pose_xyzw()
    _log(f"Object pose after 2s: {obj_pose_after}")
    _log(f"PASS: Object stable at z={obj_pose_after[2]:.3f}" if obj_pose_after[2] > 0.3
         else f"FAIL: Object fell! z={obj_pose_after[2]:.3f}")

    # Manual action test (Phase 8d)
    _log("\n--- Manual action test: move arm joint 1 by 0.1 rad ---")
    q0 = env.get_robot_state()[0].copy()
    test_targets = q0.copy()
    test_targets[0] += 0.1
    env.set_joint_position_targets(test_targets)
    for _ in range(60):
        env.step(render=False)
    q_after = env.get_robot_state()[0]
    error = abs(q_after[0] - test_targets[0])
    _log(f"Commanded: {test_targets[0]:.3f}, Achieved: {q_after[0]:.3f}, Error: {error:.4f}")
    _log("PASS: Joint tracking OK" if error < 0.01 else f"WARN: Tracking error {error:.4f} > 0.01")

    env.close()
    _log("\n=== Phase 5-6 tests complete ===")


if __name__ == "__main__":
    run_phase5_test()

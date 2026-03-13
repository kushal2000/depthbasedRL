"""Debug script: manually teleport a part along its insertion trajectory.

Spawns an IsaacGym env in a subprocess (reusing eval_assembly's pattern),
then teleports the object step-by-step along the insertion waypoints.
Reports per-step desired vs actual pose to diagnose V-HACD collision issues.

Usage:
    python fabrica/debug_insertion.py \
        --config-path pretrained_policy/config.yaml \
        --checkpoint-path pretrained_policy/model.pth \
        --assembly beam --part 2 \
        --steps-per-waypoint 50 \
        --start-waypoint 8 \
        --port 8080
"""

import argparse
import json
import math
import multiprocessing
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

from fabrica.viser_utils import COLORS

sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

# ===================================================================
# Constants (same as eval_assembly)
# ===================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_Z = 0.38
Z_OFFSET = 0.03
N_ACT = 29
CONTROL_DT = 1.0 / 60.0

ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"

_ARM_DEFAULT = np.array([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308])
_ARM_DEFAULT[1] -= np.deg2rad(10)
_ARM_DEFAULT[3] += np.deg2rad(10)
DEFAULT_DOF_POS = np.zeros(N_ACT)
DEFAULT_DOF_POS[:7] = _ARM_DEFAULT

BASE_OVERRIDES = {
    "task.env.resetPositionNoiseX": 0.0,
    "task.env.resetPositionNoiseY": 0.0,
    "task.env.resetPositionNoiseZ": 0.0,
    "task.env.randomizeObjectRotation": False,
    "task.env.resetDofPosRandomIntervalFingers": 0.0,
    "task.env.resetDofPosRandomIntervalArm": 0.0,
    "task.env.resetDofVelRandomInterval": 0.0,
    "task.env.tableResetZRange": 0.0,
    "task.env.numEnvs": 1,
    "task.env.envSpacing": 0.4,
    "task.env.capture_video": False,
    "task.env.useActionDelay": False,
    "task.env.useObsDelay": False,
    "task.env.useObjectStateDelayNoise": False,
    "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
    "task.env.resetWhenDropped": False,
    "task.env.armMovingAverage": 0.1,
    "task.env.evalSuccessTolerance": 0.01,
    "task.env.successSteps": 1,
    "task.env.fixedSizeKeypointReward": True,
    "task.env.useFixedInitObjectPose": True,
    "task.env.startArmHigher": True,
    "task.env.forceScale": 0.0,
    "task.env.torqueScale": 0.0,
    "task.env.linVelImpulseScale": 0.0,
    "task.env.angVelImpulseScale": 0.0,
    "task.env.forceOnlyWhenLifted": True,
    "task.env.torqueOnlyWhenLifted": True,
    "task.env.linVelImpulseOnlyWhenLifted": True,
    "task.env.angVelImpulseOnlyWhenLifted": True,
    "task.env.forceProbRange": [0.0001, 0.0001],
    "task.env.torqueProbRange": [0.0001, 0.0001],
    "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
    "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
}


# ===================================================================
# Helpers (from eval_assembly)
# ===================================================================

def _load_assembly_order(assembly: str) -> List[str]:
    path = ASSETS_DIR / assembly / "assembly_order.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["steps"]


def _load_trajectory(assembly: str, pid: str) -> Optional[dict]:
    path = REPO_ROOT / "fabrica" / "trajectories" / f"{assembly}_{pid}" / "pick_place.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _get_available_parts(assembly: str) -> List[str]:
    order = _load_assembly_order(assembly)
    available = []
    for pid in order:
        name = f"{assembly}_{pid}"
        traj = REPO_ROOT / "fabrica" / "trajectories" / name / "pick_place.json"
        urdf = ASSETS_DIR / "environments" / name / "pick_place.urdf"
        if traj.exists() and urdf.exists():
            available.append(pid)
    return available


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


def _quat_xyzw_to_rpy(q) -> Tuple[float, float, float]:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def _generate_table_urdf(assembly: str, active_pid: str, completed_pids: List[str]) -> str:
    env_dir = ASSETS_DIR / "environments" / f"{assembly}_{active_pid}"
    env_dir.mkdir(parents=True, exist_ok=True)
    out_path = env_dir / "scene.urdf"

    lines = [
        '<?xml version="1.0"?>',
        f'<robot name="table_{assembly}_{active_pid}_scene">',
        '  <link name="box">',
        '    <visual>',
        '      <material name="wood"><color rgba="0.82 0.56 0.35 1.0"/></material>',
        '      <origin xyz="0 0 0"/>',
        '      <geometry><box size="0.475 0.4 0.3"/></geometry>',
        '    </visual>',
        '    <collision>',
        '      <origin xyz="0 0 0"/>',
        '      <geometry><box size="0.475 0.4 0.3"/></geometry>',
        '    </collision>',
        '    <inertial>',
        '      <mass value="500"/>',
        '      <friction value="1.0"/>',
        '      <inertia ixx="1000.0" ixy="0.0" ixz="0.0" iyy="1000.0" iyz="0.0" izz="1000.0"/>',
        '    </inertial>',
        '  </link>',
    ]

    for pid in completed_pids:
        traj = _load_trajectory(assembly, pid)
        if traj is None:
            continue
        goal_pose = traj["goals"][-1]
        mesh_rel = f"../../{assembly}/{pid}/{pid}_canonical.obj"
        rx, ry, rz = goal_pose[0], goal_pose[1], goal_pose[2] - TABLE_Z
        rpy = _quat_xyzw_to_rpy(goal_pose[3:7])

        lines.extend([
            f'  <link name="part_{pid}">',
            '    <visual>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
            f'      <material name="placed_{pid}"><color rgba="0.6 0.6 0.6 1.0"/></material>',
            '    </visual>',
            '    <collision>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
            '    </collision>',
            '  </link>',
            f'  <joint name="part_{pid}_joint" type="fixed">',
            '    <parent link="box"/>',
            f'    <child link="part_{pid}"/>',
            f'    <origin xyz="{rx} {ry} {rz}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>',
            '  </joint>',
        ])

    lines.append('</robot>')
    out_path.write_text('\n'.join(lines))
    return f"urdf/fabrica/environments/{assembly}_{active_pid}/scene.urdf"


# ===================================================================
# Trajectory interpolation
# ===================================================================

def slerp(q0, q1, t):
    """Spherical linear interpolation between quaternions (xyzw format)."""
    q0 = np.array(q0, dtype=np.float64)
    q1 = np.array(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


def interpolate_waypoints(waypoints: List[List[float]], steps_per: int) -> List[np.ndarray]:
    """Interpolate between waypoints. Each waypoint is [x,y,z, qx,qy,qz,qw].

    Returns list of poses as 7-element arrays.
    """
    poses = []
    for i in range(len(waypoints) - 1):
        wp0 = np.array(waypoints[i])
        wp1 = np.array(waypoints[i + 1])
        for s in range(steps_per):
            t = s / steps_per
            pos = wp0[:3] + t * (wp1[:3] - wp0[:3])
            quat = slerp(wp0[3:7], wp1[3:7], t)
            poses.append(np.concatenate([pos, quat]))
    # Add final waypoint
    poses.append(np.array(waypoints[-1]))
    return poses


# ===================================================================
# SUBPROCESS -- IsaacGym teleport worker
# ===================================================================

def teleport_worker(conn, assembly, part_id, config_path, checkpoint_path, table_urdf_rel,
                     object_name_override=None, headless=True):
    """Child process: create env, receive pose commands, teleport object, report back."""
    try:
        from isaacgym import gymapi  # noqa: F401 isort:skip
        import torch  # noqa: E401
        from isaacgym import gymtorch
        from deployment.isaac.isaac_env import create_env
        import fabrica.objects  # noqa: F401

        device = "cuda" if torch.cuda.is_available() else "cpu"
        object_name = object_name_override or f"{assembly}_{part_id}"
        use_sdf = object_name.endswith("_sdf")
        traj_name = f"{assembly}_{part_id}"
        traj_path = REPO_ROOT / "fabrica" / "trajectories" / traj_name / "pick_place.json"

        with open(traj_path) as f:
            traj = json.load(f)
        traj["start_pose"][2] += Z_OFFSET

        env = create_env(
            config_path=str(config_path), headless=headless, device=device,
            overrides={
                **BASE_OVERRIDES,
                "task.env.objectName": object_name,
                "task.env.useFixedGoalStates": True,
                "task.env.fixedGoalStates": traj["goals"],
                "task.env.asset.table": table_urdf_rel,
                "task.env.tableResetZ": TABLE_Z,
                "task.env.objectStartPose": traj["start_pose"],
                "task.env.finalGoalSuccessTolerance": 0.01,
                **({"task.env.useSDF": True} if use_sdf else {}),
            },
        )

        # Initial reset
        obs = env.step(torch.zeros((env.num_envs, N_ACT), device=device))[0]["obs"]
        init_obj_pose = env.object_state[0, :7].cpu().numpy()
        conn.send(("ready", init_obj_pose))

        # Main loop: receive pose commands, keep viewer alive
        while True:
            # Non-blocking check so we can pump the viewer
            if conn.poll(0.016):  # ~60 Hz
                msg = conn.recv()
                if msg[0] == "teleport":
                    desired_pose = np.array(msg[1], dtype=np.float32)  # [x,y,z, qx,qy,qz,qw]
                    n_substeps = msg[2] if len(msg) > 2 else 1

                    # Set object pose in root_state_tensor
                    pose_tensor = torch.tensor(desired_pose, dtype=torch.float32, device=device)
                    env.root_state_tensor[env.object_indices[0], 0:7] = pose_tensor
                    # Zero velocities
                    env.root_state_tensor[env.object_indices[0], 7:13] = 0.0
                    # Apply
                    env.deferred_set_actor_root_state_tensor_indexed(
                        [env.object_indices[0:1]]
                    )
                    env.set_actor_root_state_tensor_indexed()

                    # Step physics
                    for _ in range(n_substeps):
                        env.gym.simulate(env.sim)
                    if device == "cpu":
                        env.gym.fetch_results(env.sim, True)

                    # Read back actual pose
                    env.gym.refresh_actor_root_state_tensor(env.sim)
                    actual_pose = env.root_state_tensor[env.object_indices[0], 0:7].cpu().numpy()

                    delta = np.linalg.norm(desired_pose[:3] - actual_pose[:3])
                    conn.send(("result", desired_pose.tolist(), actual_pose.tolist(), float(delta)))

                elif msg[0] == "quit":
                    break

            # Always render to keep the viewer responsive
            if not headless:
                env.render()

    except Exception as exc:
        conn.send(("error", f"{exc}\n{traceback.format_exc()}"))

    conn.close()


# ===================================================================
# MAIN PROCESS -- viser GUI + orchestration
# ===================================================================

class InsertionDebugger:

    def __init__(self, config_path: str, checkpoint_path: str,
                 assembly: str, part_id: str,
                 steps_per_waypoint: int = 50,
                 start_waypoint: int = 8,
                 end_waypoint: int = None,
                 port: int = 8080,
                 object_name_override: str = None,
                 table_urdf_override: str = None,
                 headless: bool = True):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.assembly = assembly
        self.part_id = part_id
        self.steps_per_waypoint = steps_per_waypoint
        self.start_waypoint = start_waypoint
        self.port = port
        self.object_name_override = object_name_override
        self.table_urdf_override = table_urdf_override
        self.headless = headless

        # Load trajectory
        traj = _load_trajectory(assembly, part_id)
        if traj is None:
            raise FileNotFoundError(f"No trajectory for {assembly}_{part_id}")
        self.trajectory = traj
        all_goals = traj["goals"]

        self.end_waypoint = end_waypoint if end_waypoint is not None else len(all_goals) - 1
        waypoints = all_goals[self.start_waypoint:self.end_waypoint + 1]
        print(f"Interpolating waypoints {self.start_waypoint}..{self.end_waypoint} "
              f"({len(waypoints)} waypoints, {steps_per_waypoint} steps each)")
        for i, wp in enumerate(waypoints):
            widx = self.start_waypoint + i
            print(f"  WP{widx}: pos=({wp[0]:.4f}, {wp[1]:.4f}, {wp[2]:.4f}) "
                  f"quat=({wp[3]:.4f}, {wp[4]:.4f}, {wp[5]:.4f}, {wp[6]:.4f})")

        self.poses = interpolate_waypoints(waypoints, steps_per_waypoint)
        self.total_steps = len(self.poses)
        print(f"Total interpolated steps: {self.total_steps}")

        # State
        self._proc = None
        self._conn = None
        self._env_ready = False
        self._current_step = 0
        self._playing = False
        self._waiting_for_result = False
        self._results = []  # (step, desired, actual, delta)

        # Viser
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self._dyn = []
        self._obj_frame = None
        self._goal_frame = None

        self._build_gui()
        self._setup_static_scene()
        self._load_env()  # auto-load since assembly/part are fixed via CLI

    def _build_gui(self):
        self.server.gui.add_markdown("# Insertion Debug\n"
                                     f"### {self.assembly} / Part {self.part_id}")

        with self.server.gui.add_folder("Controls", expand_by_default=True):
            self._slider = self.server.gui.add_slider(
                "Step", min=0, max=max(self.total_steps - 1, 1),
                step=1, initial_value=0,
            )

            self._btn_play = self.server.gui.add_button("Play")
            self._btn_play.on_click(lambda _: self._toggle_play())
            self._btn_step = self.server.gui.add_button("Step Forward")
            self._btn_step.on_click(lambda _: self._step_forward())
            self._btn_reset = self.server.gui.add_button("Reset to Start")
            self._btn_reset.on_click(lambda _: self._reset_to_start())

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self._md_status = self.server.gui.add_markdown("**Status:** Not loaded")
            self._md_pose = self.server.gui.add_markdown("**Pose:** --")
            self._md_delta = self.server.gui.add_markdown("**Delta:** --")
            self._md_summary = self.server.gui.add_markdown("**Summary:** --")

    def _setup_static_scene(self):
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            client.camera.position = (0.0, -0.5, 0.8)
            client.camera.look_at = (-0.12, 0.04, 0.6)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self.server.scene.add_frame(
            "/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        robot = ViserUrdf(
            self.server,
            REPO_ROOT / "assets" / "urdf" / "kuka_sharpa_description" / "iiwa14_left_sharpa_adjusted_restricted.urdf",
            root_node_name="/robot",
        )
        robot.update_cfg(DEFAULT_DOF_POS)

        self.server.scene.add_frame(
            "/table", position=(0, 0, TABLE_Z), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self.server.scene.add_box(
            "/table/wood", color=(180, 130, 70),
            dimensions=(0.475, 0.4, 0.3), position=(0, 0, 0),
            side="double", opacity=0.9,
        )

        self._setup_context_parts()

    def _setup_context_parts(self):
        """Show completed parts and goal ghost."""
        available = _get_available_parts(self.assembly)
        part_id_to_idx = {pid: i for i, pid in enumerate(available)}

        if self.part_id in available:
            si = available.index(self.part_id)
            completed_pids = available[:si]
        else:
            completed_pids = []

        # Completed parts at goal pose
        for cpid in completed_pids:
            traj = _load_trajectory(self.assembly, cpid)
            if traj is None:
                continue
            mesh_path = ASSETS_DIR / self.assembly / cpid / f"{cpid}_canonical.obj"
            if not mesh_path.exists():
                continue
            mesh = trimesh.load(str(mesh_path), force="mesh")
            gp = traj["goals"][-1]
            idx = part_id_to_idx.get(cpid, 0)
            color = COLORS[idx % len(COLORS)]
            frame_name = f"/completed_{cpid}"
            frame = self.server.scene.add_frame(
                frame_name,
                position=(gp[0], gp[1], gp[2]),
                wxyz=quat_xyzw_to_wxyz(gp[3:7]),
                show_axes=False,
            )
            self._dyn.append(frame)
            h = self.server.scene.add_mesh_simple(
                f"{frame_name}/mesh",
                vertices=np.array(mesh.vertices, dtype=np.float32),
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=color,
            )
            self._dyn.append(h)

        # Active part frame
        self._obj_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.05, axes_radius=0.001,
        )
        self._dyn.append(self._obj_frame)

        # Load active part mesh for visualization
        from fabrica.objects import FABRICA_NAME_TO_OBJECT
        object_name = f"{self.assembly}_{self.part_id}"
        obj_urdf = FABRICA_NAME_TO_OBJECT[object_name].urdf_path
        idx = part_id_to_idx.get(self.part_id, 0)
        color = COLORS[idx % len(COLORS)]
        color_override = tuple(int(c * 255) for c in color)
        ViserUrdf(self.server, obj_urdf, root_node_name="/object",
                  mesh_color_override=color_override)

        # Goal ghost at final waypoint
        final_goal = self.trajectory["goals"][-1]
        self._goal_frame = self.server.scene.add_frame(
            "/goal",
            position=(final_goal[0], final_goal[1], final_goal[2]),
            wxyz=quat_xyzw_to_wxyz(final_goal[3:7]),
            show_axes=True, axes_length=0.05, axes_radius=0.001,
        )
        self._dyn.append(self._goal_frame)
        ViserUrdf(self.server, obj_urdf, root_node_name="/goal",
                  mesh_color_override=(0, 255, 0, 0.5))

        # Place the object at the first interpolated pose
        p = self.poses[0]
        self._obj_frame.position = tuple(p[:3])
        self._obj_frame.wxyz = quat_xyzw_to_wxyz(p[3:7])

    # -- Subprocess management --

    def _load_env(self):
        self._kill_subprocess()
        self._md_status.content = "**Status:** Loading IsaacGym environment..."

        available = _get_available_parts(self.assembly)
        if self.part_id in available:
            si = available.index(self.part_id)
            completed_pids = available[:si]
        else:
            completed_pids = []

        if self.table_urdf_override:
            table_urdf_rel = self.table_urdf_override
        else:
            table_urdf_rel = _generate_table_urdf(self.assembly, self.part_id, completed_pids)

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=teleport_worker,
            args=(child_conn, self.assembly, self.part_id,
                  self.config_path, self.checkpoint_path, table_urdf_rel,
                  self.object_name_override, self.headless),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        print(f"[debug] Spawned subprocess pid={self._proc.pid}")

    def _kill_subprocess(self):
        if self._conn is not None:
            try:
                self._conn.send(("quit",))
            except (BrokenPipeError, OSError):
                pass
            self._conn.close()
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join()
            self._proc = None
        self._env_ready = False
        self._playing = False

    # -- Commands --

    def _toggle_play(self):
        if not self._env_ready:
            self._md_status.content = "**Status:** Load environment first."
            return
        self._playing = not self._playing
        self._btn_play.name = "Pause" if self._playing else "Play"

    def _step_forward(self):
        if not self._env_ready or self._waiting_for_result:
            self._md_status.content = "**Status:** Load environment first." if not self._env_ready else "**Status:** Waiting for result..."
            return
        self._playing = False
        self._btn_play.name = "Play"
        self._do_teleport_step()

    def _reset_to_start(self):
        self._current_step = 0
        self._slider.value = 0
        self._playing = False
        self._btn_play.name = "Play"
        self._results.clear()
        self._md_summary.content = "**Summary:** Reset"
        p = self.poses[0]
        if self._obj_frame is not None:
            self._obj_frame.position = tuple(p[:3])
            self._obj_frame.wxyz = quat_xyzw_to_wxyz(p[3:7])

    def _do_teleport_step(self):
        if self._current_step >= self.total_steps:
            self._playing = False
            self._btn_play.name = "Play"
            self._print_summary()
            return
        if self._waiting_for_result:
            return

        pose = self.poses[self._current_step]
        try:
            self._conn.send(("teleport", pose.tolist()))
            self._waiting_for_result = True
        except (BrokenPipeError, OSError):
            self._md_status.content = "**Status:** Connection lost."
            self._playing = False
            return

    def _handle_result(self, msg):
        desired = np.array(msg[1])
        actual = np.array(msg[2])
        delta = msg[3]

        step = self._current_step
        self._results.append((step, desired, actual, delta))

        # Update viser
        if self._obj_frame is not None:
            self._obj_frame.position = tuple(actual[:3])
            self._obj_frame.wxyz = quat_xyzw_to_wxyz(actual[3:7])

        # Update slider
        self._slider.value = step

        # Console output
        collision_flag = " !! COLLISION" if delta > 0.001 else ""
        print(f"Step {step:03d}/{self.total_steps} | "
              f"desired z={desired[2]:.4f} | actual z={actual[2]:.4f} | "
              f"delta={delta:.4f}{collision_flag}")

        # GUI update
        self._md_pose.content = (
            f"**Desired:** ({desired[0]:.4f}, {desired[1]:.4f}, {desired[2]:.4f})  \n"
            f"**Actual:** ({actual[0]:.4f}, {actual[1]:.4f}, {actual[2]:.4f})"
        )
        delta_str = f"**Delta:** {delta:.4f}m"
        if delta > 0.001:
            delta_str += " **COLLISION**"
        self._md_delta.content = delta_str

        self._md_status.content = f"**Status:** Step {step}/{self.total_steps}"

        self._current_step += 1

        if self._current_step >= self.total_steps:
            self._playing = False
            self._btn_play.name = "Play"
            self._print_summary()

    def _print_summary(self):
        if not self._results:
            return
        collision_steps = [(s, d, a, delta) for s, d, a, delta in self._results if delta > 0.001]
        print("\n" + "=" * 60)
        print("INSERTION DEBUG SUMMARY")
        print("=" * 60)
        print(f"Total steps: {len(self._results)}")
        print(f"Collision steps (delta > 0.001m): {len(collision_steps)}")

        if collision_steps:
            first = collision_steps[0]
            max_delta = max(collision_steps, key=lambda x: x[3])
            print(f"First collision at step {first[0]}, z={first[1][2]:.4f}, delta={first[3]:.4f}")
            print(f"Max delta at step {max_delta[0]}, z={max_delta[1][2]:.4f}, delta={max_delta[3]:.4f}")

            # Find the lowest z the part can reach before collision
            last_ok = None
            for s, d, a, delta in self._results:
                if delta <= 0.001:
                    last_ok = d
            if last_ok is not None:
                print(f"Lowest z without collision: {last_ok[2]:.4f}")
            self._md_summary.content = (
                f"**Summary:** {len(collision_steps)} collision steps. "
                f"First at z={first[1][2]:.4f}. "
                f"Max delta={max_delta[3]:.4f}"
            )
        else:
            print("No collisions detected -- part fits cleanly!")
            self._md_summary.content = "**Summary:** No collisions detected!"
        print("=" * 60 + "\n")

    # -- Poll loop --

    def _poll(self):
        if self._conn is None:
            return
        try:
            while self._conn.poll(0):
                msg = self._conn.recv()
                tag = msg[0]
                if tag == "ready":
                    self._env_ready = True
                    self._current_step = 0
                    self._results.clear()
                    self._md_status.content = "**Status:** Ready -- click Play or Step Forward"
                    print("[debug] Environment ready")
                elif tag == "result":
                    self._waiting_for_result = False
                    self._handle_result(msg)
                elif tag == "error":
                    self._env_ready = False
                    self._playing = False
                    self._md_status.content = f"**Status:** Error: {msg[1][:200]}"
                    print(f"[debug] Subprocess error:\n{msg[1]}")
        except (EOFError, ConnectionResetError, OSError):
            self._conn = None
            self._env_ready = False
            self._playing = False

    def run(self):
        print()
        print("  +-------------------------------------------------+")
        print("  |        Insertion Debug (Teleport Mode)           |")
        print(f"  |     http://localhost:{self.port:<26}|")
        print(f"  |     {self.assembly} / Part {self.part_id:<30}|")
        print("  +-------------------------------------------------+")
        print()

        try:
            while True:
                self._poll()
                if self._playing and self._env_ready and not self._waiting_for_result:
                    self._do_teleport_step()
                    time.sleep(1.0 / 30.0)  # ~30 Hz playback
                else:
                    time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\n[debug] Shutting down...")
            self._kill_subprocess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug insertion collisions via teleportation")
    parser.add_argument("--config-path", type=str, default="pretrained_policy/config.yaml")
    parser.add_argument("--checkpoint-path", type=str, default="pretrained_policy/model.pth")
    parser.add_argument("--assembly", type=str, default="beam")
    parser.add_argument("--part", type=str, default="2")
    parser.add_argument("--steps-per-waypoint", type=int, default=50)
    parser.add_argument("--start-waypoint", type=int, default=8,
                        help="First waypoint index (inclusive)")
    parser.add_argument("--end-waypoint", type=int, default=None,
                        help="Last waypoint index (inclusive, default: last)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--object-name", type=str, default=None,
                        help="Override object name (e.g. beam_2_sdf)")
    parser.add_argument("--table-urdf", type=str, default=None,
                        help="Override table URDF path relative to assets root "
                             "(e.g. urdf/fabrica/environments/beam_2/pick_place_sdf.urdf)")
    parser.add_argument("--no-headless", action="store_true",
                        help="Show IsaacGym viewer window")
    args = parser.parse_args()

    def _resolve(p):
        path = Path(p)
        if path.exists():
            return str(path)
        path = REPO_ROOT / p
        if path.exists():
            return str(path)
        raise FileNotFoundError(p)

    InsertionDebugger(
        config_path=_resolve(args.config_path),
        checkpoint_path=_resolve(args.checkpoint_path),
        assembly=args.assembly,
        part_id=args.part,
        steps_per_waypoint=args.steps_per_waypoint,
        start_waypoint=args.start_waypoint,
        end_waypoint=args.end_waypoint,
        port=args.port,
        object_name_override=args.object_name,
        table_urdf_override=args.table_urdf,
        headless=not args.no_headless,
    ).run()

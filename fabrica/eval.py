"""Fabrica Beam Assembly – Eval Script
======================================
Loads beam_6 into the SimToolReal Isaac Gym environment with the fixture on
the table, runs the pretrained policy on a pick-and-place trajectory, and
visualises in viser.

Usage:
    python fabrica/eval.py \
        --config-path pretrained_policy/config.yaml \
        --checkpoint-path pretrained_policy/model.pth
"""

import argparse
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_Z = 0.38
Z_OFFSET = 0.03

# Default joint positions matching IsaacGym reset
_ARM_DEFAULT = np.array([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308])
_ARM_DEFAULT[1] -= np.deg2rad(10)  # startArmHigher
_ARM_DEFAULT[3] += np.deg2rad(10)  # startArmHigher
DEFAULT_DOF_POS = np.zeros(29)
DEFAULT_DOF_POS[:7] = _ARM_DEFAULT


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


# ═══════════════════════════════════════════════════════════════════
# SUBPROCESS  -- IsaacGym simulation (all heavy imports stay here)
# ═══════════════════════════════════════════════════════════════════

def _sim_get_state(env, obs, joint_lower, joint_upper, n_act):
    """Extract visualisation state from the env."""
    obs_np = obs[0].cpu().numpy()
    joint_pos = 0.5 * (obs_np[:n_act] + 1.0) * (joint_upper - joint_lower) + joint_lower
    return (
        joint_pos,
        env.object_state[0, :7].cpu().numpy(),
        env.goal_pose[0].cpu().numpy(),
        env.obj_keypoint_pos[0].cpu().numpy(),       # (num_keypoints, 3)
        env.goal_keypoint_pos[0].cpu().numpy(),       # (num_keypoints, 3)
    )


def sim_worker(conn, object_name, table_urdf, traj_path,
               config_path, checkpoint_path):
    """Child process: creates env, runs episodes on command."""
    try:
        from isaacgym import gymapi  # noqa: F401 isort:skip
    except ImportError:
        conn.send(("error",
                   "Isaac Gym is not installed. Download Isaac Gym Preview 4 "
                   "from https://developer.nvidia.com/isaac-gym-preview-4 and "
                   "install with: cd isaacgym/python && pip install -e ."))
        return

    import torch
    from deployment.rl_player import RlPlayer
    from deployment.isaac.isaac_env import create_env

    # Register fabrica objects into the global NAME_TO_OBJECT
    import fabrica.objects  # noqa: F401

    n_act = 29
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        with open(traj_path) as f:
            traj_data = json.load(f)
        traj_data["start_pose"][2] += Z_OFFSET

        env = create_env(
            config_path=str(config_path), headless=True, device=device,
            overrides={
                # Turn off randomization noise
                "task.env.resetPositionNoiseX": 0.0,
                "task.env.resetPositionNoiseY": 0.0,
                "task.env.resetPositionNoiseZ": 0.0,
                "task.env.randomizeObjectRotation": False,
                "task.env.resetDofPosRandomIntervalFingers": 0.0,
                "task.env.resetDofPosRandomIntervalArm": 0.0,
                "task.env.resetDofVelRandomInterval": 0.0,
                "task.env.tableResetZRange": 0.0,
                # Object
                "task.env.objectName": object_name,
                # Set up environment parameters
                "task.env.numEnvs": 1,
                "task.env.envSpacing": 0.4,
                "task.env.capture_video": False,
                # Goal settings
                "task.env.useFixedGoalStates": True,
                "task.env.fixedGoalStates": traj_data["goals"],
                # Delays and noise
                "task.env.useActionDelay": False,
                "task.env.useObsDelay": False,
                "task.env.useObjectStateDelayNoise": False,
                "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
                # Reset
                "task.env.resetWhenDropped": False,
                # Moving average
                "task.env.armMovingAverage": 0.1,
                # Success criteria
                "task.env.evalSuccessTolerance": 0.01,
                "task.env.successSteps": 1,
                "task.env.fixedSizeKeypointReward": True,
                # Table
                "task.env.asset.table": str(table_urdf),
                "task.env.tableResetZ": TABLE_Z,
                # Initialization
                "task.env.useFixedInitObjectPose": True,
                "task.env.objectStartPose": traj_data["start_pose"],
                "task.env.startArmHigher": True,
                # Forces/torques (all zero for eval)
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
            },
        )

        joint_lower = env.arm_hand_dof_lower_limits[:n_act].cpu().numpy()
        joint_upper = env.arm_hand_dof_upper_limits[:n_act].cpu().numpy()

        # Load policy
        env.set_env_state(torch.load(checkpoint_path)[0]["env_state"])
        policy = RlPlayer(140, n_act, config_path, checkpoint_path, device, env.num_envs)

        # Initial reset
        obs_dict, _, _, _ = env.step(torch.zeros((env.num_envs, n_act), device=device))
        obs = obs_dict["obs"]

        init_state = _sim_get_state(env, obs, joint_lower, joint_upper, n_act)
        conn.send(("ready", init_state))

    except Exception:
        import traceback
        conn.send(("error", traceback.format_exc()))
        return

    # ── Command loop ─────────────────────────────────────
    while True:
        try:
            cmd = conn.recv()
        except (EOFError, ConnectionResetError):
            break

        if cmd == "quit":
            break

        if cmd == "run":
            policy.reset()
            obs_dict, _, _, _ = env.step(torch.zeros((env.num_envs, n_act), device=device))
            obs = obs_dict["obs"]

            control_dt = 1.0 / 60.0
            step = 0
            done = False
            paused = False

            while not done:
                # Drain commands
                while conn.poll(0):
                    c = conn.recv()
                    if c == "stop":
                        conn.send(("stopped",))
                        done = True
                        break
                    elif c == "pause":
                        paused = True
                    elif c == "resume":
                        paused = False
                    elif c == "quit":
                        return
                if done:
                    break

                if paused:
                    time.sleep(0.05)
                    continue

                t0 = time.time()

                action = policy.get_normalized_action(obs, deterministic_actions=True)
                obs_dict, _, done_tensor, _ = env.step(action)
                obs = obs_dict["obs"]
                done = done_tensor[0].item()
                step += 1

                state = _sim_get_state(env, obs, joint_lower, joint_upper, n_act)
                successes = int(env.successes[0].item())
                max_succ = env.max_consecutive_successes

                conn.send(("state", state, successes, max_succ, step))

                elapsed = time.time() - t0
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)

            goal_pct = 100 * int(env.successes[0].item()) / env.max_consecutive_successes
            conn.send(("done", goal_pct, step))


# ═══════════════════════════════════════════════════════════════════
# MAIN PROCESS  -- viser visualisation
# ═══════════════════════════════════════════════════════════════════

class FabricaDemo:
    def __init__(self, config_path, checkpoint_path, object_name, task_name, port):
        self.object_name = object_name
        self.task_name = task_name
        self.port = port

        self.server = viser.ViserServer(host="0.0.0.0", port=port)

        self._episode_running = False
        self._is_paused = False

        # Scene handles
        self.robot = None
        self._obj_frame = None
        self._goal_frame = None
        self._obj_kp_handles = []
        self._goal_kp_handles = []

        # Stats
        self.ep_count = 0
        self.ep_goals = []
        self.ep_lengths = []

        self._build_gui()
        self._setup_static_scene()

        # Launch sim subprocess immediately
        traj_path = REPO_ROOT / "fabrica" / "trajectories" / object_name / f"{task_name}.json"
        table_urdf = f"urdf/fabrica/environments/{object_name}/{task_name}.urdf"
        assert traj_path.exists(), f"Trajectory not found: {traj_path}"

        self._md_status.content = f"**Status:** Loading {object_name} / {task_name} ..."

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=sim_worker,
            args=(child_conn, object_name, table_urdf, str(traj_path),
                  config_path, checkpoint_path),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        print(f"[launcher] Spawned subprocess pid={self._proc.pid}")

    def _build_gui(self):
        self.server.gui.add_markdown(
            f"# Fabrica Beam Assembly\n"
            f"### {self.object_name} / {self.task_name}"
        )

        with self.server.gui.add_folder("Controls", expand_by_default=True):
            self._btn_run = self.server.gui.add_button("Run Episode")
            self._btn_run.on_click(lambda _: self._cmd_run())
            self._btn_pause = self.server.gui.add_button("Pause")
            self._btn_pause.on_click(lambda _: self._cmd_pause())
            self._btn_stop = self.server.gui.add_button("Stop Episode")
            self._btn_stop.on_click(lambda _: self._cmd_stop())

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self._md_status = self.server.gui.add_markdown("**Status:** Starting...")
            self._md_prog = self.server.gui.add_markdown("**Progress:** --")
            self._md_stats = self.server.gui.add_markdown("**Stats:** No episodes yet")
            self._md_obj = self.server.gui.add_markdown("**Object Pos:** --")

    def _setup_static_scene(self):
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            client.camera.position = (0.0, -1.0, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.5)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        robot_urdf = (
            REPO_ROOT / "assets" / "urdf" / "kuka_sharpa_description"
            / "iiwa14_left_sharpa_adjusted_restricted.urdf"
        )
        self.server.scene.add_frame(
            "/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self.robot = ViserUrdf(self.server, robot_urdf, root_node_name="/robot")
        self.robot.update_cfg(DEFAULT_DOF_POS)

        # Table
        self.server.scene.add_frame(
            "/table", position=(0, 0, TABLE_Z), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self.server.scene.add_box(
            "/table/wood", color=(180, 130, 70), dimensions=(0.475, 0.4, 0.3),
            position=(0, 0, 0), side="double", opacity=0.9,
        )
        # Fixture on table
        fixture_urdf = REPO_ROOT / "assets" / "urdf" / "fabrica" / "beam" / "fixture" / "fixture.urdf"
        self.server.scene.add_frame(
            "/table/fixture", position=(0.12, -0.152, 0.15), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        ViserUrdf(self.server, fixture_urdf, root_node_name="/table/fixture")

    def _setup_object_goal(self, num_keypoints):
        from fabrica.objects import FABRICA_NAME_TO_OBJECT
        obj_urdf = FABRICA_NAME_TO_OBJECT[self.object_name].urdf_path

        self._obj_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        ViserUrdf(self.server, obj_urdf, root_node_name="/object")

        self._goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        ViserUrdf(self.server, obj_urdf, root_node_name="/goal",
                  mesh_color_override=(0, 255, 0, 0.5))

        # Keypoint spheres
        self._obj_kp_handles = []
        self._goal_kp_handles = []
        for i in range(num_keypoints):
            h = self.server.scene.add_icosphere(
                f"/obj_kp/{i}", radius=0.005, color=(255, 0, 0),
            )
            self._obj_kp_handles.append(h)
            h = self.server.scene.add_icosphere(
                f"/goal_kp/{i}", radius=0.005, color=(0, 255, 0), opacity=0.5,
            )
            self._goal_kp_handles.append(h)

    def _update_viz(self, state_tuple):
        joint_pos, obj_pose, goal_pose, obj_kps, goal_kps = state_tuple
        self.robot.update_cfg(joint_pos)
        if self._obj_frame is not None:
            self._obj_frame.position = tuple(obj_pose[:3])
            self._obj_frame.wxyz = quat_xyzw_to_wxyz(obj_pose[3:7])
        if self._goal_frame is not None:
            self._goal_frame.position = tuple(goal_pose[:3])
            self._goal_frame.wxyz = quat_xyzw_to_wxyz(goal_pose[3:7])
        for i, h in enumerate(self._obj_kp_handles):
            h.position = tuple(obj_kps[i])
        for i, h in enumerate(self._goal_kp_handles):
            h.position = tuple(goal_kps[i])

    # ── Commands ──────────────────────────────────────────

    def _send(self, msg):
        try:
            self._conn.send(msg)
        except (BrokenPipeError, OSError):
            pass

    def _cmd_run(self):
        if self._episode_running:
            return
        self._episode_running = True
        self._is_paused = False
        self._btn_pause.name = "Pause"
        self._md_status.content = "**Status:** Running episode..."
        self._send("run")

    def _cmd_pause(self):
        if not self._episode_running:
            return
        self._is_paused = not self._is_paused
        self._send("pause" if self._is_paused else "resume")
        self._btn_pause.name = "Resume" if self._is_paused else "Pause"

    def _cmd_stop(self):
        if self._episode_running:
            self._send("stop")

    # ── Message handling ───────────────────────────────────────

    def _handle(self, msg):
        tag = msg[0]

        if tag == "ready":
            init_state = msg[1]
            num_keypoints = init_state[3].shape[0]  # obj_kps shape is (num_keypoints, 3)
            self._setup_object_goal(num_keypoints)
            self._update_viz(init_state)
            self._md_status.content = "**Status:** Ready -- click **Run Episode**"
            print("[launcher] Environment ready")

        elif tag == "state":
            state, successes, max_succ, step = msg[1], msg[2], msg[3], msg[4]
            self._update_viz(state)
            pct = 100 * successes / max_succ if max_succ > 0 else 0
            self._md_prog.content = (
                f"**Time:** {step / 60.0:.1f}s &nbsp;|&nbsp; "
                f"**Goal:** {successes}/{max_succ} ({pct:.0f}%)"
            )
            obj_pos = state[1][:3]
            self._md_obj.content = (
                f"**Object Pos:** {obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}"
            )

        elif tag == "done":
            goal_pct, steps = msg[1], msg[2]
            self._episode_running = False
            self.ep_goals.append(goal_pct)
            self.ep_lengths.append(steps)
            self.ep_count += 1
            avg_g = np.mean(self.ep_goals)
            avg_t = np.mean(self.ep_lengths) / 60.0
            self._md_stats.content = (
                f"**Episodes:** {self.ep_count} &nbsp;|&nbsp; "
                f"**Avg Goal:** {avg_g:.1f}% &nbsp;|&nbsp; "
                f"**Avg Time:** {avg_t:.1f}s"
            )
            self._md_status.content = (
                f"**Status:** Done -- {steps / 60.0:.1f}s, {goal_pct:.0f}% goals"
            )
            print(f"[launcher] Episode done: {goal_pct:.0f}% goals in {steps / 60.0:.1f}s")

        elif tag == "stopped":
            self._episode_running = False
            self._md_status.content = "**Status:** Episode stopped."

        elif tag == "error":
            self._episode_running = False
            self._md_status.content = f"**Status:** Error -- {msg[1][:200]}"
            print(f"[launcher] Subprocess error:\n{msg[1]}")

    def _poll(self):
        try:
            while self._conn.poll(0):
                self._handle(self._conn.recv())
        except (EOFError, ConnectionResetError, OSError):
            if self._proc is not None and not self._proc.is_alive():
                self._md_status.content = "**Status:** Subprocess exited unexpectedly."
                self._episode_running = False

    def run(self):
        print()
        print("  +-------------------------------------------------+")
        print("  |     Fabrica Beam Assembly – Policy Demo          |")
        print(f"  |     http://localhost:{self.port:<26}|")
        print("  +-------------------------------------------------+")
        print()

        try:
            while True:
                self._poll()
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\n[launcher] Shutting down...")
            try:
                self._conn.send("quit")
            except (BrokenPipeError, OSError):
                pass
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fabrica Beam Assembly Eval")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config-path", type=str, default="pretrained_policy/config.yaml")
    parser.add_argument("--checkpoint-path", type=str, default="pretrained_policy/model.pth")
    parser.add_argument("--object-name", type=str, default="beam_6")
    parser.add_argument("--task-name", type=str, default="pick_place")
    args = parser.parse_args()

    FabricaDemo(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        object_name=args.object_name,
        task_name=args.task_name,
        port=args.port,
    ).run()

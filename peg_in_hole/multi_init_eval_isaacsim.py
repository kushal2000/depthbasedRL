#!/usr/bin/env python3
"""Interactive multi-init peg-in-hole evaluation for IsaacSim.

This mirrors ``peg_multi_init_eval.py``: the main process hosts a viser UI and
the sim runs in a child process. Scene, tolerance, policy, and goal-mode changes
reload the worker; peg-index changes apply on the next reset.

Usage:
    python peg_in_hole/multi_init_eval_isaacsim.py \
        --checkpoint-path pretrained_policy/model.pth \
        --port 8043
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "peg_in_hole"
SCENES_NPZ = ASSETS_DIR / "scenes" / "scenes.npz"
ROBOT_URDF = (
    REPO_ROOT
    / "assets"
    / "urdf"
    / "kuka_sharpa_description"
    / "iiwa14_left_sharpa_adjusted_restricted.urdf"
)
PEG_URDF = ASSETS_DIR / "peg" / "peg.urdf"

N_ACT = 29
CONTROL_DT = 1.0 / 60.0
GOAL_MODES = ("dense", "preInsertAndFinal", "finalGoalOnly")

_ARM_DEFAULT = np.array([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308])
_ARM_DEFAULT[1] -= np.deg2rad(10)
_ARM_DEFAULT[3] += np.deg2rad(10)
DEFAULT_DOF_POS = np.zeros(N_ACT)
DEFAULT_DOF_POS[:7] = _ARM_DEFAULT


def _main_python() -> Path:
    return REPO_ROOT / ".venv" / "bin" / "python"


def _isaacsim_python() -> Path:
    return REPO_ROOT / ".venv_isaacsim" / "bin" / "python"


def _ensure_viser_main() -> None:
    if "--worker" in sys.argv:
        return
    try:
        import viser  # noqa: F401
        from viser.extras import ViserUrdf  # noqa: F401
    except ModuleNotFoundError:
        python = _main_python()
        if python.exists() and Path(sys.executable).resolve() != python.resolve():
            os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
        raise


def _load_scenes() -> dict[str, np.ndarray]:
    if not SCENES_NPZ.exists():
        raise FileNotFoundError(f"{SCENES_NPZ} not found. Run generate_scenes.py first.")
    data = np.load(str(SCENES_NPZ))
    return {
        "start_poses": data["start_poses"],
        "goals": data["goals"],
        "traj_lengths": data["traj_lengths"],
        "hole_positions": data["hole_positions"],
        "tolerance_pool_m": data["tolerance_pool_m"],
        "scene_tolerance_indices": data["scene_tolerance_indices"],
    }


def _parse_override_value(value: str) -> Any:
    if value in ("True", "true"):
        return True
    if value in ("False", "false"):
        return False
    if value in ("None", "null"):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _set_cfg_path(cfg: Any, path: str, value: Any) -> None:
    path = path.removeprefix("env.")
    parts = path.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _xyzw_to_wxyz(quat: np.ndarray) -> tuple[float, float, float, float]:
    return (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))


def _first_env_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _first_env_bool(value: Any) -> bool:
    return bool(_first_env_float(value))


def _episode_final_float(env: Any, key: str, default: float = 0.0) -> float:
    values = getattr(env, "extras", {}).get("episode_final", {})
    if key not in values:
        return default
    return _first_env_float(values[key])


def _launch_isaac_app(headless: bool):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    launcher_args = parser.parse_args(["--headless"] if headless else [])
    return AppLauncher(launcher_args).app


def _apply_eval_overrides(env_cfg: Any, args: argparse.Namespace) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 1.2
    env_cfg.sim.device = args.sim_device

    env_cfg.peg_in_hole.goal_mode = args.goal_mode
    env_cfg.peg_in_hole.force_scene_tol_combo = (args.scene_idx, args.tol_slot)
    env_cfg.peg_in_hole.force_peg_idx = args.peg_idx
    env_cfg.peg_in_hole.goal_xy_obs_noise = args.goal_xy_obs_noise
    env_cfg.peg_in_hole.enable_retract = not args.no_retract
    env_cfg.peg_in_hole.retract_distance_threshold = args.retract_distance_threshold
    env_cfg.peg_in_hole.retract_success_tolerance = args.retract_success_tolerance

    env_cfg.termination.eval_success_tolerance = args.success_tolerance
    env_cfg.termination.force_consecutive_near_goal_steps = not args.no_force_consecutive

    reset = env_cfg.reset
    reset.reset_position_noise_x = 0.0
    reset.reset_position_noise_y = 0.0
    reset.reset_position_noise_z = 0.0
    reset.reset_dof_pos_random_interval_arm = 0.0
    reset.reset_dof_pos_random_interval_fingers = 0.0
    reset.reset_dof_vel_random_interval = 0.0
    reset.table_reset_z_range = 0.0

    dr = env_cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.object_scale_noise_multiplier_range = (1.0, 1.0)
    dr.joint_velocity_obs_noise_std = 0.0
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (0.0001, 0.0001)
    dr.torque_prob_range = (0.0001, 0.0001)

    for key, value in json.loads(args.overrides_json).items():
        _set_cfg_path(env_cfg, key, value)


def _load_player(env: Any, args: argparse.Namespace):
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rl_games.torch_runner import Runner

    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env

    agent_cfg = load_cfg_from_registry(args.task, args.agent)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["device"] = args.rl_device
    agent_cfg["params"]["config"]["device_name"] = args.rl_device

    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(
        env,
        rl_device=args.rl_device,
        clip_obs=clip_obs,
        clip_actions=clip_actions,
    )

    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(args.checkpoint)
    player.has_batch_dimension = True
    return wrapped, player


def _tensor_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _pose_xyzw(pos: Any, quat_wxyz: Any) -> np.ndarray:
    quat = _tensor_np(quat_wxyz)
    pose = np.zeros(7, dtype=np.float32)
    pose[:3] = _tensor_np(pos)
    pose[3:] = quat[[1, 2, 3, 0]]
    return pose


def _termination_reason(env: Any) -> str:
    reasons = getattr(env, "_termination_reasons", {})
    active = [name for name, value in reasons.items() if _first_env_bool(value)]
    if active:
        return ",".join(active)
    if _first_env_bool(getattr(env, "reset_time_outs", False)):
        return "timeout"
    if _first_env_bool(getattr(env, "reset_terminated", False)):
        return "terminated"
    return "max_steps"


def _sim_state(env: Any, step: int) -> dict[str, Any]:
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import _keypoints_world

    origin = env.scene.env_origins[0]
    obj_pos = env.object.data.root_pos_w - env.scene.env_origins
    goal_pos = env.goal_viz.data.root_pos_w - env.scene.env_origins

    kp_offsets = (
        env._keypoint_offsets_fixed
        if env.cfg.reward.fixed_size_keypoint_reward
        else env._keypoint_offsets
    )
    obj_kp = _keypoints_world(obj_pos, env.object.data.root_quat_w, kp_offsets)
    goal_kp = _keypoints_world(goal_pos, env.goal_viz.data.root_quat_w, kp_offsets)
    kp_dist = (obj_kp - goal_kp).norm(dim=-1)

    ft_pos = (
        env.robot.data.body_state_w[:, env._fingertip_body_ids, 0:3]
        - env.scene.env_origins.unsqueeze(1)
    )
    ft_dist = (ft_pos - obj_pos.unsqueeze(1)).norm(dim=-1)

    return {
        "joint_pos": _tensor_np(env.robot.data.joint_pos[0, env._perm_lab_to_canon]),
        "robot_pose": _pose_xyzw(
            env.robot.data.root_pos_w[0] - origin,
            env.robot.data.root_quat_w[0],
        ),
        "object_pose": _pose_xyzw(
            env.object.data.root_pos_w[0] - origin,
            env.object.data.root_quat_w[0],
        ),
        "goal_pose": _pose_xyzw(
            env.goal_viz.data.root_pos_w[0] - origin,
            env.goal_viz.data.root_quat_w[0],
        ),
        "table_pose": _pose_xyzw(
            env.table.data.root_pos_w[0] - origin,
            env.table.data.root_quat_w[0],
        ),
        "obj_keypoints": _tensor_np(obj_kp[0]),
        "goal_keypoints": _tensor_np(goal_kp[0]),
        "keypoints_max_dist": float(kp_dist[0].max().detach().cpu().item()),
        "success_tolerance": float(
            env._current_success_tolerance * env.cfg.reward.keypoint_scale
        ),
        "near_goal_steps": int(env._near_goal_steps[0].detach().cpu().item()),
        "progress": int(env.episode_length_buf[0].detach().cpu().item()),
        "max_episode_length": int(env.max_episode_length),
        "reset_pending": bool(env.reset_buf[0].detach().cpu().item())
        if hasattr(env, "reset_buf")
        else False,
        "successes": int(env._successes[0].detach().cpu().item()),
        "max_goals": int(env.env_max_goals[0].detach().cpu().item()),
        "retract_phase": bool(env.retract_phase[0].detach().cpu().item()),
        "retract_success": bool(env.retract_succeeded[0].detach().cpu().item()),
        "mean_fingertip_dist": float(ft_dist[0].mean().detach().cpu().item()),
        "step": int(step),
    }


def _run_episode(conn: Any, env: Any, wrapped: Any, player: Any, args: argparse.Namespace):
    player.reset()
    obs = player.env_reset(wrapped)

    done = False
    paused = False
    step = 0
    total_reward = 0.0
    peak_successes = 0
    max_goals_seen = max(1, int(_first_env_float(env.env_max_goals)))
    retract_ok = False

    while not done and step < args.max_steps:
        while conn.poll(0):
            cmd = conn.recv()
            if cmd == "pause":
                paused = True
            elif cmd == "resume":
                paused = False
            elif cmd == "stop":
                conn.send(("stopped",))
                return obs
            elif isinstance(cmd, tuple) and cmd[0] == "set_peg_idx":
                env.cfg.peg_in_hole.force_peg_idx = int(cmd[1])

        if paused:
            time.sleep(0.05)
            continue

        t0 = time.time()
        action = player.get_action(obs, is_deterministic=not args.stochastic)
        obs, reward, dones, _ = player.env_step(wrapped, action)
        total_reward += _first_env_float(reward)
        done = _first_env_bool(dones)
        step += 1

        if done:
            successes_now = int(_first_env_float(env._prev_episode_successes))
            max_goals_now = int(_first_env_float(env.prev_episode_env_max_goals))
        else:
            successes_now = int(_first_env_float(env._successes))
            max_goals_now = int(_first_env_float(env.env_max_goals))

        peak_successes = max(peak_successes, successes_now)
        max_goals_seen = max(max_goals_seen, max_goals_now)
        retract_ok = retract_ok or (_episode_final_float(env, "retract_success") > 0.5)

        state = _sim_state(env, step)
        state["successes"] = successes_now
        state["max_goals"] = max_goals_now
        state["retract_success"] = state["retract_success"] or retract_ok
        conn.send(("state", state))

        elapsed = time.time() - t0
        if (sleep_s := CONTROL_DT - elapsed) > 0:
            time.sleep(sleep_s)

    goal_pct = 100.0 * peak_successes / max(1, max_goals_seen)
    conn.send(
        (
            "done",
            {
                "goal_pct": goal_pct,
                "steps": step,
                "retract_success": bool(retract_ok),
                "return": float(total_reward),
                "termination": _termination_reason(env) if done else "max_steps",
            },
        )
    )
    return obs


def _worker_main(args: argparse.Namespace) -> None:
    conn = Client((args.ipc_host, args.ipc_port), authkey=args.ipc_auth.encode())
    app = None
    env = None
    try:
        conn.send(("status", "Booting IsaacSim..."))
        app = _launch_isaac_app(headless=args.worker_headless)

        import gymnasium as gym
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        import isaacsimenvs  # noqa: F401

        conn.send(("status", "Creating env..."))
        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
        _apply_eval_overrides(env_cfg, args)
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped

        conn.send(("status", "Loading policy..."))
        wrapped, player = _load_player(env, args)
        player.reset()
        player.env_reset(wrapped)

        conn.send(("ready", _sim_state(inner, 0)))
        while True:
            cmd = conn.recv()
            if cmd == "run":
                _run_episode(conn, inner, wrapped, player, args)
            elif isinstance(cmd, tuple) and cmd[0] == "set_peg_idx":
                inner.cfg.peg_in_hole.force_peg_idx = int(cmd[1])
            elif cmd == "quit":
                break
    except Exception as exc:
        conn.send(("error", f"{exc}\n{traceback.format_exc()}"))
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None and hasattr(app, "close"):
                app.close()
            conn.close()


class PegMultiInitIsaacSimDemo:
    def __init__(
        self,
        policies: Dict[str, Tuple[Optional[str], str]],
        *,
        port: int,
        headless: bool,
        goal_mode: str,
        initial_policy: Optional[str],
        task: str,
        agent: str,
        rl_device: str,
        sim_device: str,
        isaacsim_python: Path,
        extra_overrides: dict[str, Any],
    ) -> None:
        import viser
        from viser.extras import ViserUrdf

        if goal_mode not in GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {GOAL_MODES}")
        if not policies:
            raise ValueError("At least one policy is required.")

        self.viser = viser
        self.ViserUrdf = ViserUrdf
        self.server = viser.ViserServer(host="0.0.0.0", port=port)

        self.policies = policies
        self.initial_policy = (
            initial_policy if initial_policy in policies else next(iter(policies))
        )
        self.headless = headless
        self.goal_mode = goal_mode
        self.task = task
        self.agent = agent
        self.rl_device = rl_device
        self.sim_device = sim_device
        self.isaacsim_python = isaacsim_python
        self.extra_overrides = extra_overrides
        self.scenes = _load_scenes()
        self.num_scenes = int(self.scenes["start_poses"].shape[0])
        self.num_pegs = int(self.scenes["start_poses"].shape[1])
        self.num_tol_slots = int(self.scenes["scene_tolerance_indices"].shape[1])

        self._proc: Optional[subprocess.Popen] = None
        self._conn = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False
        self._dyn: list[Any] = []
        self._obj_frame = None
        self._goal_frame = None
        self._table_frame = None
        self._robot_frame = None
        self._robot_urdf = None
        self._obj_keypoints: list[Any] = []
        self._goal_keypoints: list[Any] = []
        self.ep_count = 0

        self._build_gui()
        self._setup_static_scene()

    def _build_gui(self) -> None:
        self.server.gui.add_markdown(
            "# Peg-in-Hole Multi-Init Eval\n"
            "### IsaacGym-trained policy running in IsaacSim"
        )

        with self.server.gui.add_folder("Task Selection", expand_by_default=True):
            self._dd_policy = self.server.gui.add_dropdown(
                "Policy", options=list(self.policies.keys()), initial_value=self.initial_policy
            )
            self._dd_scene = self.server.gui.add_dropdown(
                "Scene idx", options=[str(i) for i in range(self.num_scenes)], initial_value="0"
            )
            self._dd_peg = self.server.gui.add_dropdown(
                "Peg idx", options=[str(i) for i in range(self.num_pegs)], initial_value="0"
            )
            self._dd_tol = self.server.gui.add_dropdown(
                "Tol slot idx",
                options=[str(i) for i in range(self.num_tol_slots)],
                initial_value="0",
            )
            self._dd_goal_mode = self.server.gui.add_dropdown(
                "Goal mode", options=list(GOAL_MODES), initial_value=self.goal_mode
            )
            self._btn_load = self.server.gui.add_button("Load / reload env")
            self._btn_load.on_click(lambda _: self._load_env())
            self._md_status = self.server.gui.add_markdown("**Status:** Ready")

        with self.server.gui.add_folder("Episode Controls", expand_by_default=True):
            self._btn_run = self.server.gui.add_button("Run Episode")
            self._btn_run.on_click(lambda _: self._cmd_run())
            self._btn_pause = self.server.gui.add_button("Pause")
            self._btn_pause.on_click(lambda _: self._cmd_pause())
            self._btn_stop = self.server.gui.add_button("Stop")
            self._btn_stop.on_click(lambda _: self._cmd_stop())

        with self.server.gui.add_folder("Display", expand_by_default=True):
            self._cb_keypoints = self.server.gui.add_checkbox(
                "Show keypoints", initial_value=True
            )
            self._cb_keypoints.on_update(lambda _: self._apply_keypoint_visibility())

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self._md_task = self.server.gui.add_markdown("**Task:** --")
            self._md_prog = self.server.gui.add_markdown("**Progress:** --")
            self._md_diag = self.server.gui.add_markdown("**Goal dist:** --")
            self._md_retract = self.server.gui.add_markdown("**Retract:** --")
            self._md_stats = self.server.gui.add_markdown("**Stats:** No episodes yet")
            self._md_tol_value = self.server.gui.add_markdown("**Tol value:** --")

        self._dd_peg.on_update(lambda _: self._apply_peg_change())

    def _setup_static_scene(self) -> None:
        @self.server.on_client_connect
        def _(client) -> None:
            client.camera.position = (0.0, -1.0, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.5)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self._robot_frame = self.server.scene.add_frame(
            "/robot", position=(0, 0, 0), wxyz=(1, 0, 0, 0), show_axes=False
        )
        self._robot_urdf = self.ViserUrdf(
            self.server, ROBOT_URDF, root_node_name="/robot"
        )
        self._robot_urdf.update_cfg(DEFAULT_DOF_POS)

    def _clear_dynamic(self) -> None:
        for handle in self._dyn:
            try:
                handle.remove()
            except Exception:
                pass
        self._dyn.clear()
        self._obj_frame = None
        self._goal_frame = None
        self._table_frame = None
        self._obj_keypoints.clear()
        self._goal_keypoints.clear()

    def _setup_scene_objects(self, scene_idx: int, tol_slot: int) -> None:
        self._clear_dynamic()

        self._table_frame = self.server.scene.add_frame(
            "/table", position=(0, 0, 0), wxyz=(1, 0, 0, 0), show_axes=False
        )
        self._dyn.append(self._table_frame)
        table_urdf = ASSETS_DIR / "scenes" / f"scene_{scene_idx:04d}" / f"scene_tol{tol_slot:02d}.urdf"
        self._dyn.append(self.ViserUrdf(self.server, table_urdf, root_node_name="/table"))

        self._obj_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.08, axes_radius=0.001
        )
        self._dyn.append(self._obj_frame)
        self._dyn.append(
            self.ViserUrdf(
                self.server,
                PEG_URDF,
                root_node_name="/object",
                mesh_color_override=(204, 40, 40),
            )
        )

        self._goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.08, axes_radius=0.001
        )
        self._dyn.append(self._goal_frame)
        self._dyn.append(
            self.ViserUrdf(
                self.server,
                PEG_URDF,
                root_node_name="/goal",
                mesh_color_override=(0, 255, 0, 0.45),
            )
        )

        tol_m = float(
            self.scenes["tolerance_pool_m"][
                self.scenes["scene_tolerance_indices"][scene_idx, tol_slot]
            ]
        )
        self._md_tol_value.content = (
            f"**Tol value:** {tol_m * 1000:.4f} mm "
            f"(slot {tol_slot}, scene {scene_idx})"
        )

    def _setup_keypoints(self, count: int) -> None:
        for handle in self._obj_keypoints + self._goal_keypoints:
            try:
                handle.remove()
            except Exception:
                pass
        self._obj_keypoints.clear()
        self._goal_keypoints.clear()
        for i in range(count):
            self._obj_keypoints.append(
                self.server.scene.add_icosphere(
                    f"/obj_kp/{i}", radius=0.005, color=(255, 0, 0)
                )
            )
            self._goal_keypoints.append(
                self.server.scene.add_icosphere(
                    f"/goal_kp/{i}", radius=0.005, color=(0, 255, 0), opacity=0.5
                )
            )
        self._apply_keypoint_visibility()

    def _apply_keypoint_visibility(self) -> None:
        visible = self._cb_keypoints.value
        for handle in self._obj_keypoints + self._goal_keypoints:
            handle.visible = visible

    def _kill_subprocess(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send("quit")
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
            self._proc = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False

    def _load_env(self) -> None:
        self._kill_subprocess()

        scene_idx = int(self._dd_scene.value)
        peg_idx = int(self._dd_peg.value)
        tol_slot = int(self._dd_tol.value)
        goal_mode = self._dd_goal_mode.value
        policy_name = self._dd_policy.value
        _, checkpoint = self.policies[policy_name]

        label = (
            f"{policy_name} | scene {scene_idx} | peg {peg_idx} "
            f"| tol slot {tol_slot} | goals: {goal_mode}"
        )
        self._md_status.content = f"**Status:** Loading *{label}* ..."
        self._md_task.content = f"**Task:** {label}"
        self._md_retract.content = "**Retract:** --"
        self._md_stats.content = "**Stats:** No episodes yet"
        self.ep_count = 0
        self._robot_urdf.update_cfg(DEFAULT_DOF_POS)
        self._setup_scene_objects(scene_idx, tol_slot)

        auth = os.urandom(16).hex()
        listener = Listener(("127.0.0.1", 0), authkey=auth.encode())
        listener._listener._socket.settimeout(30.0)
        host, port = listener.address

        cmd = [
            str(self.isaacsim_python),
            str(Path(__file__).resolve()),
            "--worker",
            "--ipc-host",
            str(host),
            "--ipc-port",
            str(port),
            "--ipc-auth",
            auth,
            "--checkpoint",
            checkpoint,
            "--task",
            self.task,
            "--agent",
            self.agent,
            "--scene-idx",
            str(scene_idx),
            "--peg-idx",
            str(peg_idx),
            "--tol-slot",
            str(tol_slot),
            "--goal-mode",
            goal_mode,
            "--rl-device",
            self.rl_device,
            "--sim-device",
            self.sim_device,
            "--overrides-json",
            json.dumps(self.extra_overrides),
        ]
        if self.headless:
            cmd.append("--worker-headless")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
        try:
            self._conn = listener.accept()
        except socket.timeout as exc:
            self._kill_subprocess()
            raise TimeoutError("Timed out waiting for IsaacSim worker to connect") from exc
        finally:
            listener.close()
        print(f"[launcher] Spawned pid={self._proc.pid} scene={scene_idx} tol={tol_slot} peg={peg_idx}")

    def _apply_peg_change(self) -> None:
        if self._conn is not None and self._env_ready:
            peg_idx = int(self._dd_peg.value)
            try:
                self._conn.send(("set_peg_idx", peg_idx))
                print(f"[launcher] Queued peg_idx={peg_idx} for next reset")
            except (BrokenPipeError, EOFError, OSError):
                pass

    def _send(self, msg: Any) -> None:
        if self._conn is not None:
            try:
                self._conn.send(msg)
            except (BrokenPipeError, EOFError, OSError):
                pass

    def _cmd_run(self) -> None:
        if not self._env_ready:
            self._md_status.content = "**Status:** Load an environment first."
            return
        if self._episode_running:
            return
        self._episode_running = True
        self._is_paused = False
        self._btn_pause.name = "Pause"
        self._md_status.content = "**Status:** Running episode..."
        self._md_retract.content = "**Retract:** --"
        self._send("run")

    def _cmd_pause(self) -> None:
        if not self._episode_running:
            return
        self._is_paused = not self._is_paused
        self._send("pause" if self._is_paused else "resume")
        self._btn_pause.name = "Resume" if self._is_paused else "Pause"

    def _cmd_stop(self) -> None:
        if self._episode_running:
            self._send("stop")

    def _update_viz(self, state: dict[str, Any]) -> None:
        self._robot_urdf.update_cfg(np.asarray(state["joint_pos"], dtype=float))
        self._robot_frame.position = tuple(state["robot_pose"][:3])
        self._robot_frame.wxyz = _xyzw_to_wxyz(state["robot_pose"][3:7])

        if self._table_frame is not None:
            self._table_frame.position = tuple(state["table_pose"][:3])
            self._table_frame.wxyz = _xyzw_to_wxyz(state["table_pose"][3:7])
        if self._obj_frame is not None:
            self._obj_frame.position = tuple(state["object_pose"][:3])
            self._obj_frame.wxyz = _xyzw_to_wxyz(state["object_pose"][3:7])
        if self._goal_frame is not None:
            self._goal_frame.position = tuple(state["goal_pose"][:3])
            self._goal_frame.wxyz = _xyzw_to_wxyz(state["goal_pose"][3:7])

        if not self._obj_keypoints:
            self._setup_keypoints(len(state["obj_keypoints"]))
        for handle, pos in zip(self._obj_keypoints, state["obj_keypoints"]):
            handle.position = tuple(pos)
        for handle, pos in zip(self._goal_keypoints, state["goal_keypoints"]):
            handle.position = tuple(pos)

    def _handle(self, msg: Any) -> None:
        tag = msg[0]
        if tag == "status":
            self._md_status.content = f"**Status:** {msg[1]}"
        elif tag == "ready":
            state = msg[1]
            self._update_viz(state)
            self._env_ready = True
            self._md_status.content = "**Status:** Ready - click **Run Episode**"
            print("[launcher] Environment ready")
        elif tag == "state":
            state = msg[1]
            self._update_viz(state)
            successes = state["successes"]
            max_goals = state["max_goals"]
            pct = 100.0 * successes / max(1, max_goals)
            self._md_prog.content = (
                f"**Time:** {state['step'] / 60.0:.1f}s | "
                f"**Goal:** {successes}/{max_goals} ({pct:.0f}%)"
            )
            kp_max = state["keypoints_max_dist"]
            tol = state["success_tolerance"]
            in_tol = "yes" if kp_max <= tol else "no"
            self._md_diag.content = (
                f"**Goal dist:** {kp_max * 1000:.1f} mm ({in_tol}) "
                f"| tol {tol * 1000:.1f} mm | near-goal steps: "
                f"**{state['near_goal_steps']}**  \n"
                f"**progress_buf:** {state['progress']}/{state['max_episode_length']} "
                f"| reset_buf: **{state['reset_pending']}**"
            )
            if state["retract_success"]:
                self._md_retract.content = (
                    f"**Retract:** SUCCESS "
                    f"(hand dist {state['mean_fingertip_dist']:.3f}m)"
                )
            elif state["retract_phase"]:
                self._md_retract.content = (
                    f"**Retract:** IN PROGRESS "
                    f"(hand dist {state['mean_fingertip_dist']:.3f}m)"
                )
            else:
                self._md_retract.content = (
                    f"**Retract:** not yet "
                    f"(hand dist {state['mean_fingertip_dist']:.3f}m)"
                )
        elif tag == "done":
            result = msg[1]
            self._episode_running = False
            self.ep_count += 1
            self._md_stats.content = (
                f"**Episodes:** {self.ep_count} | "
                f"**Last goal:** {result['goal_pct']:.0f}% | "
                f"**Last time:** {result['steps'] / 60.0:.1f}s | "
                f"**Return:** {result['return']:.1f}"
            )
            self._md_status.content = (
                f"**Status:** Done - {result['termination']}, "
                f"{result['goal_pct']:.0f}% goals | "
                f"Retract {'OK' if result['retract_success'] else 'FAIL'}"
            )
            print(
                "[launcher] Episode done: "
                f"{result['goal_pct']:.0f}% goals, "
                f"{result['steps'] / 60.0:.1f}s, "
                f"retract={result['retract_success']}"
            )
        elif tag == "stopped":
            self._episode_running = False
            self._md_status.content = "**Status:** Episode stopped."
        elif tag == "error":
            self._env_ready = False
            self._episode_running = False
            self._md_status.content = f"**Status:** Error - {msg[1][:200]}"
            print(f"[launcher] Worker error:\n{msg[1]}")

    def _poll(self) -> None:
        if self._conn is None:
            return
        try:
            while self._conn.poll(0):
                self._handle(self._conn.recv())
        except (EOFError, ConnectionResetError, OSError):
            self._conn = None
            self._env_ready = False
            self._episode_running = False
            self._md_status.content = "**Status:** Worker connection closed."

    def run(self, port: int) -> None:
        print()
        print(f"  Peg-in-Hole IsaacSim Multi-Init Eval   http://localhost:{port}")
        print()
        try:
            while True:
                self._poll()
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\n[launcher] Shutting down...")
            self._kill_subprocess()


def _resolve_path(path: str) -> str:
    p = Path(path).expanduser()
    if p.exists():
        return str(p)
    p = REPO_ROOT / path
    if p.exists():
        return str(p)
    raise FileNotFoundError(path)


def _resolve_policies(args: argparse.Namespace) -> Dict[str, Tuple[Optional[str], str]]:
    policies: Dict[str, Tuple[Optional[str], str]] = {}
    if args.policies_dir is not None:
        root = Path(args.policies_dir).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        if not root.exists():
            raise FileNotFoundError(f"--policies-dir not found: {root}")
        allowed = set(args.policies) if args.policies else None
        for subdir in sorted(root.iterdir()):
            if allowed is not None and subdir.name not in allowed:
                continue
            ckpt = subdir / "model.pth"
            cfg = subdir / "config.yaml"
            if ckpt.exists():
                policies[subdir.name] = (str(cfg) if cfg.exists() else None, str(ckpt))
        if not policies:
            raise FileNotFoundError(f"No policy subfolders with model.pth in {root}")
        return policies

    checkpoint = args.checkpoint_path or args.checkpoint
    if checkpoint is None:
        raise ValueError("Provide --checkpoint-path, --checkpoint, or --policies-dir.")
    name = Path(checkpoint).expanduser().parent.name or "policy"
    policies[name] = (
        _resolve_path(args.config_path) if args.config_path else None,
        _resolve_path(checkpoint),
    )
    return policies


def _parse_main_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8043)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--policies-dir", type=str, default=None)
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument("--initial-policy", type=str, default=None)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="dense")
    parser.add_argument("--task", default="Isaacsimenvs-PegInHole-Direct-v0")
    parser.add_argument("--agent", default="rl_games_sapg_cfg_entry_point")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--isaacsim-python", type=str, default=str(_isaacsim_python()))
    parser.add_argument(
        "--override",
        nargs=2,
        action="append",
        default=[],
        metavar=("CFG_PATH", "VALUE"),
        help="Extra IsaacSim env cfg override, e.g. peg_in_hole.goal_xy_obs_noise 0.0",
    )
    return parser.parse_args()


def _parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--ipc-host", required=True)
    parser.add_argument("--ipc-port", type=int, required=True)
    parser.add_argument("--ipc-auth", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--scene-idx", type=int, required=True)
    parser.add_argument("--peg-idx", type=int, required=True)
    parser.add_argument("--tol-slot", type=int, required=True)
    parser.add_argument("--goal-mode", choices=GOAL_MODES, required=True)
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--worker-headless", action="store_true")
    parser.add_argument("--overrides-json", default="{}")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--success-tolerance", type=float, default=0.01)
    parser.add_argument("--goal-xy-obs-noise", type=float, default=0.0)
    parser.add_argument("--retract-distance-threshold", type=float, default=0.1)
    parser.add_argument("--retract-success-tolerance", type=float, default=0.005)
    parser.add_argument("--no-retract", action="store_true")
    parser.add_argument("--no-force-consecutive", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    if "--worker" in sys.argv:
        _worker_main(_parse_worker_args())
        return

    _ensure_viser_main()
    args = _parse_main_args()
    overrides = {
        key: _parse_override_value(value)
        for key, value in args.override
    }
    policies = _resolve_policies(args)
    demo = PegMultiInitIsaacSimDemo(
        policies=policies,
        port=args.port,
        headless=not args.no_headless,
        goal_mode=args.goal_mode,
        initial_policy=args.initial_policy,
        task=args.task,
        agent=args.agent,
        rl_device=args.rl_device,
        sim_device=args.sim_device,
        isaacsim_python=Path(args.isaacsim_python),
        extra_overrides=overrides,
    )
    demo.run(args.port)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Multi-init, multi-goal, multi-tolerance peg-in-hole evaluation (viser).

Mirrors ``peg_eval.py`` architecture (main-process viser GUI + IsaacGym
subprocess) but adapted for PegInHoleEnv's 3-axis scene schema
(``scene_idx × peg_idx × tol_slot_idx``) from ``scenes.npz``.

UI dropdowns:
  - Policy   — pick from --policies-dir / --config-path subfolders
  - Scene    — hole position (0..N-1)
  - Peg      — start pose within scene (0..M-1)
  - Tol slot — scene URDF clearance variant (0..K-1)
  - Goal mode — dense / preInsertAndFinal / finalGoalOnly

Scene, tol-slot, and goal-mode changes require the sim subprocess to be
restarted because each bakes a scene URDF / goal tensor at env-creation.
Peg changes are state-only (next reset picks the new peg).

Usage:
    python peg_in_hole/peg_multi_init_eval.py \\
        --config-path hardware_rollouts/Apr17_pegInHole/peg_1mm_with_dr/config.yaml \\
        --checkpoint-path hardware_rollouts/Apr17_pegInHole/peg_1mm_with_dr/model.pth \\
        --port 8043
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "peg_in_hole"
SCENES_NPZ = ASSETS_DIR / "scenes" / "scenes.npz"

TABLE_Z = 0.38
N_ACT = 29
OBS_DIM = 140
CONTROL_DT = 1.0 / 60.0

_ARM_DEFAULT = np.array([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308])
_ARM_DEFAULT[1] -= np.deg2rad(10)
_ARM_DEFAULT[3] += np.deg2rad(10)
DEFAULT_DOF_POS = np.zeros(N_ACT)
DEFAULT_DOF_POS[:7] = _ARM_DEFAULT

GOAL_MODES = ["dense", "preInsertAndFinal", "finalGoalOnly"]

# Peg-in-hole geometry constants (must match create_peg_and_holes.py)
HOLE_FOOTPRINT = (0.08, 0.08)
HOLE_SLOT_CORE = (0.02, 0.03)
HOLE_FLOOR_THICKNESS = 0.01
HOLE_DEPTH = 0.05
HOLE_SCENE_Z = 0.15

BASE_OVERRIDES = {
    # Deterministic-ish eval: kill reset noise and DR.
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
    "task.env.forceConsecutiveNearGoalSteps": True,
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
    "task.env.viserViz": False,
}


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


# ===================================================================
# Scene data loading
# ===================================================================

def _load_scenes() -> dict:
    if not SCENES_NPZ.exists():
        raise FileNotFoundError(
            f"{SCENES_NPZ} not found. Run generate_scenes.py first."
        )
    d = np.load(str(SCENES_NPZ))
    return {
        "start_poses": d["start_poses"],             # (N, M, 7)
        "goals": d["goals"],                          # (N, M, T, 7)
        "traj_lengths": d["traj_lengths"],           # (N, M)
        "hole_positions": d["hole_positions"],       # (N, 3)
        "tolerance_pool_m": d["tolerance_pool_m"],    # (pool_size,)
        "scene_tolerance_indices": d["scene_tolerance_indices"],  # (N, K)
    }


def _hole_mesh_for(scene_idx: int, tol_slot: int, scenes: dict) -> trimesh.Trimesh:
    """Build a trimesh of the hole block (outer + slot walls) for the given
    (scene, tol_slot). Origin at the hole's scene-local base corner."""
    tol_m = float(scenes["tolerance_pool_m"][scenes["scene_tolerance_indices"][scene_idx, tol_slot]])
    slot_x = HOLE_SLOT_CORE[0] + 2 * tol_m
    slot_y = HOLE_SLOT_CORE[1] + 2 * tol_m
    ox, oy = HOLE_FOOTPRINT
    t = HOLE_FLOOR_THICKNESS
    d = HOLE_DEPTH

    boxes = [((0.0, 0.0, t / 2), (ox, oy, t))]
    zc = t + d / 2
    ew = (ox - slot_x) / 2
    if ew > 1e-6:
        boxes.append(((slot_x / 2 + ew / 2, 0.0, zc), (ew, oy, d)))
        boxes.append(((-(slot_x / 2 + ew / 2), 0.0, zc), (ew, oy, d)))
    nl = (oy - slot_y) / 2
    if nl > 1e-6:
        boxes.append(((0.0, slot_y / 2 + nl / 2, zc), (slot_x, nl, d)))
        boxes.append(((0.0, -(slot_y / 2 + nl / 2), zc), (slot_x, nl, d)))

    meshes = []
    for (cx, cy, cz), ext in boxes:
        m = trimesh.creation.box(extents=np.asarray(ext, dtype=float))
        m.apply_translation((cx, cy, cz))
        meshes.append(m)
    return trimesh.util.concatenate(meshes)


# ===================================================================
# SUBPROCESS -- IsaacGym simulation
# ===================================================================

def _create_env(config_path, headless, device, overrides):
    from deployment.rl_player_utils import read_cfg_omegaconf
    from deployment.isaac.isaac_env import merge_cfg_with_default_config, create_env_from_cfg
    from omegaconf import OmegaConf

    cfg = read_cfg_omegaconf(config_path=config_path, device=device)
    cfg = merge_cfg_with_default_config(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.task.name = "PegInHoleEnv"
    cfg.task_name = "PegInHoleEnv"
    pih_defaults = {
        "scenesPath": "assets/urdf/peg_in_hole/scenes/scenes.npz",
        "goalMode": "dense",
        "objectName": "peg",
        "useFixedGoalStates": True,
        "useFixedInitObjectPose": True,
        "enableRetract": True,
        "retractRewardScale": 1.0,
        "retractDistanceThreshold": 0.1,
        "retractSuccessBonus": 1000.0,
        "retractSuccessTolerance": 0.005,
        "forceSceneTolCombo": None,
        "forcePegIdx": None,
    }
    for k, v in pih_defaults.items():
        OmegaConf.update(cfg, f"task.env.{k}", v, force_add=True)
    return create_env_from_cfg(cfg=cfg, headless=headless, overrides=overrides)


def _sim_get_state(env, obs, joint_lower, joint_upper):
    obs_np = obs[0].cpu().numpy()
    joint_pos = 0.5 * (obs_np[:N_ACT] + 1.0) * (joint_upper - joint_lower) + joint_lower
    return (
        joint_pos,
        env.object_state[0, :7].cpu().numpy(),
        env.goal_pose[0].cpu().numpy(),
        env.obj_keypoint_pos[0].cpu().numpy(),
        env.goal_keypoint_pos[0].cpu().numpy(),
        bool(env.retract_phase[0].item()),
        bool(env.retract_succeeded[0].item()),
        float(env.curr_fingertip_distances[0].mean().item()),
    )


def _sim_reset(env, device):
    import torch
    obs, _, _, _ = env.step(torch.zeros((env.num_envs, N_ACT), device=device))
    return obs["obs"]


def _sim_episode(conn, env, policy, joint_lower, joint_upper, device):
    import time as _time
    import torch  # noqa: F401
    policy.reset()
    obs = _sim_reset(env, device)

    step, done, paused = 0, False, False
    while not done:
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
                # Runtime peg_idx change (applied next reset)
                env.cfg["env"]["forcePegIdx"] = int(cmd[1])

        if paused:
            _time.sleep(0.05)
            continue

        t0 = _time.time()
        state = _sim_get_state(env, obs, joint_lower, joint_upper)
        action = policy.get_normalized_action(obs, deterministic_actions=True)
        obs_dict, _, done_tensor, _ = env.step(action)
        obs = obs_dict["obs"]
        done = done_tensor[0].item()
        step += 1

        conn.send((
            "state", state,
            int(env.successes[0].item()),
            int(env.env_max_goals[0].item()) if hasattr(env, "env_max_goals") else env.max_consecutive_successes,
            step,
        ))

        elapsed = _time.time() - t0
        if (sleep := CONTROL_DT - elapsed) > 0:
            _time.sleep(sleep)

    goal_pct = 100 * int(env.successes[0].item()) / max(1, int(env.env_max_goals[0].item()))
    conn.send(("done", goal_pct, step, bool(env.retract_succeeded[0].item())))
    return obs


def sim_worker(conn, config_path, checkpoint_path, scene_idx, tol_slot_idx,
               peg_idx, goal_mode, extra_overrides=None, headless=True):
    try:
        from isaacgym import gymapi  # noqa: F401 isort:skip
        import torch
        from deployment.rl_player import RlPlayer
        import peg_in_hole.objects  # noqa: F401  (registers peg)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        overrides = {
            **BASE_OVERRIDES,
            "task.env.forceSceneTolCombo": [int(scene_idx), int(tol_slot_idx)],
            "task.env.forcePegIdx": int(peg_idx),
            "task.env.goalMode": goal_mode,
            **(extra_overrides or {}),
        }
        env = _create_env(config_path=str(config_path), headless=headless,
                          device=device, overrides=overrides)

        joint_lower = env.arm_hand_dof_lower_limits[:N_ACT].cpu().numpy()
        joint_upper = env.arm_hand_dof_upper_limits[:N_ACT].cpu().numpy()
        env.set_env_state(torch.load(checkpoint_path, map_location=device)[0]["env_state"])
        policy = RlPlayer(OBS_DIM, N_ACT, config_path, checkpoint_path, device, env.num_envs)

        obs = _sim_reset(env, device)
        init_state = _sim_get_state(env, obs, joint_lower, joint_upper)
        conn.send(("ready", init_state))

        while True:
            cmd = conn.recv()
            if cmd == "run":
                obs = _sim_episode(conn, env, policy, joint_lower, joint_upper, device)
            elif isinstance(cmd, tuple) and cmd[0] == "set_peg_idx":
                env.cfg["env"]["forcePegIdx"] = int(cmd[1])
            elif cmd == "quit":
                break
    except Exception as exc:
        conn.send(("error", f"{exc}\n{traceback.format_exc()}"))
    conn.close()


# ===================================================================
# MAIN PROCESS -- viser GUI
# ===================================================================

class PegMultiInitDemo:
    _PH = "-- Select --"

    def __init__(self, policies: Dict[str, Tuple[str, str]],
                 port: int = 8043, headless: bool = True,
                 goal_mode: str = "dense", initial_policy: Optional[str] = None,
                 extra_overrides: Optional[dict] = None):
        if goal_mode not in GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {GOAL_MODES}")
        if not policies:
            raise ValueError("policies dict must be non-empty")
        self.policies = policies
        self.initial_policy = initial_policy if initial_policy in policies else next(iter(policies))
        self.port = port
        self.headless = headless
        self.goal_mode = goal_mode
        self.extra_overrides = extra_overrides or {}
        self.server = viser.ViserServer(host="0.0.0.0", port=port)

        self._proc = None
        self._conn = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False

        self.scenes = _load_scenes()
        self.N = self.scenes["start_poses"].shape[0]
        self.M = self.scenes["start_poses"].shape[1]
        self.K = self.scenes["scene_tolerance_indices"].shape[1]

        self.ep_count = 0

        self.robot = None
        self._dyn = []
        self._obj_frame = None
        self._goal_frame = None
        self._obj_keypoints = []
        self._goal_keypoints = []

        self._build_gui()
        self._setup_static_scene()

    # ── GUI ──────────────────────────────────────────────────────

    def _build_gui(self):
        self.server.gui.add_markdown(
            "# Peg-in-Hole Multi-Init Eval\n"
            "### Pretrained policy on (scene × peg × tolerance) scenes"
        )

        with self.server.gui.add_folder("Task Selection", expand_by_default=True):
            self._dd_policy = self.server.gui.add_dropdown(
                "Policy", options=list(self.policies.keys()),
                initial_value=self.initial_policy,
            )
            self._dd_scene = self.server.gui.add_dropdown(
                "Scene idx", options=[str(i) for i in range(self.N)], initial_value="0",
            )
            self._dd_peg = self.server.gui.add_dropdown(
                "Peg idx", options=[str(i) for i in range(self.M)], initial_value="0",
            )
            self._dd_tol = self.server.gui.add_dropdown(
                "Tol slot idx", options=[str(i) for i in range(self.K)], initial_value="0",
            )
            self._dd_goal_mode = self.server.gui.add_dropdown(
                "Goal mode", options=GOAL_MODES, initial_value=self.goal_mode,
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
            self._cb_keypoints = self.server.gui.add_checkbox("Show keypoints", initial_value=True)
            self._cb_keypoints.on_update(lambda _: self._apply_keypoint_visibility())

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self._md_task = self.server.gui.add_markdown("**Task:** --")
            self._md_prog = self.server.gui.add_markdown("**Progress:** --")
            self._md_retract = self.server.gui.add_markdown("**Retract:** --")
            self._md_stats = self.server.gui.add_markdown("**Stats:** No episodes yet")
            self._md_tol_value = self.server.gui.add_markdown("**Tol value:** --")

        # peg dropdown can change mid-run (state only, no subprocess restart)
        self._dd_peg.on_update(lambda _: self._apply_peg_change())

    def _setup_static_scene(self):
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            client.camera.position = (0.0, -1.0, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.5)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        self.server.scene.add_frame(
            "/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self.robot = ViserUrdf(
            self.server,
            REPO_ROOT / "assets" / "urdf" / "kuka_sharpa_description" / "iiwa14_left_sharpa_adjusted_restricted.urdf",
            root_node_name="/robot",
        )
        self.robot.update_cfg(DEFAULT_DOF_POS)

        # Table (shared across scenes)
        self.server.scene.add_frame(
            "/table", position=(0, 0, TABLE_Z), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self.server.scene.add_box(
            "/table/wood", color=(180, 130, 70),
            dimensions=(0.475, 0.4, 0.3), position=(0, 0, 0),
            side="double", opacity=0.9,
        )

    def _clear_dynamic(self):
        for h in self._dyn:
            try:
                h.remove()
            except Exception:
                pass
        self._dyn.clear()
        self._obj_frame = self._goal_frame = None
        self._obj_keypoints.clear()
        self._goal_keypoints.clear()

    def _setup_scene_objects(self, scene_idx: int, tol_slot: int):
        """Render peg + goal ghost + hole mesh at the selected scene/tol."""
        self._clear_dynamic()

        peg_urdf = ASSETS_DIR / "peg" / "peg.urdf"

        self._obj_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        self._dyn.append(self._obj_frame)
        ViserUrdf(self.server, peg_urdf, root_node_name="/object",
                  mesh_color_override=(204, 40, 40))

        self._goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        self._dyn.append(self._goal_frame)
        ViserUrdf(self.server, peg_urdf, root_node_name="/goal",
                  mesh_color_override=(0, 255, 0, 0.5))

        # Hole: scene's hole XY + tol_slot's geometry. Place at world pose.
        hole_x, hole_y, _ = self.scenes["hole_positions"][scene_idx]
        hole_mesh = _hole_mesh_for(scene_idx, tol_slot, self.scenes)

        hole_frame = self.server.scene.add_frame(
            "/hole",
            position=(float(hole_x), float(hole_y), TABLE_Z + HOLE_SCENE_Z),
            wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self._dyn.append(hole_frame)
        self._dyn.append(self.server.scene.add_mesh_simple(
            "/hole/mesh",
            vertices=np.array(hole_mesh.vertices, dtype=np.float32),
            faces=np.array(hole_mesh.faces, dtype=np.uint32),
            color=(120, 120, 120),
        ))

        tol_m = float(self.scenes["tolerance_pool_m"][
            self.scenes["scene_tolerance_indices"][scene_idx, tol_slot]
        ])
        self._md_tol_value.content = f"**Tol value:** {tol_m * 1000:.4f} mm (slot {tol_slot}, scene {scene_idx})"

    def _setup_keypoints(self, num_keypoints):
        for kp in self._obj_keypoints + self._goal_keypoints:
            try:
                kp.remove()
            except Exception:
                pass
        self._obj_keypoints.clear()
        self._goal_keypoints.clear()
        for i in range(num_keypoints):
            self._obj_keypoints.append(
                self.server.scene.add_icosphere(f"/obj_kp/{i}", radius=0.005, color=(255, 0, 0))
            )
            self._goal_keypoints.append(
                self.server.scene.add_icosphere(f"/goal_kp/{i}", radius=0.005, color=(0, 255, 0), opacity=0.5)
            )
        self._apply_keypoint_visibility()

    def _apply_keypoint_visibility(self):
        visible = self._cb_keypoints.value
        for kp in self._obj_keypoints + self._goal_keypoints:
            kp.visible = visible

    # ── Subprocess management ────────────────────────────────────

    def _kill_subprocess(self):
        if self._conn is not None:
            try:
                self._conn.send("quit")
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
        self._episode_running = False
        self._is_paused = False

    def _load_env(self):
        self._kill_subprocess()

        scene_idx = int(self._dd_scene.value)
        peg_idx = int(self._dd_peg.value)
        tol_slot = int(self._dd_tol.value)
        goal_mode = self._dd_goal_mode.value
        policy_name = self._dd_policy.value
        config_path, checkpoint_path = self.policies[policy_name]

        label = (
            f"{policy_name} | scene {scene_idx} | peg {peg_idx} | tol slot {tol_slot}"
            f" | goals: {goal_mode}"
        )
        self._md_status.content = f"**Status:** Loading *{label}* ..."
        self._md_task.content = f"**Task:** {label}"
        self._md_retract.content = "**Retract:** --"
        self.ep_count = 0
        self._md_stats.content = "**Stats:** No episodes yet"

        self.robot.update_cfg(DEFAULT_DOF_POS)
        self._setup_scene_objects(scene_idx, tol_slot)

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=sim_worker,
            args=(child_conn, config_path, checkpoint_path,
                  scene_idx, tol_slot, peg_idx, goal_mode,
                  self.extra_overrides, self.headless),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        print(f"[launcher] Spawned pid={self._proc.pid} scene={scene_idx} tol={tol_slot} peg={peg_idx}")

    def _apply_peg_change(self):
        if self._conn is not None and self._env_ready:
            new_peg = int(self._dd_peg.value)
            try:
                self._conn.send(("set_peg_idx", new_peg))
                print(f"[launcher] Queued peg_idx={new_peg} for next reset")
            except (BrokenPipeError, OSError):
                pass

    def _send(self, msg):
        if self._conn is not None:
            try:
                self._conn.send(msg)
            except (BrokenPipeError, OSError):
                pass

    def _cmd_run(self):
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

    def _cmd_pause(self):
        if not self._episode_running:
            return
        self._is_paused = not self._is_paused
        self._send("pause" if self._is_paused else "resume")
        self._btn_pause.name = "Resume" if self._is_paused else "Pause"

    def _cmd_stop(self):
        if self._episode_running:
            self._send("stop")

    def _update_viz(self, state_tuple):
        joint_pos, obj_pose, goal_pose = state_tuple[0], state_tuple[1], state_tuple[2]
        self.robot.update_cfg(joint_pos)

        if self._obj_frame is not None:
            self._obj_frame.position = tuple(obj_pose[:3])
            self._obj_frame.wxyz = quat_xyzw_to_wxyz(obj_pose[3:7])
        if self._goal_frame is not None:
            self._goal_frame.position = tuple(goal_pose[:3])
            self._goal_frame.wxyz = quat_xyzw_to_wxyz(goal_pose[3:7])

        if len(state_tuple) > 3:
            obj_kps, goal_kps = state_tuple[3], state_tuple[4]
            for handle, pos in zip(self._obj_keypoints, obj_kps):
                handle.position = tuple(pos)
            for handle, pos in zip(self._goal_keypoints, goal_kps):
                handle.position = tuple(pos)

    def _handle(self, msg):
        tag = msg[0]
        if tag == "ready":
            init_state = msg[1]
            if len(init_state) > 3:
                self._setup_keypoints(init_state[3].shape[0])
            self._update_viz(init_state)
            self._env_ready = True
            self._md_status.content = "**Status:** Ready — click **Run Episode**"
            print("[launcher] Environment ready")
        elif tag == "state":
            state, successes, max_succ, step = msg[1], msg[2], msg[3], msg[4]
            self._update_viz(state)
            pct = 100 * successes / max_succ if max_succ > 0 else 0
            self._md_prog.content = (
                f"**Time:** {step / 60.0:.1f}s &nbsp;|&nbsp; "
                f"**Goal:** {successes}/{max_succ} ({pct:.0f}%)"
            )
            if len(state) >= 8:
                retract_phase, retract_ok, mean_ft_dist = state[5], state[6], state[7]
                if retract_ok:
                    self._md_retract.content = f"**Retract:** SUCCESS (hand dist {mean_ft_dist:.3f}m)"
                elif retract_phase:
                    self._md_retract.content = f"**Retract:** IN PROGRESS (hand dist {mean_ft_dist:.3f}m)"
        elif tag == "done":
            goal_pct, steps, retract_ok = msg[1], msg[2], msg[3]
            self._episode_running = False
            self.ep_count += 1
            self._md_stats.content = (
                f"**Episodes:** {self.ep_count} &nbsp;|&nbsp; "
                f"**Last goal:** {goal_pct:.0f}% &nbsp;|&nbsp; "
                f"**Last time:** {steps / 60.0:.1f}s"
            )
            self._md_status.content = (
                f"**Status:** Done — {steps / 60.0:.1f}s, {goal_pct:.0f}% goals"
                f" | Retract {'OK' if retract_ok else 'FAIL'}"
            )
            self._md_retract.content = f"**Retract:** {'SUCCESS' if retract_ok else 'FAILED'}"
            print(f"[launcher] Episode done: {goal_pct:.0f}% goals, {steps / 60.0:.1f}s")
        elif tag == "stopped":
            self._episode_running = False
            self._md_status.content = "**Status:** Episode stopped."
        elif tag == "error":
            self._env_ready = False
            self._episode_running = False
            self._md_status.content = f"**Status:** Error — {msg[1][:200]}"
            print(f"[launcher] Subprocess error:\n{msg[1]}")

    def _poll(self):
        if self._conn is None:
            return
        try:
            while self._conn.poll(0):
                self._handle(self._conn.recv())
        except (EOFError, ConnectionResetError, OSError):
            self._conn = None
            if self._proc is not None and not self._proc.is_alive():
                self._md_status.content = "**Status:** Subprocess exited unexpectedly."
                self._proc = None
                self._env_ready = False
                self._episode_running = False

    def run(self):
        print()
        print(f"  Peg-in-Hole Multi-Init Eval   http://localhost:{self.port}")
        print()
        try:
            while True:
                self._poll()
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\n[launcher] Shutting down...")
            self._kill_subprocess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8043)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--policies-dir", type=str, default=None,
                        help="Directory of policy subfolders (each with config.yaml + model.pth).")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="Policy names (filters --policies-dir).")
    parser.add_argument("--initial-policy", type=str, default=None)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="dense")
    parser.add_argument("--override", nargs=2, action="append", default=[],
                        metavar=("KEY", "VALUE"),
                        help="Extra task-env overrides (repeated).")
    args = parser.parse_args()

    def _resolve(p):
        path = Path(p)
        if path.exists():
            return str(path)
        path = REPO_ROOT / p
        if path.exists():
            return str(path)
        raise FileNotFoundError(p)

    extra_overrides = {}
    for key, val in args.override:
        for cast in (int, float):
            try:
                val = cast(val)
                break
            except ValueError:
                continue
        if val == "True":
            val = True
        elif val == "False":
            val = False
        extra_overrides[key] = val

    policies: Dict[str, Tuple[str, str]] = {}
    if args.policies_dir is not None:
        pdir = Path(args.policies_dir)
        if not pdir.is_absolute():
            pdir = REPO_ROOT / pdir
        if not pdir.exists():
            raise FileNotFoundError(f"--policies-dir not found: {pdir}")
        name_filter = set(args.policies) if args.policies else None
        for sub in sorted(pdir.iterdir()):
            if name_filter is not None and sub.name not in name_filter:
                continue
            cfg = sub / "config.yaml"
            ckpt = sub / "model.pth"
            if cfg.exists() and ckpt.exists():
                policies[sub.name] = (str(cfg), str(ckpt))
        if not policies:
            raise FileNotFoundError(f"No policy subfolders in {pdir}")
    else:
        if args.config_path is None or args.checkpoint_path is None:
            raise ValueError("Provide either --policies-dir or both --config-path and --checkpoint-path")
        name = Path(args.config_path).parent.name or "policy"
        policies[name] = (_resolve(args.config_path), _resolve(args.checkpoint_path))

    PegMultiInitDemo(
        policies=policies,
        port=args.port,
        headless=not args.no_headless,
        goal_mode=args.goal_mode,
        initial_policy=args.initial_policy,
        extra_overrides=extra_overrides,
    ).run()

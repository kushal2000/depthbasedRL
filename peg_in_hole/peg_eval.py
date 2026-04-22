"""Peg-in-Hole evaluation with pretrained policy.

Mirrors fabrica/fabrica_eval.py: viser GUI main process + IsaacGym subprocess
running FabricaEnv. A single "Tolerance" dropdown replaces the fabrica
assembly/part dropdowns.

Usage:
    python peg_in_hole/peg_eval.py \
        --config-path pretrained_policy/config.yaml \
        --checkpoint-path pretrained_policy/model.pth
"""

import argparse
import json
import math
import multiprocessing
import re
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

GOAL_MODES = ["dense", "final_only", "pre_insert_and_final"]
_GOAL_MODE_OVERRIDE_KEY = {
    "final_only": "task.env.finalGoalOnly",
    "pre_insert_and_final": "task.env.preInsertAndFinal",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "peg_in_hole"

TABLE_Z = 0.38
Z_OFFSET = 0.03
N_ACT = 29
OBS_DIM = 140
CONTROL_DT = 1.0 / 60.0

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
    # Disable the env-side debug viser (each sim_worker subprocess would
    # otherwise spawn its own viser server and the port numbers drift).
    "task.env.viserViz": False,
    # Enable the table+hole force sensor so the peak insertion force can be
    # observed during eval. NOTE: with the default TGS
    # num_position_iterations=8, the reported force is ~8× the physical
    # value (see peg_multi_init_eval.py comments for the diagnostic
    # walkthrough). Interpret readings accordingly.
    "task.env.withTableForceSensor": True,
    # Eval-only: disable the 100 N force-reset trigger so we can see the
    # full peak force during insertion. Training keeps the default.
    "task.env.tableForceResetThreshold": 1.0e6,
}

_TOL_DIR_RE = re.compile(r"^hole_tol([0-9p]+)mm$")


def _get_available_tolerances() -> List[str]:
    """Return tolerance tags (e.g. ['0p1', '0p5', '1', '5']) whose hole dir has
    both scene.urdf and trajectories/peg/pick_place.json.
    """
    holes_root = ASSETS_DIR / "holes"
    if not holes_root.exists():
        return []
    out = []
    for sub in sorted(holes_root.iterdir()):
        m = _TOL_DIR_RE.match(sub.name)
        if not m:
            continue
        scene = sub / "scene.urdf"
        traj = sub / "trajectories" / "peg" / "pick_place.json"
        if scene.exists() and traj.exists():
            out.append(m.group(1))
    # Sort numerically (0p1 < 0p5 < 1 < 5)
    def _key(tag):
        return float(tag.replace("p", "."))
    return sorted(out, key=_key)


def _load_trajectory(tol_tag: str) -> Optional[dict]:
    path = ASSETS_DIR / "holes" / f"hole_tol{tol_tag}mm" / "trajectories" / "peg" / "pick_place.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


def _scene_urdf_rel(tol_tag: str) -> str:
    return f"urdf/peg_in_hole/holes/hole_tol{tol_tag}mm/scene.urdf"


# ===================================================================
# SUBPROCESS -- IsaacGym simulation (FabricaEnv forced)
# ===================================================================

def _create_fabrica_env(config_path, headless, device, overrides):
    from deployment.rl_player_utils import read_cfg_omegaconf
    from deployment.isaac.isaac_env import merge_cfg_with_default_config, create_env_from_cfg
    from omegaconf import OmegaConf

    cfg = read_cfg_omegaconf(config_path=config_path, device=device)
    cfg = merge_cfg_with_default_config(cfg)

    OmegaConf.set_struct(cfg, False)
    cfg.task.name = "FabricaEnv"
    cfg.task_name = "FabricaEnv"
    fabrica_defaults = {
        "enableRetract": True,
        "retractDistanceThreshold": 0.5,
        "retractRewardScale": 1.0,
        "retractSuccessBonus": 0.0,
        "multiPart": False,
        "multiInitStates": False,
        "finalGoalOnly": False,
        "preInsertAndFinal": False,
        "objectNames": None,
    }
    for k, v in fabrica_defaults.items():
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
        # Diagnostic: max keypoint distance, current success tolerance,
        # near-goal-step counter, progress_buf, max_episode_length, reset_buf.
        float(env.keypoints_max_dist[0].item()),
        float(env.success_tolerance * env.keypoint_scale),
        int(env.near_goal_steps[0].item()) if hasattr(env, "near_goal_steps") else 0,
        int(env.progress_buf[0].item()),
        int(env.max_episode_length),
        bool(env.reset_buf[0].item()),
        # Table force sensor (covers table + hole body since they're one
        # URDF). World-frame 3-vector [fx, fy, fz] in Newtons; smoothed.
        env.table_sensor_forces_smoothed[0, :3].cpu().numpy()
        if hasattr(env, "table_sensor_forces_smoothed")
        else np.zeros(3, dtype=np.float32),
    )


def _sim_reset(env, device):
    import torch
    obs, _, _, _ = env.step(torch.zeros((env.num_envs, N_ACT), device=device))
    return obs["obs"]


def _setup_collision_camera(env):
    from isaacgym import gymapi
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 720
    cam_props.use_collision_geometry = True
    env._collision_cam = env.gym.create_camera_sensor(env.envs[0], cam_props)
    env._collision_cam_props = cam_props
    cam_target = gymapi.Vec3(0.0, 0.0, 0.53)
    cam_pos = cam_target + gymapi.Vec3(0.0, -1.0, 0.5)
    env.gym.set_camera_location(env._collision_cam, env.envs[0], cam_pos, cam_target)


def _capture_frame(env):
    from isaacgym import gymapi
    env.gym.render_all_camera_sensors(env.sim)
    cam = getattr(env, "_collision_cam", env.camera_handle)
    cam_props = getattr(env, "_collision_cam_props", env.camera_properties)
    color_image = env.gym.get_camera_image(env.sim, env.envs[0], cam, gymapi.IMAGE_COLOR)
    if color_image.size == 0:
        return None
    return color_image.reshape(cam_props.height, cam_props.width, 4)


def _save_video(frames, video_dir, tol_tag, ep_num):
    import imageio
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    path = video_dir / f"peg_tol{tol_tag}mm_ep{ep_num:03d}.mp4"
    imageio.mimsave(str(path), frames, fps=int(1.0 / CONTROL_DT))
    print(f"[sim_worker] Saved video: {path}")


def _sim_episode(conn, env, policy, joint_lower, joint_upper, device,
                 record_video=False, video_dir=None, tol_tag=None, ep_num=0):
    import time, torch  # noqa: E401

    policy.reset()
    obs = _sim_reset(env, device)
    frames = []

    if record_video:
        env.enable_viewer_sync = True

    step, done, paused = 0, False, False
    retract_ok = False
    last_arm_pos = None
    cum_total = 0.0
    cum_breakdown: dict = {}

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

        if paused:
            time.sleep(0.05)
            continue

        t0 = time.time()
        state = _sim_get_state(env, obs, joint_lower, joint_upper)
        last_arm_pos = env.arm_hand_dof_pos[0, :N_ACT].cpu().numpy().tolist()

        action = policy.get_normalized_action(obs, deterministic_actions=True)
        obs_dict, _, done_tensor, _ = env.step(action)
        obs = obs_dict["obs"]
        done = done_tensor[0].item()
        step += 1

        if env.extras.get("retract_success_ratio", 0) > 0.5:
            retract_ok = True

        cum_total += float(env.rew_buf[0].item())
        step_ec = env.extras.get("episode_cumulative", {}) or {}
        for _k, _v in step_ec.items():
            try:
                _val = float(_v[0].item()) if hasattr(_v, "__len__") else float(_v.item())
            except Exception:
                try:
                    _val = float(_v)
                except Exception:
                    continue
            cum_breakdown[_k] = cum_breakdown.get(_k, 0.0) + _val

        if record_video:
            frame = _capture_frame(env)
            if frame is not None:
                frames.append(frame)

        state_with_retract = state[:6] + (retract_ok,) + state[7:]
        conn.send((
            "state", state_with_retract,
            int(env.successes[0].item()),
            env.max_consecutive_successes,
            step,
            float(env.keypoints_max_dist[0].item()),
            cum_total,
            dict(cum_breakdown),
        ))

        elapsed = time.time() - t0
        if (sleep := CONTROL_DT - elapsed) > 0:
            time.sleep(sleep)

    if record_video and frames:
        _save_video(frames, video_dir, tol_tag, ep_num)

    goal_pct = 100 * int(env.successes[0].item()) / env.max_consecutive_successes
    final_arm_pos = last_arm_pos or env.arm_hand_dof_pos[0, :N_ACT].cpu().numpy().tolist()
    conn.send(("done", goal_pct, step, retract_ok, final_arm_pos))
    return obs


def sim_worker(conn, tol_tag, config_path, checkpoint_path, scene_urdf_rel,
               final_goal_tolerance=None, extra_overrides=None,
               headless=True, record_video=False, video_dir=None):
    try:
        from isaacgym import gymapi  # noqa: F401 isort:skip
        import json, torch  # noqa: E401
        from deployment.rl_player import RlPlayer
        import peg_in_hole.objects  # noqa: F401  (registers peg into NAME_TO_OBJECT)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        object_name = "peg"
        traj_path = ASSETS_DIR / "holes" / f"hole_tol{tol_tag}mm" / "trajectories" / "peg" / "pick_place.json"

        with open(traj_path) as f:
            traj = json.load(f)

        print(f"[sim_worker] tolerance: {tol_tag}mm")
        print(f"[sim_worker] scene_urdf: {scene_urdf_rel}")
        print(f"[sim_worker] object_name: {object_name}")
        print(f"[sim_worker] start_pose: {traj['start_pose']}")
        print(f"[sim_worker] #goals: {len(traj['goals'])}")

        env = _create_fabrica_env(
            config_path=str(config_path), headless=headless, device=device,
            overrides={
                **BASE_OVERRIDES,
                "task.env.objectName": object_name,
                "task.env.useFixedGoalStates": True,
                "task.env.fixedGoalStates": traj["goals"],
                "task.env.asset.table": scene_urdf_rel,
                "task.env.tableResetZ": TABLE_Z,
                "task.env.objectStartPose": traj["start_pose"],
                **({"task.env.finalGoalSuccessTolerance": final_goal_tolerance}
                   if final_goal_tolerance is not None else {}),
                **(extra_overrides or {}),
            },
        )

        joint_lower = env.arm_hand_dof_lower_limits[:N_ACT].cpu().numpy()
        joint_upper = env.arm_hand_dof_upper_limits[:N_ACT].cpu().numpy()

        env.set_env_state(torch.load(checkpoint_path, map_location=device)[0]["env_state"])

        policy = RlPlayer(OBS_DIM, N_ACT, config_path, checkpoint_path, device, env.num_envs)

        if record_video:
            _setup_collision_camera(env)

        obs = _sim_reset(env, device)
        init_state = _sim_get_state(env, obs, joint_lower, joint_upper)
        conn.send(("ready", init_state))

        ep_num = 0
        while True:
            cmd = conn.recv()
            if cmd == "run":
                obs = _sim_episode(
                    conn, env, policy, joint_lower, joint_upper, device,
                    record_video=record_video, video_dir=video_dir,
                    tol_tag=tol_tag, ep_num=ep_num,
                )
                ep_num += 1
            elif cmd == "quit":
                break

    except Exception as exc:
        conn.send(("error", f"{exc}\n{traceback.format_exc()}"))

    conn.close()


# ===================================================================
# MAIN PROCESS -- viser GUI
# ===================================================================

class PegInHoleDemo:

    def __init__(self, policies: dict, port: int = 8082,
                 final_goal_tolerance: float = None,
                 extra_overrides: dict = None, headless: bool = True,
                 record_video: bool = False, video_dir: str = None,
                 goal_mode: str = "dense",
                 initial_policy: str = None):
        if goal_mode not in GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {GOAL_MODES}, got {goal_mode}")
        if not policies:
            raise ValueError("policies dict must be non-empty")
        self.policies = policies
        if initial_policy is None or initial_policy not in policies:
            initial_policy = next(iter(policies))
        self.initial_policy = initial_policy
        self.port = port
        self.final_goal_tolerance = final_goal_tolerance
        self.record_video = record_video
        self.video_dir = video_dir or str(REPO_ROOT / "peg_in_hole" / "eval_videos")
        self.extra_overrides = extra_overrides or {}
        self.headless = headless
        self.goal_mode = goal_mode
        self.server = viser.ViserServer(host="0.0.0.0", port=port)

        self._proc = None
        self._conn = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False
        self._pending_tol = ""

        self.ep_count = 0
        self.ep_goals = []
        self.ep_lengths = []

        self.robot = None
        self._dyn = []
        self._obj_frame = None
        self._goal_frame = None
        self._obj_keypoints = []
        self._goal_keypoints = []

        self._trajectory = None  # dict | None

        self._tolerances = _get_available_tolerances()
        if not self._tolerances:
            raise FileNotFoundError(
                f"No valid hole variants in {ASSETS_DIR / 'holes'} "
                "(need scene.urdf + trajectories/peg/pick_place.json)"
            )

        self._build_gui()
        self._setup_static_scene()

    def _build_gui(self):
        self.server.gui.add_markdown("# Peg-in-Hole Eval\n### Pretrained policy on insertion task")

        _PH = "-- Select --"
        with self.server.gui.add_folder("Task Selection", expand_by_default=True):
            self._dd_policy = self.server.gui.add_dropdown(
                "Policy", options=list(self.policies.keys()),
                initial_value=self.initial_policy,
            )
            tol_labels = [_PH] + [f"{t.replace('p', '.')} mm" for t in self._tolerances]
            self._dd_tol = self.server.gui.add_dropdown(
                "Tolerance", options=tol_labels, initial_value=_PH,
            )
            self._dd_goal_mode = self.server.gui.add_dropdown(
                "Goal Mode", options=GOAL_MODES, initial_value=self.goal_mode,
            )
            self._btn_load = self.server.gui.add_button("Load Environment")
            self._btn_load.on_click(lambda _: self._load_env())
            self._md_status = self.server.gui.add_markdown("**Status:** Ready")

        with self.server.gui.add_folder("Episode Controls", expand_by_default=True):
            self._btn_run = self.server.gui.add_button("Run Episode")
            self._btn_run.on_click(lambda _: self._cmd_run())
            self._btn_pause = self.server.gui.add_button("Pause")
            self._btn_pause.on_click(lambda _: self._cmd_pause())
            self._btn_stop = self.server.gui.add_button("Stop Episode")
            self._btn_stop.on_click(lambda _: self._cmd_stop())

        with self.server.gui.add_folder("Display", expand_by_default=True):
            self._cb_keypoints = self.server.gui.add_checkbox(
                "Show Keypoints", initial_value=True,
            )
            self._cb_keypoints.on_update(lambda _: self._toggle_keypoints())

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self._md_task = self.server.gui.add_markdown("**Task:** --")
            self._md_prog = self.server.gui.add_markdown("**Progress:** --")
            self._md_diag = self.server.gui.add_markdown("**Goal dist:** --")
            self._md_retract = self.server.gui.add_markdown("**Retract:** --")
            self._md_force = self.server.gui.add_markdown("**Table force:** --")
            self._md_stats = self.server.gui.add_markdown("**Stats:** No episodes yet")
            self._md_obj = self.server.gui.add_markdown("**Object Pos:** --")
            self._md_dist = self.server.gui.add_markdown("**Dist to Goal:** --")
            self._md_reward = self.server.gui.add_markdown("**Cum Reward:** --")

        self._peak_force = 0.0

    def _tol_from_label(self, label: str) -> Optional[str]:
        if not label.endswith(" mm"):
            return None
        numeric = label[:-3].replace(".", "p")
        return numeric if numeric in self._tolerances else None

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

    def _setup_object_goal(self, tol_tag: str):
        """Render peg + goal ghost + hole walls under /hole."""
        self._clear_dynamic()

        peg_urdf = ASSETS_DIR / "peg" / "peg.urdf"

        # Active peg
        self._obj_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        self._dyn.append(self._obj_frame)
        ViserUrdf(self.server, peg_urdf, root_node_name="/object",
                  mesh_color_override=(204, 40, 40))

        # Goal ghost
        self._goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        self._dyn.append(self._goal_frame)
        ViserUrdf(self.server, peg_urdf, root_node_name="/goal",
                  mesh_color_override=(0, 255, 0, 0.5))

        # Static hole mesh (at fabrica fixture location)
        hole_obj = ASSETS_DIR / "holes" / f"hole_tol{tol_tag}mm" / "hole.obj"
        if hole_obj.exists():
            mesh = trimesh.load(str(hole_obj), force="mesh")
            hole_frame = self.server.scene.add_frame(
                f"/hole",
                position=(-0.05, 0.08, TABLE_Z + 0.15),
                wxyz=(1, 0, 0, 0),
                show_axes=False,
            )
            self._dyn.append(hole_frame)
            self._dyn.append(self.server.scene.add_mesh_simple(
                "/hole/mesh",
                vertices=np.array(mesh.vertices, dtype=np.float32),
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=(120, 120, 120),
            ))

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

    def _toggle_keypoints(self):
        self._apply_keypoint_visibility()

    def _apply_keypoint_visibility(self):
        visible = self._cb_keypoints.value
        for kp in self._obj_keypoints + self._goal_keypoints:
            kp.visible = visible

    # -- Subprocess management ------------------------------------------

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
        tol = self._tol_from_label(self._dd_tol.value)
        if tol is None:
            self._md_status.content = "**Status:** Please select a tolerance."
            return

        self._kill_subprocess()
        self._pending_tol = tol

        goal_mode = self._dd_goal_mode.value
        policy_name = self._dd_policy.value
        config_path, checkpoint_path = self.policies[policy_name]
        label = f"{policy_name} | tol {tol.replace('p', '.')}mm | goals: {goal_mode}"
        self._md_status.content = f"**Status:** Loading *{label}* ..."
        self._md_task.content = f"**Task:** {label}"
        self._md_retract.content = "**Retract:** --"

        self.ep_count = 0
        self.ep_goals.clear()
        self.ep_lengths.clear()
        self._md_stats.content = "**Stats:** No episodes yet"

        self.robot.update_cfg(DEFAULT_DOF_POS)

        self._trajectory = _load_trajectory(tol)
        scene_urdf_rel = _scene_urdf_rel(tol)

        scene_overrides = dict(self.extra_overrides)
        mode_key = _GOAL_MODE_OVERRIDE_KEY.get(goal_mode)
        if mode_key is not None:
            scene_overrides[mode_key] = True

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=sim_worker,
            args=(child_conn, tol,
                  config_path, checkpoint_path, scene_urdf_rel,
                  self.final_goal_tolerance, scene_overrides,
                  self.headless, self.record_video, self.video_dir),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        print(f"[launcher] Spawned subprocess pid={self._proc.pid} for tol={tol}mm")

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
        self._peak_force = 0.0
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
            self._setup_object_goal(self._pending_tol)
            if len(init_state) > 3:
                self._setup_keypoints(init_state[3].shape[0])
            self._update_viz(init_state)
            self._env_ready = True
            self._md_status.content = "**Status:** Ready -- click **Run Episode**"
            print("[launcher] Environment ready")

        elif tag == "state":
            state, successes, max_succ, step = msg[1], msg[2], msg[3], msg[4]
            dist_to_goal = msg[5] if len(msg) > 5 else None
            cum_total = msg[6] if len(msg) > 6 else None
            cum_breakdown = msg[7] if len(msg) > 7 else None
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
            if dist_to_goal is not None:
                self._md_dist.content = f"**Dist to Goal:** {dist_to_goal:.4f}"
            if cum_total is not None:
                lines = [f"**Cum Reward:** {cum_total:.1f}"]
                if cum_breakdown:
                    order = [
                        "lifting_rew", "lift_bonus_rew",
                        "keypoint_rew", "bonus_rew",
                        "fingertip_delta_rew",
                        "kuka_actions_penalty", "hand_actions_penalty",
                        "retract_rew",
                    ]
                    for _k in order:
                        if _k in cum_breakdown:
                            lines.append(f"&nbsp;&nbsp;{_k}: {cum_breakdown[_k]:.1f}")
                self._md_reward.content = "\n\n".join(lines)

            if len(state) >= 8:
                retract_phase, retract_succeeded, mean_ft_dist = state[5], state[6], state[7]
                if retract_succeeded:
                    self._md_retract.content = f"**Retract:** SUCCESS (hand dist: {mean_ft_dist:.3f}m)"
                elif retract_phase:
                    self._md_retract.content = f"**Retract:** IN PROGRESS (hand dist: {mean_ft_dist:.3f}m)"
            if len(state) >= 14:
                kp_max_dist = state[8]
                tol_m = state[9]
                near_steps = state[10]
                progress = state[11]
                max_ep_len = state[12]
                reset_pending = state[13]
                in_tol = "✓" if kp_max_dist <= tol_m else "✗"
                self._md_diag.content = (
                    f"**Goal dist:** {kp_max_dist*1000:.1f} mm {in_tol}  "
                    f"&nbsp;(tol {tol_m*1000:.1f} mm)  "
                    f"&nbsp;near-goal-steps: **{near_steps}**  \n"
                    f"**progress_buf:** {progress}/{max_ep_len}  "
                    f"&nbsp;reset_buf: **{reset_pending}**"
                )
            if len(state) >= 15:
                force_vec = np.asarray(state[14], dtype=np.float32)
                force_mag = float(np.linalg.norm(force_vec))
                if force_mag > self._peak_force:
                    self._peak_force = force_mag
                self._md_force.content = (
                    f"**Table force:** {force_mag:.2f} N  "
                    f"&nbsp;(peak this episode: **{self._peak_force:.2f} N**)  \n"
                    f"&nbsp;[fx, fy, fz] = "
                    f"[{force_vec[0]:+.2f}, {force_vec[1]:+.2f}, {force_vec[2]:+.2f}] N"
                )

        elif tag == "done":
            goal_pct, steps = msg[1], msg[2]
            retract_ok = msg[3] if len(msg) > 3 else False
            self._episode_running = False
            self.ep_goals.append(goal_pct)
            self.ep_lengths.append(steps)
            self.ep_count += 1
            avg_g = np.mean(self.ep_goals)
            avg_t = np.mean(self.ep_lengths) / 60.0

            retract_str = f" | Retract: {'OK' if retract_ok else 'FAIL'}"
            self._md_retract.content = f"**Retract:** {'SUCCESS' if retract_ok else 'FAILED'}"
            self._md_stats.content = (
                f"**Episodes:** {self.ep_count} &nbsp;|&nbsp; "
                f"**Avg Goal:** {avg_g:.1f}% &nbsp;|&nbsp; "
                f"**Avg Time:** {avg_t:.1f}s"
            )
            self._md_status.content = (
                f"**Status:** Done -- {steps / 60.0:.1f}s, {goal_pct:.0f}% goals{retract_str}"
            )
            print(f"[launcher] Episode done: {goal_pct:.0f}% goals in {steps / 60.0:.1f}s{retract_str}")

        elif tag == "stopped":
            self._episode_running = False
            self._md_status.content = "**Status:** Episode stopped."

        elif tag == "error":
            self._env_ready = False
            self._episode_running = False
            self._md_status.content = f"**Status:** Error -- {msg[1][:200]}"
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
        print("  +-------------------------------------------------+")
        print("  |  Peg-in-Hole Evaluation (pretrained policy)      |")
        print(f"  |     http://localhost:{self.port:<26}|")
        print("  +-------------------------------------------------+")
        print()

        try:
            while True:
                self._poll()
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\n[launcher] Shutting down...")
            self._kill_subprocess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peg-in-Hole evaluation (pretrained policy)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--policies-dir", type=str, default=None,
                        help="Directory of policy subfolders (each with config.yaml + model.pth).")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="Policy names (filters --policies-dir).")
    parser.add_argument("--initial-policy", type=str, default=None)
    parser.add_argument("--config-path", type=str, default="pretrained_policy/config.yaml")
    parser.add_argument("--checkpoint-path", type=str, default="pretrained_policy/model.pth")
    parser.add_argument("--final-goal-tolerance", type=float, default=None)
    parser.add_argument("--no-headless", action="store_true",
                        help="Show IsaacGym viewer window")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="dense")
    parser.add_argument("--override", nargs=2, action="append", default=[],
                        metavar=("KEY", "VALUE"),
                        help="Extra config overrides, e.g. --override task.sim.physx.num_position_iterations 32")
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

    policies = {}
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
            raise FileNotFoundError(
                f"No policy subfolders in {pdir} (need config.yaml + model.pth each)"
            )
    else:
        name = Path(args.config_path).parent.name or "policy"
        policies[name] = (_resolve(args.config_path), _resolve(args.checkpoint_path))

    PegInHoleDemo(
        policies=policies,
        port=args.port,
        final_goal_tolerance=args.final_goal_tolerance,
        extra_overrides=extra_overrides,
        headless=not args.no_headless,
        record_video=args.record_video,
        video_dir=args.video_dir,
        goal_mode=args.goal_mode,
        initial_policy=args.initial_policy,
    ).run()

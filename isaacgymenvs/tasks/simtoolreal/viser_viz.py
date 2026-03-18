"""Viser-based training viewer with online and offline modes.

Online mode:  Push live env state to viser every control step.
Offline mode: Replay saved .pt episode dumps with scrubbing.

Designed to be reusable by any IsaacGym training env that provides
the required state interface (joint positions, object/goal poses).
"""

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import trimesh
import viser
import yourdfpy
from viser.extras import ViserUrdf


# ── Constants ─────────────────────────────────────────────────

TABLE_Z = 0.38
N_ACT = 29  # arm (7) + hand (22)
CONTROL_DT = 1.0 / 60.0

_ARM_DEFAULT = np.array([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308])
_ARM_DEFAULT[1] -= np.deg2rad(10)
_ARM_DEFAULT[3] += np.deg2rad(10)
DEFAULT_DOF_POS = np.zeros(N_ACT)
DEFAULT_DOF_POS[:7] = _ARM_DEFAULT


def _quat_xyzw_to_wxyz(q):
    """Convert IsaacGym quaternion [x,y,z,w] to viser [w,x,y,z]."""
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


# ── Main class ────────────────────────────────────────────────


class TrainingViserViewer:
    """In-process viser viewer for training visualization.

    Parameters
    ----------
    port : int
        Viser server port.
    dump_dir : str
        Directory for offline .pt episode dumps.
    object_name : str
        IsaacGym object name, e.g. "beam_2_coacd".
    robot_urdf_path : Path
        Path to the robot URDF.
    object_urdf_path : Path
        Path to the object URDF (used by ViserUrdf for rendering).
    num_envs : int
        Total number of parallel envs.
    num_dofs : int
        Number of DOFs (arm + hand).
    """

    def __init__(
        self,
        port: int,
        dump_dir: str,
        object_name: str,
        robot_urdf_path: Path,
        object_urdf_path: Path,
        scene_urdf_path: Path,
        num_envs: int,
        num_dofs: int,
    ):
        self.port = port
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.object_name = object_name
        self.robot_urdf_path = robot_urdf_path
        self.object_urdf_path = object_urdf_path
        self.scene_urdf_path = scene_urdf_path
        self.num_envs = num_envs
        self.num_dofs = num_dofs

        # State
        self.recording_env_idx: int = 0
        self.online_env_idx: int = 0
        self._mode = "Online"  # "Online" or "Offline"
        self._episode_count = 0

        # Offline recording buffers
        self._recording = False
        self._rec_root_states: List[torch.Tensor] = []
        self._rec_dof_states: List[torch.Tensor] = []

        # Offline playback state
        self._loaded_episode: Optional[Dict] = None
        self._playing = False
        self._playback_speed = 1.0
        self._loop = False
        self._current_step = 0

        # Viser handles
        self._obj_frame = None
        self._goal_frame = None
        self.robot = None

        # Create server and build scene
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self._build_gui()
        self._setup_scene()
        self._start_file_watcher()
        self._start_playback_thread()

        print(f"[ViserViz] Server running on http://0.0.0.0:{port}")

    # ── Properties ────────────────────────────────────────────

    @property
    def is_online_mode(self) -> bool:
        return self._mode == "Online"

    @property
    def is_offline_recording(self) -> bool:
        return self._recording

    # ── GUI ────────────────────────────────────────────────────

    def _build_gui(self):
        self.server.gui.add_markdown("# Training Viewer")

        with self.server.gui.add_folder("Mode", expand_by_default=True):
            self._dd_mode = self.server.gui.add_dropdown(
                "Mode", options=["Online", "Offline"], initial_value="Online",
            )
            self._dd_mode.on_update(lambda _: self._on_mode_change())

        with self.server.gui.add_folder("Recording", expand_by_default=True):
            self._input_rec_idx = self.server.gui.add_number(
                "Recording Env Index",
                initial_value=0, min=0, max=self.num_envs - 1, step=1,
            )
            self._input_rec_idx.on_update(lambda _: self._on_rec_idx_change())
            self._md_rec_status = self.server.gui.add_markdown("**Recording:** idle")

        with self.server.gui.add_folder("Online", expand_by_default=True):
            self._input_online_idx = self.server.gui.add_number(
                "View Env Index",
                initial_value=0, min=0, max=self.num_envs - 1, step=1,
            )
            self._input_online_idx.on_update(lambda _: self._on_online_idx_change())
            self._md_online_step = self.server.gui.add_markdown("**Step:** –")
            self._md_online_goal = self.server.gui.add_markdown("**Goal:** –")
            self._md_online_obj = self.server.gui.add_markdown("**Obj pos:** –")
            self._md_online_dist = self.server.gui.add_markdown("**Dist:** –")
            self._md_online_tolerance = self.server.gui.add_markdown("**Tolerance:** –")

        with self.server.gui.add_folder("Offline", expand_by_default=False):
            self._dd_episode = self.server.gui.add_dropdown(
                "Episode", options=["-- none --"], initial_value="-- none --",
            )
            self._dd_episode.on_update(lambda _: self._on_episode_select())
            self._btn_refresh = self.server.gui.add_button("Refresh List")
            self._btn_refresh.on_click(lambda _: self._refresh_episode_list())
            self._slider_step = self.server.gui.add_slider(
                "Timestep", min=0, max=1, step=1, initial_value=0,
            )
            self._slider_step.on_update(lambda _: self._on_slider_change())
            self._btn_play = self.server.gui.add_button("Play")
            self._btn_play.on_click(lambda _: self._toggle_play())
            self._btn_reset = self.server.gui.add_button("Reset")
            self._btn_reset.on_click(lambda _: self._reset_playback())
            self._input_speed = self.server.gui.add_number(
                "Speed", initial_value=1.0, min=0.25, max=4.0, step=0.25,
            )
            self._input_speed.on_update(lambda _: self._on_speed_change())
            self._cb_loop = self.server.gui.add_checkbox("Loop", initial_value=False)
            self._cb_loop.on_update(lambda _: self._on_loop_change())
            self._md_episode_info = self.server.gui.add_markdown("**Episode:** none loaded")
            self._md_offline_frame = self.server.gui.add_markdown("**Frame:** –")
            self._md_offline_obj = self.server.gui.add_markdown("**Obj pos:** –")
            self._md_offline_dist = self.server.gui.add_markdown("**Dist to goal:** –")

    # ── Static scene ──────────────────────────────────────────

    def _setup_scene(self):
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            client.camera.position = (0.0, -1.0, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.5)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        # Robot
        self.server.scene.add_frame(
            "/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self.robot = ViserUrdf(
            self.server,
            self.robot_urdf_path,
            root_node_name="/robot",
        )
        self.robot.update_cfg(DEFAULT_DOF_POS)

        # Scene (table + already-placed parts in fixture)
        self._table_frame = self.server.scene.add_frame(
            "/table", position=(0, 0, TABLE_Z), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        if self.scene_urdf_path.exists():
            self._load_scene_urdf()
        else:
            # Fallback: plain box if scene URDF not found
            self.server.scene.add_box(
                "/table/wood", color=(180, 130, 70),
                dimensions=(0.475, 0.4, 0.3), position=(0, 0, 0),
                side="double", opacity=0.9,
            )

        # Object and goal frames (using ViserUrdf to render the same URDF as training)
        self._obj_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.08, axes_radius=0.002,
        )
        self._goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.08, axes_radius=0.002,
        )

        if self.object_urdf_path.exists():
            ViserUrdf(
                self.server, self.object_urdf_path,
                root_node_name="/object",
                mesh_color_override=(100, 150, 255),
            )
            ViserUrdf(
                self.server, self.object_urdf_path,
                root_node_name="/goal",
                mesh_color_override=(0, 255, 0, 0.4),
            )
        else:
            print(f"[ViserViz] Warning: object URDF not found at {self.object_urdf_path}")

    def _load_scene_urdf(self):
        """Parse scene URDF and add each visual mesh at correct joint offset."""
        urdf = yourdfpy.URDF.load(str(self.scene_urdf_path))

        # Add the base link (box) as a box primitive
        self.server.scene.add_box(
            "/table/box", color=(210, 143, 89),
            dimensions=(0.475, 0.4, 0.3), position=(0, 0, 0),
            side="double", opacity=0.9,
        )

        # Add each child link mesh at its joint origin
        for jname, joint in urdf.joint_map.items():
            child_link = joint.child
            link = urdf.link_map[child_link]
            if not link.visuals:
                continue
            visual = link.visuals[0]
            if visual.geometry.mesh is None:
                continue

            mesh_file = visual.geometry.mesh.filename
            mesh_path = self.scene_urdf_path.parent / mesh_file
            if not mesh_path.exists():
                continue

            mesh = trimesh.load(str(mesh_path), force="mesh")

            # Joint origin gives position/rotation relative to parent (box)
            pos = joint.origin[:3, 3].tolist()
            rot_matrix = joint.origin[:3, :3]
            from scipy.spatial.transform import Rotation
            quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()
            quat_wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])

            self.server.scene.add_mesh_trimesh(
                f"/table/{child_link}",
                mesh=mesh,
                position=tuple(pos),
                wxyz=quat_wxyz,
            )

    # ── GUI callbacks ─────────────────────────────────────────

    def _on_mode_change(self):
        self._mode = self._dd_mode.value
        self._playing = False

    def _on_rec_idx_change(self):
        self.recording_env_idx = int(self._input_rec_idx.value)

    def _on_online_idx_change(self):
        self.online_env_idx = int(self._input_online_idx.value)

    def _on_episode_select(self):
        name = self._dd_episode.value
        if name == "-- none --":
            return
        filepath = self.dump_dir / name
        if filepath.exists():
            self._load_episode(filepath)

    def _on_slider_change(self):
        if self._loaded_episode is not None:
            self._current_step = int(self._slider_step.value)
            self._render_offline_step(self._current_step)

    def _toggle_play(self):
        self._playing = not self._playing
        self._btn_play.name = "Pause" if self._playing else "Play"

    def _reset_playback(self):
        self._playing = False
        self._current_step = 0
        self._slider_step.value = 0
        self._btn_play.name = "Play"
        if self._loaded_episode is not None:
            self._render_offline_step(0)

    def _on_speed_change(self):
        self._playback_speed = max(0.25, float(self._input_speed.value))

    def _on_loop_change(self):
        self._loop = self._cb_loop.value

    # ── Online mode ───────────────────────────────────────────

    def update_online(self, dof_positions, object_pose_xyzw, goal_pose_xyzw, table_pose_xyzw=None, info=None):
        """Called every control step. Push live state to viser.

        Parameters
        ----------
        dof_positions : np.ndarray, shape [num_dofs]
        object_pose_xyzw : np.ndarray, shape [7] (xyz + quat xyzw)
        goal_pose_xyzw : np.ndarray, shape [7]
        table_pose_xyzw : np.ndarray, shape [7], optional
        info : dict, optional
            Training info: control_steps, successes, max_successes,
            progress_buf, closest_dist, success_tolerance.
        """
        if not self.is_online_mode:
            return

        # Update robot joint positions
        if self.robot is not None:
            self.robot.update_cfg(dof_positions[:N_ACT])

        # Update object frame
        if self._obj_frame is not None:
            self._obj_frame.position = tuple(object_pose_xyzw[:3].tolist())
            self._obj_frame.wxyz = _quat_xyzw_to_wxyz(object_pose_xyzw[3:7])

        # Update goal frame
        if self._goal_frame is not None:
            self._goal_frame.position = tuple(goal_pose_xyzw[:3].tolist())
            self._goal_frame.wxyz = _quat_xyzw_to_wxyz(goal_pose_xyzw[3:7])

        # Update table frame (position may be randomized per env)
        if table_pose_xyzw is not None and self._table_frame is not None:
            self._table_frame.position = tuple(table_pose_xyzw[:3].tolist())
            self._table_frame.wxyz = _quat_xyzw_to_wxyz(table_pose_xyzw[3:7])

        # Update info panels
        if info is not None:
            step = info.get("control_steps", 0)
            progress = info.get("progress_buf", 0)
            self._md_online_step.content = f"**Step:** {step} | ep frame {progress}"

            succ = info.get("successes", 0)
            max_succ = info.get("max_successes", 1)
            pct = succ / max(max_succ, 1) * 100
            self._md_online_goal.content = f"**Goal:** {succ}/{max_succ} ({pct:.0f}%)"

            ox, oy, oz = object_pose_xyzw[:3]
            self._md_online_obj.content = f"**Obj pos:** ({ox:.3f}, {oy:.3f}, {oz:.3f})"

            dist = info.get("closest_dist", -1)
            self._md_online_dist.content = f"**Dist:** {dist:.4f}" if dist >= 0 else "**Dist:** –"

            tol = info.get("success_tolerance", -1)
            self._md_online_tolerance.content = f"**Tolerance:** {tol:.4f}" if tol >= 0 else "**Tolerance:** –"

    # ── Offline recording ─────────────────────────────────────

    def start_recording(self):
        """Signal to start accumulating states for offline dump."""
        self._recording = True
        self._rec_root_states.clear()
        self._rec_dof_states.clear()
        self._md_rec_status.content = "**Recording:** capturing..."

    def stop_recording(self):
        """Stop accumulating without saving (e.g. capture window ended without reset)."""
        self._recording = False
        self._rec_root_states.clear()
        self._rec_dof_states.clear()
        self._md_rec_status.content = "**Recording:** idle"

    def accumulate_step(self, root_state_per_env, dof_state_per_env):
        """Append one timestep of state for the tracked env.

        Parameters
        ----------
        root_state_per_env : Tensor, shape [num_actors_per_env, 13]
        dof_state_per_env : Tensor, shape [num_dofs, 2]
        """
        if not self._recording:
            return
        self._rec_root_states.append(root_state_per_env.clone().cpu())
        self._rec_dof_states.append(dof_state_per_env.clone().cpu())

    def on_episode_end(self, env_idx: int, metadata: dict):
        """Called on episode reset. Save accumulated states if recording.

        Parameters
        ----------
        env_idx : int
            The env that just reset.
        metadata : dict
            Episode metadata (object_name, success, control_steps,
            object_local_idx, goal_local_idx, num_dofs, etc.)
        """
        if not self._recording or len(self._rec_root_states) == 0:
            return

        root_traj = torch.stack(self._rec_root_states)  # [T, num_actors, 13]
        dof_traj = torch.stack(self._rec_dof_states)     # [T, num_dofs, 2]

        metadata["episode_length"] = len(self._rec_root_states)
        dump_data = {
            "root_state_tensor": root_traj,
            "dof_state": dof_traj,
            "metadata": metadata,
        }

        filename = f"episode_{self._episode_count:06d}_step{metadata.get('control_steps', 0)}.pt"
        torch.save(dump_data, self.dump_dir / filename)
        self._episode_count += 1

        print(f"[ViserViz] Saved offline episode: {filename} "
              f"(T={metadata['episode_length']}, success={metadata.get('success', '?')})")

        # Reset recording
        self._recording = False
        self._rec_root_states.clear()
        self._rec_dof_states.clear()
        self._md_rec_status.content = f"**Recording:** saved {filename}"

    # ── Offline playback ──────────────────────────────────────

    def _load_episode(self, filepath: Path):
        """Load a .pt episode dump for offline replay."""
        self._playing = False
        self._btn_play.name = "Play"
        try:
            data = torch.load(filepath, map_location="cpu", weights_only=False)
        except Exception as e:
            self._md_episode_info.content = f"**Error:** {e}"
            return

        self._loaded_episode = data
        meta = data.get("metadata", {})
        T = data["root_state_tensor"].shape[0]

        self._slider_step.max = max(T - 1, 1)
        self._slider_step.value = 0
        self._current_step = 0

        success_str = "Yes" if meta.get("success") else "No"
        self._md_episode_info.content = (
            f"**Object:** {meta.get('object_name', '?')}\n\n"
            f"**Length:** {meta.get('episode_length', T)} steps\n\n"
            f"**Success:** {success_str}\n\n"
            f"**Train step:** {meta.get('control_steps', '?')}"
        )

        # Reset offline panels
        self._md_offline_frame.content = "**Frame:** –"
        self._md_offline_obj.content = "**Obj pos:** –"
        self._md_offline_dist.content = "**Dist to goal:** –"

        self._render_offline_step(0)

    def _render_offline_step(self, step_idx: int):
        """Render a single timestep from the loaded episode."""
        if self._loaded_episode is None:
            return

        root = self._loaded_episode["root_state_tensor"]  # [T, num_actors, 13]
        dof = self._loaded_episode["dof_state"]            # [T, num_dofs, 2]
        meta = self._loaded_episode.get("metadata", {})
        T = root.shape[0]
        step_idx = min(step_idx, T - 1)

        obj_idx = meta.get("object_local_idx", 1)
        goal_idx = meta.get("goal_local_idx", 2)
        table_idx = meta.get("table_local_idx", None)

        # Joint positions
        dof_pos = dof[step_idx, :, 0].numpy()
        if self.robot is not None:
            self.robot.update_cfg(dof_pos[:N_ACT])

        # Object pose
        obj_state = root[step_idx, obj_idx, :7].numpy()
        if self._obj_frame is not None:
            self._obj_frame.position = tuple(obj_state[:3].tolist())
            self._obj_frame.wxyz = _quat_xyzw_to_wxyz(obj_state[3:7])

        # Goal pose
        goal_state = root[step_idx, goal_idx, :7].numpy()
        if self._goal_frame is not None:
            self._goal_frame.position = tuple(goal_state[:3].tolist())
            self._goal_frame.wxyz = _quat_xyzw_to_wxyz(goal_state[3:7])

        # Update offline status panels
        dist = np.linalg.norm(obj_state[:3] - goal_state[:3])
        ox, oy, oz = obj_state[:3]
        self._md_offline_frame.content = f"**Frame:** {step_idx + 1}/{T}"
        self._md_offline_obj.content = f"**Obj pos:** ({ox:.3f}, {oy:.3f}, {oz:.3f})"
        self._md_offline_dist.content = f"**Dist to goal:** {dist:.4f}"

        # Table pose
        if table_idx is not None and self._table_frame is not None:
            table_state = root[step_idx, table_idx, :7].numpy()
            self._table_frame.position = tuple(table_state[:3].tolist())
            self._table_frame.wxyz = _quat_xyzw_to_wxyz(table_state[3:7])

    # ── Background threads ────────────────────────────────────

    def _start_playback_thread(self):
        def _loop():
            while True:
                if self._playing and self._loaded_episode is not None and not self.is_online_mode:
                    T = self._loaded_episode["root_state_tensor"].shape[0]
                    self._current_step += 1
                    if self._current_step >= T:
                        if self._loop:
                            self._current_step = 0
                        else:
                            self._current_step = T - 1
                            self._playing = False
                            self._btn_play.name = "Play"
                    self._slider_step.value = self._current_step
                    self._render_offline_step(self._current_step)
                    time.sleep(CONTROL_DT / self._playback_speed)
                else:
                    time.sleep(0.05)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def _start_file_watcher(self):
        def _watch():
            known = set()
            while True:
                try:
                    current = set(p.name for p in self.dump_dir.glob("episode_*.pt"))
                    if current != known:
                        known = current
                        self._refresh_episode_list()
                except Exception:
                    pass
                time.sleep(5.0)

        t = threading.Thread(target=_watch, daemon=True)
        t.start()

    def _refresh_episode_list(self):
        files = sorted(self.dump_dir.glob("episode_*.pt"), reverse=True)
        names = [f.name for f in files]
        if not names:
            names = ["-- none --"]
        self._dd_episode.options = names

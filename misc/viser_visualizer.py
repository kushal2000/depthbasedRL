"""Viser trajectory visualizer + WandB diagnostics callbacks for training.

Extracted from train_isaaclab.py and train_isaacgym.py to keep those files
focused on env setup / config / GPU isolation.

Two backends (Isaac Lab vs IsaacGym) access state differently:
  - Isaac Lab: gym_env.unwrapped, world-frame positions, subtract env_origins
  - IsaacGym: raw env directly, env-local positions
"""

import threading
import time as _time
from pathlib import Path

import numpy as np


# ── Shared constants ──

URDF_PATH = Path("assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf")

FINGERTIP_NAMES_IG = [
    "left_index_DP", "left_middle_DP", "left_ring_DP",
    "left_thumb_DP", "left_pinky_DP",
]
FT_COLORS = [(255, 50, 50), (50, 50, 255), (255, 255, 50), (255, 50, 255), (50, 255, 255)]


def _quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


# ─────────────────────────────────────────────────────────────────────────────
# Shared playback loop — same for both backends
# ─────────────────────────────────────────────────────────────────────────────

def _run_playback_loop(lock, playback_traj_ref, render_fn, fps=60):
    """Continuously loop through the latest trajectory, calling render_fn(frame)."""
    frame_dt = 1.0 / fps
    while True:
        with lock:
            traj = list(playback_traj_ref[0])
        if len(traj) == 0:
            _time.sleep(0.5)
            continue
        for f in traj:
            render_fn(f)
            _time.sleep(frame_dt)


def _create_scene(server, ft_names):
    """Create shared viser scene elements (table, object, goal, fingertips, keypoints)."""
    server.scene.add_frame("/robot", position=(0.0, 0.8, 0.0))

    server.scene.add_box("/table", dimensions=(0.475, 0.4, 0.3),
                         color=(209, 143, 89), position=(0.0, 0.0, 0.38))
    obj_handle = server.scene.add_box("/object", dimensions=(0.141, 0.03, 0.027),
                                      color=(255, 80, 0))
    goal_handle = server.scene.add_box("/goal", dimensions=(0.141, 0.03, 0.027),
                                       color=(0, 255, 0), opacity=0.4)
    ft_handles = [server.scene.add_icosphere(f"/fingertip/{n}", radius=0.008, color=FT_COLORS[i])
                  for i, n in enumerate(ft_names)]
    obj_kp_handles = [server.scene.add_icosphere(f"/keypoints/obj_{i}", radius=0.006, color=(255, 160, 0))
                      for i in range(4)]
    goal_kp_handles = [server.scene.add_icosphere(f"/keypoints/goal_{i}", radius=0.006, color=(0, 200, 0), opacity=0.6)
                       for i in range(4)]

    epoch_label = server.gui.add_text("Epoch", initial_value="--", disabled=True)
    reward_label = server.gui.add_text("Reward", initial_value="--", disabled=True)
    env_label = server.gui.add_text("Env ID", initial_value="0", disabled=True)

    return obj_handle, goal_handle, ft_handles, obj_kp_handles, goal_kp_handles, epoch_label, reward_label, env_label


# ─────────────────────────────────────────────────────────────────────────────
# Isaac Lab backend
# ─────────────────────────────────────────────────────────────────────────────

def setup_isaaclab_viser(gym_env, interval=10, port=8012, fps=60):
    """Set up viser trajectory recording + looped playback for Isaac Lab.

    Returns (step_callback, epoch_callback) to attach to SAPGAgent.
    """
    import viser
    from viser.extras import ViserUrdf
    from envs.isaaclab.sim_tool_real_cfg import FINGERTIP_NAMES

    server = viser.ViserServer(port=port)
    viser_urdf = ViserUrdf(server, URDF_PATH, root_node_name="/robot")

    # Build canonical joint name list for dict-based update_cfg
    arm_joint_names = [f"iiwa14_joint_{i}" for i in range(1, 8)]
    unwrapped = gym_env.unwrapped
    env_joint_names = list(unwrapped.robot.data.joint_names)
    env_arm_ids = [env_joint_names.index(n) for n in arm_joint_names]
    env_arm_set = set(env_arm_ids)
    env_hand_ids = [i for i in range(len(env_joint_names)) if i not in env_arm_set]
    env_canonical_names = [env_joint_names[i] for i in env_arm_ids + env_hand_ids]

    handles = _create_scene(server, FINGERTIP_NAMES)
    obj_h, goal_h, ft_hs, okp_hs, gkp_hs, epoch_lbl, reward_lbl, env_lbl = handles
    print(f"[Viser] Trajectory visualizer at http://localhost:{port}")

    lock = threading.Lock()
    recording_buf = []
    playback_traj_ref = [[]]  # mutable container for playback_loop
    target_env_id = [0]
    current_epoch = [0]
    current_reward = [0.0]

    def _step_callback(agent_ref, step_idx):
        eid = target_env_id[0]
        env_origin = unwrapped.scene.env_origins[eid].cpu().numpy()
        joint_pos_arr = unwrapped.arm_hand_dof_pos[eid].cpu().numpy()
        joint_dict = {name: float(joint_pos_arr[i])
                      for i, name in enumerate(env_canonical_names)}
        recording_buf.append({
            'joint_dict': joint_dict,
            'obj_pos': unwrapped.object_pos[eid].cpu().numpy() - env_origin,
            'obj_rot': unwrapped.object_rot[eid].cpu().numpy().copy(),
            'goal_pos': unwrapped.goal_states[eid, :3].cpu().numpy() - env_origin,
            'goal_rot': unwrapped.goal_states[eid, 3:7].cpu().numpy().copy(),
            'ft_pos': unwrapped.fingertip_pos_offset[eid].cpu().numpy() - env_origin,
            'obj_kp': unwrapped.obj_keypoint_pos[eid].cpu().numpy() - env_origin,
            'goal_kp': unwrapped.goal_keypoint_pos[eid].cpu().numpy() - env_origin,
        })

    def _epoch_callback(agent_ref, epoch_num, frame_count):
        current_epoch[0] = epoch_num
        if hasattr(agent_ref, 'game_rewards') and agent_ref.game_rewards.current_size > 0:
            current_reward[0] = float(agent_ref.game_rewards.get_mean())

        if epoch_num % interval == 0:
            if len(recording_buf) > 0:
                with lock:
                    playback_traj_ref[0] = list(recording_buf)
            recording_buf.clear()

            # Find env with highest z_lift for next recording window
            obj_z = unwrapped.object.data.root_pos_w[:, 2]
            init_z = unwrapped.object_init_state[:, 2]
            z_lift = obj_z - init_z
            best_env = int(z_lift.argmax().item())
            target_env_id[0] = best_env
            print(f"[Viser] Epoch {epoch_num}: {len(playback_traj_ref[0])} frames, "
                  f"next env={best_env} (z_lift={z_lift[best_env]:.4f})")

    def _render(f):
        epoch_lbl.value = f"Epoch {current_epoch[0]}"
        reward_lbl.value = f"Reward {current_reward[0]:.2f}"
        env_lbl.value = f"Env {target_env_id[0]}"
        viser_urdf.update_cfg(f['joint_dict'])
        obj_h.position = f['obj_pos']
        obj_h.wxyz = _quat_xyzw_to_wxyz(f['obj_rot'])
        goal_h.position = f['goal_pos']
        goal_h.wxyz = _quat_xyzw_to_wxyz(f['goal_rot'])
        for i, h in enumerate(ft_hs):
            h.position = f['ft_pos'][i]
        for i in range(4):
            okp_hs[i].position = f['obj_kp'][i]
            gkp_hs[i].position = f['goal_kp'][i]

    threading.Thread(target=_run_playback_loop,
                     args=(lock, playback_traj_ref, _render, fps), daemon=True).start()
    return _step_callback, _epoch_callback


# ─────────────────────────────────────────────────────────────────────────────
# IsaacGym backend
# ─────────────────────────────────────────────────────────────────────────────

def setup_isaacgym_viser(ig_env, interval=10, port=8013, fps=60):
    """Set up viser trajectory recording + looped playback for IsaacGym.

    Returns (step_callback, epoch_callback) to attach to SAPGAgent.
    """
    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=port)
    viser_urdf = ViserUrdf(server, URDF_PATH, root_node_name="/robot")

    handles = _create_scene(server, FINGERTIP_NAMES_IG)
    obj_h, goal_h, ft_hs, okp_hs, gkp_hs, epoch_lbl, reward_lbl, env_lbl = handles
    print(f"[Viser-IG] Trajectory visualizer at http://localhost:{port}")

    lock = threading.Lock()
    recording_buf = []
    playback_traj_ref = [[]]
    target_env_id = [0]
    current_epoch = [0]
    current_reward = [0.0]

    def _step_callback(agent_ref, step_idx):
        eid = target_env_id[0]
        # IsaacGym positions are already env-local — no origin subtraction needed
        recording_buf.append({
            'joint_pos': ig_env.arm_hand_dof_pos[eid].cpu().numpy().copy(),
            'obj_pos': ig_env.object_pos[eid].cpu().numpy().copy(),
            'obj_rot': ig_env.object_rot[eid].cpu().numpy().copy(),
            'goal_pos': ig_env.goal_states[eid, :3].cpu().numpy().copy(),
            'goal_rot': ig_env.goal_states[eid, 3:7].cpu().numpy().copy(),
            'ft_pos': ig_env.fingertip_pos_offset[eid].cpu().numpy().copy(),
            'obj_kp': ig_env.obj_keypoint_pos[eid].cpu().numpy().copy(),
            'goal_kp': ig_env.goal_keypoint_pos[eid].cpu().numpy().copy(),
        })

    def _epoch_callback(agent_ref, epoch_num, frame_count):
        current_epoch[0] = epoch_num
        if hasattr(agent_ref, 'game_rewards') and agent_ref.game_rewards.current_size > 0:
            current_reward[0] = float(agent_ref.game_rewards.get_mean())

        if epoch_num % interval == 0:
            if len(recording_buf) > 0:
                with lock:
                    playback_traj_ref[0] = list(recording_buf)
            recording_buf.clear()

            z_lift = ig_env.object_pos[:, 2] - ig_env.object_init_state[:, 2]
            best_env = int(z_lift.argmax().item())
            target_env_id[0] = best_env
            print(f"[Viser-IG] Epoch {epoch_num}: {len(playback_traj_ref[0])} frames, "
                  f"next env={best_env} (z_lift={z_lift[best_env]:.4f})")

    def _render(f):
        epoch_lbl.value = f"Epoch {current_epoch[0]}"
        reward_lbl.value = f"Reward {current_reward[0]:.2f}"
        env_lbl.value = f"Env {target_env_id[0]}"
        # IsaacGym uses array-based joint positions
        viser_urdf.update_cfg(f['joint_pos'])
        obj_h.position = f['obj_pos']
        obj_h.wxyz = _quat_xyzw_to_wxyz(f['obj_rot'])
        goal_h.position = f['goal_pos']
        goal_h.wxyz = _quat_xyzw_to_wxyz(f['goal_rot'])
        for i, h in enumerate(ft_hs):
            h.position = f['ft_pos'][i]
        for i in range(4):
            okp_hs[i].position = f['obj_kp'][i]
            gkp_hs[i].position = f['goal_kp'][i]

    threading.Thread(target=_run_playback_loop,
                     args=(lock, playback_traj_ref, _render, fps), daemon=True).start()
    return _step_callback, _epoch_callback


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics callbacks — log lift/reward metrics to WandB
# ─────────────────────────────────────────────────────────────────────────────

def _log_obs_components(wb, obs_components, epoch_num):
    """Log mean/std/min/max of each observation component to WandB."""
    obs_diag = {"info/epochs": epoch_num}
    for name, tensor in obs_components.items():
        t = tensor.float()
        obs_diag[f"obs_components/{name}_mean"] = t.mean().item()
        obs_diag[f"obs_components/{name}_std"] = t.std().item()
        obs_diag[f"obs_components/{name}_min"] = t.min().item()
        obs_diag[f"obs_components/{name}_max"] = t.max().item()
    return obs_diag


def _log_lift_diagnostics(wb, obj_z, init_z, lifted_frac, successes, near_goal_steps):
    """Log common lift/success metrics."""
    z_lift = obj_z - init_z
    n_lifted_5cm = (z_lift > 0.05).sum().item()
    n_lifted_15cm = (z_lift > 0.15).sum().item()

    wb.log({
        "diagnostics/object_z_mean": obj_z.mean().item(),
        "diagnostics/z_lift_mean": z_lift.mean().item(),
        "diagnostics/z_lift_max": z_lift.max().item(),
        "diagnostics/n_lifted_5cm": n_lifted_5cm,
        "diagnostics/n_lifted_15cm": n_lifted_15cm,
        "diagnostics/lifted_object_frac": lifted_frac,
        "diagnostics/successes_mean": successes.mean().item(),
        "diagnostics/successes_max": successes.max().item(),
        "diagnostics/near_goal_steps_mean": near_goal_steps.float().mean().item(),
    }, commit=False)

    return z_lift, n_lifted_5cm, n_lifted_15cm


def make_isaaclab_diag_callback(gym_env, interval=10):
    """WandB diagnostics callback for Isaac Lab training."""
    def _callback(agent_ref, epoch_num, frame):
        if epoch_num % interval != 0:
            return
        try:
            import wandb as _wb
            if _wb.run is None:
                return
        except ImportError:
            return

        unwrapped = gym_env.unwrapped

        # Reward component diagnostics
        for key, buf in unwrapped.rewards_episode.items():
            _wb.log({f"reward_components/{key}": buf.mean().item()}, commit=False)

        obj_z = unwrapped.object.data.root_pos_w[:, 2]
        init_z = unwrapped.object_init_state[:, 2]
        lifted_frac = unwrapped.lifted_object.float().mean().item()
        z_lift, n5, n15 = _log_lift_diagnostics(
            _wb, obj_z, init_z, lifted_frac, unwrapped.successes, unwrapped.near_goal_steps)

        if hasattr(unwrapped, 'curr_fingertip_distances'):
            finger_dists = unwrapped.curr_fingertip_distances
            _wb.log({
                "diagnostics/fingertip_dist_mean": finger_dists.mean().item(),
                "diagnostics/fingertip_dist_min": finger_dists.min().item(),
            }, commit=False)

        # Per-obs-component diagnostics
        # Subtract env_origins from absolute positions to match
        # IsaacGym's env-local frame for diagnostics comparison
        _env_origins = unwrapped.scene.env_origins
        obs_diag = _log_obs_components(_wb, {
            "joint_pos": unwrapped.arm_hand_dof_pos,
            "joint_vel": unwrapped.arm_hand_dof_vel,
            "palm_pos": unwrapped.palm_center_pos - _env_origins,
            "palm_rot": unwrapped._palm_state[:, 3:7],
            "palm_vel": unwrapped._palm_state[:, 7:13],
            "object_rot": unwrapped.object_state[:, 3:7],
            "object_vel": unwrapped.object_state[:, 7:13],
            "object_pos": unwrapped.object_pos - _env_origins,
            "keypoints_rel_palm": unwrapped.keypoints_rel_palm.reshape(unwrapped.num_envs, -1),
            "keypoints_rel_goal": unwrapped.keypoints_rel_goal.reshape(unwrapped.num_envs, -1),
            "fingertip_pos_rel_palm": unwrapped.fingertip_pos_rel_palm.reshape(unwrapped.num_envs, -1),
            "prev_targets": unwrapped.prev_targets,
            "cur_targets": unwrapped.cur_targets,
        }, epoch_num)

        # Joint target tracking error
        target_err = (unwrapped.cur_targets - unwrapped.arm_hand_dof_pos).abs()
        obs_diag["obs_components/target_error_arm_mean"] = target_err[:, :7].mean().item()
        obs_diag["obs_components/target_error_hand_mean"] = target_err[:, 7:].mean().item()
        obs_diag["obs_components/target_error_arm_max"] = target_err[:, :7].max().item()
        obs_diag["obs_components/target_error_hand_max"] = target_err[:, 7:].max().item()
        _wb.log(obs_diag)

        print(f"[Diag] epoch={epoch_num} z_lift_mean={z_lift.mean():.4f} "
              f"z_lift_max={z_lift.max():.4f} "
              f"n_lifted_5cm={n5} n_lifted_15cm={n15} "
              f"lifted_frac={lifted_frac:.3f} "
              f"successes_mean={unwrapped.successes.mean():.2f}")

    return _callback


def make_isaacgym_diag_callback(raw_env, interval=10):
    """WandB diagnostics callback for IsaacGym training."""
    def _callback(agent_ref, epoch_num, frame):
        if epoch_num % interval != 0:
            return
        try:
            import wandb as _wb
            _has_wandb = _wb.run is not None
        except ImportError:
            _has_wandb = False

        env = raw_env
        obj_z = env.object_pos[:, 2]
        init_z = env.object_init_state[:, 2]
        z_lift = obj_z - init_z
        n_lifted_5cm = (z_lift > 0.05).sum().item()
        n_lifted_15cm = (z_lift > 0.15).sum().item()
        lifted_frac = env.lifted_object.float().mean().item()

        if not _has_wandb:
            print(f"[Diag] epoch={epoch_num} z_lift_mean={z_lift.mean():.4f} "
                  f"z_lift_max={z_lift.max():.4f} "
                  f"n_lifted_5cm={n_lifted_5cm} n_lifted_15cm={n_lifted_15cm} "
                  f"lifted_frac={lifted_frac:.3f} "
                  f"successes_mean={env.successes.mean():.2f}", flush=True)
            return

        _log_lift_diagnostics(
            _wb, obj_z, init_z, lifted_frac, env.successes, env.near_goal_steps)

        finger_dists = env.curr_fingertip_distances
        _wb.log({
            "diagnostics/fingertip_dist_mean": finger_dists.mean().item(),
            "diagnostics/fingertip_dist_min": finger_dists.min().item(),
        }, commit=False)

        obs_diag = _log_obs_components(_wb, {
            "joint_pos": env.arm_hand_dof_pos,
            "joint_vel": env.arm_hand_dof_vel,
            "palm_pos": env.palm_center_pos,
            "palm_rot": env._palm_state[:, 3:7],
            "palm_vel": env._palm_state[:, 7:13],
            "object_rot": env.object_state[:, 3:7],
            "object_vel": env.object_state[:, 7:13],
            "object_pos": env.object_pos,
            "keypoints_rel_palm": env.keypoints_rel_palm.reshape(env.num_envs, -1),
            "keypoints_rel_goal": env.keypoints_rel_goal.reshape(env.num_envs, -1),
            "fingertip_pos_rel_palm": env.fingertip_pos_rel_palm.reshape(env.num_envs, -1),
            "prev_targets": env.prev_targets,
            "cur_targets": env.cur_targets,
        }, epoch_num)

        target_err = (env.cur_targets - env.arm_hand_dof_pos).abs()
        obs_diag["obs_components/target_error_arm_mean"] = target_err[:, :7].mean().item()
        obs_diag["obs_components/target_error_hand_mean"] = target_err[:, 7:].mean().item()
        obs_diag["obs_components/target_error_arm_max"] = target_err[:, :7].max().item()
        obs_diag["obs_components/target_error_hand_max"] = target_err[:, 7:].max().item()
        _wb.log(obs_diag)

        print(f"[Diag] epoch={epoch_num} z_lift_mean={z_lift.mean():.4f} "
              f"z_lift_max={z_lift.max():.4f} "
              f"n_lifted_5cm={n_lifted_5cm} n_lifted_15cm={n_lifted_15cm} "
              f"lifted_frac={lifted_frac:.3f} "
              f"successes_mean={env.successes.mean():.2f}", flush=True)

    return _callback

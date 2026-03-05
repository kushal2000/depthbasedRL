"""Very small Fabrica eval viewer.

Starts the Isaac Gym env, runs one deterministic episode, and streams the
robot/object state into a viser scene.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_Z = 0.38
Z_OFFSET = 0.03
N_ACT = 29
OBS_DIM = 140
CONTROL_DT = 1.0 / 60.0

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

_ARM_DEFAULT = np.array([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308])
_ARM_DEFAULT[1] -= np.deg2rad(10)
_ARM_DEFAULT[3] += np.deg2rad(10)
DEFAULT_DOF_POS = np.zeros(N_ACT)
DEFAULT_DOF_POS[:7] = _ARM_DEFAULT


def parse_args():
    parser = argparse.ArgumentParser(description="Very small Fabrica eval viewer")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config-path", type=str, default="pretrained_policy/config.yaml")
    parser.add_argument("--checkpoint-path", type=str, default="pretrained_policy/model.pth")
    parser.add_argument("--object-name", type=str, default="beam_6")
    parser.add_argument("--task-name", type=str, default="pick_place")
    parser.add_argument("--no-headless", action="store_true", help="Show Isaac Gym viewer GUI")
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if path.exists():
        return path
    path = REPO_ROOT / path_str
    if path.exists():
        return path
    raise FileNotFoundError(path_str)


def quat_xyzw_to_wxyz(quat):
    return (quat[3], quat[0], quat[1], quat[2])


def load_trajectory(path):
    with open(path) as f:
        traj = json.load(f)
    traj["start_pose"][2] += Z_OFFSET
    return traj


def make_overrides(object_name, table_urdf, traj):
    return {
        **BASE_OVERRIDES,
        "task.env.objectName": object_name,
        "task.env.useFixedGoalStates": True,
        "task.env.fixedGoalStates": traj["goals"],
        "task.env.asset.table": str(table_urdf),
        "task.env.tableResetZ": TABLE_Z,
        "task.env.objectStartPose": traj["start_pose"],
    }


def get_state(env, obs, joint_lower, joint_upper):
    obs_np = obs[0].cpu().numpy()
    joint_pos = 0.5 * (obs_np[:N_ACT] + 1.0) * (joint_upper - joint_lower) + joint_lower
    return (
        joint_pos,
        env.object_state[0, :7].cpu().numpy(),
        env.goal_pose[0].cpu().numpy(),
        env.obj_keypoint_pos[0].cpu().numpy(),
        env.goal_keypoint_pos[0].cpu().numpy(),
    )


def add_scene(server, object_name, num_keypoints):
    from fabrica.objects import FABRICA_NAME_TO_OBJECT

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (0.0, -1.0, 1.0)
        client.camera.look_at = (0.0, 0.0, 0.5)

    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    server.scene.add_frame("/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False)
    robot = ViserUrdf(
        server,
        REPO_ROOT / "assets" / "urdf" / "kuka_sharpa_description" / "iiwa14_left_sharpa_adjusted_restricted.urdf",
        root_node_name="/robot",
    )
    robot.update_cfg(DEFAULT_DOF_POS)

    server.scene.add_frame("/table", position=(0, 0, TABLE_Z), wxyz=(1, 0, 0, 0), show_axes=False)
    server.scene.add_box(
        "/table/wood",
        color=(180, 130, 70),
        dimensions=(0.475, 0.4, 0.3),
        position=(0, 0, 0),
        side="double",
        opacity=0.9,
    )
    server.scene.add_frame("/table/fixture", position=(0.12, -0.152, 0.15), wxyz=(1, 0, 0, 0), show_axes=False)
    ViserUrdf(
        server,
        REPO_ROOT / "assets" / "urdf" / "fabrica" / "beam" / "fixture" / "fixture.urdf",
        root_node_name="/table/fixture",
    )

    obj_urdf = FABRICA_NAME_TO_OBJECT[object_name].urdf_path
    obj_frame = server.scene.add_frame("/object", show_axes=True, axes_length=0.1, axes_radius=0.001)
    goal_frame = server.scene.add_frame("/goal", show_axes=True, axes_length=0.1, axes_radius=0.001)
    ViserUrdf(server, obj_urdf, root_node_name="/object")
    ViserUrdf(server, obj_urdf, root_node_name="/goal", mesh_color_override=(0, 255, 0, 0.5))

    obj_keypoints = []
    goal_keypoints = []
    for i in range(num_keypoints):
        obj_keypoints.append(server.scene.add_icosphere(f"/obj_kp/{i}", radius=0.005, color=(255, 0, 0)))
        goal_keypoints.append(server.scene.add_icosphere(f"/goal_kp/{i}", radius=0.005, color=(0, 255, 0), opacity=0.5))

    return robot, obj_frame, goal_frame, obj_keypoints, goal_keypoints


def update_scene(scene, state):
    robot, obj_frame, goal_frame, obj_keypoints, goal_keypoints = scene
    joint_pos, obj_pose, goal_pose, obj_kps, goal_kps = state

    robot.update_cfg(joint_pos)
    obj_frame.position = tuple(obj_pose[:3])
    obj_frame.wxyz = quat_xyzw_to_wxyz(obj_pose[3:7])
    goal_frame.position = tuple(goal_pose[:3])
    goal_frame.wxyz = quat_xyzw_to_wxyz(goal_pose[3:7])

    for handle, pos in zip(obj_keypoints, obj_kps):
        handle.position = tuple(pos)
    for handle, pos in zip(goal_keypoints, goal_kps):
        handle.position = tuple(pos)


def reset_env(env, torch, device):
    obs_dict, _, _, _ = env.step(torch.zeros((env.num_envs, N_ACT), device=device))
    return obs_dict["obs"]


def main():
    from isaacgym import gymapi  # noqa: F401 isort:skip
    import torch
    from deployment.isaac.isaac_env import create_env
    from deployment.rl_player import RlPlayer
    import fabrica.objects  # noqa: F401

    args = parse_args()
    config_path = resolve_path(args.config_path)
    checkpoint_path = resolve_path(args.checkpoint_path)
    traj_path = REPO_ROOT / "fabrica" / "trajectories" / args.object_name / f"{args.task_name}.json"
    table_urdf = f"urdf/fabrica/environments/{args.object_name}/{args.task_name}.urdf"
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    traj = load_trajectory(traj_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = create_env(
        config_path=str(config_path),
        headless=not args.no_headless,
        device=device,
        overrides=make_overrides(args.object_name, table_urdf, traj),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    env.set_env_state(checkpoint[0]["env_state"])
    policy = RlPlayer(OBS_DIM, N_ACT, str(config_path), str(checkpoint_path), device, env.num_envs)

    joint_lower = env.arm_hand_dof_lower_limits[:N_ACT].cpu().numpy()
    joint_upper = env.arm_hand_dof_upper_limits[:N_ACT].cpu().numpy()
    obs = reset_env(env, torch, device)
    state = get_state(env, obs, joint_lower, joint_upper)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.add_markdown(f"# Fabrica Beam Assembly\n### {args.object_name} / {args.task_name}")
    run_btn = server.gui.add_button("Run Episode")
    pause_btn = server.gui.add_button("Pause")
    status_md = server.gui.add_markdown("**Status:** Ready")
    progress_md = server.gui.add_markdown("**Progress:** --")
    object_md = server.gui.add_markdown("**Object Pos:** --")
    stats_md = server.gui.add_markdown("**Episodes:** 0")
    scene = add_scene(server, args.object_name, state[3].shape[0])
    update_scene(scene, state)

    print()
    print("  +-------------------------------------------------+")
    print("  |          Minimal Fabrica Eval Viewer            |")
    print(f"  |     http://localhost:{args.port:<26}|")
    print("  +-------------------------------------------------+")
    print()

    control = {
        "run_requested": False,
        "episode_running": False,
        "paused": False,
    }
    step = 0
    max_succ = int(env.max_consecutive_successes)
    episode_count = 0
    episode_goals = []
    episode_lengths = []

    @run_btn.on_click
    def _run(_event):
        if control["episode_running"]:
            return
        control["run_requested"] = True
        control["paused"] = False
        pause_btn.name = "Pause"
        status_md.content = "**Status:** Starting episode..."

    @pause_btn.on_click
    def _pause(_event):
        if not control["episode_running"]:
            return
        control["paused"] = not control["paused"]
        pause_btn.name = "Resume" if control["paused"] else "Pause"
        status_md.content = "**Status:** Paused" if control["paused"] else "**Status:** Running episode..."

    try:
        while True:
            if control["run_requested"] and not control["episode_running"]:
                policy.reset()
                obs = reset_env(env, torch, device)
                state = get_state(env, obs, joint_lower, joint_upper)
                update_scene(scene, state)
                step = 0
                max_succ = int(env.max_consecutive_successes)
                control["run_requested"] = False
                control["episode_running"] = True
                control["paused"] = False
                pause_btn.name = "Pause"
                progress_md.content = "**Progress:** --"
                status_md.content = "**Status:** Running episode..."

            if not control["episode_running"]:
                time.sleep(1.0 / 120.0)
                continue

            if control["paused"]:
                time.sleep(0.05)
                continue

            t0 = time.time()
            action = policy.get_normalized_action(obs, deterministic_actions=True)
            obs_dict, _, done_tensor, _ = env.step(action)
            obs = obs_dict["obs"]
            done = bool(done_tensor[0].item())
            step += 1

            state = get_state(env, obs, joint_lower, joint_upper)
            update_scene(scene, state)

            successes = int(env.successes[0].item())
            pct = 100 * successes / max_succ if max_succ > 0 else 0
            progress_md.content = f"**Time:** {step / 60.0:.1f}s | **Goal:** {successes}/{max_succ} ({pct:.0f}%)"
            obj_pos = state[1][:3]
            object_md.content = f"**Object Pos:** {obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}"

            if done:
                goal_pct = 100.0 * int(env.successes[0].item()) / max_succ if max_succ > 0 else 0.0
                episode_count += 1
                episode_goals.append(goal_pct)
                episode_lengths.append(step)
                control["episode_running"] = False
                control["paused"] = False
                pause_btn.name = "Pause"
                status_md.content = f"**Status:** Done -- {step / 60.0:.1f}s, {goal_pct:.0f}% goals"
                stats_md.content = (
                    f"**Episodes:** {episode_count} | "
                    f"**Avg Goal:** {np.mean(episode_goals):.1f}% | "
                    f"**Avg Time:** {np.mean(episode_lengths) / 60.0:.1f}s"
                )
                print(f"[launcher] Episode done: {goal_pct:.0f}% goals in {step / 60.0:.1f}s")
                continue

            if args.no_headless:
                continue
            elapsed = time.time() - t0
            if elapsed < CONTROL_DT:
                time.sleep(CONTROL_DT - elapsed)
    except KeyboardInterrupt:
        print("\n[launcher] Shutting down...")


if __name__ == "__main__":
    main()

"""Online BC/DAgger-style depth student distillation for Isaac Lab PegInHole.

This is intentionally separate from the older ``isaacsim_conversion`` trainer:
it uses Kushal's Isaac Lab environment directly, asks the env for
``get_student_obs()``, and uses a restored rl_games teacher only to produce
action labels on the student rollouts.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_DIR = Path("/juno/u/kedia/depthbasedRL/train_dir/Apr28/isaacSim_PegInHole")


def _parse_optional_pair(value: str | None) -> tuple[int, int] | None:
    if value is None or value.lower() in {"none", "null"}:
        return None
    parts = value.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"Expected two ints, got {value!r}")
    return int(parts[0]), int(parts[1])


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _load_env_cfg(task: str, teacher_config: Path | None, num_envs: int, sim_device: str):
    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    from isaacsimenvs.utils.config_utils import apply_env_cfg_dict

    spec = gym.spec(task)
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")

    task_yaml = spec.kwargs.get("env_cfg_yaml_entry_point")
    if task_yaml:
        apply_env_cfg_dict(env_cfg, _load_yaml(Path(task_yaml)))

    if teacher_config is not None:
        teacher_env = _load_yaml(teacher_config).get("env", {})
        sections = (
            "sim",
            "scene",
            "assets",
            "obs",
            "action",
            "reward",
            "reset",
            "termination",
            "domain_randomization",
            "peg_in_hole",
        )
        apply_env_cfg_dict(env_cfg, {key: teacher_env[key] for key in sections if key in teacher_env})
        # Re-apply the student-camera overlay after matching teacher dynamics.
        if task_yaml:
            task_overlay = _load_yaml(Path(task_yaml))
            student_overlay = {"student_obs": task_overlay.get("student_obs", {})}
            if "sim" in task_overlay and "render" in task_overlay["sim"]:
                student_overlay["sim"] = {"render": task_overlay["sim"]["render"]}
            apply_env_cfg_dict(env_cfg, student_overlay)

    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.sim.device = sim_device
    return env_cfg


def _load_teacher_player(env, task: str, agent: str, checkpoint: Path, rl_device: str):
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rl_games.torch_runner import Runner

    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env

    agent_cfg = load_cfg_from_registry(task, agent)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device

    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(env, rl_device=rl_device, clip_obs=clip_obs, clip_actions=clip_actions)

    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(str(checkpoint))
    player.has_batch_dimension = True
    player.reset()
    return wrapped, player


def _student_image_channels(modality: str) -> int:
    if modality == "depth":
        return 1
    if modality == "rgb":
        return 3
    if modality == "rgbd":
        return 4
    raise ValueError(f"Unsupported student image modality: {modality!r}")


def _init_wandb(args, run_dir: Path):
    if not args.wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group or None,
        name=args.wandb_name or run_dir.name,
        dir=str(run_dir),
        config=vars(args),
    )


def _log_metrics(run_dir: Path, row: dict, wandb_run=None, step: int | None = None) -> None:
    path = run_dir / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row))
        if write_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in sorted(row)})
    if wandb_run is not None:
        wandb_run.log({key: value for key, value in row.items() if isinstance(value, (int, float))}, step=step)


def _save_checkpoint(path: Path, student, optimizer, step: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "best_metric": float(best_metric),
        },
        path,
    )


def _load_student_checkpoint(path: Path | None, student, optimizer=None) -> tuple[int, float]:
    if path is None:
        return 0, -1.0
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("student", ckpt)
    student.load_state_dict(state)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0)), float(ckpt.get("best_metric", -1.0))


def _object_pos_env_frame(env) -> torch.Tensor:
    return env.object.data.root_pos_w - env.scene.env_origins


def _teacher_obs_tensor(obs) -> torch.Tensor:
    """Extract the policy observation tensor passed through the rl_games wrapper."""
    if isinstance(obs, torch.Tensor):
        return obs.float()
    if isinstance(obs, dict):
        for key in ("obs", "policy", "observations"):
            if key in obs:
                return _teacher_obs_tensor(obs[key])
    raise TypeError(
        "Expected rl_games observation to be a tensor or dict containing an "
        f"'obs' tensor, got {type(obs).__name__}."
    )


def _reset_hidden_for_done(hidden: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    done = dones.reshape(-1).bool()
    if not done.any():
        return hidden
    hidden = hidden.clone()
    hidden[done] = 0.0
    return hidden


def _done_success_stats(env, dones: torch.Tensor) -> tuple[float, float, int]:
    done = dones.reshape(-1).bool()
    count = int(done.sum().item())
    if count == 0:
        return 0.0, 0.0, 0
    successes = env._prev_episode_successes[done].float()
    max_goals = env.prev_episode_env_max_goals[done].clamp_min(1).float()
    return float(successes.mean().item()), float((successes / max_goals).mean().item()), count


def _capture_viewer_if_needed(
    *,
    env,
    frames: list,
    output_dir: Path,
    capture_len: int,
    step: int,
) -> list:
    from isaacsimenvs.tasks.simtoolreal.pose_viewer import (
        build_pose_viewer_html,
        capture_pose_viewer_frame,
        object_urdf_text_for_env,
        table_urdf_text_for_env,
    )

    frames.append(capture_pose_viewer_frame(env, 0))
    if len(frames) < capture_len:
        return frames

    output_dir.mkdir(parents=True, exist_ok=True)
    html = build_pose_viewer_html(
        frames=frames,
        object_urdf_text=object_urdf_text_for_env(env, 0),
        table_urdf_text=table_urdf_text_for_env(env, 0),
        github_raw_base=None,
        url_check="skip",
    )
    path = output_dir / f"pose_viewer_step_{step:08d}.html"
    path.write_text(html, encoding="utf-8")
    print(f"[distill_depth] wrote viewer HTML: {path}", flush=True)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("teacher_eval", "student_eval", "train_online"), default="teacher_eval")
    parser.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    parser.add_argument("--teacher_agent", default="rl_games_sapg_cfg_entry_point")
    parser.add_argument("--teacher_checkpoint", type=Path, default=DEFAULT_TEACHER_DIR / "model.pth")
    parser.add_argument("--teacher_config", type=Path, default=DEFAULT_TEACHER_DIR / "config.yaml")
    parser.add_argument("--student_checkpoint", type=Path, default=None)
    parser.add_argument("--student_input", choices=("camera", "teacher_obs"), default="camera")
    parser.add_argument("--student_arch", choices=("mono_transformer_recurrent", "mlp_recurrent"), default=None)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_iters", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--action_loss_weight", type=float, default=1.0)
    parser.add_argument("--aux_object_pos_weight", type=float, default=1.0)
    parser.add_argument("--deterministic_teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--force_scene_tol_combo", default=None)
    parser.add_argument("--force_peg_idx", type=int, default=None)
    parser.add_argument("--depth_noise_profile", choices=("off", "weak", "medium", "strong", "custom"), default=None)
    parser.add_argument("--depth_noise_strength", type=float, default=None)
    parser.add_argument(
        "--camera_pose_randomization_profile",
        choices=("off", "weak", "medium", "strong", "custom"),
        default=None,
    )
    parser.add_argument("--camera_pose_randomization_mode", choices=("startup", "reset"), default=None)
    parser.add_argument(
        "--camera_pos_noise_m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Custom per-axis camera position noise half-widths in meters.",
    )
    parser.add_argument(
        "--camera_rot_noise_deg",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Custom per-axis camera RPY noise half-widths in degrees.",
    )
    parser.add_argument("--peg_object_init_orientation_mode", choices=("scene", "yaw_only", "full"), default=None)
    parser.add_argument("--peg_object_init_position_noise_xy", type=float, nargs=2, default=None)
    parser.add_argument("--peg_object_init_position_noise_z", type=float, default=None)
    parser.add_argument("--capture_viewer", action="store_true")
    parser.add_argument("--capture_viewer_len", type=int, default=600)
    parser.add_argument("--depth_debug_interval", type=int, default=0)
    parser.add_argument("--depth_debug_env_ids", default="0,1,2,3")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="depthbasedRL-isaacsim-distill")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    # Camera jobs need Isaac Lab's camera pipeline enabled. Teacher-observation
    # jobs do not, so keep them on the cheaper non-rendering path.
    args.enable_cameras = args.student_input == "camera"
    app = AppLauncher(args).app

    import gymnasium as gym

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.distillation.depth_debug import save_depth_debug
    from isaacsimenvs.distillation.student_policy import MLPRecurrentPolicy, MonoTransformerRecurrentPolicy

    run_dir = args.run_dir or REPO_ROOT / "distillation_runs" / f"kushal_depth_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    print(f"[distill_depth] run_dir={run_dir}", flush=True)

    env_cfg = _load_env_cfg(args.task, args.teacher_config, args.num_envs, args.sim_device)
    if args.student_input == "teacher_obs":
        env_cfg.student_obs.image_enabled = False
    if args.force_scene_tol_combo is not None:
        env_cfg.peg_in_hole.force_scene_tol_combo = _parse_optional_pair(args.force_scene_tol_combo)
    if args.force_peg_idx is not None:
        env_cfg.peg_in_hole.force_peg_idx = args.force_peg_idx
    if args.depth_noise_profile is not None:
        env_cfg.student_obs.depth_noise_profile = args.depth_noise_profile
    if args.depth_noise_strength is not None:
        env_cfg.student_obs.depth_noise_strength = args.depth_noise_strength
    if args.camera_pose_randomization_profile is not None:
        env_cfg.student_obs.camera_pose_randomization_profile = args.camera_pose_randomization_profile
    if args.camera_pose_randomization_mode is not None:
        env_cfg.student_obs.camera_pose_randomization_mode = args.camera_pose_randomization_mode
    if args.camera_pos_noise_m is not None:
        env_cfg.student_obs.camera_pos_noise_m = tuple(args.camera_pos_noise_m)
    if args.camera_rot_noise_deg is not None:
        env_cfg.student_obs.camera_rot_noise_deg = tuple(args.camera_rot_noise_deg)
    if args.peg_object_init_orientation_mode is not None:
        env_cfg.peg_in_hole.object_init_orientation_mode = args.peg_object_init_orientation_mode
    if args.peg_object_init_position_noise_xy is not None:
        env_cfg.peg_in_hole.object_init_position_noise_xy = tuple(args.peg_object_init_position_noise_xy)
    if args.peg_object_init_position_noise_z is not None:
        env_cfg.peg_in_hole.object_init_position_noise_z = args.peg_object_init_position_noise_z

    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped
    wrapped, teacher = _load_teacher_player(
        env,
        task=args.task,
        agent=args.teacher_agent,
        checkpoint=args.teacher_checkpoint,
        rl_device=args.rl_device,
    )
    obs = teacher.env_reset(wrapped)

    if args.student_input == "teacher_obs":
        arch = args.student_arch or "mlp_recurrent"
        if arch != "mlp_recurrent":
            raise ValueError("--student_input teacher_obs currently requires --student_arch mlp_recurrent")
        teacher_obs = _teacher_obs_tensor(obs)
        student = MLPRecurrentPolicy(
            obs_dim=teacher_obs.shape[-1],
            action_dim=int(env_cfg.action_space),
            aux_heads={"object_pos": 3},
        ).to(inner.device)
        print(
            f"[distill_depth] teacher_obs student obs_dim={teacher_obs.shape[-1]} "
            f"action_dim={int(env_cfg.action_space)}",
            flush=True,
        )
    else:
        arch = args.student_arch or "mono_transformer_recurrent"
        if arch != "mono_transformer_recurrent":
            raise ValueError("--student_input camera currently requires --student_arch mono_transformer_recurrent")
        student_obs = inner.get_student_obs()
        image = student_obs["image"]
        proprio = student_obs["proprio"]
        student = MonoTransformerRecurrentPolicy(
            image_channels=_student_image_channels(str(env_cfg.student_obs.image_modality).lower()),
            proprio_dim=proprio.shape[-1],
            action_dim=int(env_cfg.action_space),
            aux_heads={"object_pos": 3},
        ).to(inner.device)
        print(
            f"[distill_depth] camera student image_shape={tuple(image.shape)} "
            f"proprio_dim={proprio.shape[-1]} action_dim={int(env_cfg.action_space)}",
            flush=True,
        )
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    start_step, best_metric = _load_student_checkpoint(args.student_checkpoint, student, optimizer)
    hidden = student.initial_state(inner.num_envs, inner.device)
    wandb_run = _init_wandb(args, run_dir)

    depth_debug_env_ids = [
        int(item) for item in args.depth_debug_env_ids.replace(",", " ").split() if item
    ]
    depth_debug_env_ids = [idx for idx in depth_debug_env_ids if 0 <= idx < inner.num_envs]
    viewer_frames: list = []
    interval_action_loss = 0.0
    interval_aux_loss = 0.0
    interval_step_count = 0
    interval_done_goal_idx = 0.0
    interval_done_completion = 0.0
    interval_done_count = 0
    interval_start = time.perf_counter()

    for local_step in range(args.num_iters):
        step = start_step + local_step + 1
        with torch.no_grad():
            teacher_action = teacher.get_action(obs, is_deterministic=args.deterministic_teacher)

        if args.student_input == "teacher_obs":
            student_out, next_hidden = student(_teacher_obs_tensor(obs), hidden)
            image = None
        else:
            student_obs = inner.get_student_obs()
            image = student_obs["image"]
            proprio = student_obs["proprio"]
            student_out, next_hidden = student(image, proprio, hidden)
        student_action = student_out.action
        action_loss = F.mse_loss(student_action, teacher_action, reduction="none").mean(dim=1)
        aux_pred = student_out.aux["object_pos"]
        aux_loss = F.mse_loss(aux_pred, _object_pos_env_frame(inner), reduction="none").mean(dim=1)
        loss = args.action_loss_weight * action_loss.mean() + args.aux_object_pos_weight * aux_loss.mean()

        if args.mode == "train_online":
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            hidden = next_hidden.detach()
            action_for_env = student_action.detach()
        elif args.mode == "student_eval":
            hidden = next_hidden.detach()
            action_for_env = student_action.detach()
        else:
            hidden = hidden.detach()
            action_for_env = teacher_action

        obs, reward, dones, infos = teacher.env_step(wrapped, action_for_env)
        hidden = _reset_hidden_for_done(hidden, dones)
        done_goal_idx, done_completion, done_count = _done_success_stats(inner, dones)
        if done_count:
            interval_done_goal_idx += done_goal_idx * done_count
            interval_done_completion += done_completion * done_count
            interval_done_count += done_count

        if args.capture_viewer:
            viewer_frames = _capture_viewer_if_needed(
                env=inner,
                frames=viewer_frames,
                output_dir=run_dir / "interactive_viewer",
                capture_len=args.capture_viewer_len,
                step=step,
            )

        if (
            args.student_input == "camera"
            and args.depth_debug_interval > 0
            and (step == 1 or step % args.depth_debug_interval == 0)
        ):
            raw_depth = inner.student_camera.data.output["distance_to_image_plane"]
            save_depth_debug(
                output_dir=run_dir / "depth_debug",
                step=step,
                env_ids=depth_debug_env_ids or [0],
                raw_depth=raw_depth,
                noisy_depth=getattr(inner, "_student_depth_noisy_m", None),
                policy_depth=image,
                near=float(env_cfg.student_obs.depth_min_m),
                far=float(env_cfg.student_obs.depth_max_m),
            )

        interval_action_loss += float(action_loss.mean().detach().cpu().item())
        interval_aux_loss += float(aux_loss.mean().detach().cpu().item())
        interval_step_count += 1

        should_log = step == 1 or step % args.log_interval == 0 or local_step == args.num_iters - 1
        if should_log:
            elapsed = max(time.perf_counter() - interval_start, 1e-6)
            current_goal_idx = float(inner._successes.float().mean().detach().cpu().item())
            current_completion = float(
                (inner._successes.float() / inner.env_max_goals.clamp_min(1).float()).mean().detach().cpu().item()
            )
            recent_goal_idx = interval_done_goal_idx / max(interval_done_count, 1)
            recent_completion = interval_done_completion / max(interval_done_count, 1)
            row = {
                "step": step,
                "mode": args.mode,
                "action_loss": interval_action_loss / max(interval_step_count, 1),
                "action_rmse": math.sqrt(max(interval_action_loss / max(interval_step_count, 1), 0.0)),
                "aux_object_pos_loss": interval_aux_loss / max(interval_step_count, 1),
                "aux_object_pos_rmse_m": math.sqrt(max(interval_aux_loss / max(interval_step_count, 1), 0.0)),
                "current_goal_idx_avg": current_goal_idx,
                "current_goal_completion_ratio_avg": current_completion,
                "recent_reset_goal_idx_avg": recent_goal_idx,
                "recent_reset_goal_completion_ratio_avg": recent_completion,
                "recent_reset_count": interval_done_count,
                "env_steps_per_s": inner.num_envs * interval_step_count / elapsed,
            }
            print(
                "[distill_depth] "
                f"step={step} mode={args.mode} current_goal_idx={current_goal_idx:.2f} "
                f"recent_reset_goal_idx={recent_goal_idx:.2f} recent_reset_count={interval_done_count} "
                f"action_rmse={row['action_rmse']:.4f} aux_pos_rmse_cm={100.0 * row['aux_object_pos_rmse_m']:.2f}",
                flush=True,
            )
            _log_metrics(run_dir, row, wandb_run=wandb_run, step=step)
            interval_action_loss = 0.0
            interval_aux_loss = 0.0
            interval_step_count = 0
            interval_done_goal_idx = 0.0
            interval_done_completion = 0.0
            interval_done_count = 0
            interval_start = time.perf_counter()

            selection_metric = recent_completion if row["recent_reset_count"] > 0 else current_completion
            if args.mode == "train_online" and selection_metric >= best_metric:
                best_metric = selection_metric
                _save_checkpoint(run_dir / "checkpoints" / "student_best.pt", student, optimizer, step, best_metric)

        if args.mode == "train_online" and step % args.save_interval == 0:
            _save_checkpoint(run_dir / "checkpoints" / "student_latest.pt", student, optimizer, step, best_metric)

    if viewer_frames:
        _capture_viewer_if_needed(
            env=inner,
            frames=viewer_frames,
            output_dir=run_dir / "interactive_viewer",
            capture_len=len(viewer_frames),
            step=start_step + args.num_iters,
        )
    if args.mode == "train_online":
        _save_checkpoint(run_dir / "checkpoints" / "student_latest.pt", student, optimizer, start_step + args.num_iters, best_metric)

    env.close()
    if wandb_run is not None:
        wandb_run.finish()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from deployment.rl_player import RlPlayer
from isaacgymenvs.utils.observation_action_utils_sharpa import compute_joint_pos_targets
from isaacsim_conversion.distill_env import IsaacSimDistillEnv
from isaacsim_conversion.isaacsim_env import _log
from isaacsim_conversion.student_policy import (
    MonoTransformerRecurrentPolicy,
    preprocess_image,
    resize_image,
)
from isaacsim_conversion.task_utils import CameraPose, load_task_spec, load_yaml


def launch_app():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "teacher_eval", "student_eval", "mixed_eval", "camera_debug"], default="train")
    parser.add_argument("--task_source", choices=["fabrica", "dextoolbench"], default="dextoolbench")
    parser.add_argument("--assembly", default="beam")
    parser.add_argument("--part_id", default="2")
    parser.add_argument("--collision_method", default="coacd")
    parser.add_argument("--object_category", default="hammer")
    parser.add_argument("--object_name", default="claw_hammer")
    parser.add_argument("--task_name", default="swing_down")
    parser.add_argument("--teacher_checkpoint", default="pretrained_policy/model.pth")
    parser.add_argument("--teacher_config", default="pretrained_policy/config.yaml")
    parser.add_argument("--student_checkpoint", default=None)
    parser.add_argument("--student_arch", default="mono_transformer_recurrent")
    parser.add_argument("--student_modality", choices=["depth", "rgb", "rgbd"], default=None)
    parser.add_argument("--distill_config", default="isaacsim_conversion/configs/hammer_distill.yaml")
    parser.add_argument("--camera_config", default="isaacsim_conversion/configs/hammer_camera.yaml")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    return app_launcher.app, args


_app, _args = launch_app()


@dataclass
class DistillSettings:
    num_episodes: int = 10
    max_steps: int = 700
    action_loss_weight: float = 1.0
    aux_object_pos_weight: float = 1.0
    hand_moving_average: float = 0.1
    arm_moving_average: float = 0.1
    dof_speed_scale: float = 1.5
    control_dt: float = 1.0 / 60.0
    beta_mode: str = "metric_driven"
    beta_start: float = 1.0
    beta_end: float = 0.0
    beta_decay: float = 0.1
    beta_metric_threshold: float = 0.8
    beta_update_episodes: int = 2
    checkpoint_interval: int = 5
    eval_interval: int = 5
    image_height: int = 224
    image_width: int = 224
    learning_rate: float = 1e-4


class BetaScheduler:
    def __init__(self, cfg: DistillSettings):
        self.cfg = cfg
        self.beta = cfg.beta_start
        self.episode_counter = 0
        self.window: list[float] = []

    def value(self) -> float:
        if self.cfg.beta_mode == "always_teacher":
            return 1.0
        if self.cfg.beta_mode == "always_student":
            return 0.0
        return self.beta

    def update(self, goal_completion_ratio: float):
        self.episode_counter += 1
        self.window.append(goal_completion_ratio)
        if self.cfg.beta_mode == "fixed_decay":
            self.beta = max(self.cfg.beta_end, self.beta - self.cfg.beta_decay)
            return
        if self.cfg.beta_mode != "metric_driven":
            return
        if len(self.window) < self.cfg.beta_update_episodes:
            return
        avg_metric = float(np.mean(self.window[-self.cfg.beta_update_episodes :]))
        if avg_metric >= self.cfg.beta_metric_threshold:
            self.beta = max(self.cfg.beta_end, self.beta - self.cfg.beta_decay)


def load_distill_settings(path: Path, args) -> DistillSettings:
    cfg = load_yaml(path)
    settings = DistillSettings(**cfg)
    if args.max_steps is not None:
        settings.max_steps = args.max_steps
    if args.num_episodes is not None:
        settings.num_episodes = args.num_episodes
    return settings


def load_camera_pose(path: Path) -> tuple[str, CameraPose]:
    cfg = load_yaml(path)
    modality = cfg.get("modality", "depth")
    pose_cfg = cfg.get("pose", {})
    return modality, CameraPose(
        pos=tuple(float(x) for x in pose_cfg["pos"]),
        quat_wxyz=tuple(float(x) for x in pose_cfg["quat_wxyz"]),
        convention=str(cfg.get("convention", pose_cfg.get("convention", "ros"))),
    )


def resolve_repo_path(repo_root: Path, maybe_relative: str | None) -> str | None:
    if maybe_relative is None:
        return None
    if maybe_relative.startswith("/"):
        return maybe_relative
    return str(repo_root / maybe_relative)


def choose_stepping_action(beta: float, teacher_action: torch.Tensor, student_action: torch.Tensor) -> torch.Tensor:
    if beta >= 1.0:
        return teacher_action
    if beta <= 0.0:
        return student_action
    use_teacher = torch.rand(teacher_action.shape[0], device=teacher_action.device) < beta
    mask = use_teacher.unsqueeze(-1)
    return torch.where(mask, teacher_action, student_action)


def stack_student_image(student_obs: dict[str, torch.Tensor], modality: str, settings: DistillSettings) -> torch.Tensor:
    images = student_obs["images"]
    if modality == "depth":
        image = preprocess_image(images["depth"], modality)
    elif modality == "rgb":
        image = preprocess_image(images["rgb"], modality)
    elif modality == "rgbd":
        image = torch.cat([images["rgb"], images["depth"]], dim=1)
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return resize_image(image, settings.image_height, settings.image_width)


def build_student(args, env: IsaacSimDistillEnv, settings: DistillSettings) -> MonoTransformerRecurrentPolicy:
    if args.student_arch != "mono_transformer_recurrent":
        raise ValueError(f"Unsupported student_arch for now: {args.student_arch}")
    image_channels = {"depth": 1, "rgb": 3, "rgbd": 4}[env.camera_modality]
    return MonoTransformerRecurrentPolicy(
        image_channels=image_channels,
        proprio_dim=env.student_proprio_dim,
        action_dim=env.action_dim,
        aux_heads={"object_pos": 3},
    ).to(env.env.sim.device)


def save_checkpoint(path: Path, student: torch.nn.Module, optimizer: torch.optim.Optimizer, episode: int, best_metric: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "best_metric": best_metric,
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_student_checkpoint(path: str | None, student: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> tuple[int, float]:
    if not path:
        return 0, -1.0
    ckpt = torch.load(path, map_location="cpu")
    student.load_state_dict(ckpt["student_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt.get("episode", 0)), float(ckpt.get("best_metric", -1.0))


def ensure_run_dir(repo_root: Path, args, settings: DistillSettings) -> Path:
    if args.run_dir:
        run_dir = Path(resolve_repo_path(repo_root, args.run_dir))
    else:
        run_name = f"{args.object_category}_{args.object_name}_{args.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = repo_root / "distillation_runs" / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "camera_debug").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "resolved_distill_config.yaml", "w") as f:
        yaml.safe_dump(settings.__dict__, f, sort_keys=False)
    return run_dir


def log_csv_row(csv_path: Path, row: dict[str, float | int | str]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def force_exit(code: int = 0):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def run_episode(
    mode: str,
    env: IsaacSimDistillEnv,
    teacher: RlPlayer,
    student: MonoTransformerRecurrentPolicy,
    optimizer: torch.optim.Optimizer | None,
    settings: DistillSettings,
    beta_scheduler: BetaScheduler,
) -> dict[str, float]:
    teacher.reset()
    env.reset()
    student_hidden = student.initial_state(batch_size=1, device=env.env.sim.device)
    total_action_loss = 0.0
    total_aux_loss = 0.0
    total_steps = 0

    for step_i in range(settings.max_steps):
        sim_state = env.compute_sim_state()
        teacher_obs = env.build_teacher_obs(sim_state)
        teacher_obs_tensor = torch.from_numpy(teacher_obs).float().to(env.env.sim.device)
        teacher_action = teacher.get_normalized_action(teacher_obs_tensor, deterministic_actions=True)

        student_obs = env.build_student_obs(sim_state, camera_modality=env.camera_modality)
        student_image = stack_student_image(student_obs, env.camera_modality, settings)
        student_out, student_hidden = student(student_image, student_obs["proprio"], student_hidden)
        student_action = student_out.action

        beta = beta_scheduler.value() if mode in ("train", "mixed_eval") else (1.0 if mode == "teacher_eval" else 0.0)
        stepping_action = choose_stepping_action(beta, teacher_action, student_action)
        targets = compute_joint_pos_targets(
            actions=stepping_action.detach().cpu().numpy(),
            prev_targets=env.prev_targets,
            hand_moving_average=settings.hand_moving_average,
            arm_moving_average=settings.arm_moving_average,
            hand_dof_speed_scale=settings.dof_speed_scale,
            dt=settings.control_dt,
        )
        env.apply_action(targets[0])
        env.step(render=not _args.headless)
        next_sim_state = env.compute_sim_state()
        finished = env.maybe_advance_goal(next_sim_state)

        action_loss = torch.mean((student_action - teacher_action) ** 2)
        aux_pred = student_out.aux.get("object_pos")
        gt_object_pos = torch.from_numpy(sim_state.object_pos_world_env_frame[None]).float().to(env.env.sim.device)
        aux_loss = torch.mean((aux_pred - gt_object_pos) ** 2) if aux_pred is not None else torch.tensor(0.0, device=env.env.sim.device)
        total_loss = settings.action_loss_weight * action_loss + settings.aux_object_pos_weight * aux_loss

        if mode == "train" and optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        total_action_loss += float(action_loss.item())
        total_aux_loss += float(aux_loss.item())
        total_steps += 1

        if step_i % 60 == 0:
            _log(
                f"[distill step {step_i:5d}] mode={mode}, goal={env.goal_idx}/{len(env.task_spec.goals)}, "
                f"kp_dist={next_sim_state.kp_dist:.4f}, beta={beta:.3f}"
            )
        if finished:
            break

    metrics = env.compute_progress_metrics(env.compute_sim_state())
    metrics.update(
        {
            "episode_steps": float(total_steps),
            "action_loss": total_action_loss / max(total_steps, 1),
            "aux_object_pos_loss": total_aux_loss / max(total_steps, 1),
            "beta": beta_scheduler.value(),
        }
    )
    return metrics


def save_camera_debug(env: IsaacSimDistillEnv, run_dir: Path):
    import imageio.v3 as iio

    outputs = env.camera.data.output
    if outputs.get("rgb") is not None:
        rgb = outputs["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
        iio.imwrite(run_dir / "camera_debug" / "rgb.png", rgb)
    if outputs.get("distance_to_image_plane") is not None:
        depth = outputs["distance_to_image_plane"][0].cpu().numpy()
        if depth.ndim == 3:
            depth = depth[..., 0]
        finite_mask = np.isfinite(depth)
        if np.any(finite_mask):
            finite_depth = depth[finite_mask]
            depth_min = float(np.min(finite_depth))
            depth_max = float(np.max(finite_depth))
            denom = max(depth_max - depth_min, 1e-6)
            depth_vis = np.zeros_like(depth, dtype=np.float32)
            depth_vis[finite_mask] = np.clip((depth[finite_mask] - depth_min) / denom, 0.0, 1.0)
        else:
            depth_vis = np.zeros_like(depth, dtype=np.float32)
        iio.imwrite(run_dir / "camera_debug" / "depth.png", (depth_vis * 255).astype(np.uint8))


def main():
    args = _args
    app = _app
    repo_root = Path(__file__).resolve().parent.parent
    teacher_config = Path(resolve_repo_path(repo_root, args.teacher_config))
    teacher_checkpoint = resolve_repo_path(repo_root, args.teacher_checkpoint)
    settings = load_distill_settings(Path(resolve_repo_path(repo_root, args.distill_config)), args)
    camera_modality, camera_pose = load_camera_pose(Path(resolve_repo_path(repo_root, args.camera_config)))
    if args.student_modality:
        camera_modality = args.student_modality

    task_spec = load_task_spec(
        repo_root=repo_root,
        task_source=args.task_source,
        assembly=args.assembly,
        part_id=args.part_id,
        collision_method=args.collision_method,
        object_category=args.object_category,
        object_name=args.object_name,
        task_name=args.task_name,
        teacher_config_path=teacher_config,
    )
    env = IsaacSimDistillEnv(
        task_spec=task_spec,
        app=app,
        headless=args.headless,
        camera_modality=camera_modality,
        camera_pose_override=camera_pose,
    )
    teacher = RlPlayer(
        num_observations=140,
        num_actions=29,
        config_path=str(teacher_config),
        checkpoint_path=teacher_checkpoint,
        device=str(env.env.sim.device),
        num_envs=1,
    )
    student = build_student(args, env, settings)
    optimizer = torch.optim.Adam(student.parameters(), lr=settings.learning_rate)
    if args.student_checkpoint:
        load_student_checkpoint(resolve_repo_path(repo_root, args.student_checkpoint), student, optimizer)
    run_dir = ensure_run_dir(repo_root, args, settings)
    beta_scheduler = BetaScheduler(settings)

    _log(f"Teacher checkpoint: {teacher_checkpoint}")
    _log(f"Teacher config: {teacher_config}")
    _log(f"Run dir: {run_dir}")
    _log(f"Student arch/modality: {args.student_arch}/{camera_modality}")

    if args.mode == "camera_debug":
        env.reset()
        env.step(render=not args.headless)
        save_camera_debug(env, run_dir)
        _log(f"Saved camera debug outputs under {run_dir / 'camera_debug'}")
        force_exit(0)

    best_metric = -1.0
    for episode in range(settings.num_episodes if args.mode == "train" else 1):
        metrics = run_episode(args.mode, env, teacher, student, optimizer if args.mode == "train" else None, settings, beta_scheduler)
        _log(
            f"[episode {episode}] mode={args.mode}, goal_completion_ratio={metrics['goal_completion_ratio']:.3f}, "
            f"goal_idx={metrics['goal_idx']:.0f}, action_loss={metrics['action_loss']:.6f}, "
            f"aux_object_pos_loss={metrics['aux_object_pos_loss']:.6f}, beta={metrics['beta']:.3f}"
        )
        log_csv_row(
            run_dir / "metrics.csv",
            {
                "episode": episode,
                "mode": args.mode,
                **metrics,
            },
        )
        save_camera_debug(env, run_dir)
        if args.mode == "train":
            beta_scheduler.update(metrics["goal_completion_ratio"])
            save_checkpoint(run_dir / "checkpoints" / "student_latest.pt", student, optimizer, episode, best_metric)
            if metrics["goal_completion_ratio"] >= best_metric:
                best_metric = metrics["goal_completion_ratio"]
                save_checkpoint(run_dir / "checkpoints" / "student_best.pt", student, optimizer, episode, best_metric)
            if (episode + 1) % settings.checkpoint_interval == 0:
                save_checkpoint(run_dir / "checkpoints" / f"student_step_{episode + 1}.pt", student, optimizer, episode, best_metric)

    force_exit(0)


if __name__ == "__main__":
    main()

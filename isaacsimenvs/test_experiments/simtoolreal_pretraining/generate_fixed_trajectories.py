"""Pre-generate a pool of fixed goal trajectories for the trajectory_count ablation.

Each trajectory is a length-K chain of (pos, quat_wxyz) goal poses in env-local
coordinates: the first goal is uniformly random in the workspace (matching
``sample_absolute_goal_pose``); each subsequent goal is a bounded random walk
from the previous one (matching ``sample_delta_goal_pose``).

The output JSON is consumed by SimToolRealEnv when
``cfg.reset.fixed_trajectory_file`` is set. Different ablation runs share the
same JSON and pick the first ``N`` trajectories via
``cfg.reset.fixed_trajectory_count``, so trajectory id ``i`` is identical
across N = 1 / 10 / 100 / 1000 runs.

Math is inlined (rather than imported from ``goal_sampling.py``) so the script
runs without the Isaac Sim Kit runtime. The arithmetic is identical:
* ``_random_orientation`` — Shoemake uniform-random unit quat (wxyz).
* ``_quat_from_angle_axis`` / ``_quat_mul`` — wxyz-convention quaternion ops.

Defaults mirror ``SimToolRealEnvCfg.reset`` so the pool statistics match the
baseline's live samplers exactly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


# ── Quaternion helpers (wxyz convention) ────────────────────────────────


def _random_orientation(n: int, device: torch.device) -> torch.Tensor:
    """Uniform random unit quaternion, Shoemake's algorithm. Returns (n, 4)."""
    u1 = torch.rand(n, device=device)
    u2 = torch.rand(n, device=device) * 2.0 * math.pi
    u3 = torch.rand(n, device=device) * 2.0 * math.pi
    s1 = torch.sqrt(1.0 - u1)
    s2 = torch.sqrt(u1)
    w = s2 * torch.cos(u3)
    x = s1 * torch.sin(u2)
    y = s1 * torch.cos(u2)
    z = s2 * torch.sin(u3)
    return torch.stack([w, x, y, z], dim=-1)


def _quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """angle: (N,), axis: (N, 3) unit-norm. Returns (N, 4) wxyz."""
    half = angle * 0.5
    w = torch.cos(half).unsqueeze(-1)
    xyz = axis * torch.sin(half).unsqueeze(-1)
    return torch.cat([w, xyz], dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion product (wxyz). Both (N, 4)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


# ── Workspace sampling (mirrors goal_sampling.py) ───────────────────────


def _scale_workspace_bounds(
    mins: torch.Tensor, maxs: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    center = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins) * scale
    return center - half, center + half


def _sample_absolute(
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
    n: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mins_t = torch.as_tensor(mins, device=device, dtype=torch.float32)
    maxs_t = torch.as_tensor(maxs, device=device, dtype=torch.float32)
    lo, hi = _scale_workspace_bounds(mins_t, maxs_t, scale)
    pos = lo + (hi - lo) * torch.rand(n, 3, device=device)
    return pos, _random_orientation(n, device=device)


def _sample_delta(
    prev_pos: torch.Tensor,
    prev_quat: torch.Tensor,
    delta_distance: float,
    delta_rotation_degrees: float,
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = prev_pos.device
    n = prev_pos.shape[0]
    mins_t = torch.as_tensor(mins, device=device, dtype=torch.float32)
    maxs_t = torch.as_tensor(maxs, device=device, dtype=torch.float32)
    lo, hi = _scale_workspace_bounds(mins_t, maxs_t, scale)

    pos_noise = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * delta_distance
    new_pos = torch.clamp(prev_pos + pos_noise, lo, hi)

    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=device), dim=-1)
    angle = (torch.rand(n, device=device) * 2.0 - 1.0) * delta_rotation_degrees * (
        math.pi / 180.0
    )
    dq = _quat_from_angle_axis(angle, axis)
    new_quat = _quat_mul(dq, prev_quat)
    return new_pos, new_quat


# ── Top-level ───────────────────────────────────────────────────────────


def generate_trajectories(
    n_trajectories: int,
    k_goals: int,
    seed: int,
    workspace_mins: tuple[float, float, float],
    workspace_maxs: tuple[float, float, float],
    workspace_scale: float,
    delta_distance: float,
    delta_rotation_degrees: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pos, quat_wxyz)`` of shape ``(N, K, 3)`` and ``(N, K, 4)``."""
    torch.manual_seed(seed)

    pos = torch.empty(n_trajectories, k_goals, 3, device=device)
    quat = torch.empty(n_trajectories, k_goals, 4, device=device)

    pos0, quat0 = _sample_absolute(
        workspace_mins, workspace_maxs, workspace_scale,
        n=n_trajectories, device=device,
    )
    pos[:, 0] = pos0
    quat[:, 0] = quat0

    for k in range(1, k_goals):
        pos_k, quat_k = _sample_delta(
            prev_pos=pos[:, k - 1],
            prev_quat=quat[:, k - 1],
            delta_distance=delta_distance,
            delta_rotation_degrees=delta_rotation_degrees,
            mins=workspace_mins,
            maxs=workspace_maxs,
            scale=workspace_scale,
        )
        pos[:, k] = pos_k
        quat[:, k] = quat_k

    return pos, quat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="number of trajectories")
    parser.add_argument("--k", type=int, default=10, help="goals per trajectory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workspace_mins",
        type=float,
        nargs=3,
        default=(-0.35, -0.2, 0.6),
        help="default matches SimToolRealEnvCfg.reset.target_volume_mins",
    )
    parser.add_argument(
        "--workspace_maxs",
        type=float,
        nargs=3,
        default=(0.35, 0.2, 0.95),
        help="default matches SimToolRealEnvCfg.reset.target_volume_maxs",
    )
    parser.add_argument("--workspace_scale", type=float, default=1.0)
    parser.add_argument("--delta_distance", type=float, default=0.1)
    parser.add_argument("--delta_rotation_degrees", type=float, default=90.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "tasks/simtoolreal/data/simtoolreal_trajectories_n1000_k10_seed0.json",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    pos, quat = generate_trajectories(
        n_trajectories=args.n,
        k_goals=args.k,
        seed=args.seed,
        workspace_mins=tuple(args.workspace_mins),
        workspace_maxs=tuple(args.workspace_maxs),
        workspace_scale=args.workspace_scale,
        delta_distance=args.delta_distance,
        delta_rotation_degrees=args.delta_rotation_degrees,
        device=device,
    )

    payload = {
        "metadata": {
            "n_trajectories": args.n,
            "k_goals_per_trajectory": args.k,
            "seed": args.seed,
            "workspace_mins": list(args.workspace_mins),
            "workspace_maxs": list(args.workspace_maxs),
            "workspace_scale": args.workspace_scale,
            "delta_distance": args.delta_distance,
            "delta_rotation_degrees": args.delta_rotation_degrees,
            "schema": "pos (N,K,3) env-local meters; quat_wxyz (N,K,4) unit",
        },
        "pos": pos.tolist(),
        "quat_wxyz": quat.tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f)

    size_kb = args.output.stat().st_size / 1024
    print(f"wrote {args.output}  ({args.n}x{args.k},  {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

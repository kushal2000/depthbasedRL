"""Goal-pose samplers for SimToolReal.

Pure-torch helpers — no env references — so they're straightforward to unit
test without spinning up Isaac Sim. Two modes, matching legacy
``_reset_target`` at ``isaacgymenvs/tasks/simtoolreal/env.py:1434-1527``:

- :func:`sample_absolute_goal_pose` — uniform sample inside the scaled
  workspace box.
- :func:`sample_delta_goal_pose` — bounded random walk from the previous goal
  (chain-like continuation); matches env.py:1395-1424 ``_sample_delta_goal``.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_from_angle_axis, quat_mul, random_orientation


def _scale_workspace_bounds(
    mins: torch.Tensor, maxs: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shrink/expand the workspace box around its center by ``scale``.

    For ``scale=1.0`` this is the identity. ``scale < 1`` shrinks the box;
    ``scale > 1`` expands it. Preserves the center.
    """
    center = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins) * scale
    return center - half, center + half


def sample_absolute_goal_pose(
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
    n_envs: int,
    device: torch.device,
    randomize_rotation: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform sample inside the scaled workspace box.

    Returns
    -------
    pos : (n_envs, 3)
    quat_wxyz : (n_envs, 4) — random orientation if ``randomize_rotation``,
        else identity.
    """
    mins_t = torch.as_tensor(mins, device=device, dtype=torch.float32)
    maxs_t = torch.as_tensor(maxs, device=device, dtype=torch.float32)
    lo, hi = _scale_workspace_bounds(mins_t, maxs_t, scale)
    pos = lo + (hi - lo) * torch.rand(n_envs, 3, device=device)
    if randomize_rotation:
        quat = random_orientation(n_envs, device=device)
    else:
        quat = torch.zeros(n_envs, 4, device=device)
        quat[:, 0] = 1.0  # wxyz identity
    return pos, quat


def sample_delta_goal_pose(
    prev_pos: torch.Tensor,              # (N, 3)
    prev_quat_wxyz: torch.Tensor,         # (N, 4)
    delta_distance: float,
    delta_rotation_degrees: float,
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perturb the previous goal by a bounded random walk.

    Position: ``prev_pos + uniform(-delta, +delta)``, clamped to the scaled
    workspace box. Rotation: random-axis rotation by a uniform angle in
    ``±delta_rotation_degrees``, composed onto ``prev_quat_wxyz``.
    """
    device = prev_pos.device
    n = prev_pos.shape[0]

    mins_t = torch.as_tensor(mins, device=device, dtype=torch.float32)
    maxs_t = torch.as_tensor(maxs, device=device, dtype=torch.float32)
    lo, hi = _scale_workspace_bounds(mins_t, maxs_t, scale)

    # Position: symmetric uniform perturbation clamped to workspace.
    pos_noise = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * delta_distance
    new_pos = torch.clamp(prev_pos + pos_noise, lo, hi)

    # Rotation: random unit axis × uniform angle in ±delta_rotation_degrees.
    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=device), dim=-1)
    angle = (torch.rand(n, device=device) * 2.0 - 1.0) * delta_rotation_degrees * (
        math.pi / 180.0
    )
    dq = quat_from_angle_axis(angle, axis)
    new_quat = quat_mul(dq, prev_quat_wxyz)
    return new_pos, new_quat


__all__ = ["sample_absolute_goal_pose", "sample_delta_goal_pose"]

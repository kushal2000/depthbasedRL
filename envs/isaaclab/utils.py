"""Shared utilities for Isaac Lab environments.

Quaternion convention helpers and math utilities used across the
Isaac Lab SimToolReal environment.
"""

import torch
from torch import Tensor


# ── Quaternion convention conversion ──────────────────────────────────
# IsaacGym uses xyzw, Isaac Lab uses wxyz.

def quat_xyzw_to_wxyz(q: Tensor) -> Tensor:
    """Convert quaternion from xyzw (IsaacGym) to wxyz (Isaac Lab) convention."""
    return torch.cat([q[..., 3:4], q[..., :3]], dim=-1)


def quat_wxyz_to_xyzw(q: Tensor) -> Tensor:
    """Convert quaternion from wxyz (Isaac Lab) to xyzw (IsaacGym) convention."""
    return torch.cat([q[..., 1:], q[..., :1]], dim=-1)


# ── Quaternion math (wxyz convention, matching Isaac Lab) ─────────────

def quat_mul_wxyz(q1: Tensor, q2: Tensor) -> Tensor:
    """Multiply two quaternions in wxyz convention."""
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
    return torch.cat([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_conjugate_wxyz(q: Tensor) -> Tensor:
    """Conjugate of a quaternion in wxyz convention."""
    return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)


def quat_rotate_wxyz(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector v by quaternion q (wxyz convention).

    Args:
        q: Quaternions (..., 4) in wxyz
        v: Vectors (..., 3)
    Returns:
        Rotated vectors (..., 3)
    """
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    return a + b + c


def quat_from_angle_axis(angle: Tensor, axis: Tensor) -> Tensor:
    """Create quaternion from angle (radians) and axis (wxyz convention).

    Args:
        angle: (...,) angles in radians
        axis: (..., 3) unit axis vectors
    Returns:
        Quaternions (..., 4) in wxyz convention
    """
    half_angle = angle.unsqueeze(-1) * 0.5
    return torch.cat([
        torch.cos(half_angle),
        axis * torch.sin(half_angle),
    ], dim=-1)


# ── Tensor utilities ──────────────────────────────────────────────────

def scale(x: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Scale x from [-1, 1] to [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Scale x from [lower, upper] to [-1, 1]."""
    return 2.0 * (x - lower) / (upper - lower) - 1.0


def tensor_clamp(t: Tensor, min_t: Tensor, max_t: Tensor) -> Tensor:
    """Clamp tensor t element-wise between min_t and max_t."""
    return torch.max(torch.min(t, max_t), min_t)


def torch_rand_float(low: float, high: float, shape: tuple, device: str) -> Tensor:
    """Generate uniform random floats in [low, high)."""
    return (high - low) * torch.rand(shape, device=device) + low


def get_axis_params(value: float, axis_idx: int, x_value: float = 0.0, dtype=float):
    """Construct axis-aligned parameter list (used for up_axis initialization)."""
    params = [x_value, 0.0, 0.0]
    params[axis_idx] = value
    return params

"""Obs builders + step-shared intermediates for SimToolReal.

Public entry points used by :class:`SimToolRealEnv`:
  - :func:`compute_obs_dim` — field-list → total dim (used in __init__).
  - :func:`compute_intermediate_values` — runs at the start of ``_get_dones``;
    caches ``_keypoints_max_dist``, ``_curr_fingertip_distances``, ``_near_goal``,
    updates ``_near_goal_steps``, sets ``_is_success``.
  - :func:`build_observations` — runs in ``_get_observations``; returns
    ``{"policy": (N, 140), "critic": (N, 162)}``.

Module-level constants:
  - ``OBS_FIELD_SIZES`` — maps obs field names (keys in ``cfg.obs.obs_list`` /
    ``cfg.obs.state_list``) to tensor dims.
  - ``NUM_JOINTS``, ``NUM_FINGERTIPS``, ``NUM_KEYPOINTS``, ``KEYPOINT_CORNERS``.

Per-field tensor math lives in :func:`build_observations`; DR perturbations
are isolated in :func:`_apply_object_state_dr` and :func:`_apply_obs_delay`
(module-private).
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import convert_quat, quat_apply, quat_from_angle_axis, quat_mul


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------


NUM_JOINTS: int = 29
NUM_FINGERTIPS: int = 5
NUM_KEYPOINTS: int = 4

# Palm-local offset (legacy env.py:309) used to compute palm_center_pos:
#   palm_center_pos = palm_body_pos + quat_apply(palm_rot, PALM_CENTER_OFFSET).
# This is the reference point used for palm_pos obs and for
# fingertip_pos_rel_palm / keypoints_rel_palm — the legacy palm "body" is
# at the wrist, the policy was trained against the palm center (~16cm
# along the palm-local z) so we must replicate the offset.
PALM_CENTER_OFFSET: tuple[float, float, float] = (-0.0, -0.02, 0.16)

# Per-fingertip offset in fingertip-local frame (legacy env.py:298-307).
# The DP body origin sits ~at the joint, so legacy shifts each fingertip
# reference forward by 2cm along finger-local x and 2mm along y to land
# near the actual finger pad. Same offset for all 5 SHARPA fingertips.
FINGERTIP_OFFSET: tuple[float, float, float] = (0.02, 0.002, 0.0)

# Object-frame corner offsets before scaling (legacy env.py:1205-1211).
KEYPOINT_CORNERS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, 1, -1),
    (-1, -1, 1),
    (-1, -1, -1),
)

OBS_FIELD_SIZES: dict[str, int] = {
    "joint_pos": NUM_JOINTS,
    "joint_vel": NUM_JOINTS,
    "prev_action_targets": NUM_JOINTS,
    "palm_pos": 3,
    "palm_rot": 4,
    "palm_vel": 6,
    "object_rot": 4,
    "object_vel": 6,
    "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS,  # 15
    "keypoints_rel_palm": 3 * NUM_KEYPOINTS,  # 12
    "keypoints_rel_goal": 3 * NUM_KEYPOINTS,  # 12
    "object_scales": 3,
    "closest_keypoint_max_dist": 1,
    "closest_fingertip_dist": NUM_FINGERTIPS,  # 5
    "lifted_object": 1,
    "progress": 1,
    "successes": 1,
    "reward": 1,
}


def compute_obs_dim(field_list) -> int:
    """Return total tensor dim for an ordered list of obs field names."""
    return sum(OBS_FIELD_SIZES[f] for f in field_list)


def _stack_obs_dict(obs_dict: dict[str, torch.Tensor], field_list) -> torch.Tensor:
    """Concatenate named obs tensors into ``(num_envs, total_dim)``."""
    return torch.cat(
        [obs_dict[f].reshape(obs_dict[f].shape[0], -1) for f in field_list],
        dim=-1,
    )


# ----------------------------------------------------------------------------
# Quaternion / keypoint helpers
# ----------------------------------------------------------------------------


def _perturb_quat(q_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    """Apply a small random-axis rotation to a batch of wxyz quaternions
    (port of legacy env.py:6002-6029 ``sample_delta_quat_xyzw``)."""
    n = q_wxyz.shape[0]
    axis = torch.nn.functional.normalize(
        torch.randn(n, 3, device=q_wxyz.device), dim=-1
    )
    angle = torch.empty(n, device=q_wxyz.device).uniform_(
        -max_deg, max_deg
    ) * (math.pi / 180.0)
    dq = quat_from_angle_axis(angle, axis)
    return quat_mul(dq, q_wxyz)


def _keypoints_world(
    center_pos: torch.Tensor,    # (N, 3)
    center_rot: torch.Tensor,    # (N, 4) wxyz
    kp_offsets: torch.Tensor,    # (N, K, 3)
) -> torch.Tensor:
    """Rotate + translate object-frame keypoint offsets into world/env-local.

    Returns ``(N, K, 3)`` — one rotated-and-translated keypoint per corner.
    """
    n_envs, k, _ = kp_offsets.shape
    rot_r = center_rot.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    offsets_r = kp_offsets.reshape(-1, 3)
    return center_pos.unsqueeze(1) + quat_apply(rot_r, offsets_r).reshape(n_envs, k, 3)


# ----------------------------------------------------------------------------
# Step-shared intermediate values (feeds _get_dones + _get_rewards)
# ----------------------------------------------------------------------------


def compute_intermediate_values(env) -> None:
    """Compute step-level shared state; mutates env in-place.

    Writes to env:
      - ``_keypoints_max_dist``       (N,)
      - ``_curr_fingertip_distances`` (N, NUM_FINGERTIPS)
      - ``_near_goal``                (N,) bool
      - ``_near_goal_steps``          (N,) long — updated in place
      - ``_is_success``               (N,) bool
    """
    from .reward_utils import update_near_goal_steps  # local import to avoid cycle

    rew_cfg = env.cfg.reward
    term_cfg = env.cfg.termination
    env_origins = env.scene.env_origins

    obj_pos = env.object.data.root_pos_w - env_origins
    obj_rot = env.object.data.root_quat_w
    goal_pos = env.goal_viz.data.root_pos_w - env_origins
    goal_rot = env.goal_viz.data.root_quat_w

    ft_state = env.robot.data.body_state_w[:, env._fingertip_body_ids, :]
    ft_pos = ft_state[:, :, 0:3] - env_origins.unsqueeze(1)
    env._curr_fingertip_distances = torch.norm(
        ft_pos - obj_pos.unsqueeze(1), dim=-1
    )  # (N, 5)

    # Reward keypoints — fixed-size variant if configured; else per-env scaled
    # (no DR noise — obs side adds noise separately).
    if rew_cfg.fixed_size_keypoint_reward:
        kp_offsets = env._keypoint_offsets_fixed
    else:
        kp_offsets = env._keypoint_offsets

    obj_kp = _keypoints_world(obj_pos, obj_rot, kp_offsets)
    goal_kp = _keypoints_world(goal_pos, goal_rot, kp_offsets)

    env._keypoints_max_dist = torch.norm(obj_kp - goal_kp, dim=-1).max(dim=-1).values

    # Lazy-init the "closest-so-far" trackers from the current value when the
    # sentinel (<0) is set — mirrors legacy env.py:3152-3160. Without this,
    # initializing the tracker to a fake "very far" value (e.g. 10.0) and
    # then doing ``delta = closest - current`` on the next step produces a
    # spurious one-shot reward of `(10.0 - small) * keypoint_rew_scale`,
    # which on a goal-hit reset reads as a +2000 reward spike that legacy
    # gym never gave (gym uses -1 sentinel and inits to current here).
    sentinel = env._closest_keypoint_max_dist < 0.0
    env._closest_keypoint_max_dist = torch.where(
        sentinel, env._keypoints_max_dist, env._closest_keypoint_max_dist
    )
    sentinel_ft = env._closest_fingertip_dist < 0.0
    env._closest_fingertip_dist = torch.where(
        sentinel_ft, env._curr_fingertip_distances, env._closest_fingertip_dist
    )

    tol = env._current_success_tolerance * rew_cfg.keypoint_scale
    env._near_goal = env._keypoints_max_dist <= tol
    env._near_goal_steps = update_near_goal_steps(
        near_goal=env._near_goal,
        near_goal_steps=env._near_goal_steps,
        force_consecutive=term_cfg.force_consecutive_near_goal_steps,
    )
    env._is_success = env._near_goal_steps >= term_cfg.success_steps


# ----------------------------------------------------------------------------
# Observation builder (Phase D)
# ----------------------------------------------------------------------------


def _apply_object_state_dr(env, obj_pos, obj_rot, obj_linvel, obj_angvel):
    """Push clean object state into queue, sample per-env delay, return
    noisy ``(pos, rot, vel)``. Only called when ``use_object_state_delay_noise``
    is True."""
    dr = env.cfg.domain_randomization
    state_stack = torch.cat([obj_pos, obj_rot, obj_linvel, obj_angvel], dim=-1)  # (N, 13)
    env._object_state_queue = torch.roll(env._object_state_queue, shifts=1, dims=1)
    env._object_state_queue[:, 0, :] = state_stack
    idx = torch.randint(
        0, env._object_state_queue.shape[1], (env.num_envs,), device=env.device
    )
    delayed = env._object_state_queue[
        torch.arange(env.num_envs, device=env.device), idx
    ]
    noisy_pos = delayed[:, 0:3] + torch.randn_like(delayed[:, 0:3]) * dr.object_state_xyz_noise_std
    noisy_rot = _perturb_quat(delayed[:, 3:7], dr.object_state_rotation_noise_degrees)
    noisy_vel = delayed[:, 7:13]
    return noisy_pos, noisy_rot, noisy_vel


def _apply_obs_delay(env, policy_tensor: torch.Tensor) -> torch.Tensor:
    """Push policy tensor into obs queue, sample per-env delay, return delayed."""
    env._obs_queue = torch.roll(env._obs_queue, shifts=1, dims=1)
    env._obs_queue[:, 0, :] = policy_tensor
    idx = torch.randint(
        0, env._obs_queue.shape[1], (env.num_envs,), device=env.device
    )
    return env._obs_queue[
        torch.arange(env.num_envs, device=env.device), idx
    ]


def build_observations(env) -> dict[str, torch.Tensor]:
    """Assemble asymmetric actor-critic observations with obs-side DR.

    Returns ``{"policy": (N, 140), "critic": (N, 162)}``.
    """
    dr = env.cfg.domain_randomization
    env_origins = env.scene.env_origins

    # --- Raw state reads ---
    # Reorder Lab-parser joint tensors → canonical (legacy isaacgymenvs)
    # order so the policy sees joint_pos/joint_vel/prev_action_targets in
    # the same layout it was trained against. Internal storage
    # (env.robot.data.joint_pos, env._prev_targets) stays in Lab order.
    #
    # joint_pos is normalized to [-1, 1] via unscale(q, lower, upper) to
    # match legacy env.py:3164 — joint_vel and prev_action_targets stay
    # raw because legacy emits those raw too.
    joint_pos_raw = env.robot.data.joint_pos[:, env._perm_lab_to_canon]
    joint_pos = (
        2.0 * (joint_pos_raw - env._joint_lower_canon)
        / (env._joint_upper_canon - env._joint_lower_canon)
        - 1.0
    )
    joint_vel = env.robot.data.joint_vel[:, env._perm_lab_to_canon]
    prev_targets_canon = env._prev_targets[:, env._perm_lab_to_canon]

    palm_state = env.robot.data.body_state_w[:, env._palm_body_id, :]  # (N, 13)
    palm_pos_w = palm_state[:, 0:3]
    palm_rot = palm_state[:, 3:7]  # wxyz (Isaac Lab convention)
    palm_vel = palm_state[:, 7:13]

    # palm_center_pos = palm_body_pos + R(palm_rot) @ PALM_CENTER_OFFSET.
    # Legacy env (isaacgymenvs/.../env.py:3032) uses this — not the raw
    # palm body — as the reference for palm_pos obs and for
    # fingertip_pos_rel_palm / keypoints_rel_palm.
    palm_offset_local = torch.tensor(
        PALM_CENTER_OFFSET, device=env.device, dtype=torch.float32
    ).expand(env.num_envs, 3)
    palm_center_pos_w = palm_pos_w + quat_apply(palm_rot, palm_offset_local)
    palm_pos = palm_center_pos_w - env_origins

    ft_state = env.robot.data.body_state_w[:, env._fingertip_body_ids, :]  # (N, 5, 13)
    ft_body_pos_w = ft_state[:, :, 0:3]
    ft_body_rot_w = ft_state[:, :, 3:7]  # wxyz

    # Per-fingertip pad offset (legacy env.py:3058-3062). Push each
    # fingertip reference forward by FINGERTIP_OFFSET in finger-local
    # frame so it lands near the actual pad rather than the DP joint.
    ft_offset_local = torch.tensor(
        FINGERTIP_OFFSET, device=env.device, dtype=torch.float32
    ).expand(env.num_envs, NUM_FINGERTIPS, 3)
    ft_pos_w = ft_body_pos_w + quat_apply(
        ft_body_rot_w.reshape(-1, 4), ft_offset_local.reshape(-1, 3)
    ).reshape(env.num_envs, NUM_FINGERTIPS, 3)

    obj_pos = env.object.data.root_pos_w - env_origins
    obj_rot = env.object.data.root_quat_w  # wxyz
    obj_linvel = env.object.data.root_lin_vel_w
    obj_angvel = env.object.data.root_ang_vel_w
    obj_vel = torch.cat([obj_linvel, obj_angvel], dim=-1)

    goal_pos = env.goal_viz.data.root_pos_w - env_origins
    goal_rot = env.goal_viz.data.root_quat_w  # wxyz

    # --- Object-state delay + noise (policy side only) ---
    if dr.use_object_state_delay_noise:
        noisy_obj_pos, noisy_obj_rot, noisy_obj_vel = _apply_object_state_dr(
            env, obj_pos, obj_rot, obj_linvel, obj_angvel
        )
    else:
        noisy_obj_pos, noisy_obj_rot, noisy_obj_vel = obj_pos, obj_rot, obj_vel

    # --- Keypoint world positions (clean, noisy-obj, goal) ---
    # Per-env scale-noise multiplier applied to offsets (legacy env.py:3093-3104).
    kp_offsets = env._keypoint_offsets * env._object_scale_multiplier.unsqueeze(1)
    obj_kp = _keypoints_world(obj_pos, obj_rot, kp_offsets)
    goal_kp = _keypoints_world(goal_pos, goal_rot, kp_offsets)
    noisy_obj_kp = _keypoints_world(noisy_obj_pos, noisy_obj_rot, kp_offsets)

    keypoints_rel_palm_clean = obj_kp - palm_pos.unsqueeze(1)
    keypoints_rel_palm_noisy = noisy_obj_kp - palm_pos.unsqueeze(1)
    keypoints_rel_goal_clean = obj_kp - goal_kp
    keypoints_rel_goal_noisy = noisy_obj_kp - goal_kp

    # Fingertip-rel-palm: world-frame diff (legacy env.py:3085 — no palm-frame rotation).
    fingertip_pos_rel_palm = (
        (ft_pos_w - env_origins.unsqueeze(1)) - palm_pos.unsqueeze(1)
    )  # (N, 5, 3)

    object_scales_obs = env._object_scale_per_env * env._object_scale_multiplier

    # Quaternion convention: Isaac Lab stores (w, x, y, z); the legacy
    # isaacgymenvs obs path emits (x, y, z, w) (Isaac Gym convention) and
    # the pretrained policy was trained against that. Convert at the obs
    # boundary only — internal math (quat_apply / quat_mul / keypoints /
    # _perturb_quat) keeps wxyz throughout.
    palm_rot_xyzw = convert_quat(palm_rot, to="xyzw")
    obj_rot_xyzw = convert_quat(obj_rot, to="xyzw")
    noisy_obj_rot_xyzw = convert_quat(noisy_obj_rot, to="xyzw")

    # --- Build the clean dict (used for critic stack) ---
    obs_clean: dict[str, torch.Tensor] = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "prev_action_targets": prev_targets_canon,
        "palm_pos": palm_pos,
        "palm_rot": palm_rot_xyzw,
        "palm_vel": palm_vel,
        "object_rot": obj_rot_xyzw,
        "object_vel": obj_vel,
        "fingertip_pos_rel_palm": fingertip_pos_rel_palm,
        "keypoints_rel_palm": keypoints_rel_palm_clean,
        "keypoints_rel_goal": keypoints_rel_goal_clean,
        "object_scales": object_scales_obs,
        "closest_keypoint_max_dist": env._closest_keypoint_max_dist.unsqueeze(-1),
        "closest_fingertip_dist": env._closest_fingertip_dist,
        "lifted_object": env._lifted_object.float().unsqueeze(-1),
        "progress": torch.log(env.episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1),
        "successes": torch.log(env._successes.float() + 1.0).unsqueeze(-1),
        "reward": (env.reward_buf * 0.01).unsqueeze(-1),
    }

    # --- Policy-side DR: object fields + joint-vel Gaussian ---
    obs_noisy = dict(obs_clean)
    obs_noisy["object_rot"] = noisy_obj_rot_xyzw
    obs_noisy["object_vel"] = noisy_obj_vel
    obs_noisy["keypoints_rel_palm"] = keypoints_rel_palm_noisy
    obs_noisy["keypoints_rel_goal"] = keypoints_rel_goal_noisy
    if dr.joint_velocity_obs_noise_std > 0:
        obs_noisy["joint_vel"] = (
            joint_vel + torch.randn_like(joint_vel) * dr.joint_velocity_obs_noise_std
        )

    # --- Stack ---
    state_tensor = _stack_obs_dict(obs_clean, env.cfg.obs.state_list)
    policy_tensor = _stack_obs_dict(obs_noisy, env.cfg.obs.obs_list)

    # --- Obs delay on the policy tensor ---
    if dr.use_obs_delay:
        policy_tensor = _apply_obs_delay(env, policy_tensor)

    # --- Clamp ---
    clip = env.cfg.obs.clamp_abs_observations
    policy_tensor = policy_tensor.clamp(-clip, clip)
    state_tensor = state_tensor.clamp(-clip, clip)

    return {"policy": policy_tensor, "critic": state_tensor}


__all__ = [
    "NUM_JOINTS",
    "NUM_FINGERTIPS",
    "NUM_KEYPOINTS",
    "KEYPOINT_CORNERS",
    "OBS_FIELD_SIZES",
    "compute_obs_dim",
    "compute_intermediate_values",
    "build_observations",
]

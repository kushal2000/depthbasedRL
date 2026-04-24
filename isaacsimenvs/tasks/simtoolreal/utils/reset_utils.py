"""Reset + state-allocation helpers for SimToolReal.

Public entry points used by :class:`SimToolRealEnv`:
  - :func:`allocate_state_buffers` — one-shot ``__init__`` buffer allocation
    (body/joint-id caches, joint-limit tensors, action-pipeline buffers,
    DR queues + priors, reward trackers, curriculum state).
  - :func:`reset_env_state` — full per-env reset, called from ``_reset_idx``
    after ``super()._reset_idx``.
  - :func:`reset_goal_trackers` — partial (mid-episode) reset: tracker clear
    + goal-pose resample. Called from :func:`compute_terminations` on
    ``is_success``.

Legacy reference: isaacgymenvs/tasks/simtoolreal/env.py:3611-3796 (reset_idx).
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import random_orientation

from .action_utils import sample_log_uniform
from .goal_sampling import sample_absolute_goal_pose, sample_delta_goal_pose
from .obs_utils import KEYPOINT_CORNERS, NUM_FINGERTIPS
from .scene_utils import (
    ARM_JOINT_REGEX,
    FINGERTIP_BODY_REGEX,
    HAND_JOINT_REGEX,
    JOINT_NAMES_CANONICAL,
    PALM_BODY_NAME,
)


# ----------------------------------------------------------------------------
# __init__ buffer allocation
# ----------------------------------------------------------------------------


def allocate_state_buffers(env) -> None:
    """Populate every per-env buffer + index cache used by the hooks.

    Called once from ``__init__`` after ``super().__init__`` (which runs
    ``_setup_scene`` and makes ``env.robot``/``env.object``/``env.goal_viz``
    available).
    """
    dr = env.cfg.domain_randomization
    rew = env.cfg.reward

    # --- Joint/body id caches (Phase C + D) ---
    env._arm_joint_ids = env.robot.find_joints(ARM_JOINT_REGEX)[0]      # 7
    env._hand_joint_ids = env.robot.find_joints(HAND_JOINT_REGEX)[0]     # 22
    env._palm_body_id = env.robot.find_bodies(PALM_BODY_NAME)[0][0]
    env._fingertip_body_ids = env.robot.find_bodies(FINGERTIP_BODY_REGEX)[0]  # 5
    assert len(env._fingertip_body_ids) == NUM_FINGERTIPS

    # --- Joint-order permutation tensors (Lab parser <-> legacy isaacgym) ---
    # The pretrained policy + external code expect joints in
    # JOINT_NAMES_CANONICAL order, but Isaac Lab parses them differently
    # (BFS across fingers vs canonical DFS within finger). The env stores
    # all joint state in Lab order; these perms convert at the policy I/O
    # boundary only (action input, joint obs output).
    #
    #   actions_lab = actions_canon[:, _perm_canon_to_lab]
    #     -> for each Lab position j, source the action from canonical
    #        index `canonical_names.index(lab_names[j])`.
    #
    #   tensor_canon = tensor_lab[:, _perm_lab_to_canon]
    #     -> for each canonical position k, source from Lab index
    #        `lab_names.index(canonical_names[k])`.
    lab_names = list(env.robot.data.joint_names)
    assert len(lab_names) == len(JOINT_NAMES_CANONICAL), (
        f"robot has {len(lab_names)} joints, canonical list has "
        f"{len(JOINT_NAMES_CANONICAL)}"
    )
    missing = set(JOINT_NAMES_CANONICAL) - set(lab_names)
    assert not missing, f"canonical joints absent from robot: {missing}"
    env._perm_canon_to_lab = torch.tensor(
        [JOINT_NAMES_CANONICAL.index(n) for n in lab_names],
        device=env.device, dtype=torch.long,
    )
    env._perm_lab_to_canon = torch.tensor(
        [lab_names.index(n) for n in JOINT_NAMES_CANONICAL],
        device=env.device, dtype=torch.long,
    )

    # --- Canonical-order joint limits (for joint_pos obs unscale) ---
    # Legacy isaacgymenvs obs builder normalizes joint_pos to [-1, 1] via
    # unscale(q, lower, upper) — we must match that for the pretrained
    # policy. Cache once in canonical order; per-env limits are identical
    # across envs (same robot URDF), so we take env_0's row.
    limits = env.robot.data.joint_pos_limits  # (N, num_joints, 2), Lab order
    env._joint_lower_canon = limits[0, :, 0][env._perm_lab_to_canon]  # (29,)
    env._joint_upper_canon = limits[0, :, 1][env._perm_lab_to_canon]  # (29,)

    # --- Joint limits (raw URDF values, not soft). ---
    limits = env.robot.data.joint_pos_limits  # (N, num_joints, 2)
    env._arm_lower = limits[:, env._arm_joint_ids, 0]
    env._arm_upper = limits[:, env._arm_joint_ids, 1]
    env._hand_lower = limits[:, env._hand_joint_ids, 0]
    env._hand_upper = limits[:, env._hand_joint_ids, 1]

    # --- Action-pipeline buffers (Phase C) ---
    action_space = env.cfg.action_space
    env._cur_targets = torch.zeros(env.num_envs, action_space, device=env.device)
    env._prev_targets = torch.zeros(env.num_envs, action_space, device=env.device)
    env._action_queue = torch.zeros(
        env.num_envs,
        max(1, dr.action_delay_max),
        action_space,
        device=env.device,
    )

    # --- Keypoint offsets in object local frame ---
    corners = torch.tensor(
        KEYPOINT_CORNERS, device=env.device, dtype=torch.float32
    )  # (4, 3)
    # Per-env scaled (always used for obs + for reward when fixed flag is False).
    per_env_half = (
        env._object_scale_per_env * rew.object_base_size * rew.keypoint_scale * 0.5
    )
    env._keypoint_offsets = corners.unsqueeze(0) * per_env_half.unsqueeze(1)  # (N, 4, 3)
    # Fixed-size offsets (reward when fixed_size_keypoint_reward=True).
    fixed_half = 0.5 * torch.tensor(rew.fixed_size, device=env.device)
    offsets_fixed = corners * fixed_half.unsqueeze(0)
    env._keypoint_offsets_fixed = offsets_fixed.unsqueeze(0).expand(
        env.num_envs, -1, -1
    ).contiguous()

    # --- Per-env DR priors (re-sampled on reset; seeded once here) ---
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier = torch.empty(
        env.num_envs, 3, device=env.device
    ).uniform_(lo, hi)

    # --- Reward / termination trackers (Phase E/F) ---
    env._lifted_object = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._closest_keypoint_max_dist = torch.full(
        (env.num_envs,), 10.0, device=env.device
    )
    env._closest_fingertip_dist = torch.full(
        (env.num_envs, NUM_FINGERTIPS), 10.0, device=env.device
    )
    env._successes = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._near_goal_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )

    # --- Tolerance curriculum state (Phase H) ---
    env._current_success_tolerance: float = env.cfg.termination.success_tolerance
    env._prev_episode_successes = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._frame_counter: int = 0
    env._last_curriculum_update: int = 0

    # --- Object lifted-reward reference z (updated on each _reset_object_pose) ---
    init_z = env.cfg.reset.table_reset_z + env.cfg.reset.table_object_z_offset
    env._object_init_z = torch.full(
        (env.num_envs,), init_z, device=env.device
    )

    # --- DR rolling buffers (Phase D) ---
    env._object_state_queue = torch.zeros(
        env.num_envs,
        max(1, dr.object_state_delay_max),
        13,  # pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        device=env.device,
    )
    env._obs_queue = torch.zeros(
        env.num_envs,
        max(1, dr.obs_delay_max),
        env.cfg.observation_space,
        device=env.device,
    )

    # --- Wrench DR state (Phase C/D) ---
    env._random_force_prob = sample_log_uniform(
        dr.force_prob_range, env.num_envs
    ).to(env.device)
    env._random_torque_prob = sample_log_uniform(
        dr.torque_prob_range, env.num_envs
    ).to(env.device)
    env._object_forces = torch.zeros(env.num_envs, 1, 3, device=env.device)
    env._object_torques = torch.zeros(env.num_envs, 1, 3, device=env.device)
    env._object_mass = env.object.data.default_mass[:, 0:1].to(env.device)  # (N, 1)

    # --- Step-shared caches populated by compute_intermediate_values (Phase F) ---
    env._keypoints_max_dist = torch.zeros(env.num_envs, device=env.device)
    env._curr_fingertip_distances = torch.zeros(
        env.num_envs, NUM_FINGERTIPS, device=env.device
    )
    env._near_goal = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._is_success = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )

    # DirectRLEnv only sets `reward_buf` after first `_get_rewards`; defensive
    # init so first `_get_observations` (called during reset) can read it.
    env.reward_buf = torch.zeros(env.num_envs, device=env.device)


# ----------------------------------------------------------------------------
# Per-phase reset helpers
# ----------------------------------------------------------------------------


def _randomize_robot_dof_state(env, env_ids: torch.Tensor) -> None:
    """Multiplicative-on-range DOF noise around default + symmetric velocity
    noise (legacy env.py:3708-3738). Also re-seeds ``_prev_targets`` so the
    arm velocity-delta accumulator starts from the reset DOF pose."""
    cfg = env.cfg.reset
    default_pos = env.robot.data.default_joint_pos[env_ids]  # (n, num_dofs)
    lower = env.robot.data.joint_pos_limits[env_ids, :, 0]
    upper = env.robot.data.joint_pos_limits[env_ids, :, 1]

    delta_min = lower - default_pos
    delta_max = upper - default_pos
    rand = torch.rand_like(default_pos)

    noise_coeff = torch.zeros_like(default_pos)
    noise_coeff[:, env._arm_joint_ids] = cfg.reset_dof_pos_random_interval_arm
    noise_coeff[:, env._hand_joint_ids] = cfg.reset_dof_pos_random_interval_fingers

    joint_pos = default_pos + noise_coeff * (
        delta_min + (delta_max - delta_min) * rand
    )
    joint_pos = torch.clamp(joint_pos, lower, upper)

    joint_vel = (torch.rand_like(default_pos) * 2.0 - 1.0) * cfg.reset_dof_vel_random_interval

    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    env._prev_targets[env_ids] = joint_pos
    env._cur_targets[env_ids] = joint_pos


def _reset_object_pose(env, env_ids: torch.Tensor) -> None:
    """Reset object to table-relative position + configured noise + optional
    random quat. Updates ``env._object_init_z`` for the lifting reward
    (legacy env.py:3488-3568)."""
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    rand = torch.rand(n, 3, device=env.device) * 2.0 - 1.0  # symmetric uniform
    z_table_jitter = (torch.rand(n, device=env.device) * 2.0 - 1.0) * cfg.table_reset_z_range

    pos_local = torch.zeros(n, 3, device=env.device)
    pos_local[:, 0] = rand[:, 0] * cfg.reset_position_noise_x
    pos_local[:, 1] = rand[:, 1] * cfg.reset_position_noise_y
    pos_local[:, 2] = (
        cfg.table_reset_z
        + cfg.table_object_z_offset
        + z_table_jitter
        + rand[:, 2] * cfg.reset_position_noise_z
    )
    pos_world = pos_local + env_origins

    if cfg.randomize_object_rotation:
        quat = random_orientation(n, device=env.device)
    else:
        quat = torch.zeros(n, 4, device=env.device)
        quat[:, 0] = 1.0  # identity wxyz

    pose = torch.cat([pos_world, quat], dim=-1)  # (n, 7)
    env.object.write_root_pose_to_sim(pose, env_ids=env_ids)
    env.object.write_root_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=env_ids
    )

    env._object_init_z[env_ids] = pos_local[:, 2]


def _reset_goal_pose(env, env_ids: torch.Tensor, mode: str) -> None:
    """Resample the goal pose and push it to ``env.goal_viz``.

    ``mode="absolute"`` samples fresh in the scaled workspace box;
    ``mode="delta"`` perturbs the current goal (chain continuation).
    """
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    if mode == "delta":
        prev_pos_local = env.goal_viz.data.root_pos_w[env_ids] - env_origins
        prev_quat = env.goal_viz.data.root_quat_w[env_ids]
        new_pos_local, new_quat = sample_delta_goal_pose(
            prev_pos=prev_pos_local,
            prev_quat_wxyz=prev_quat,
            delta_distance=cfg.delta_goal_distance,
            delta_rotation_degrees=cfg.delta_rotation_degrees,
            mins=cfg.target_volume_mins,
            maxs=cfg.target_volume_maxs,
            scale=cfg.target_volume_region_scale,
        )
    else:  # "absolute" or anything unrecognized
        new_pos_local, new_quat = sample_absolute_goal_pose(
            mins=cfg.target_volume_mins,
            maxs=cfg.target_volume_maxs,
            scale=cfg.target_volume_region_scale,
            n_envs=n,
            device=env.device,
            randomize_rotation=True,
        )

    pose = torch.cat([new_pos_local + env_origins, new_quat], dim=-1)
    env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)


# ----------------------------------------------------------------------------
# Reset orchestrators (public)
# ----------------------------------------------------------------------------


def reset_goal_trackers(env, env_ids: torch.Tensor) -> None:
    """Mid-episode reset: clear per-target trackers + resample goal pose.

    Called on goal-hit from ``compute_terminations``. Uses the configured
    ``cfg.reset.goal_sampling_type`` so delta-mode chains from the previous
    goal while absolute-mode samples fresh.
    """
    env._closest_keypoint_max_dist[env_ids] = 10.0
    env._closest_fingertip_dist[env_ids] = 10.0
    env._near_goal_steps[env_ids] = 0
    _reset_goal_pose(env, env_ids, mode=env.cfg.reset.goal_sampling_type)


def reset_env_state(env, env_ids: torch.Tensor) -> None:
    """Full per-env reset (after ``super()._reset_idx``).

    Order:
      1. DOF noise + object pose + goal pose (absolute).
      2. Capture ``_prev_episode_successes`` BEFORE zeroing ``_successes``
         (needed by the tolerance curriculum).
      3. Clear per-env trackers.
      4. Zero DR queues + wrench state.
      5. Re-sample per-env DR priors.
    """
    n = env_ids.numel()

    _randomize_robot_dof_state(env, env_ids)
    _reset_object_pose(env, env_ids)
    _reset_goal_pose(env, env_ids, mode="absolute")  # full reset → always absolute

    # Capture ending episode's success count for curriculum (legacy env.py:3630).
    env._prev_episode_successes[env_ids] = env._successes[env_ids]

    # Per-env trackers.
    env._lifted_object[env_ids] = False
    env._closest_keypoint_max_dist[env_ids] = 10.0
    env._closest_fingertip_dist[env_ids] = 10.0
    env._successes[env_ids] = 0
    env._near_goal_steps[env_ids] = 0

    # DR rolling buffers + wrench state.
    env._action_queue[env_ids] = 0.0
    env._obs_queue[env_ids] = 0.0
    env._object_state_queue[env_ids] = 0.0
    env._object_forces[env_ids] = 0.0
    env._object_torques[env_ids] = 0.0

    # Re-sample per-env DR priors.
    dr = env.cfg.domain_randomization
    env._random_force_prob[env_ids] = sample_log_uniform(
        dr.force_prob_range, n
    ).to(env.device)
    env._random_torque_prob[env_ids] = sample_log_uniform(
        dr.torque_prob_range, n
    ).to(env.device)
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier[env_ids] = torch.empty(
        n, 3, device=env.device
    ).uniform_(lo, hi)


__all__ = [
    "allocate_state_buffers",
    "reset_env_state",
    "reset_goal_trackers",
]

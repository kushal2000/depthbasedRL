"""Pure-torch reward-term helpers for SimToolReal.

Each function takes tensors/scalars only — no env reference — and returns
updated tensors or tuples. Legacy sources cited per-term (all line numbers are
``isaacgymenvs/tasks/simtoolreal/env.py``).

The reward loop lives in :meth:`SimToolRealEnv._get_rewards`; this module is
pure math so it can be unit-tested without spinning up Isaac Sim.
"""

from __future__ import annotations

import torch


def lifting_reward(
    object_z: torch.Tensor,            # (N,)
    object_init_z: torch.Tensor,       # (N,)
    prev_lifted: torch.Tensor,         # (N,) bool
    lifting_bonus_threshold: float,
    lifting_bonus: float,
    lifting_rew_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port of env.py:2425-2449.

    Returns ``(lifting_rew, lift_bonus_rew, new_lifted)``:
      - ``lifting_rew`` — ``clip(0.05 + dz, 0, 0.5)``, zeroed once the object is
        lifted, then multiplied by ``lifting_rew_scale``.
      - ``lift_bonus_rew`` — one-shot ``lifting_bonus`` on the step the object
        first crosses ``lifting_bonus_threshold``.
      - ``new_lifted`` — latching OR: once True, stays True until an external
        reset.
    """
    z_lift = 0.05 + object_z - object_init_z
    lift_rew = torch.clamp(z_lift, 0.0, 0.5)
    lifted = (z_lift > lifting_bonus_threshold) | prev_lifted
    just_crossed = lifted & ~prev_lifted
    lift_bonus = lifting_bonus * just_crossed.float()
    lift_rew = lift_rew * (~lifted).float()
    return lift_rew * lifting_rew_scale, lift_bonus, lifted


def distance_delta_reward(
    curr_fingertip_dist: torch.Tensor,    # (N, num_fingertips)
    closest_fingertip_dist: torch.Tensor,  # (N, num_fingertips)
    lifted: torch.Tensor,                  # (N,) bool
    rew_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Port of env.py:2388-2415. Returns ``(reward, new_closest_fingertip_dist)``.

    Reward = ``sum_i clamp(closest_i - curr_i, 0, 10) * (~lifted) * rew_scale``
    — rewards only steps when a fingertip gets closer than its best-so-far;
    disabled after lifting.
    """
    deltas = closest_fingertip_dist - curr_fingertip_dist
    new_closest = torch.minimum(closest_fingertip_dist, curr_fingertip_dist)
    deltas = torch.clamp(deltas, 0.0, 10.0)
    rew = deltas.sum(dim=-1) * (~lifted).float()
    return rew * rew_scale, new_closest


def keypoint_reward(
    keypoints_max_dist: torch.Tensor,         # (N,)
    closest_keypoint_max_dist: torch.Tensor,   # (N,)
    lifted: torch.Tensor,                      # (N,) bool
    rew_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Port of env.py:2451-2479. Returns ``(reward, new_closest_keypoint_max_dist)``.

    Reward = ``clamp(closest - curr, 0, 100) * lifted * rew_scale`` — only
    rewards when the current max-keypoint distance is smaller than any
    previously achieved, and only while the object is lifted.
    """
    delta = closest_keypoint_max_dist - keypoints_max_dist
    new_closest = torch.minimum(closest_keypoint_max_dist, keypoints_max_dist)
    delta = torch.clamp(delta, 0.0, 100.0)
    rew = delta * lifted.float()
    return rew * rew_scale, new_closest


def action_penalty(
    joint_vel: torch.Tensor,        # (N, num_dofs)
    arm_ids,
    hand_ids,
    kuka_scale: float,
    hand_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Port of env.py:2481-2494.

    Note: legacy variable name says ``action_penalty`` but the formula is the
    **L1 norm of joint velocities** (not raw actions), scaled and negated.
    """
    kuka = -kuka_scale * joint_vel[:, arm_ids].abs().sum(dim=-1)
    hand = -hand_scale * joint_vel[:, hand_ids].abs().sum(dim=-1)
    return kuka, hand


def update_near_goal_steps(
    near_goal: torch.Tensor,        # (N,) bool
    near_goal_steps: torch.Tensor,   # (N,) long
    force_consecutive: bool,
) -> torch.Tensor:
    """Port of env.py:2694-2699. Returns the updated counter.

    When ``force_consecutive`` is True, the counter is zeroed on any step
    where ``near_goal`` is False (require consecutive stepping).
    """
    ng = near_goal.long()
    if force_consecutive:
        return (near_goal_steps + ng) * ng
    return near_goal_steps + ng


def reach_goal_bonus(
    near_goal: torch.Tensor,        # (N,) bool
    is_success: torch.Tensor,        # (N,) bool
    reach_goal_bonus_value: float,
    success_steps: int,
    force_consecutive: bool,
) -> torch.Tensor:
    """Port of env.py:2733-2737.

    - ``force_consecutive``: lump-sum ``reach_goal_bonus`` once per success.
    - else: amortized across the window → ``near_goal * bonus / success_steps``
      each step.
    """
    if force_consecutive:
        return is_success.float() * reach_goal_bonus_value
    return near_goal.float() * (reach_goal_bonus_value / success_steps)


def compute_rewards(env) -> torch.Tensor:
    """Top-level reward builder — sums the four reward terms + action penalties
    using cached intermediate values from ``compute_intermediate_values``.

    Mutates env state: ``_lifted_object`` (latching OR),
    ``_closest_fingertip_dist`` and ``_closest_keypoint_max_dist`` (min-so-far
    trackers).
    """
    rew_cfg = env.cfg.reward
    term_cfg = env.cfg.termination
    env_origins = env.scene.env_origins
    obj_pos = env.object.data.root_pos_w - env_origins

    lift_rew, lift_bonus_rew, new_lifted = lifting_reward(
        object_z=obj_pos[:, 2],
        object_init_z=env._object_init_z,
        prev_lifted=env._lifted_object,
        lifting_bonus_threshold=rew_cfg.lifting_bonus_threshold,
        lifting_bonus=rew_cfg.lifting_bonus,
        lifting_rew_scale=rew_cfg.lifting_rew_scale,
    )
    env._lifted_object = new_lifted

    ft_rew, new_closest_ft = distance_delta_reward(
        curr_fingertip_dist=env._curr_fingertip_distances,
        closest_fingertip_dist=env._closest_fingertip_dist,
        lifted=env._lifted_object,
        rew_scale=rew_cfg.distance_delta_rew_scale,
    )
    env._closest_fingertip_dist = new_closest_ft

    kp_rew, new_closest_kp = keypoint_reward(
        keypoints_max_dist=env._keypoints_max_dist,
        closest_keypoint_max_dist=env._closest_keypoint_max_dist,
        lifted=env._lifted_object,
        rew_scale=rew_cfg.keypoint_rew_scale,
    )
    env._closest_keypoint_max_dist = new_closest_kp

    kuka_pen, hand_pen = action_penalty(
        joint_vel=env.robot.data.joint_vel,
        arm_ids=env._arm_joint_ids,
        hand_ids=env._hand_joint_ids,
        kuka_scale=rew_cfg.kuka_actions_penalty_scale,
        hand_scale=rew_cfg.hand_actions_penalty_scale,
    )

    bonus = reach_goal_bonus(
        near_goal=env._near_goal,
        is_success=env._is_success,
        reach_goal_bonus_value=rew_cfg.reach_goal_bonus,
        success_steps=term_cfg.success_steps,
        force_consecutive=term_cfg.force_consecutive_near_goal_steps,
    )

    return lift_rew + lift_bonus_rew + ft_rew + kp_rew + kuka_pen + hand_pen + bonus


__all__ = [
    "lifting_reward",
    "distance_delta_reward",
    "keypoint_reward",
    "action_penalty",
    "update_near_goal_steps",
    "reach_goal_bonus",
    "compute_rewards",
]

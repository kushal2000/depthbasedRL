"""Termination + tolerance-curriculum helpers for SimToolReal.

Both entry points are called from :meth:`SimToolRealEnv._get_dones`:

  1. :func:`update_tolerance_curriculum` — advances ``_current_success_tolerance``
     toward ``target_success_tolerance`` once per curriculum-interval when the
     policy averages ≥3 successes per episode (legacy
     ``isaacgymenvs/tasks/simtoolreal/utils.py:tolerance_curriculum``).
  2. :func:`compute_terminations` — reads the cached intermediate values,
     updates ``_successes``, handles mid-episode goal-hit (tracker reset +
     ``episode_length_buf`` zero), and returns
     ``(terminated, truncated)`` (legacy env.py:2496-2560, minus the
     peg-in-hole / resetWhenDropped causes).
"""

from __future__ import annotations

import torch

from .reset_utils import reset_goal_trackers


def update_tolerance_curriculum(env) -> None:
    """Shrink ``env._current_success_tolerance`` toward the floor when the
    policy consistently hits goals. Mutates env state in-place.

    Gating:
      - Frame cadence: at least ``tolerance_curriculum_interval`` policy
        steps since the last update (global counter, not per-env).
      - Performance: ``mean(env._prev_episode_successes) >= 3.0``.
    """
    env._frame_counter += 1
    term = env.cfg.termination
    if env._frame_counter - env._last_curriculum_update < term.tolerance_curriculum_interval:
        return
    if env._prev_episode_successes.float().mean().item() < 3.0:
        return
    new_tol = env._current_success_tolerance * term.tolerance_curriculum_increment
    new_tol = max(min(new_tol, term.success_tolerance), term.target_success_tolerance)
    env._current_success_tolerance = new_tol
    env._last_curriculum_update = env._frame_counter


def compute_terminations(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(terminated, truncated)`` bool tensors of shape ``(N,)``.

    Reads cached intermediates (``_is_success``, ``_curr_fingertip_distances``)
    populated by :func:`compute_intermediate_values`. Mutates env on goal-hit:
    increments ``_successes``, resets per-target trackers, zeroes
    ``episode_length_buf`` for the hit envs (legacy env.py:2503-2505).
    """
    term_cfg = env.cfg.termination
    env_origins = env.scene.env_origins
    is_success = env._is_success

    # Authoritative updates on goal-hit.
    env._successes = env._successes + is_success.long()
    goal_reset_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
    if goal_reset_ids.numel() > 0:
        reset_goal_trackers(env, goal_reset_ids)
        # Extend episode: zero the length buf so truncation doesn't fire
        # on a step that just reached the goal.
        env.episode_length_buf[goal_reset_ids] = 0

    # Termination causes (hardcoded thresholds; legacy env.py:2500, 2529).
    object_z_local = env.object.data.root_pos_w[:, 2] - env_origins[:, 2]
    fall = object_z_local < 0.1

    if term_cfg.max_consecutive_successes > 0:
        max_successes_reached = env._successes >= term_cfg.max_consecutive_successes
    else:
        max_successes_reached = torch.zeros_like(fall)

    hand_far = env._curr_fingertip_distances.max(dim=-1).values > 1.5

    terminated = fall | max_successes_reached | hand_far
    truncated = env.episode_length_buf >= env.max_episode_length
    return terminated, truncated


__all__ = ["update_tolerance_curriculum", "compute_terminations"]

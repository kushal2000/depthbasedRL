"""Action-pipeline helpers for SimToolReal (Phase C + force/torque DR).

Two entry points, both called from :meth:`SimToolRealEnv._pre_physics_step`:
  - :func:`apply_action_pipeline` — action delay + arm velocity-delta + hand
    absolute-scale + EMA smoothing + clamp + ``_prev_targets`` update.
  - :func:`apply_wrench_dr` — decayed force/torque impulses on the object
    body, gated by ``_lifted_object``.

Legacy references (isaacgymenvs/tasks/simtoolreal/env.py):
  - 3841-3854: action delay queue + per-step lag sampling.
  - 3886-3895: arm velocity-delta (useRelativeControl=False branch).
  - 3904-3918: hand absolute-scale + EMA.
  - 4001-4060: force/torque impulses (decay, sample, lift-gate).
"""

from __future__ import annotations

import math

import torch


def sample_log_uniform(lo_hi: tuple[float, float], n: int) -> torch.Tensor:
    """Log-uniform sample on CPU (caller moves to device)."""
    lo, hi = lo_hi
    return torch.exp(
        torch.empty(n).uniform_(math.log(lo + 1e-12), math.log(hi + 1e-12))
    )


def apply_action_pipeline(env, actions: torch.Tensor) -> None:
    """Run the action pipeline in-place on the env state.

    Reads/writes ``env._action_queue``, ``env._prev_targets``, ``env._cur_targets``.
    Uses the arm vs hand joint id masks cached in Phase C.
    """
    # Replay override — eval_simtoolreal.py --replay_targets_npz path. Caller
    # writes a Lab-order target to env._replay_target_lab_order; we skip the
    # action pipeline and push that straight into _cur_targets so
    # _apply_action sends it verbatim to PhysX. Used to isolate PhysX 4 vs
    # PhysX 5 physics response from action-pipeline math.
    replay_target = getattr(env, "_replay_target_lab_order", None)
    if replay_target is not None:
        env._cur_targets[:] = replay_target
        env._prev_targets = env._cur_targets.clone()
        return

    dr = env.cfg.domain_randomization
    act_cfg = env.cfg.action
    dt = env.step_dt  # policy step (= decimation * sim.dt = 1/60 s)

    actions = actions.clone().to(env.device)

    # 0. Convert action from canonical (legacy isaacgymenvs/pretrained-policy)
    #    order to Lab parser order. The arm/hand split below expects Lab-order
    #    storage: positions 0-6 are arm (matches both orderings), positions
    #    7-28 are the 22 hand joints in Lab BFS order. _perm_canon_to_lab
    #    realigns canonical-DFS hand actions to Lab-BFS positions.
    actions = actions[:, env._perm_canon_to_lab]

    # 1. Action delay — roll queue, insert newest, sample per-env lag each step.
    if dr.use_action_delay and dr.action_delay_max > 0:
        env._action_queue = torch.roll(env._action_queue, shifts=1, dims=1)
        env._action_queue[:, 0, :] = actions
        delay_idx = torch.randint(
            0, env._action_queue.shape[1], (env.num_envs,), device=env.device
        )
        actions = env._action_queue[
            torch.arange(env.num_envs, device=env.device), delay_idx
        ]

    # 2. Arm (action[:7]) — velocity-delta accumulator (env.py:3886-3895).
    arm_action = actions[:, :7]
    arm_raw = env._prev_targets[:, :7] + act_cfg.dof_speed_scale * dt * arm_action
    arm_raw = torch.clamp(arm_raw, env._arm_lower, env._arm_upper)
    arm_smoothed = (
        act_cfg.arm_moving_average * arm_raw
        + (1.0 - act_cfg.arm_moving_average) * env._prev_targets[:, :7]
    )
    arm_smoothed = torch.clamp(arm_smoothed, env._arm_lower, env._arm_upper)

    # 3. Hand (action[7:]) — absolute scale [-1,1] → [lower, upper] (env.py:3904-3918).
    hand_action = actions[:, 7:]
    hand_raw = env._hand_lower + 0.5 * (hand_action + 1.0) * (
        env._hand_upper - env._hand_lower
    )
    hand_smoothed = (
        act_cfg.hand_moving_average * hand_raw
        + (1.0 - act_cfg.hand_moving_average) * env._prev_targets[:, 7:]
    )
    hand_smoothed = torch.clamp(hand_smoothed, env._hand_lower, env._hand_upper)

    # 4. Combine into _cur_targets at joint-id positions.
    env._cur_targets[:, env._arm_joint_ids] = arm_smoothed
    env._cur_targets[:, env._hand_joint_ids] = hand_smoothed

    # 5. Update prev_targets for next-step EMA + Phase D's obs builder.
    env._prev_targets = env._cur_targets.clone()


def apply_wrench_dr(env) -> None:
    """Sample + apply random force/torque impulses on the object body.

    Uses ``env._lifted_object`` from the PREVIOUS step's state (legacy has this
    one-step delay too since ``_lifted_object`` updates in the reward, which
    runs after this hook).
    """
    dr = env.cfg.domain_randomization
    dt_pol = env.step_dt

    # Decay previous wrench (env.py:4002-4004).
    if dr.force_decay > 0.0:
        env._object_forces *= dr.force_decay ** (dt_pol / dr.force_decay_interval)
    else:
        env._object_forces.zero_()
    if dr.torque_decay > 0.0:
        env._object_torques *= dr.torque_decay ** (dt_pol / dr.torque_decay_interval)
    else:
        env._object_torques.zero_()

    # Sample new wrench with per-env probability.
    force_fire = (
        torch.rand(env.num_envs, device=env.device) < env._random_force_prob
    ).view(-1, 1, 1)
    torque_fire = (
        torch.rand(env.num_envs, device=env.device) < env._random_torque_prob
    ).view(-1, 1, 1)
    mass = env._object_mass.unsqueeze(-1)  # (N, 1, 1)
    new_force = (
        torch.randn(env.num_envs, 1, 3, device=env.device) * mass * dr.force_scale
    )
    new_torque = (
        torch.randn(env.num_envs, 1, 3, device=env.device) * mass * dr.torque_scale
    )
    env._object_forces = torch.where(force_fire, new_force, env._object_forces)
    env._object_torques = torch.where(torque_fire, new_torque, env._object_torques)

    if dr.force_only_when_lifted:
        env._object_forces *= env._lifted_object.float().view(-1, 1, 1)
    if dr.torque_only_when_lifted:
        env._object_torques *= env._lifted_object.float().view(-1, 1, 1)

    env.object.set_external_force_and_torque(
        forces=env._object_forces,
        torques=env._object_torques,
        is_global=True,
    )


__all__ = ["apply_action_pipeline", "apply_wrench_dr", "sample_log_uniform"]

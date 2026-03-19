"""FabricaEnv: SimToolReal subclass with retract-after-insertion reward.

After the robot completes all insertion goals, it enters a retract phase
where it is rewarded for moving its hand away from the object while the
object remains at the final goal pose.

Controlled by config:
  env.enableRetract (bool): Master switch. When False, behaves identically
      to SimToolReal (episode resets after final goal).
  env.retractRewardScale (float): Scale factor for retract distance reward.
  env.retractDistanceThreshold (float): Min hand-object distance to count
      as successfully retracted.
  env.retractSuccessBonus (float): One-time bonus when hand clears threshold.
"""

import torch
from torch import Tensor
from typing import Tuple

from isaacgymenvs.tasks.simtoolreal.env import SimToolReal


class FabricaEnv(SimToolReal):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        self.enable_retract = cfg["env"].get("enableRetract", False)
        self.retract_reward_scale = cfg["env"].get("retractRewardScale", 1.0)
        self.retract_distance_threshold = cfg["env"].get("retractDistanceThreshold", 0.1)
        self.retract_success_bonus = cfg["env"].get("retractSuccessBonus", 1000.0)

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        # Per-env retract state
        self.retract_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.retract_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Register retract_rew in episode reward tracking
        self.rewards_episode["retract_rew"] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def _compute_resets(self, is_success):
        """Override to enter retract phase instead of resetting on max successes."""
        if not self.enable_retract:
            return super()._compute_resets(is_success)

        ones = torch.ones_like(self.reset_buf)
        zeros = torch.zeros_like(self.reset_buf)

        object_z_low = torch.where(self.object_pos[:, 2] < 0.1, ones, zeros)

        if self.max_consecutive_successes > 0:
            self.progress_buf = torch.where(
                is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf
            )
            # Reset on retract success, not max_consecutive_successes
            max_consecutive_successes_reached = torch.where(
                self.retract_succeeded, ones, zeros
            )
        else:
            max_consecutive_successes_reached = zeros

        max_episode_length_reached = torch.where(
            self.progress_buf >= self.max_episode_length - 1, ones, zeros
        )

        if self.with_table_force_sensor:
            TABLE_FORCE_THRESHOLD = 100.0
            table_force_too_high = torch.where(
                self.max_table_sensor_force_norm_smoothed > TABLE_FORCE_THRESHOLD,
                ones, zeros,
            )
        else:
            table_force_too_high = zeros

        # During retract phase, don't reset for hand far from object
        hand_far_from_object = torch.where(
            (self.curr_fingertip_distances.max(dim=-1).values > 1.5) & ~self.retract_phase,
            ones, zeros,
        )

        if self.cfg["env"]["resetWhenDropped"]:
            dropped_z = self.object_init_state[:, 2]
            dropped = (
                torch.where(self.object_pos[:, 2] < dropped_z, ones, zeros)
                * self.lifted_object
            )
        else:
            dropped = zeros

        resets = (
            self.reset_buf
            | object_z_low
            | max_consecutive_successes_reached
            | max_episode_length_reached
            | table_force_too_high
            | hand_far_from_object
            | dropped
        )
        resets = self._extra_reset_rules(resets)

        # Clear retract state for envs that are resetting
        self.retract_phase[resets.bool()] = False
        self.retract_succeeded[resets.bool()] = False

        return resets

    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        """Override to add retract phase after all insertion goals are reached.

        For envs NOT in retract phase: identical to SimToolReal.
        For envs IN retract phase: action penalties + retract reward only.

        Retract phase is detected at the END of reward computation so the
        transition frame (hitting final goal) gets the full base reward
        including the goal bonus. Retract reward starts next frame.
        """
        if not self.enable_retract:
            return super().compute_kuka_reward()

        # ── Base reward components (identical to SimToolReal) ──
        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew, hand_delta_penalty = self._distance_delta_rewards(lifted_object)
        keypoint_rew, keypoint_rew_fixed_size = self._keypoint_reward(lifted_object)
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            keypoint_rew = keypoint_rew_fixed_size

        if self.cfg["env"].get("finalGoalToleranceCurriculumEnabled", False):
            final_tol = self.final_goal_success_tolerance
        else:
            final_tol = self.cfg["env"].get("finalGoalSuccessTolerance", None)
        if final_tol is not None and self.cfg["env"]["useFixedGoalStates"]:
            is_final_goal = self.successes == (self.max_consecutive_successes - 1)
            base_tol = self.success_tolerance * self.keypoint_scale
            tight_tol = final_tol * self.keypoint_scale
            keypoint_success_tolerance = torch.where(is_final_goal, tight_tol, base_tol)
        else:
            keypoint_success_tolerance = self.success_tolerance * self.keypoint_scale

        near_goal: Tensor = self.keypoints_max_dist <= keypoint_success_tolerance
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            near_goal = self.keypoints_max_dist_fixed_size <= keypoint_success_tolerance

        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            self.near_goal_steps = (self.near_goal_steps + near_goal) * near_goal
        else:
            self.near_goal_steps += near_goal

        is_success = self.near_goal_steps >= self.success_steps
        goal_resets = is_success.clone()
        self.successes += is_success

        # Suppress goal cycling for retract envs (already in retract from prev frame)
        goal_resets[self.retract_phase] = False
        self.reset_goal_buf[:] = goal_resets

        object_lin_vel_penalty = -torch.sum(torch.square(self.object_linvel), dim=-1)
        object_ang_vel_penalty = -torch.sum(torch.square(self.object_angvel), dim=-1)

        self.rewards_episode["raw_fingertip_delta_rew"] += fingertip_delta_rew
        self.rewards_episode["raw_hand_delta_penalty"] += hand_delta_penalty
        self.rewards_episode["raw_lifting_rew"] += lifting_rew
        self.rewards_episode["raw_keypoint_rew"] += keypoint_rew
        self.rewards_episode["raw_object_lin_vel_penalty"] += object_lin_vel_penalty
        self.rewards_episode["raw_object_ang_vel_penalty"] += object_ang_vel_penalty

        fingertip_delta_rew *= self.distance_delta_rew_scale
        hand_delta_penalty *= self.distance_delta_rew_scale * 0  # currently disabled
        lifting_rew *= self.lifting_rew_scale
        keypoint_rew *= self.keypoint_rew_scale
        object_lin_vel_penalty *= self.object_lin_vel_penalty_scale
        object_ang_vel_penalty *= self.object_ang_vel_penalty_scale

        kuka_actions_penalty, hand_actions_penalty = self._action_penalties()

        bonus_rew = near_goal * (self.reach_goal_bonus / self.success_steps)
        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            bonus_rew = is_success * self.reach_goal_bonus
        # No goal bonus during retract phase
        bonus_rew[self.retract_phase] = 0.0

        base_reward = (
            fingertip_delta_rew
            + hand_delta_penalty
            + lifting_rew
            + lift_bonus_rew
            + keypoint_rew
            + kuka_actions_penalty
            + hand_actions_penalty
            + bonus_rew
            + object_lin_vel_penalty
            + object_ang_vel_penalty
        )

        # ── Retract reward (for envs already in retract phase) ──
        # TODO: If the agent learns to knock the object off the goal to trigger
        # a timeout reset (avoiding the harder retract task), consider adding:
        # 1. Immediate episode reset when object leaves goal during retract, and/or
        # 2. A negative penalty for object displacement during retract phase
        retract_rew = torch.zeros_like(base_reward)
        if self.retract_phase.any():
            # Object must still be at final goal for retract reward
            retract_tol = tight_tol if final_tol is not None else (self.success_tolerance * self.keypoint_scale)
            object_at_goal = (self.keypoints_max_dist <= retract_tol).float()

            # Reward proportional to hand-object distance (gated on object at goal)
            mean_fingertip_dist = self.curr_fingertip_distances.mean(dim=-1)
            retract_dist_rew = mean_fingertip_dist * self.retract_reward_scale * object_at_goal

            # One-time bonus when hand clears threshold (gated on object at goal)
            just_retracted = (
                (mean_fingertip_dist > self.retract_distance_threshold)
                & self.retract_phase
                & ~self.retract_succeeded
                & object_at_goal.bool()
            )
            self.retract_succeeded |= just_retracted
            retract_bonus = just_retracted.float() * self.retract_success_bonus

            retract_rew = (retract_dist_rew + retract_bonus) * self.retract_phase.float()

        # Retract envs: action penalties + retract reward
        # Non-retract envs: full base reward
        retract_env_reward = kuka_actions_penalty + hand_actions_penalty + retract_rew
        reward = torch.where(self.retract_phase, retract_env_reward, base_reward)

        self.rew_buf[:] = reward

        # ── Detect retract phase entry + clamp (AFTER reward computation) ──
        # This ensures the transition frame gets the full base reward with
        # final goal bonus. Retract reward starts next frame.
        self.retract_phase |= (self.successes >= self.max_consecutive_successes)
        self.successes.clamp_(max=self.max_consecutive_successes - 1)

        resets = self._compute_resets(is_success)
        self.reset_buf[:] = resets

        if self.cfg["env"]["forceNoReset"]:
            self.reset_buf[:] = False

        # ── Logging ──
        self.extras["successes"] = self.prev_episode_successes
        self.extras["success_ratio"] = (
            self.prev_episode_successes.mean().item() / self.max_consecutive_successes
        )
        self.extras["closest_keypoint_max_dist"] = self.prev_episode_closest_keypoint_max_dist
        self.extras["all_goals_hit_ratio"] = (
            self.prev_episode_successes >= self.max_consecutive_successes
        ).float().mean().item()
        self.extras["final_goal_tolerance"] = self.final_goal_success_tolerance
        self.true_objective = self._true_objective()
        self.extras["true_objective"] = self.true_objective
        self.extras["retract_phase_ratio"] = self.retract_phase.float().mean().item()
        self.extras["retract_success_ratio"] = self.retract_succeeded.float().mean().item()

        rewards = [
            (fingertip_delta_rew, "fingertip_delta_rew"),
            (hand_delta_penalty, "hand_delta_penalty"),
            (lifting_rew, "lifting_rew"),
            (lift_bonus_rew, "lift_bonus_rew"),
            (keypoint_rew, "keypoint_rew"),
            (kuka_actions_penalty, "kuka_actions_penalty"),
            (hand_actions_penalty, "hand_actions_penalty"),
            (bonus_rew, "bonus_rew"),
            (object_lin_vel_penalty, "object_lin_vel_penalty"),
            (object_ang_vel_penalty, "object_ang_vel_penalty"),
            (retract_rew, "retract_rew"),
            (reward, "total_reward"),
        ]

        episode_cumulative = dict()
        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
            episode_cumulative[rew_name] = rew_value
        self.extras["rewards_episode"] = self.rewards_episode
        self.extras["episode_cumulative"] = episode_cumulative

        return self.rew_buf, is_success

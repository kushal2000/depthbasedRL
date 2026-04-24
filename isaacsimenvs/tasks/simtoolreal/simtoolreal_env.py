"""SimToolReal DirectRLEnv — thin hook dispatcher.

All phase-specific math lives in :mod:`.utils`. This class is just the
DirectRLEnv wiring and state holder: it (a) overrides obs/state_space from
the configured field lists so rl_games sees the right shapes, (b) allocates
per-env buffers after ``super().__init__`` runs ``_setup_scene``, and (c)
dispatches each hook to the appropriate utility module:

  - ``_setup_scene``        → :func:`utils.scene_utils.setup_scene`
  - ``_pre_physics_step``   → :func:`utils.action_utils.apply_action_pipeline`
                              + :func:`utils.action_utils.apply_wrench_dr`
  - ``_apply_action``       → direct call on the articulation (one line)
  - ``_get_observations``   → :func:`utils.obs_utils.build_observations`
  - ``_get_rewards``        → :func:`utils.reward_utils.compute_rewards`
  - ``_get_dones``          → :func:`utils.termination_utils.update_tolerance_curriculum`
                              + :func:`utils.obs_utils.compute_intermediate_values`
                              + :func:`utils.termination_utils.compute_terminations`
  - ``_reset_idx``          → :func:`utils.reset_utils.reset_env_state`

Mid-episode goal-hit triggers :func:`utils.reset_utils.reset_goal_trackers`
directly from :func:`utils.termination_utils.compute_terminations` — no
method on this class wraps it.

**Startup cost warning:** ``setup_scene`` uses ``force_usd_conversion=True``
on every procedural URDF, which regenerates 100 × N_handle_head_types USDs
from scratch each launch (default 600). Expect several minutes first time.
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv

from .simtoolreal_env_cfg import SimToolRealEnvCfg
from .utils.action_utils import apply_action_pipeline, apply_wrench_dr
from .utils.obs_utils import build_observations, compute_intermediate_values, compute_obs_dim
from .utils.reset_utils import allocate_state_buffers, reset_env_state
from .utils.reward_utils import compute_rewards
from .utils.scene_utils import setup_scene
from .utils.termination_utils import compute_terminations, update_tolerance_curriculum


__all__ = ["SimToolRealEnv", "SimToolRealEnvCfg"]


class SimToolRealEnv(DirectRLEnv):
    cfg: SimToolRealEnvCfg

    def __init__(
        self, cfg: SimToolRealEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        # Override obs/state space from configured field lists before
        # DirectRLEnv / rl_games observes the configclass.
        cfg.observation_space = compute_obs_dim(cfg.obs.obs_list)
        cfg.state_space = compute_obs_dim(cfg.obs.state_list)

        super().__init__(cfg, render_mode, **kwargs)  # runs _setup_scene
        allocate_state_buffers(self)

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        apply_action_pipeline(self, actions)
        apply_wrench_dr(self)

    def _apply_action(self) -> None:
        # Called decimation times per policy step; idempotent.
        self.robot.set_joint_position_target(self._cur_targets)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return build_observations(self)

    def _get_rewards(self) -> torch.Tensor:
        return compute_rewards(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        compute_intermediate_values(self)
        return compute_terminations(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        reset_env_state(
            self,
            torch.as_tensor(env_ids, device=self.device, dtype=torch.long),
        )

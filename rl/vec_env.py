"""Self-contained vec-env wrapper for the rl/ agent — no rl_games dependency.

Wraps the IsaacGym VecTask (or Isaac Lab DirectRLEnv) and provides
env-level CUDA profiling of the step() call to identify simulation
bottlenecks.
"""

import gymnasium.spaces as spaces
import torch

from rl.timing import CudaTimer


class IsaacVecEnv:
    """Thin wrapper around an IsaacGym VecTask environment.

    Drop-in replacement for rl_games' ``RLGPUEnv`` with optional CUDA
    event profiling of the hot-path inside ``step()``.
    """

    def __init__(self, env):
        self.env = env
        device = getattr(env, 'device', 'cuda:0')
        self.timer = CudaTimer(device, enabled=True)

    # ── Core API (matches RLGPUEnv interface) ─────────────────────────

    def step(self, actions):
        t = self.timer

        # DR actions (usually off, but keep for correctness)
        if self.env.dr_randomizations.get('actions', None):
            actions = self.env.dr_randomizations['actions']['noise_lambda'](actions)

        t.start('step/pre_physics')
        action_tensor = torch.clamp(actions, -self.env.clip_actions, self.env.clip_actions)
        self.env.pre_physics_step(action_tensor, None)
        t.stop('step/pre_physics')

        # Physics simulation + render
        t.start('step/simulate')
        for _ in range(self.env.control_freq_inv):
            if self.env.force_render:
                self.env.render()
            self.env.gym.simulate(self.env.sim)
        t.stop('step/simulate')

        # CPU-mode sync
        if self.env.device == 'cpu':
            self.env.gym.fetch_results(self.env.sim, True)

        t.start('step/post_physics')
        self.env.post_physics_step()
        t.stop('step/post_physics')

        self.env.control_steps += 1

        t.start('step/bookkeeping')
        # Timeout buf
        self.env.timeout_buf = (
            (self.env.progress_buf >= self.env.max_episode_length - 1)
            & (self.env.reset_buf != 0)
        )

        # DR observations
        if self.env.dr_randomizations.get('observations', None):
            self.env.obs_buf = self.env.dr_randomizations['observations']['noise_lambda'](self.env.obs_buf)

        self.env.extras['time_outs'] = self.env.timeout_buf.to(self.env.rl_device)

        obs_dict = {}
        obs_dict['obs'] = torch.clamp(self.env.obs_buf, -self.env.clip_obs, self.env.clip_obs).to(self.env.rl_device)
        if self.env.num_states > 0:
            obs_dict['states'] = self.env.get_state()
        if getattr(self.env, 'use_depth_camera', False):
            obs_dict['depth'] = self.env.depth_buf.to(self.env.rl_device)
        t.stop('step/bookkeeping')

        return (
            obs_dict,
            self.env.rew_buf.to(self.env.rl_device),
            self.env.reset_buf.to(self.env.rl_device),
            self.env.extras,
        )

    def reset(self):
        obs_dict = self.env.reset()
        if getattr(self.env, 'use_depth_camera', False):
            self.env._render_depth_cameras()
            obs_dict['depth'] = self.env.depth_buf.to(self.env.rl_device)
        return obs_dict

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {
            'action_space': self.env.action_space,
            'observation_space': self.env.observation_space,
        }
        if hasattr(self.env, 'amp_observation_space'):
            info['amp_observation_space'] = self.env.amp_observation_space
        if self.env.num_states > 0:
            info['state_space'] = self.env.state_space
        return info

    def set_train_info(self, env_frames, *args, **kwargs):
        if hasattr(self.env, 'set_train_info'):
            self.env.set_train_info(env_frames, *args, **kwargs)

    def get_env_state(self):
        if hasattr(self.env, 'get_env_state'):
            return self.env.get_env_state()
        return None

    def set_env_state(self, env_state):
        if hasattr(self.env, 'set_env_state'):
            self.env.set_env_state(env_state)

    def flush_profile(self) -> dict:
        """Synchronize GPU and return accumulated env-step timings (ms).

        Call once per epoch from the agent's train loop.
        """
        return self.timer.flush()


class IsaacLabVecEnv:
    """Thin wrapper around Isaac Lab DirectRLEnv for the custom RL pipeline.

    Makes Isaac Lab envs drop-in compatible with SAPGAgent, which expects
    the same interface as IsaacVecEnv (obs_dict with 'obs', 'states', 'depth').
    """

    def __init__(self, env):
        self.env = env
        # gymnasium.make() wraps in OrderEnforcing — get device from unwrapped
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        self.device = unwrapped.device
        self.timer = CudaTimer(self.device, enabled=True)

        # Match the interface expected by SAPGAgent
        cfg = unwrapped.cfg
        self.num_envs = unwrapped.num_envs
        self.num_actions = cfg.action_space
        self.num_obs = cfg.observation_space
        self.num_states = cfg.state_space

        self.observation_space = spaces.Box(-float('inf'), float('inf'), (self.num_obs,))
        self.state_space = spaces.Box(-float('inf'), float('inf'), (self.num_states,))
        self.action_space = spaces.Box(-1.0, 1.0, (self.num_actions,))
        self.clip_actions = 1.0

    # ── Core API (matches IsaacVecEnv / RLGPUEnv interface) ──────────

    def step(self, actions):
        t = self.timer

        t.start('step/env_step')
        obs_dict, reward, terminated, truncated, info = self.env.step(actions)
        t.stop('step/env_step')

        t.start('step/bookkeeping')
        done = terminated | truncated
        info['time_outs'] = truncated.to(self.device)

        result = {
            'obs': obs_dict['policy'].to(self.device),
        }
        if 'critic' in obs_dict:
            result['states'] = obs_dict['critic'].to(self.device)
        if 'depth' in obs_dict:
            result['depth'] = obs_dict['depth'].to(self.device)
        t.stop('step/bookkeeping')

        return (
            result,
            reward.to(self.device),
            done.to(self.device),
            info,
        )

    def reset(self):
        obs_dict, info = self.env.reset()
        result = {
            'obs': obs_dict['policy'].to(self.device),
        }
        if 'critic' in obs_dict:
            result['states'] = obs_dict['critic'].to(self.device)
        if 'depth' in obs_dict:
            result['depth'] = obs_dict['depth'].to(self.device)
        return result

    def reset_done(self):
        return self.reset()

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {
            'action_space': self.action_space,
            'observation_space': self.observation_space,
        }
        if self.num_states > 0:
            info['state_space'] = self.state_space
        return info

    def set_train_info(self, env_frames, *args, **kwargs):
        pass

    def get_env_state(self):
        return None

    def set_env_state(self, env_state):
        pass

    def flush_profile(self) -> dict:
        """Synchronize GPU and return accumulated env-step timings (ms)."""
        return self.timer.flush()

"""Minimal SAPG PPO agent — numerically identical to rl_games for our config.

Ported from:
  - rl_games/common/a2c_common.py (A2CBase, ContinuousA2CBase)
  - rl_games/algos_torch/a2c_continuous.py (A2CAgent)
  - rl_games/algos_torch/central_value.py (CentralValueTrain)
"""

import copy
import math
import os
import time
from collections import deque, OrderedDict

import gym
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from rl.network import ActorNetwork, CriticNetwork, RunningMeanStd


# ── Helpers ──────────────────────────────────────────────────────────────────

def swap_and_flatten01(arr):
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


def filter_leader(val, orig_len, repeat_idxs, num_blocks):
    if len(val) > 1:
        bsize = orig_len // num_blocks
        filtered = []
        for i, idx in enumerate(repeat_idxs):
            if idx == 0:
                filtered.append(val[i * orig_len:(i + 1) * orig_len])
            else:
                filtered.append(val[i * orig_len + (idx - 1) * bsize:i * orig_len + idx * bsize])
        return torch.cat(filtered, dim=0)
    else:
        bsize = orig_len // num_blocks
        filtered = []
        for i, idx in enumerate(repeat_idxs):
            if idx == 0:
                filtered.append(val[:, i * orig_len:(i + 1) * orig_len])
            else:
                filtered.append(val[:, i * orig_len + (idx - 1) * bsize:i * orig_len + idx * bsize])
        return torch.cat(filtered, dim=1)


def shuffle_batch(batch_dict, horizon_length):
    device = batch_dict['returns'].device
    n = len(batch_dict['returns']) // horizon_length
    indices = (
        torch.randperm(n, device=device).reshape(-1, 1) * horizon_length
        + torch.arange(horizon_length, device=device).reshape(1, -1)
    )
    flat = indices.reshape(-1)
    for key in batch_dict:
        if key == 'rnn_states':
            if batch_dict[key] is None:
                continue
            batch_dict[key] = [s[:, indices[:, 0] // horizon_length] for s in batch_dict[key]]
        elif key in ['played_frames', 'step_time']:
            continue
        else:
            batch_dict[key] = batch_dict[key][flat]
    return batch_dict


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    return action * d + m


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=True):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = (c1 + c2 + c3).sum(dim=-1)
    return kl.mean() if reduce else kl


def flatten_dict(d, prefix='', separator='/'):
    """Flatten nested dicts: {'a': {'b': 1}} -> {'a/b': 1}."""
    res = {}
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            res.update(flatten_dict(value, prefix + key + separator, separator))
        else:
            res[prefix + key] = value
    return res


def remove_envs_from_info(infos, num_envs):
    """Remove first num_envs entries from info tensors (for ignore_env_boundary)."""
    for key in list(infos.keys()):
        if isinstance(infos[key], dict):
            infos[key] = remove_envs_from_info(infos[key], num_envs)
        elif isinstance(infos[key], (list, np.ndarray, torch.Tensor)):
            if key in ['successes', 'closest_keypoint_max_dist']:
                block_size = len(infos[key]) - num_envs
                if block_size > 0 and len(infos[key]) % block_size == 0:
                    for i in range(len(infos[key]) // block_size):
                        infos[f"{key}_per_block/block_{i}"] = infos[key][i * block_size:(i + 1) * block_size]
            infos[key] = infos[key][num_envs:]
    return infos


# ── Env Info Observer ────────────────────────────────────────────────────────

class EnvInfoObserver:
    """Processes env infos and logs metrics — mirrors RLGPUAlgoObserver."""

    def __init__(self, writer, device, games_to_track=3000, ignore_env_boundary=0):
        self.writer = writer
        self.device = device
        self.games_to_track = games_to_track
        self.ignore_env_boundary = ignore_env_boundary
        self.ep_infos = []
        self.direct_info = {}
        self.episode_cumulative = {}
        self.episode_cumulative_avg = {}
        self.new_finished_episodes = False

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict):
            return

        if self.ignore_env_boundary > 0:
            infos = remove_envs_from_info(copy.deepcopy(infos), self.ignore_env_boundary)
            done_indices = done_indices[done_indices >= self.ignore_env_boundary].unsqueeze(-1) - self.ignore_env_boundary

        if 'episode' in infos:
            self.ep_infos.append(infos['episode'])

        if 'episode_cumulative' in infos:
            for key, value in infos['episode_cumulative'].items():
                if key not in self.episode_cumulative:
                    self.episode_cumulative[key] = torch.zeros_like(value)
                self.episode_cumulative[key] += value

            for done_idx in done_indices:
                self.new_finished_episodes = True
                done_idx = done_idx.item()
                for key, value in infos['episode_cumulative'].items():
                    if key not in self.episode_cumulative_avg:
                        self.episode_cumulative_avg[key] = deque([], maxlen=self.games_to_track)
                    self.episode_cumulative_avg[key].append(self.episode_cumulative[key][done_idx].item())
                    self.episode_cumulative[key][done_idx] = 0

        # Flatten nested dicts into summary keys
        if len(infos) > 0 and isinstance(infos, dict):
            infos_flat = flatten_dict(infos, prefix='', separator='/')
            self.direct_info = {}
            for k, v in infos_flat.items():
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.direct_info[k] = v

        # Special handling for successes, keypoint dist, etc.
        for tag in ['successes', 'closest_keypoint_max_dist', 'discounted_reward']:
            if tag in infos:
                self.direct_info[tag] = infos[tag].mean()
                self.direct_info[f'{tag}_median'] = torch.median(infos[tag]).item()
                self.direct_info[f'{tag}_max'] = infos[tag].max()
                for key in infos:
                    if key.startswith(f'{tag}_per_block'):
                        self.direct_info[key] = torch.mean(infos[key]).item()

        if 'true_objective' in infos:
            self.direct_info['true_objective_mean'] = infos['true_objective'].mean()
            self.direct_info['true_objective_max'] = infos['true_objective'].max()

    def log_stats(self, frame, epoch_num, total_time):
        if self.writer is None:
            return

        # Episode info
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_infos:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, frame)
            self.ep_infos.clear()

        # Episode cumulative metrics
        if self.new_finished_episodes:
            for key in self.episode_cumulative_avg:
                self.writer.add_scalar(f'episode_cumulative/{key}', np.mean(self.episode_cumulative_avg[key]), frame)
                self.writer.add_scalar(f'episode_cumulative_min/{key}_min', np.min(self.episode_cumulative_avg[key]), frame)
                self.writer.add_scalar(f'episode_cumulative_max/{key}_max', np.max(self.episode_cumulative_avg[key]), frame)
            self.new_finished_episodes = False

        # Direct info (scalars from env)
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, frame)
            self.writer.add_scalar(f'{k}/iter', v, frame)
            self.writer.add_scalar(f'{k}/time', v, frame)


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super().__init__()
        self.max_size = max_size
        self.register_buffer("current_size", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        size = values.size(0)
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0).reshape(self.mean.shape)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = torch.tensor(size_sum, dtype=torch.int32)
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = torch.tensor(0, dtype=torch.int32)
        self.mean.fill_(0)

    def get_mean(self):
        return self.mean.cpu().numpy()


# ── PPO Dataset ──────────────────────────────────────────────────────────────

class PPODataset:
    def __init__(self, batch_size, minibatch_size, is_rnn, device, seq_length):
        self.is_rnn = is_rnn
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.length = batch_size // minibatch_size
        total_games = batch_size // seq_length
        self.num_games_batch = minibatch_size // seq_length
        self.values_dict = None
        self.last_range = (0, 0)

    def update_values_dict(self, values_dict):
        self.values_dict = values_dict
        if values_dict is not None and 'returns' in values_dict:
            self.length = len(values_dict['returns']) // self.minibatch_size

    def update_mu_sigma(self, mu, sigma):
        start, end = self.last_range
        self.values_dict['mu'][start:end] = mu
        self.values_dict['sigma'][start:end] = sigma

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_rnn:
            return self._get_item_rnn(idx)
        return self._get_item(idx)

    def _get_item_rnn(self, idx):
        gstart = idx * self.num_games_batch
        gend = (idx + 1) * self.num_games_batch
        if idx == self.length - 1:
            gend = len(self.values_dict['returns']) // self.seq_length
        start = gstart * self.seq_length
        end = gend * self.seq_length
        self.last_range = (start, end)

        input_dict = {}
        for k, v in self.values_dict.items():
            if k == 'rnn_states':
                continue
            if v is not None:
                input_dict[k] = v[start:end]
            else:
                input_dict[k] = None

        rnn_states = self.values_dict['rnn_states']
        input_dict['rnn_states'] = [s[:, gstart:gend, :].contiguous() for s in rnn_states]
        return input_dict

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        if idx == self.length - 1:
            end = len(self.values_dict['returns'])
        self.last_range = (start, end)
        input_dict = {}
        for k, v in self.values_dict.items():
            if k != 'rnn_states' and v is not None:
                input_dict[k] = v[start:end]
        return input_dict


# ── Experience Buffer ────────────────────────────────────────────────────────

class ExperienceBuffer:
    def __init__(self, env_info, num_actors, horizon_length, has_central_value, device, extra_obs_dim=None):
        self.device = device
        self.num_actors = num_actors
        self.horizon_length = horizon_length
        self.has_central_value = has_central_value
        num_agents = env_info.get('agents', 1)
        batch_size = num_actors * num_agents
        obs_base_shape = (horizon_length, batch_size)
        state_base_shape = (horizon_length, num_actors)
        action_space = env_info['action_space']
        obs_space = env_info['observation_space']
        actions_num = action_space.shape[0]
        value_size = env_info.get('value_size', 1)

        def _make(shape, dtype=torch.float32, base=obs_base_shape):
            return torch.zeros(base + shape, dtype=dtype, device=device)

        obs_shape = obs_space.shape
        if extra_obs_dim is not None:
            obs_shape = (obs_shape[0] + extra_obs_dim,)
        state_space = env_info.get('state_space', None)

        self.tensor_dict = {}
        self.tensor_dict['obses'] = _make(obs_shape)
        if has_central_value and state_space is not None:
            state_shape = state_space.shape
            if extra_obs_dim is not None:
                state_shape = (state_shape[0] + extra_obs_dim,)
            self.tensor_dict['states'] = _make(state_shape, base=state_base_shape)
        self.tensor_dict['rewards'] = _make((value_size,))
        self.tensor_dict['intr_rewards'] = _make((1,))
        self.tensor_dict['values'] = _make((value_size,))
        self.tensor_dict['neglogpacs'] = _make(())
        self.tensor_dict['dones'] = _make((), dtype=torch.uint8)
        self.tensor_dict['actions'] = _make((actions_num,))
        self.tensor_dict['mus'] = _make((actions_num,))
        self.tensor_dict['sigmas'] = _make((actions_num,))

    def update_data(self, name, index, val):
        self.tensor_dict[name][index, :] = val

    def get_transformed_list(self, transform_op, tensor_list):
        res = {}
        for k in tensor_list:
            v = self.tensor_dict.get(k)
            if v is None:
                continue
            res[k] = transform_op(v)
        return res


# ── SAPGAgent ────────────────────────────────────────────────────────────────

class SAPGAgent:
    """PPO + SAPG agent that trains on an Isaac Gym vec env."""

    def __init__(self, vec_env, config, writer=None):
        self.vec_env = vec_env
        self.writer = writer
        self.config = c = config

        self.ppo_device = c.get('device', 'cuda:0')
        self.env_info = vec_env.get_env_info()
        self.num_actors = c['num_actors']
        self.num_agents = self.env_info.get('agents', 1)
        self.value_size = self.env_info.get('value_size', 1)
        self.observation_space = self.env_info['observation_space']
        self.obs_shape = self.observation_space.shape
        self.action_space = self.env_info['action_space']
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.ppo_device)
        self.state_space = self.env_info.get('state_space', None)
        self.state_shape = self.state_space.shape if self.state_space is not None else None

        # Hyperparameters
        self.horizon_length = c['horizon_length']
        self.seq_length = c.get('seq_length', 4)
        self.gamma = c['gamma']
        self.tau = c['tau']
        self.e_clip = c['e_clip']
        self.clip_value = c['clip_value']
        self.critic_coef = c['critic_coef']
        self.bounds_loss_coef = c.get('bounds_loss_coef', None)
        self.grad_norm = c['grad_norm']
        self.truncate_grads = c.get('truncate_grads', False)
        self.normalize_input = c.get('normalize_input', False)
        self.normalize_value = c.get('normalize_value', False)
        self.normalize_advantage = c.get('normalize_advantage', True)
        self.entropy_coef = c.get('entropy_coef', 0.0)
        self.mini_epochs_num = c['mini_epochs']
        self.minibatch_size = c['minibatch_size']
        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors
        self.num_minibatches = self.batch_size // self.minibatch_size
        self.value_bootstrap = c.get('value_bootstrap', None)
        self.mixed_precision = c.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision and self.ppo_device != 'cpu')
        self.weight_decay = c.get('weight_decay', 0.0)
        self.last_lr = float(c['learning_rate'])
        self.max_epochs = c.get('max_epochs', -1)
        self.max_frames = c.get('max_frames', -1)
        self.save_freq = c.get('save_frequency', 0)
        self.save_best_after = c.get('save_best_after', 100)
        self.print_stats = c.get('print_stats', True)
        self.games_to_track = c.get('games_to_track', 3000)

        self.bound_loss_type = c.get('bound_loss_type', 'bound')
        self.use_experimental_cv = c.get('use_experimental_cv', True)

        # Reward shaper
        reward_shaper_cfg = c.get('reward_shaper', {})
        self.reward_scale = reward_shaper_cfg.get('scale_value', 1.0)

        # LR schedule
        lr_schedule = c.get('lr_schedule', None)
        self.is_adaptive_lr = lr_schedule == 'adaptive'
        self.schedule_type = c.get('schedule_type', 'legacy')
        if self.is_adaptive_lr:
            self.kl_threshold = c['kl_threshold']

        # SAPG exploration
        self.expl_type = c.get('expl_type', 'none')
        self.use_others_experience = c.get('use_others_experience', 'none')
        self.off_policy_ratio = c.get('off_policy_ratio', 1.0)
        self.ignore_env_boundary = c.get('good_reset_boundary', 0)

        if self.expl_type.startswith('mixed_expl'):
            self.intr_coef_block_size = c.get('expl_coef_block_size', 4096)
            assert self.num_actors % self.intr_coef_block_size == 0
            env_ids = torch.arange(self.num_actors // self.intr_coef_block_size).repeat_interleave(
                self.intr_coef_block_size
            ).to(self.ppo_device)
            embedding_genvec = torch.linspace(50.0, 0.0, self.num_actors // self.intr_coef_block_size).to(
                self.ppo_device
            )[env_ids]

            if 'disjoint' in self.expl_type or 'learn_param' in self.expl_type:
                self.intr_reward_coef_embd = embedding_genvec.reshape(-1, 1)
            else:
                raise NotImplementedError

            expl_reward_type = c.get('expl_reward_type', 'none')
            if expl_reward_type == 'entropy':
                self.intr_reward_coef = (
                    torch.linspace(0.5, 0.0, self.num_actors // self.intr_coef_block_size).to(self.ppo_device)[env_ids]
                    * c.get('expl_reward_coef_scale', 1.0)
                )
            elif expl_reward_type == 'none':
                self.intr_reward_coef = torch.zeros(self.num_actors, device=self.ppo_device)
            else:
                raise NotImplementedError
            self.intr_reward_model = None

            self.ignore_env_boundary = max(self.ignore_env_boundary, self.num_actors - self.intr_coef_block_size)
        else:
            self.intr_reward_coef = None
            self.intr_reward_coef_embd = None
            self.intr_reward_model = None

        # Policy index (from experiment name)
        exp_name = c.get('full_experiment_name') or c.get('name', '00_default')
        self.policy_idx = int(exp_name.split('_')[0])
        self.experiment_name = exp_name

        # Directories
        self.train_dir = c.get('train_dir', 'runs')
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        os.makedirs(self.nn_dir, exist_ok=True)

        # ── Build networks ──────────────────────────────────────────────

        extra_obs_dim = self.intr_reward_coef_embd.shape[-1] if self.intr_reward_coef_embd is not None else None
        coef_ids = self.intr_reward_coef_embd[::self.intr_coef_block_size, 0] if self.intr_reward_coef_embd is not None else None

        obs_dim = self.obs_shape[0]
        if extra_obs_dim is not None:
            obs_dim += extra_obs_dim

        self.model = ActorNetwork(
            obs_dim=obs_dim,
            action_dim=self.actions_num,
            num_actors=self.num_actors * self.num_agents,
            coef_ids=coef_ids,
            coef_id_idx=self.obs_shape[0],
            param_size=32,
            rnn_units=1024,
            rnn_layers=1,
            mlp_units=(1024, 1024, 512, 512),
        ).to(self.ppo_device)

        self.is_rnn = True  # Always LSTM for this config
        self.has_central_value = self.state_space is not None

        if self.has_central_value:
            state_dim = self.state_shape[0]
            if extra_obs_dim is not None:
                state_dim += extra_obs_dim
            self.central_value_net = CriticNetwork(
                state_dim=state_dim,
                mlp_units=(1024, 1024, 512, 512),
            ).to(self.ppo_device)
        else:
            self.central_value_net = None

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value

        # ── Optimizers ──────────────────────────────────────────────────

        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.has_central_value:
            cv_config = c.get('central_value_config', {})
            cv_lr = float(cv_config.get('learning_rate', self.last_lr))
            cv_weight_decay = cv_config.get('weight_decay', 0.0)
            self.cv_optimizer = optim.Adam(self.central_value_net.parameters(), cv_lr, eps=1e-08, weight_decay=cv_weight_decay)
            self.cv_mini_epochs = cv_config.get('mini_epochs', 2)
            self.cv_minibatch_size = min(cv_config.get('minibatch_size', self.minibatch_size), self.batch_size_envs)
            self.cv_num_minibatches = self.batch_size_envs // self.cv_minibatch_size
            self.cv_clip_value = cv_config.get('clip_value', True)
            self.cv_e_clip = cv_config.get('e_clip', 0.2)
            self.cv_grad_norm = cv_config.get('grad_norm', 1.0)
            self.cv_truncate_grads = cv_config.get('truncate_grads', False)
            self.cv_normalize_input = cv_config.get('normalize_input', True)
            self.cv_lr = cv_lr
        else:
            self.cv_optimizer = None

        # Value normalizer alias
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.value_mean_std if self.has_central_value else self.model.value_mean_std

        # ── Trackers ────────────────────────────────────────────────────
        self.game_rewards = AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_shaped_rewards = AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths = AverageMeter(1, self.games_to_track).to(self.ppo_device)

        # Env info observer (for episode_cumulative, successes, etc.)
        self.env_observer = EnvInfoObserver(
            writer=self.writer,
            device=self.ppo_device,
            games_to_track=self.games_to_track,
            ignore_env_boundary=self.ignore_env_boundary,
        )

        self.epoch_num = 0
        self.frame = 0
        self.total_time = 0
        self.mean_rewards = self.last_mean_rewards = -1e9
        self.obs = None
        self.dones = None
        self.rnn_states = None
        self.current_rewards = None

        # Central value RNN states (if critic had RNN, but it doesn't in this config)
        self.cv_rnn_states = None

    # ── Env interaction helpers ──────────────────────────────────────────

    def _cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            return obs.to(self.ppo_device)
        return torch.FloatTensor(obs).to(self.ppo_device)

    def _obs_to_tensors(self, obs):
        if isinstance(obs, dict):
            upd = {k: self._cast_obs(v) for k, v in obs.items()}
        else:
            upd = self._cast_obs(obs)
        if not isinstance(obs, dict) or 'obs' not in obs:
            upd = {'obs': upd}
        return upd

    def env_reset(self):
        obs = self.vec_env.reset()
        obs = self._obs_to_tensors(obs)
        if self.intr_reward_coef_embd is not None:
            obs['obs'] = torch.cat([obs['obs'], self.intr_reward_coef_embd], dim=1)
            obs['states'] = torch.cat([obs['states'], self.intr_reward_coef_embd], dim=1)
        return obs

    def env_step(self, actions):
        clamped = torch.clamp(actions, -1.0, 1.0)
        rescaled = rescale_actions(self.actions_low, self.actions_high, clamped)
        obs, rewards, dones, infos = self.vec_env.step(rescaled.to(self.vec_env.env.device))
        obs = self._obs_to_tensors(obs)
        if self.value_size == 1:
            rewards = rewards.unsqueeze(1)
        rewards = rewards.to(self.ppo_device)
        dones = dones.to(self.ppo_device)

        intr_rewards = torch.zeros_like(rewards)

        tr_obs = self._obs_to_tensors(obs)
        if self.intr_reward_coef_embd is not None:
            tr_obs['obs'] = torch.cat([tr_obs['obs'], self.intr_reward_coef_embd], dim=1)
            tr_obs['states'] = torch.cat([tr_obs['states'], self.intr_reward_coef_embd], dim=1)

        return tr_obs, rewards, intr_rewards, dones, infos

    # ── Init tensors ─────────────────────────────────────────────────────

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        extra_obs_dim = self.intr_reward_coef_embd.shape[-1] if self.intr_reward_coef_embd is not None else None
        self.experience_buffer = ExperienceBuffer(
            self.env_info,
            num_actors=self.num_actors,
            horizon_length=self.horizon_length,
            has_central_value=self.has_central_value,
            device=self.ppo_device,
            extra_obs_dim=extra_obs_dim,
        )

        if self.current_rewards is None:
            self.current_rewards = torch.zeros((batch_size, self.value_size), dtype=torch.float32, device=self.ppo_device)
            self.current_shaped_rewards = torch.zeros((batch_size, self.value_size), dtype=torch.float32, device=self.ppo_device)
            self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
            self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

        if self.rnn_states is None:
            self.rnn_states = self.model.get_default_rnn_state()
        self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

        total_agents = self.num_agents * self.num_actors
        num_seqs = self.horizon_length // self.seq_length
        self.mb_rnn_states = [
            torch.zeros((num_seqs, s.size(0), total_agents, s.size(2)), dtype=torch.float32, device=self.ppo_device)
            for s in self.rnn_states
        ]

        self.tensor_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas', 'obses', 'states', 'dones']

        self.dataset = PPODataset(self.batch_size, self.minibatch_size, self.is_rnn, self.ppo_device, self.seq_length)

        if self.has_central_value:
            self.cv_dataset = PPODataset(self.batch_size_envs, self.cv_minibatch_size, False, self.ppo_device, self.seq_length)

    # ── Action selection ─────────────────────────────────────────────────

    def get_action_values(self, obs, rnn_states=None):
        processed_obs = obs['obs']
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'raw_obs': processed_obs,
            'rnn_states': rnn_states,
        }
        with torch.no_grad():
            # Forward actor
            mu, logstd, value, states = self.model(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            selected_action = distr.sample()
            neglogp = ActorNetwork.neglogp(selected_action, mu, sigma, logstd)

            res_dict = {
                'neglogpacs': neglogp.squeeze(),
                'values': self.model.denorm_value(value),
                'actions': selected_action,
                'mus': mu,
                'sigmas': sigma,
                'rnn_states': states,
            }

            if self.has_central_value:
                cv_input = {
                    'is_train': False,
                    'obs': obs['states'],
                }
                cv_result = self.central_value_net(cv_input)
                res_dict['values'] = cv_result['values']

        return res_dict

    def get_values(self, obs, rnn_states):
        with torch.no_grad():
            if self.has_central_value:
                self.central_value_net.eval()
                cv_input = {
                    'is_train': False,
                    'obs': obs['states'],
                }
                value = self.central_value_net(cv_input)['values']
            else:
                self.model.eval()
                processed_obs = obs['obs']
                input_dict = {
                    'is_train': False,
                    'prev_actions': None,
                    'obs': processed_obs,
                    'raw_obs': processed_obs,
                    'rnn_states': rnn_states,
                }
                mu, logstd, value, states = self.model(input_dict)
                value = self.model.denorm_value(value)
            return value

    # ── GAE ───────────────────────────────────────────────────────────────

    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t + 1]
                nextvalues = mb_extrinsic_values[t + 1]
            nextnonterminal = nextnonterminal.unsqueeze(1)
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    # ── Experience collection ────────────────────────────────────────────

    def play_steps(self):
        mb_rnn_states = self.mb_rnn_states
        rnn_state_buffer = [
            torch.zeros((self.horizon_length, *s.shape), dtype=s.dtype, device=s.device)
            for s in self.rnn_states
        ]
        step_time = 0.0

        for n in range(self.horizon_length):
            if n % self.seq_length == 0:
                for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                    mb_s[n // self.seq_length, :, :, :] = s

            for i, s in enumerate(self.rnn_states):
                rnn_state_buffer[i][n, :, :, :] = s

            res_dict = self.get_action_values(self.obs, self.rnn_states)
            self.rnn_states = res_dict['rnn_states']

            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones.byte())
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, intr_rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time += time.time() - step_time_start

            shaped_rewards = rewards * self.reward_scale
            intr_rewards = intr_rewards * self.reward_scale

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self._cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('intr_rewards', n, intr_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            if len(all_done_indices) > 0:
                for s in self.rnn_states:
                    s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

            if len(env_done_indices[env_done_indices >= self.ignore_env_boundary]) > 0:
                indices = env_done_indices[env_done_indices >= self.ignore_env_boundary].view(-1, 1)
                self.game_rewards.update(self.current_rewards[indices])
                self.game_shaped_rewards.update(self.current_shaped_rewards[indices])
                self.game_lengths.update(self.current_lengths[indices])
            self.env_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs, self.rnn_states)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_intr_rewards = self.experience_buffer.tensor_dict['intr_rewards']

        if self.intr_reward_coef is not None:
            mb_total_rewards = mb_rewards + self.intr_reward_coef.unsqueeze(0).unsqueeze(2) * mb_intr_rewards
        else:
            mb_total_rewards = mb_rewards

        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_total_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        states = []
        for mb_s in mb_rnn_states:
            t_size = mb_s.size(0) * mb_s.size(2)
            h_size = mb_s.size(3)
            states.append(mb_s.permute(1, 2, 0, 3).reshape(-1, t_size, h_size))
        batch_dict['rnn_states'] = states
        batch_dict['step_time'] = step_time

        extras = {
            'rewards': mb_rewards,
            'obs': self.experience_buffer.tensor_dict['obses'],
            'last_obs': self.obs,
            'states': self.experience_buffer.tensor_dict.get('states', None),
            'dones': mb_fdones,
            'last_dones': fdones,
            'rnn_states': rnn_state_buffer,
            'last_rnn_states': self.rnn_states,
            'mb_intr_rewards': mb_intr_rewards if self.intr_reward_model is not None else None,
            'mb_extr_rewards': mb_rewards,
        }
        return batch_dict, extras

    # ── SAPG off-policy augmentation ─────────────────────────────────────

    def augment_batch_for_mixed_expl(self, batch_dict, extras, repeat_idxs=None):
        new_batch_dict = {}
        num_blocks = self.num_actors // self.intr_coef_block_size
        if repeat_idxs is None:
            num_repeat = min(num_blocks, int(self.off_policy_ratio) + 1)
            repeat_idxs = [0] + [int(x) for x in np.random.choice(range(1, num_blocks), num_repeat - 1, replace=False)]

        for key, val in batch_dict.items():
            if key in ['played_frames', 'step_time']:
                new_batch_dict[key] = val
            elif key == 'obses':
                intr_coef_embd = torch.cat([
                    torch.roll(self.intr_reward_coef_embd, self.intr_coef_block_size * i, dims=0)
                    for i in repeat_idxs
                ], dim=0)
                obses = torch.cat([val] * len(repeat_idxs), dim=0)
                obses[:, -self.intr_reward_coef_embd.shape[-1]:] = intr_coef_embd.repeat_interleave(
                    self.horizon_length, dim=0
                )
                mask = torch.zeros(len(obses), dtype=torch.bool, device=obses.device)
                mask[len(val):] = True
                if self.use_others_experience == 'lf':
                    obses = filter_leader(obses, len(val), repeat_idxs, num_blocks)
                    mask = filter_leader(mask, len(val), repeat_idxs, num_blocks)
                new_batch_dict[key] = obses
                new_batch_dict['off_policy_mask'] = mask
            elif key == 'states':
                intr_coef_embd = torch.cat([
                    torch.roll(self.intr_reward_coef_embd, self.intr_coef_block_size * i, dims=0)
                    for i in repeat_idxs
                ], dim=0)
                states = torch.cat([val] * len(repeat_idxs), dim=0)
                states[:, -self.intr_reward_coef_embd.shape[-1]:] = intr_coef_embd.repeat_interleave(
                    self.horizon_length, dim=0
                )
                if self.use_others_experience == 'lf':
                    states = filter_leader(states, len(val), repeat_idxs, num_blocks)
                new_batch_dict[key] = states
            elif key in ['values', 'returns']:
                pass
            elif key == 'rnn_states':
                if val is not None:
                    new_batch_dict[key] = [torch.cat([val[i]] * len(repeat_idxs), dim=1) for i in range(len(val))]
                    if self.use_others_experience == 'lf':
                        new_batch_dict[key] = [
                            filter_leader(new_batch_dict[key][i], val[i].shape[1], repeat_idxs, num_blocks)
                            for i in range(len(val))
                        ]
                else:
                    new_batch_dict[key] = None
            else:
                new_batch_dict[key] = torch.cat([val] * len(repeat_idxs), dim=0)
                if self.use_others_experience == 'lf':
                    new_batch_dict[key] = filter_leader(new_batch_dict[key], len(val), repeat_idxs, num_blocks)

        new_returns_list = [batch_dict['returns']]
        new_values_list = [batch_dict['values']]

        for r_k in repeat_idxs[1:]:
            mb_rewards = extras['rewards']
            mb_obs = extras['obs']
            last_obs_and_states = extras['last_obs']
            last_rnn_states = extras['last_rnn_states']
            mb_states = extras['states']
            mb_rnn_states = extras['rnn_states']

            mb_obs[:, :, -self.intr_reward_coef_embd.shape[-1]:] = torch.roll(
                self.intr_reward_coef_embd, self.intr_coef_block_size * r_k, dims=0
            )
            last_obs_and_states['obs'][:, -self.intr_reward_coef_embd.shape[-1]:] = torch.roll(
                self.intr_reward_coef_embd, self.intr_coef_block_size * r_k, dims=0
            )

            flattened_rnn_states = [
                rnn_s.transpose(0, 1).reshape(rnn_s.transpose(0, 1).shape[0], -1, *rnn_s.shape[3:])
                for rnn_s in mb_rnn_states
            ] if mb_rnn_states is not None else None

            flattened_mb_obs = mb_obs.reshape(-1, *mb_obs.shape[2:])
            flattened_mb_states = mb_states.reshape(-1, *mb_states.shape[2:]) if mb_states is not None else None

            mb_values = []
            for i in range((flattened_mb_obs.shape[0] + 8191) // 8192):
                mb_values.append(self.get_values(
                    {
                        'obs': flattened_mb_obs[i * 8192:(i + 1) * 8192],
                        'states': flattened_mb_states[i * 8192:(i + 1) * 8192] if mb_states is not None else None,
                    },
                    rnn_states=[s[:, i * 8192:(i + 1) * 8192] for s in flattened_rnn_states] if flattened_rnn_states is not None else None,
                ))
            mb_values = torch.cat(mb_values, dim=0)
            last_values = self.get_values(last_obs_and_states, last_rnn_states)

            mb_values = mb_values.reshape(*mb_obs.shape[:2], *mb_values.shape[1:])
            mb_values = torch.cat([mb_values, last_values.unsqueeze(0)], dim=0)

            mb_fdones = extras['dones']
            fdones = extras['last_dones']
            mb_fdones_cat = torch.cat([mb_fdones, fdones.unsqueeze(0)], dim=0)

            mb_returns = (
                mb_rewards
                + (
                    torch.roll(self.intr_reward_coef, self.intr_coef_block_size * r_k, dims=0).unsqueeze(0).unsqueeze(2)
                    * extras['mb_intr_rewards']
                    if extras['mb_intr_rewards'] is not None
                    else 0
                )
                + self.gamma * mb_values[1:] * (1 - mb_fdones_cat[1:]).unsqueeze(-1)
            )

            new_returns_list.append(swap_and_flatten01(mb_returns))
            new_values_list.append(swap_and_flatten01(mb_values[:-1]))

        new_batch_dict['returns'] = torch.cat(new_returns_list, dim=0)
        new_batch_dict['values'] = torch.cat(new_values_list, dim=0)
        if self.use_others_experience == 'lf':
            new_batch_dict['returns'] = filter_leader(new_batch_dict['returns'], len(batch_dict['returns']), repeat_idxs, num_blocks)
            new_batch_dict['values'] = filter_leader(new_batch_dict['values'], len(batch_dict['values']), repeat_idxs, num_blocks)

        # Reset obs and last obs in extras
        extras['obs'][:, :, -self.intr_reward_coef_embd.shape[-1]:] = self.intr_reward_coef_embd
        extras['last_obs']['obs'][:, -self.intr_reward_coef_embd.shape[-1]:] = self.intr_reward_coef_embd

        return new_batch_dict

    # ── Prepare dataset ──────────────────────────────────────────────────

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)

        advantages = returns - values

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            mean, std = advantages.mean(), advantages.std()
            advantages = (advantages - mean) / (std + 1e-8)

        dataset_dict = {
            'old_values': values,
            'old_logp_actions': neglogpacs,
            'advantages': advantages,
            'returns': returns,
            'actions': actions,
            'obs': obses,
            'dones': dones,
            'rnn_states': rnn_states,
            'rnn_masks': None,
            'mu': mus,
            'sigma': sigmas,
            'off_policy_mask': batch_dict.get('off_policy_mask', None),
        }
        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            cv_dict = {
                'old_values': values,
                'advantages': advantages,
                'returns': returns,
                'actions': actions,
                'obs': batch_dict['states'],
                'dones': dones,
                'rnn_masks': None,
            }
            self.cv_dataset.update_values_dict(cv_dict)

    # ── Policy update ────────────────────────────────────────────────────

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
            'raw_obs': obs_batch,
            'rnn_states': input_dict.get('rnn_states'),
            'seq_length': self.seq_length,
            'dones': input_dict.get('dones'),
        }

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            mu, logstd, value, states = self.model(batch_dict)
            sigma = torch.exp(logstd)
            entropy = torch.distributions.Normal(mu, sigma, validate_args=False).entropy().sum(dim=-1)
            action_log_probs = ActorNetwork.neglogp(actions_batch, mu, sigma, logstd)

            # PPO actor loss
            ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
            a_loss = torch.max(-surr1, -surr2)

            # Value loss (actor's value head)
            if self.has_value_loss:
                if self.clip_value:
                    value_pred_clipped = value_preds_batch + (value - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
                    value_losses = (value - return_batch) ** 2
                    value_losses_clipped = (value_pred_clipped - return_batch) ** 2
                    c_loss = torch.max(value_losses, value_losses_clipped)
                else:
                    c_loss = (return_batch - value) ** 2
            else:
                c_loss = torch.zeros((len(value), 1), device=self.ppo_device)

            # Bounds loss
            if self.bounds_loss_coef is not None:
                soft_bound = 1.1
                mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
            else:
                b_loss = torch.zeros(len(mu), device=self.ppo_device)

            # Entropy coefficient (per-actor for SAPG)
            if self.expl_type.startswith('mixed_expl') and self.config.get('expl_reward_type') == 'entropy':
                ec_candidates = self.intr_reward_coef[::self.intr_coef_block_size]
                ec_identifiers = self.intr_reward_coef_embd[::self.intr_coef_block_size, 0].reshape(-1, 1)
                ec_indices = torch.argmax((obs_batch[:, -self.intr_reward_coef_embd.shape[1]] == ec_identifiers).float(), dim=0)
                entropy_coef = ec_candidates[ec_indices]
            else:
                entropy_coef = self.entropy_coef

            # Combine losses (apply_masks with mask=None -> just mean)
            a_loss = torch.mean(a_loss.unsqueeze(1))
            c_loss = torch.mean(c_loss)
            entropy_loss = torch.mean((entropy_coef * entropy).unsqueeze(1))
            b_loss = torch.mean(b_loss.unsqueeze(1))

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy_loss + b_loss * self.bounds_loss_coef

        # Backward + grad clip + step
        for param in self.model.parameters():
            param.grad = None

        self.scaler.scale(loss).backward()

        # Collect all grads before clipping (for auxiliary stats)
        all_grads_list = []
        for param in self.model.parameters():
            if param.grad is not None:
                all_grads_list.append(param.grad.view(-1))
        all_grads = torch.cat(all_grads_list)

        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # KL divergence
        with torch.no_grad():
            kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce=True)

        # Build extras dict for auxiliary stats
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        contrib = torch.logical_and(ratio < 1.0 + curr_e_clip, ratio > 1.0 - curr_e_clip).float()

        off_policy_mask = input_dict.get('off_policy_mask', None)
        if off_policy_mask is not None and os.getenv('LOG_OFF_POLICY_GRADS'):
            contrib_off = torch.masked_select(contrib, off_policy_mask)
            contrib_on = torch.masked_select(contrib, ~off_policy_mask)
            contrib_off = torch.nan_to_num(contrib_off.mean())
            contrib_on = torch.nan_to_num(contrib_on.mean())
            extras = {
                "off_policy_contrib": contrib_off.item(),
                "on_policy_contrib": contrib_on.item(),
                "off_policy_grads": all_grads.detach().cpu(),
                "on_policy_grads": all_grads.detach().cpu(),
            }
        else:
            extras = {
                "on_policy_contrib": contrib.mean().item(),
                "off_policy_contrib": 0,
                "on_policy_grads": all_grads.detach().cpu(),
                "off_policy_grads": torch.zeros_like(all_grads).cpu(),
            }

        if self.expl_type.startswith('mixed_expl'):
            bl_ids = self.intr_reward_coef_embd[::self.intr_coef_block_size, 0].reshape(-1, 1)
            bl_idxs = torch.argmax((obs_batch[:, -self.intr_reward_coef_embd.shape[1]] == bl_ids).float(), dim=0)
            extras["entropies"] = [
                torch.nan_to_num(entropy[bl_idxs == i].detach().mean()).item()
                for i in range(self.num_actors // self.intr_coef_block_size)
            ]

        return (
            a_loss, c_loss,
            torch.mean(entropy),
            kl_dist, self.last_lr, lr_mul,
            mu.detach(), sigma.detach(), b_loss,
            extras,
        )

    def train_actor_critic(self, input_dict):
        return self.calc_gradients(input_dict)

    # ── Central value training ───────────────────────────────────────────

    def train_central_value(self):
        self.central_value_net.train()
        loss_total = 0
        for _ in range(self.cv_mini_epochs):
            for idx in range(len(self.cv_dataset)):
                loss_total += self._train_cv_batch(self.cv_dataset[idx])
            if self.cv_normalize_input:
                self.central_value_net.running_mean_std.eval()

        avg_loss = loss_total / (self.cv_mini_epochs * self.cv_num_minibatches)
        if self.writer is not None:
            self.writer.add_scalar('losses/cval_loss', avg_loss, self.frame)
            self.writer.add_scalar('info/cval_lr', self.cv_lr, self.frame)

    def _train_cv_batch(self, batch):
        obs_batch = batch['obs']
        value_preds_batch = batch['old_values']
        returns_batch = batch['returns']

        res = self.central_value_net({'obs': obs_batch, 'is_train': True})
        values = res['values']

        if self.cv_clip_value:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.cv_e_clip, self.cv_e_clip)
            value_losses = (values - returns_batch) ** 2
            value_losses_clipped = (value_pred_clipped - returns_batch) ** 2
            loss = torch.max(value_losses, value_losses_clipped)
        else:
            loss = (returns_batch - values) ** 2

        loss = torch.mean(loss)

        for param in self.central_value_net.parameters():
            param.grad = None
        loss.backward()

        if self.cv_truncate_grads:
            nn.utils.clip_grad_norm_(self.central_value_net.parameters(), self.cv_grad_norm)

        self.cv_optimizer.step()
        return loss.item()

    # ── LR update ────────────────────────────────────────────────────────

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # ── Train epoch ──────────────────────────────────────────────────────

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

        self.model.eval()
        if self.has_central_value:
            self.central_value_net.eval()

        play_time_start = time.time()
        with torch.no_grad():
            orig_batch_dict, ps_extras = self.play_steps()

            if self.expl_type.startswith('mixed_expl') and self.use_others_experience != 'none':
                batch_dict = self.augment_batch_for_mixed_expl(orig_batch_dict, ps_extras)
            else:
                batch_dict = orig_batch_dict
            if self.expl_type.startswith('mixed_expl'):
                batch_dict = shuffle_batch(batch_dict, self.seq_length)

        play_time_end = time.time()
        update_time_start = time.time()

        self.model.train()
        if self.has_central_value:
            self.central_value_net.train()

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)

        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        extra_infos = {
            'on_policy_contrib': [],
            'off_policy_contrib': [],
            'on_policy_grads': [],
            'off_policy_grads': [],
            'entropies': [],
            'mb_intr_rewards': ps_extras['mb_intr_rewards'],
            'mb_extr_rewards': ps_extras['rewards'],
        }

        for mini_ep in range(self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss, extras = self.train_actor_critic(self.dataset[i])
                extra_infos['on_policy_contrib'].append(extras['on_policy_contrib'])
                extra_infos['on_policy_grads'].append(extras['on_policy_grads'])
                extra_infos['off_policy_contrib'].append(extras['off_policy_contrib'])
                extra_infos['off_policy_grads'].append(extras['off_policy_grads'])
                if 'entropies' in extras:
                    extra_infos['entropies'].append(extras['entropies'])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    av_kls = kl
                    if self.is_adaptive_lr:
                        self.last_lr, self.entropy_coef = self._scheduler_update(self.last_lr, self.entropy_coef, av_kls.item())
                    self.update_lr(self.last_lr)

            av_kls = torch.mean(torch.stack(ep_kls))
            if self.schedule_type == 'standard':
                if self.is_adaptive_lr:
                    self.last_lr, self.entropy_coef = self._scheduler_update(self.last_lr, self.entropy_coef, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            if self.normalize_input:
                self.model.running_mean_std.eval()

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul, extra_infos

    def _scheduler_update(self, current_lr, entropy_coef, kl_dist):
        """Adaptive LR scheduler."""
        lr = current_lr
        if kl_dist > 2.0 * self.kl_threshold:
            lr = max(lr / 1.5, 1e-6)
        if kl_dist < 0.5 * self.kl_threshold:
            lr = min(lr * 1.5, 1e-2)
        return lr, entropy_coef

    # ── Checkpointing ────────────────────────────────────────────────────

    def save(self, fn):
        state = self.get_full_state_weights()
        print(f"=> saving checkpoint '{fn}.pth'")
        torch.save({0: state}, fn + '.pth')

    def restore(self, fn, set_epoch=True):
        print(f"=> loading checkpoint '{fn}'")
        checkpoint = torch.load(fn, weights_only=False)
        state = checkpoint[0] if 0 in checkpoint else checkpoint
        self.set_full_state_weights(state, set_epoch=set_epoch)

    def get_full_state_weights(self):
        state = {
            'model': self.model.state_dict(),
            'epoch': self.epoch_num,
            'frame': self.frame,
            'optimizer': self.optimizer.state_dict(),
            'last_mean_rewards': self.last_mean_rewards,
            'rnn_states': self.rnn_states,
            'dones': self.dones,
            'obs': self.obs,
            'current_rewards': self.current_rewards,
            'current_shaped_rewards': self.current_shaped_rewards,
            'current_lengths': self.current_lengths,
        }
        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()
            state['cv_optimizer'] = self.cv_optimizer.state_dict()
        if self.mixed_precision:
            state['scaler'] = self.scaler.state_dict()
        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        self.model.load_state_dict(weights['model'])
        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']
        self.optimizer.load_state_dict(weights['optimizer'])
        self.last_lr = weights['optimizer']['param_groups'][0]['lr']
        self.last_mean_rewards = weights.get('last_mean_rewards', -1e9)

        if self.has_central_value and 'assymetric_vf_nets' in weights:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])
        if 'cv_optimizer' in weights and self.cv_optimizer is not None:
            self.cv_optimizer.load_state_dict(weights['cv_optimizer'])

        SKIP = "current_rewards" not in weights or weights["current_rewards"].shape[0] != self.num_actors
        if not SKIP:
            for key in ['rnn_states', 'dones', 'obs', 'current_rewards', 'current_shaped_rewards', 'current_lengths']:
                if key in weights:
                    setattr(self, key, weights[key])
        else:
            print("Skipping loading of runtime state (shape mismatch)")

        if self.mixed_precision and 'scaler' in weights:
            self.scaler.load_state_dict(weights['scaler'])

    # ── Main training loop ───────────────────────────────────────────────

    def train(self):
        self.init_tensors()

        if self.obs is None:
            self.obs = self.env_reset()
        else:
            self.obs = self._obs_to_tensors(self.obs)

        self.curr_frames = self.batch_size_envs
        total_time = 0

        while True:
            self.epoch_num += 1
            epoch_num = self.epoch_num

            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul, extra_infos = self.train_epoch()
            self.total_time += sum_time
            total_time = self.total_time
            frame = self.frame

            self.dataset.update_values_dict(None)
            should_exit = False

            curr_frames = self.curr_frames
            self.frame += curr_frames

            # Scaled times (play_time includes both env step + inference)
            scaled_play_time = max(play_time, 1e-9)
            scaled_time = max(play_time + update_time, 1e-9)
            step_time_safe = max(step_time, 1e-9)

            # ── Logging ──
            if self.writer is not None:
                # Performance metrics
                self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
                self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
                self.writer.add_scalar('performance/step_fps', curr_frames / step_time_safe, frame)
                self.writer.add_scalar('performance/rl_update_time', update_time, frame)
                self.writer.add_scalar('performance/step_inference_time', play_time, frame)
                self.writer.add_scalar('performance/step_time', step_time, frame)

                # Loss metrics
                a_loss_mean = torch.mean(torch.stack(a_losses)).item() if a_losses else 0
                c_loss_mean = torch.mean(torch.stack(c_losses)).item() if c_losses else 0
                ent_mean = torch.mean(torch.stack(entropies)).item() if entropies else 0
                kl_mean = torch.mean(torch.stack(kls)).item() if kls else 0
                self.writer.add_scalar('losses/a_loss', a_loss_mean, frame)
                self.writer.add_scalar('losses/c_loss', c_loss_mean, frame)
                self.writer.add_scalar('losses/entropy', ent_mean, frame)

                # Info metrics
                self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
                self.writer.add_scalar('info/lr_mul', lr_mul, frame)
                self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
                self.writer.add_scalar('info/kl', kl_mean, frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)

                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch.mean(torch.stack(b_losses)).item(), frame)

            # ── Statistics ──
            if self.print_stats:
                fps_step = curr_frames / step_time_safe
                print(f"\nStatistics:")
                print(f"  fps step                    : {fps_step:,.0f}")
                epoch_str = f"{epoch_num:,.0f}"
                if self.max_epochs != -1:
                    epoch_str += f" / {self.max_epochs:,.0f}"
                print(f"  epoch                       : {epoch_str}")
                print(f"  frames                      : {frame:,.0f}")

            print(f"\nTiming:")
            print(f"  Play time               : {play_time:.3f} s")
            print(f"  Update time             : {update_time:.3f} s")
            print(f"  Time to train epoch     : {sum_time:.3f} s\n")

            # ── Rewards tracking ──
            if self.game_rewards.current_size > 0:
                mean_rewards = self.game_rewards.get_mean()
                mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()
                self.mean_rewards = mean_rewards[0]

                if self.writer is not None:
                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else f'rewards{i}'
                        self.writer.add_scalar(f'{rewards_name}/step', mean_rewards[i], frame)
                        self.writer.add_scalar(f'{rewards_name}/iter', mean_rewards[i], frame)
                        self.writer.add_scalar(f'{rewards_name}/time', mean_rewards[i], frame)
                        self.writer.add_scalar(f'shaped_{rewards_name}/step', mean_shaped_rewards[i], frame)
                        self.writer.add_scalar(f'shaped_{rewards_name}/iter', mean_shaped_rewards[i], frame)
                        self.writer.add_scalar(f'shaped_{rewards_name}/time', mean_shaped_rewards[i], frame)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, frame)

                    # Auxiliary stats (off-policy)
                    self.writer.add_histogram('auxiliary_stats/off_policy_contrib', np.array(extra_infos['off_policy_contrib']), frame)
                    self.writer.add_histogram('auxiliary_stats/on_policy_contrib', np.array(extra_infos['on_policy_contrib']), frame)

                    on_policy_grads = torch.stack(extra_infos['on_policy_grads'])
                    off_policy_grads = torch.stack(extra_infos['off_policy_grads'])
                    self.writer.add_scalar('auxiliary_stats/off_on_grad_similarity',
                                           torch.cosine_similarity(on_policy_grads, off_policy_grads).diag().mean(), frame)
                    self.writer.add_scalar('auxiliary_stats/off_on_relative_grad_norms',
                                           torch.norm(off_policy_grads, dim=-1).mean() / max(torch.norm(on_policy_grads, dim=-1).mean(), 1e-8), frame)

                    # Intrinsic reward per-block logging
                    if extra_infos['mb_intr_rewards'] is not None:
                        if hasattr(self, 'intr_coef_block_size'):
                            num_blocks = self.num_actors // self.intr_coef_block_size
                            for bl in range(num_blocks):
                                self.writer.add_scalar(f'intr_rewards/block_{bl}',
                                                       extra_infos['mb_intr_rewards'][:, self.intr_coef_block_size * bl:self.intr_coef_block_size * (bl + 1)].mean(), frame)
                        else:
                            self.writer.add_scalar('intr_rewards/block_0', extra_infos['mb_intr_rewards'].mean(), frame)
                        self.writer.add_scalar('intr_rewards/extr_rewards', extra_infos['mb_extr_rewards'].mean(), frame)

                    # Per-block entropy logging
                    if extra_infos['entropies']:
                        num_blocks = self.num_actors // self.intr_coef_block_size
                        for bl in range(num_blocks):
                            self.writer.add_scalar(f'intr_rewards/entropy_block_{bl}',
                                                   torch.tensor(extra_infos['entropies'])[:, bl].mean(), frame)

                # Env observer logging (episode_cumulative, successes, direct_info)
                self.env_observer.log_stats(frame, epoch_num, total_time)

                print(f"\nPolicy {self.policy_idx}: mean_reward={mean_rewards[0]:.2f}")

                checkpoint_name = self.experiment_name + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                if self.save_freq > 0:
                    if int(math.sqrt(epoch_num // self.save_freq)) ** 2 == epoch_num // self.save_freq and epoch_num % self.save_freq == 0:
                        self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))
                    if epoch_num % 200 == 0:
                        last_dir = os.path.join(self.experiment_dir, 'last')
                        os.makedirs(last_dir, exist_ok=True)
                        self.save(os.path.join(last_dir, 'model'))

                if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                    print('saving next best rewards: ', mean_rewards)
                    self.last_mean_rewards = mean_rewards[0]
                    self.save(os.path.join(self.nn_dir, self.experiment_name))
                    best_dir = os.path.join(self.experiment_dir, 'best')
                    os.makedirs(best_dir, exist_ok=True)

            if epoch_num >= self.max_epochs and self.max_epochs != -1:
                if self.game_rewards.current_size == 0:
                    print('WARNING: Max epochs reached before any env terminated')
                self.save(os.path.join(self.nn_dir, 'last_' + self.experiment_name + '_ep_' + str(epoch_num)))
                print('MAX EPOCHS NUM!')
                should_exit = True

            if self.frame >= self.max_frames and self.max_frames != -1:
                self.save(os.path.join(self.nn_dir, 'last_' + self.experiment_name + '_frame_' + str(self.frame)))
                print('MAX FRAMES NUM!')
                should_exit = True

            if should_exit:
                return self.last_mean_rewards, epoch_num

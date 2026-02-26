"""Minimal SAPG PPO agent — numerically identical to rl_games for our config.

Ported from:
  - rl_games/common/a2c_common.py (A2CBase, ContinuousA2CBase)
  - rl_games/algos_torch/a2c_continuous.py (A2CAgent)
  - rl_games/algos_torch/central_value.py (CentralValueTrain)
"""

import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim

from rl.buffer import ExperienceBuffer, PPODataset
from rl.debug_logging import DistributedDebugger
from rl.logging import AverageMeter, EnvInfoObserver
from rl.network import ActorNetwork, CriticNetwork
from rl.utils import (
    filter_leader,
    policy_kl,
    rescale_actions,
    shuffle_batch,
    swap_and_flatten01,
)


class SAPGAgent:
    """PPO + SAPG agent that trains on an Isaac Gym vec env."""

    SOFT_BOUND = 1.1
    VALUE_BATCH_CHUNK = 8192

    def __init__(self, vec_env, config, writer=None):
        self.vec_env = vec_env
        self.writer = writer
        self.config = c = config

        # Multi-GPU distributed setup
        self.multi_gpu = c.get('multi_gpu', False)
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.global_rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))

        if self.multi_gpu and self.world_size > 1:
            self.ppo_device = f'cuda:{self.local_rank}'
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group("nccl")
        else:
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
        if self.multi_gpu and self.world_size > 1 and self.global_rank != 0:
            self.print_stats = False
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
            self.lr_max_cap = c.get('lr_max_cap', 1e-2)

        # SAPG exploration
        self.expl_type = c.get('expl_type', 'none')
        self.is_mixed_expl = self.expl_type.startswith('mixed_expl')
        self.use_others_experience = c.get('use_others_experience', 'none')
        self.off_policy_ratio = c.get('off_policy_ratio', 1.0)
        self.ignore_env_boundary = c.get('good_reset_boundary', 0)

        if self.is_mixed_expl:
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

        # Policy index (from experiment name, e.g. "00_default" -> 0)
        exp_name = c.get('full_experiment_name') or c.get('name', '00_default')
        try:
            self.policy_idx = int(exp_name.split('_')[0])
        except ValueError:
            self.policy_idx = 0
        self.experiment_name = exp_name

        # Directories
        self.train_dir = c.get('train_dir', 'runs')
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        os.makedirs(self.nn_dir, exist_ok=True)

        # ── Depth camera config ─────────────────────────────────────────
        self.use_depth = c.get('use_depth_camera', False)
        self.depth_encoder_type = c.get('depth_encoder_type', 'scratch_cnn')
        self.depth_feature_dim = c.get('depth_feature_dim', 512)
        self.freeze_depth_encoder = c.get('freeze_depth_encoder', True)
        self.unfreeze_depth_after_epochs = c.get('unfreeze_depth_after_epochs', -1)
        self.depth_encoder_lr = float(c.get('depth_encoder_lr', 1e-5))
        self.depth_image_height = c.get('depth_image_height', 64)
        self.depth_image_width = c.get('depth_image_width', 64)

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
            use_depth=self.use_depth,
            depth_encoder_type=self.depth_encoder_type,
            depth_feature_dim=self.depth_feature_dim,
            freeze_depth_encoder=self.freeze_depth_encoder,
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

        # torch.compile for faster forward/backward (opt-in via config)
        if c.get('torch_compile', False):
            compile_mode = c.get('torch_compile_mode', 'reduce-overhead')
            self.model.actor_mlp = torch.compile(self.model.actor_mlp, mode=compile_mode)
            if self.has_central_value:
                self.central_value_net.actor_mlp = torch.compile(self.central_value_net.actor_mlp, mode=compile_mode)

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value

        # ── Optimizers ──────────────────────────────────────────────────

        if self.use_depth and self.model.depth_encoder is not None:
            backbone_param_ids = {id(p) for p in self.model.depth_encoder.backbone.parameters()}
            other_params = [p for p in self.model.parameters() if id(p) not in backbone_param_ids]
            backbone_params = list(self.model.depth_encoder.backbone.parameters())
            backbone_lr = 0.0 if self.freeze_depth_encoder else self.depth_encoder_lr
            self.optimizer = optim.Adam([
                {'params': other_params, 'lr': float(self.last_lr)},
                {'params': backbone_params, 'lr': backbone_lr},
            ], eps=1e-08, weight_decay=self.weight_decay)
        else:
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

        # Env-level profiling — logs every N epochs
        self._profile_every = c.get('profile_every', 10)

        # ── Debug logging for multi-GPU crash diagnosis ──
        debug_log_dir = os.path.join(self.experiment_dir, 'debug_logs')
        self.dbg = DistributedDebugger(
            rank=self.local_rank,
            world_size=self.world_size,
            log_dir=debug_log_dir,
            memory_interval_s=60,
        )
        self.dbg.logger.info(
            f"Agent config: num_actors={self.num_actors}, horizon={self.horizon_length}, "
            f"batch_size={self.batch_size}, minibatch_size={self.minibatch_size}, "
            f"num_minibatches={self.num_minibatches}, multi_gpu={self.multi_gpu}, "
            f"mixed_precision={self.mixed_precision}"
        )

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

        intr_rewards = self._zero_intr_rewards

        if self.intr_reward_coef_embd is not None:
            obs['obs'] = torch.cat([obs['obs'], self.intr_reward_coef_embd], dim=1)
            obs['states'] = torch.cat([obs['states'], self.intr_reward_coef_embd], dim=1)

        return obs, rewards, intr_rewards, dones, infos

    # ── Init tensors ─────────────────────────────────────────────────────

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        extra_obs_dim = self.intr_reward_coef_embd.shape[-1] if self.intr_reward_coef_embd is not None else None
        depth_image_shape = (1, self.depth_image_height, self.depth_image_width) if self.use_depth else None
        self.experience_buffer = ExperienceBuffer(
            self.env_info,
            num_actors=self.num_actors,
            horizon_length=self.horizon_length,
            has_central_value=self.has_central_value,
            device=self.ppo_device,
            extra_obs_dim=extra_obs_dim,
            depth_image_shape=depth_image_shape,
        )

        self._zero_intr_rewards = torch.zeros(
            (batch_size, self.value_size), dtype=torch.float32, device=self.ppo_device
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
        if self.use_depth:
            self.tensor_list.append('depth_images')

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
        if self.use_depth and 'depth' in obs:
            input_dict['depth_images'] = obs['depth']
        with torch.no_grad():
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
            if self.use_depth and 'depth' in self.obs:
                self.experience_buffer.update_data('depth_images', n, self.obs['depth'])

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

            chunk = self.VALUE_BATCH_CHUNK
            mb_values = []
            for i in range((flattened_mb_obs.shape[0] + chunk - 1) // chunk):
                mb_values.append(self.get_values(
                    {
                        'obs': flattened_mb_obs[i * chunk:(i + 1) * chunk],
                        'states': flattened_mb_states[i * chunk:(i + 1) * chunk] if mb_states is not None else None,
                    },
                    rnn_states=[s[:, i * chunk:(i + 1) * chunk] for s in flattened_rnn_states] if flattened_rnn_states is not None else None,
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
        if self.use_depth and 'depth_images' in batch_dict:
            dataset_dict['depth_images'] = batch_dict['depth_images']
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
        if self.use_depth and 'depth_images' in input_dict:
            batch_dict['depth_images'] = input_dict['depth_images']

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
                mu_loss_high = torch.clamp_min(mu - self.SOFT_BOUND, 0.0) ** 2
                mu_loss_low = torch.clamp_max(mu + self.SOFT_BOUND, 0.0) ** 2
                b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
            else:
                b_loss = torch.zeros(len(mu), device=self.ppo_device)

            # Entropy coefficient (per-actor for SAPG)
            if self.is_mixed_expl and self.config.get('expl_reward_type') == 'entropy':
                ec_candidates = self.intr_reward_coef[::self.intr_coef_block_size]
                ec_identifiers = self.intr_reward_coef_embd[::self.intr_coef_block_size, 0].reshape(-1, 1)
                ec_indices = torch.argmax((obs_batch[:, -self.intr_reward_coef_embd.shape[1]] == ec_identifiers).float(), dim=0)
                entropy_coef = ec_candidates[ec_indices]
            else:
                entropy_coef = self.entropy_coef

            a_loss = torch.mean(a_loss.unsqueeze(1))
            c_loss = torch.mean(c_loss)
            entropy_loss = torch.mean((entropy_coef * entropy).unsqueeze(1))
            b_loss = torch.mean(b_loss.unsqueeze(1))

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy_loss + b_loss * self.bounds_loss_coef

        for param in self.model.parameters():
            param.grad = None

        self.scaler.scale(loss).backward()

        if self.multi_gpu and self.world_size > 1:
            # All-reduce SCALED gradients first so any rank's inf propagates
            # to all ranks before unscale checks for it.
            self.dbg.log_grad_stats(self.model, "actor_pre_allreduce")
            all_grads = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])
            self.dbg.all_reduce_with_check(all_grads, "actor_gradients")
            offset = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.copy_(all_grads[offset:offset + p.numel()].view_as(p.grad) / self.world_size)
                    offset += p.numel()
            self.scaler.unscale_(self.optimizer)
            if self.truncate_grads:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        elif self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce=True)

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
            }
        else:
            extras = {
                "on_policy_contrib": contrib.mean().item(),
                "off_policy_contrib": 0,
            }

        if self.is_mixed_expl:
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

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
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
        self.scaler.scale(loss).backward()

        if self.multi_gpu and self.world_size > 1:
            self.dbg.log_grad_stats(self.central_value_net, "cv_pre_allreduce")
            all_grads = torch.cat([p.grad.view(-1) for p in self.central_value_net.parameters() if p.grad is not None])
            self.dbg.all_reduce_with_check(all_grads, "cv_gradients")
            offset = 0
            for p in self.central_value_net.parameters():
                if p.grad is not None:
                    p.grad.data.copy_(all_grads[offset:offset + p.numel()].view_as(p.grad) / self.world_size)
                    offset += p.numel()
            self.scaler.unscale_(self.cv_optimizer)
            if self.cv_truncate_grads:
                nn.utils.clip_grad_norm_(self.central_value_net.parameters(), self.cv_grad_norm)
        elif self.cv_truncate_grads:
            self.scaler.unscale_(self.cv_optimizer)
            nn.utils.clip_grad_norm_(self.central_value_net.parameters(), self.cv_grad_norm)

        self.scaler.step(self.cv_optimizer)
        self.scaler.update()
        return loss.item()

    # ── LR update ────────────────────────────────────────────────────────

    def update_lr(self, lr):
        if self.multi_gpu and self.world_size > 1:
            lr_tensor = torch.tensor([lr], device=self.ppo_device)
            self.dbg.broadcast_with_check(lr_tensor, src=0, name="lr_broadcast")
            lr = lr_tensor.item()
        # Only update the main param group (group 0); depth backbone LR (group 1) is separate
        self.optimizer.param_groups[0]['lr'] = lr

    # ── Train epoch ──────────────────────────────────────────────────────

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

        self.model.eval()
        if self.has_central_value:
            self.central_value_net.eval()

        self.dbg.log_phase("play_steps_start")
        play_time_start = time.time()
        with torch.no_grad():
            orig_batch_dict, ps_extras = self.play_steps()

            if self.is_mixed_expl and self.use_others_experience != 'none':
                # Broadcast repeat_idxs from rank 0 so all ranks use same augmentation
                if self.multi_gpu and self.world_size > 1:
                    num_blocks = self.num_actors // self.intr_coef_block_size
                    num_repeat = min(num_blocks, int(self.off_policy_ratio) + 1)
                    if self.global_rank == 0:
                        repeat_idxs = [0] + [int(x) for x in np.random.choice(range(1, num_blocks), num_repeat - 1, replace=False)]
                    else:
                        repeat_idxs = [0] * num_repeat
                    repeat_idxs_t = torch.tensor(repeat_idxs, device=self.ppo_device, dtype=torch.long)
                    self.dbg.broadcast_with_check(repeat_idxs_t, src=0, name="repeat_idxs")
                    repeat_idxs = repeat_idxs_t.tolist()
                    batch_dict = self.augment_batch_for_mixed_expl(orig_batch_dict, ps_extras, repeat_idxs=repeat_idxs)
                else:
                    batch_dict = self.augment_batch_for_mixed_expl(orig_batch_dict, ps_extras)
            else:
                batch_dict = orig_batch_dict
            if self.is_mixed_expl:
                batch_dict = shuffle_batch(batch_dict, self.seq_length)

        play_time_end = time.time()
        self.dbg.log_phase("update_start")
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
            'entropies': [],
            'mb_intr_rewards': ps_extras['mb_intr_rewards'],
            'mb_extr_rewards': ps_extras['rewards'],
        }

        for mini_ep in range(self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss, extras = self.calc_gradients(self.dataset[i])
                extra_infos['on_policy_contrib'].append(extras['on_policy_contrib'])
                extra_infos['off_policy_contrib'].append(extras['off_policy_contrib'])
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
                    if self.multi_gpu and self.world_size > 1:
                        self.dbg.all_reduce_with_check(av_kls, "kl_legacy")
                        av_kls /= self.world_size
                    if self.is_adaptive_lr:
                        self.last_lr, self.entropy_coef = self._scheduler_update(self.last_lr, self.entropy_coef, av_kls.item())
                    self.update_lr(self.last_lr)

            av_kls = torch.mean(torch.stack(ep_kls))
            if self.multi_gpu and self.world_size > 1:
                self.dbg.all_reduce_with_check(av_kls, "kl_standard")
                av_kls /= self.world_size
            if self.schedule_type == 'standard':
                if self.is_adaptive_lr:
                    self.last_lr, self.entropy_coef = self._scheduler_update(self.last_lr, self.entropy_coef, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            if self.normalize_input:
                self.model.running_mean_std.eval()

        # ── Flush env-level CUDA profiler timings ──
        profile_results = self.vec_env.flush_profile()

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        # Attach profiling results so train() can log them
        extra_infos['cuda_profile'] = profile_results

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul, extra_infos

    def _scheduler_update(self, current_lr, entropy_coef, kl_dist):
        """Adaptive LR scheduler."""
        lr = current_lr
        if kl_dist > 2.0 * self.kl_threshold:
            lr = max(lr / 1.5, 1e-6)
        if kl_dist < 0.5 * self.kl_threshold:
            lr = min(lr * 1.5, self.lr_max_cap)
        return lr, entropy_coef

    # ── Checkpointing ────────────────────────────────────────────────────

    def save(self, fn):
        if self.multi_gpu and self.world_size > 1 and self.global_rank != 0:
            return  # Only rank 0 saves
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

    # ── Multi-GPU helpers ───────────────────────────────────────────────

    def _broadcast_params(self, module, module_name="module"):
        """Broadcast all parameters and buffers from rank 0."""
        tensors = [p.data for p in module.parameters()] + [b.data for b in module.buffers()]
        if not tensors:
            return
        # Flatten into one contiguous buffer, broadcast, then copy back
        flat = torch.cat([t.contiguous().view(-1) for t in tensors])
        self.dbg.broadcast_with_check(flat, src=0, name=f"broadcast_params_{module_name}")
        offset = 0
        for t in tensors:
            numel = t.numel()
            t.copy_(flat[offset:offset + numel].view_as(t))
            offset += numel

    # ── Main training loop ───────────────────────────────────────────────

    def train(self):
        self.init_tensors()

        if self.obs is None:
            self.obs = self.env_reset()
        else:
            self.obs = self._obs_to_tensors(self.obs)

        self.curr_frames = self.batch_size_envs
        total_time = 0
        try:
            import wandb
            wandb_run = wandb.run
        except ImportError:
            wandb_run = None

        while True:
            self.epoch_num += 1
            epoch_num = self.epoch_num

            epoch_wallclock_start = time.time()

            self.dbg.log_epoch_start(epoch_num)

            # ── Depth encoder unfreeze scheduler ──
            if (self.use_depth and self.freeze_depth_encoder
                    and self.unfreeze_depth_after_epochs > 0
                    and epoch_num >= self.unfreeze_depth_after_epochs):
                for p in self.model.depth_encoder.backbone.parameters():
                    p.requires_grad = True
                self.optimizer.param_groups[1]['lr'] = self.depth_encoder_lr
                self.freeze_depth_encoder = False
                print(f"[Epoch {epoch_num}] Unfreezing depth encoder backbone, lr={self.depth_encoder_lr}")

            # Broadcast model params from rank 0 at start of each epoch
            if self.multi_gpu and self.world_size > 1:
                self._broadcast_params(self.model, "actor")
                if self.has_central_value:
                    self._broadcast_params(self.central_value_net, "critic")

            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul, extra_infos = self.train_epoch()
            self.total_time += sum_time
            total_time = self.total_time
            frame = self.frame

            self.dataset.update_values_dict(None)
            should_exit = False

            curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
            self.frame += curr_frames

            self.dbg.log_epoch_end(epoch_num)

            epoch_wallclock = max(time.time() - epoch_wallclock_start, 1e-9)
            scaled_play_time = max(play_time, 1e-9)
            scaled_time = max(play_time + update_time, 1e-9)
            step_time_safe = max(step_time, 1e-9)

            # ── Logging (rank 0 only) ──
            is_rank0 = self.global_rank == 0
            if self.writer is not None and is_rank0:
                self.writer.add_scalar('performance/wallclock_fps', curr_frames / epoch_wallclock, frame)
                self.writer.add_scalar('performance/epoch_wallclock_time', epoch_wallclock, frame)
                self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
                self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
                self.writer.add_scalar('performance/step_fps', curr_frames / step_time_safe, frame)
                self.writer.add_scalar('performance/rl_update_time', update_time, frame)
                self.writer.add_scalar('performance/step_inference_time', play_time, frame)
                self.writer.add_scalar('performance/step_time', step_time, frame)
                elapsed_min = total_time / 60.0
                self.writer.add_scalar('performance/frames_vs_minutes', self.frame, int(elapsed_min))
                if wandb_run:
                    wandb_run.log({'performance/elapsed_minutes': elapsed_min,
                                   'performance/frames_vs_minutes': self.frame}, commit=False)

                a_loss_mean = torch.mean(torch.stack(a_losses)).item() if a_losses else 0
                c_loss_mean = torch.mean(torch.stack(c_losses)).item() if c_losses else 0
                ent_mean = torch.mean(torch.stack(entropies)).item() if entropies else 0
                kl_mean = torch.mean(torch.stack(kls)).item() if kls else 0
                self.writer.add_scalar('losses/a_loss', a_loss_mean, frame)
                self.writer.add_scalar('losses/c_loss', c_loss_mean, frame)
                self.writer.add_scalar('losses/entropy', ent_mean, frame)

                self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
                self.writer.add_scalar('info/lr_mul', lr_mul, frame)
                self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
                self.writer.add_scalar('info/kl', kl_mean, frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)

                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch.mean(torch.stack(b_losses)).item(), frame)

            # ── Statistics ──
            if self.print_stats:
                fps_wallclock = curr_frames / epoch_wallclock
                fps_step = curr_frames / step_time_safe
                print(f"\nStatistics:")
                print(f"  fps wallclock               : {fps_wallclock:,.0f}")
                print(f"  fps step                    : {fps_step:,.0f}")
                epoch_str = f"{epoch_num:,.0f}"
                if self.max_epochs != -1:
                    epoch_str += f" / {self.max_epochs:,.0f}"
                print(f"  epoch                       : {epoch_str}")
                print(f"  frames                      : {frame:,.0f}")

            if is_rank0:
                print(f"\nTiming:")
                print(f"  Epoch wallclock         : {epoch_wallclock:.3f} s")
                print(f"  Play time               : {play_time:.3f} s")
                print(f"  Update time             : {update_time:.3f} s")
                print(f"  Overhead                : {epoch_wallclock - play_time - update_time:.3f} s\n")

                # ── CUDA profiler breakdown ──
                cuda_profile = extra_infos.get('cuda_profile', {})
                if cuda_profile and epoch_num % self._profile_every == 0:
                    print("CUDA Profiler (ms):")
                    total_profiled = sum(cuda_profile.values())
                    for name, ms in sorted(cuda_profile.items(), key=lambda x: -x[1]):
                        pct = 100.0 * ms / total_profiled if total_profiled > 0 else 0
                        print(f"  {name:45s}: {ms:9.1f} ms  ({pct:5.1f}%)")
                    print(f"  {'TOTAL PROFILED':45s}: {total_profiled:9.1f} ms")
                    print()

                # Log CUDA profile to wandb every epoch
                if self.writer is not None and cuda_profile:
                    for name, ms in cuda_profile.items():
                        self.writer.add_scalar(f'cuda_profile/{name}', ms, frame)

            # ── Rewards tracking (rank 0 only) ──
            if self.game_rewards.current_size > 0 and is_rank0:
                mean_rewards = self.game_rewards.get_mean()
                mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()
                self.mean_rewards = mean_rewards[0]

                if self.writer is not None:
                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else f'rewards{i}'
                        self.writer.add_scalar(rewards_name, mean_rewards[i], frame)
                        self.writer.add_scalar(f'shaped_{rewards_name}', mean_shaped_rewards[i], frame)

                    self.writer.add_scalar('episode_lengths', mean_lengths, frame)

                    self.writer.add_scalar('auxiliary_stats/off_policy_contrib', np.mean(extra_infos['off_policy_contrib']), frame)
                    self.writer.add_scalar('auxiliary_stats/on_policy_contrib', np.mean(extra_infos['on_policy_contrib']), frame)

                    if extra_infos['mb_intr_rewards'] is not None:
                        if hasattr(self, 'intr_coef_block_size'):
                            num_blocks = self.num_actors // self.intr_coef_block_size
                            for bl in range(num_blocks):
                                self.writer.add_scalar(f'intr_rewards/block_{bl}',
                                                       extra_infos['mb_intr_rewards'][:, self.intr_coef_block_size * bl:self.intr_coef_block_size * (bl + 1)].mean(), frame)
                        else:
                            self.writer.add_scalar('intr_rewards/block_0', extra_infos['mb_intr_rewards'].mean(), frame)
                        self.writer.add_scalar('intr_rewards/extr_rewards', extra_infos['mb_extr_rewards'].mean(), frame)

                    if extra_infos['entropies']:
                        num_blocks = self.num_actors // self.intr_coef_block_size
                        for bl in range(num_blocks):
                            self.writer.add_scalar(f'intr_rewards/entropy_block_{bl}',
                                                   torch.tensor(extra_infos['entropies'])[:, bl].mean(), frame)

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
                if self.multi_gpu and self.world_size > 1:
                    dist.destroy_process_group()
                return self.last_mean_rewards, epoch_num

"""Metrics and logging for SAPG PPO training."""

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from rl.utils import flatten_dict


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

        # Direct info (scalars from env) — single scalar per metric
        for k, v in self.direct_info.items():
            self.writer.add_scalar(k, v, frame)


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

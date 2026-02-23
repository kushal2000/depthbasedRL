"""Data storage and batching for PPO training."""

import torch


class PPODataset:
    def __init__(self, batch_size, minibatch_size, is_rnn, device, seq_length):
        self.is_rnn = is_rnn
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.length = batch_size // minibatch_size
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

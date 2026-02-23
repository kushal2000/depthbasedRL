"""Minimal network definitions for SAPG PPO training.

Ported from rl_games to produce numerically identical output for the specific
config used in AllegroKukaLSTMAsymmetric SAPG training.
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# RunningMeanStd — Welford online normalization
# Ported from rl_games/algos_torch/running_mean_std.py:10-94
# ---------------------------------------------------------------------------
class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05):
        super().__init__()
        self.insize = insize
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(insize, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(insize, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        return new_mean, new_var, tot_count

    def forward(self, input, denorm=False):
        if self.training:
            mean = input.mean(0)
            var = input.var(0)
            self.running_mean, self.running_var, self.count = (
                self._update_mean_var_count_from_moments(
                    self.running_mean, self.running_var, self.count,
                    mean, var, input.size(0),
                )
            )

        current_mean = self.running_mean
        current_var = self.running_var

        if denorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon) * y + current_mean.float()
        else:
            y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
            y = torch.clamp(y, min=-5.0, max=5.0)
        return y


# ---------------------------------------------------------------------------
# LSTMWithDones — zeros hidden at episode boundaries
# Ported from rl_games/common/layers/recurrent.py
# ---------------------------------------------------------------------------
def _multiply_hidden(h, mask):
    if isinstance(h, torch.Tensor):
        return h * mask
    return tuple(_multiply_hidden(v, mask) for v in h)


class LSTMWithDones(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, input, states, done_masks=None, bptt_len=0):
        if done_masks is None:
            return self.rnn(input, states)

        not_dones = 1.0 - done_masks
        has_zeros = (
            (not_dones.squeeze()[1:] == 0.0)
            .any(dim=-1)
            .nonzero()
            .squeeze()
            .cpu()
        )
        if has_zeros.dim() == 0:
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        has_zeros = [0] + has_zeros + [input.size(0)]
        out_batch = []

        for i in range(len(has_zeros) - 1):
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]
            not_done = not_dones[start_idx].float().unsqueeze(0)
            states = _multiply_hidden(states, not_done)
            out, states = self.rnn(input[start_idx:end_idx], states)
            out_batch.append(out)
        return torch.cat(out_batch, dim=0), states


# ---------------------------------------------------------------------------
# ActorNetwork
# ---------------------------------------------------------------------------
class ActorNetwork(nn.Module):
    """Actor network with LSTM, learned exploration embedding, and coef-conditioned sigma.

    Architecture (for the SAPG LSTM asymmetric config):
      obs -> replace coef_id with learned embedding -> normalize -> LSTM -> LayerNorm
           -> MLP [1024,1024,512,512] ELU -> mu head, sigma head (per-coef), value head

    The value head here is for the *actor's* own value estimate (used when
    has_value_loss=True with use_experimental_cv).
    """

    def __init__(self, obs_dim, action_dim, num_actors,
                 coef_ids, coef_id_idx, param_size=32,
                 rnn_units=1024, rnn_layers=1,
                 mlp_units=(1024, 1024, 512, 512)):
        super().__init__()

        self.coef_id_idx = coef_id_idx
        self.register_buffer("coef_ids", coef_ids)

        # Learned embedding that replaces the scalar coef_id in obs
        self.extra_params = nn.Parameter(
            torch.randn(len(coef_ids), param_size, dtype=torch.float32),
            requires_grad=True,
        )

        # Input to LSTM = obs[:coef_id_idx] + embedding (param_size) dims
        lstm_input_dim = coef_id_idx + param_size

        # Observation normalizer (normalizes only first coef_id_idx dims)
        self.running_mean_std = RunningMeanStd(coef_id_idx)

        # Value normalizer
        self.value_mean_std = RunningMeanStd(1)

        # RNN before MLP
        self.rnn = LSTMWithDones(input_size=lstm_input_dim, hidden_size=rnn_units, num_layers=rnn_layers)
        self.layer_norm = nn.LayerNorm(rnn_units)

        # MLP
        layers = []
        in_size = rnn_units
        for units in mlp_units:
            layers.append(nn.Linear(in_size, units))
            layers.append(nn.ELU())
            in_size = units
        self.actor_mlp = nn.Sequential(*layers)

        out_size = mlp_units[-1]

        # Heads
        self.mu = nn.Linear(out_size, action_dim)
        # coef-conditioned sigma: one sigma vector per coef_id
        self.sigma = nn.Parameter(
            torch.zeros(len(coef_ids), action_dim, dtype=torch.float32),
            requires_grad=True,
        )
        self.value = nn.Linear(out_size, 1)

        self.num_seqs = num_actors
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers

        # Weight initialization: default (kaiming_uniform for weights, zeros for bias)
        # then const(0) for sigma — already zeros from init above
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 'default' init in rl_games = PyTorch default (no-op)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        # mu gets 'default' init (no-op), sigma const(0) already done

    def get_default_rnn_state(self):
        return (
            torch.zeros(self.rnn_layers, self.num_seqs, self.rnn_units),
            torch.zeros(self.rnn_layers, self.num_seqs, self.rnn_units),
        )

    def norm_obs(self, observation):
        with torch.no_grad():
            return torch.cat([
                self.running_mean_std(observation[:, :self.coef_id_idx]),
                observation[:, self.coef_id_idx:],
            ], dim=1)

    def denorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, denorm=True)

    def forward(self, input_dict):
        obs = input_dict['obs']
        is_train = input_dict.get('is_train', True)
        states = input_dict.get('rnn_states', None)
        dones = input_dict.get('dones', None)
        seq_length = input_dict.get('seq_length', 1)

        # Replace coef_id with learned embedding
        raw_obs = input_dict.get('raw_obs', obs)  # pre-normalized obs for coef_id lookup
        idxs = (raw_obs[:, self.coef_id_idx].reshape(-1, 1) == self.coef_ids).float().argmax(dim=1)
        obs = torch.cat([obs[:, :self.coef_id_idx], self.extra_params[idxs]], dim=1)

        # Normalize obs
        obs = self.norm_obs(obs)

        # RNN (before MLP)
        batch_size = obs.size(0)
        num_seqs = batch_size // seq_length
        out = obs.reshape(num_seqs, seq_length, -1)

        if len(states) == 1:
            states = states[0]

        out = out.transpose(0, 1)
        if dones is not None:
            dones = dones.reshape(num_seqs, seq_length, -1)
            dones = dones.transpose(0, 1)
        out, states = self.rnn(out, states, dones, 0)
        out = out.transpose(0, 1)
        out = out.contiguous().reshape(out.size(0) * out.size(1), -1)

        out = self.layer_norm(out)
        if type(states) is not tuple:
            states = (states,)

        # MLP
        out = self.actor_mlp(out)

        # Value head
        value = self.value(out)

        # Mu head (identity activation)
        mu = self.mu(out)

        # Sigma head (coef-conditioned, identity activation)
        # Need to re-lookup idxs from the original obs for sigma
        sigma = self.sigma[idxs]

        return mu, sigma, value, states

    @staticmethod
    def neglogp(x, mean, std, logstd):
        return (
            0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
            + logstd.sum(dim=-1)
        )


# ---------------------------------------------------------------------------
# CriticNetwork (Central Value)
# ---------------------------------------------------------------------------
class CriticNetwork(nn.Module):
    """Asymmetric critic that takes privileged state observations.

    Architecture: state -> normalize -> MLP [1024,1024,512,512] ELU -> value head
    No LSTM, no extra_params.
    """

    def __init__(self, state_dim, mlp_units=(1024, 1024, 512, 512)):
        super().__init__()

        self.running_mean_std = RunningMeanStd(state_dim)
        self.value_mean_std = RunningMeanStd(1)

        layers = []
        in_size = state_dim
        for units in mlp_units:
            layers.append(nn.Linear(in_size, units))
            layers.append(nn.ELU())
            in_size = units
        self.actor_mlp = nn.Sequential(*layers)

        out_size = mlp_units[-1]
        self.value = nn.Linear(out_size, 1)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation)

    def denorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, denorm=True)

    def forward(self, input_dict):
        is_train = input_dict.get('is_train', True)
        obs = input_dict['obs']

        obs = self.norm_obs(obs)
        out = self.actor_mlp(obs)
        value = self.value(out)

        if not is_train:
            value = self.denorm_value(value)

        return {'values': value, 'rnn_states': None}

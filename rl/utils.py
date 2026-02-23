"""Pure helper functions for SAPG PPO training."""

from collections import OrderedDict

import torch


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

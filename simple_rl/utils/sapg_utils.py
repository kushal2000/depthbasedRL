from typing import Any, Dict, List

import numpy as np
import torch


def create_sinusoidal_encoding(
    genvec: torch.Tensor, embd_size: int, n: int = 100
) -> torch.Tensor:
    """
    Creates sinusoidal positional encoding for a vector of values.
    Used to encode entropy coefficient values into fixed embeddings.

    Args:
        genvec: 1D tensor of values to encode, shape (N,)
        embd_size: Embedding dimension
        n: Base for positional encoding (default 100)

    Returns:
        Tensor of shape (N, embd_size)
    """
    assert genvec.ndim == 1, f"genvec must be 1D, got {genvec.ndim}"
    N = len(genvec)
    encoding = torch.zeros(N, embd_size, device=genvec.device)
    for i in range(embd_size // 2):
        denom = n ** (2 * i / embd_size)
        encoding[:, 2 * i] = torch.sin(genvec / denom)
        encoding[:, 2 * i + 1] = torch.cos(genvec / denom)
    return encoding


def remove_envs_from_info(infos: Dict[str, Any], num_envs: int) -> Dict[str, Any]:
    for key in list(infos.keys()):
        if isinstance(infos[key], dict):
            infos[key] = remove_envs_from_info(infos[key], num_envs)
        elif isinstance(infos[key], list) or isinstance(
            infos[key], (np.ndarray, torch.Tensor)
        ):
            if key in ["successes", "closest_keypoint_max_dist"]:
                block_size = len(infos[key]) - num_envs
                if len(infos[key]) % block_size == 0:
                    for i in range(len(infos[key]) // block_size):
                        infos[f"{key}_per_block/block_{i}"] = infos[key][
                            i * block_size : (i + 1) * block_size
                        ]
            infos[key] = infos[key][num_envs:]
    return infos


def shuffle_batch(batch_dict: Dict[str, Any], horizon_length: int) -> Dict[str, Any]:
    N = len(batch_dict["returns"])
    assert N % horizon_length == 0, (
        f"N={N} must be divisible by horizon_length={horizon_length}"
    )
    batch_size = N // horizon_length
    device = batch_dict["returns"].device

    indices = torch.randperm(batch_size).to(device).reshape(
        batch_size, 1
    ) * horizon_length + torch.arange(horizon_length).to(device).reshape(
        1, horizon_length
    )
    assert indices.shape == (batch_size, horizon_length), (
        f"indices.shape={indices.shape} must be (batch_size, horizon_length)=({batch_size}, {horizon_length})"
    )

    flattened_indices = indices.reshape(-1)
    for key, val in batch_dict.items():
        if key == "rnn_states":
            if val is None:
                continue
            batch_dict[key] = [s[:, indices[:, 0] // horizon_length] for s in val]
        elif key in ["played_frames", "step_time"]:
            continue
        else:
            batch_dict[key] = val[flattened_indices]
    return batch_dict


def filter_leader(
    val: torch.Tensor, orig_len: int, sampled_block_idxs: List[int], num_blocks: int
) -> torch.Tensor:
    """
    Filters data corresponding to leader i.e. evaluation policy
    Used with mixed_expl
    """
    block_size = orig_len // num_blocks
    filtered_val_list = []
    for i, block_idx in enumerate(sampled_block_idxs):
        if block_idx == 0:  # Leader
            start_idx = i * orig_len
            end_idx = (i + 1) * orig_len
        else:  # Followers
            start_idx = i * orig_len + (block_idx - 1) * block_size
            end_idx = i * orig_len + block_idx * block_size

        filtered_val = val[start_idx:end_idx]
        filtered_val_list.append(filtered_val)

    new_val = torch.cat(filtered_val_list, dim=0)
    return new_val

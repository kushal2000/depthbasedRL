"""Drop-in replacement for deployment/rl_player.py using simple_rl.Player.

Public interface is identical to RlPlayer:
  - get_normalized_action(obs, deterministic_actions=True) -> torch.Tensor
  - reset() -> None

For SAPG checkpoints: appends conditioning_idx=0 (the leader block, integer 0)
to obs instead of the rl_games hack of appending 50.0.

Config format expected:
  The saved config.yaml must have top-level keys:
    train.ppo   -> dict_to_dataclass(d, PpoConfig)
    train.network -> dict_to_dataclass(d, NetworkConfig)
  If the config is in rl_games format (train.params.config), this player
  will fail clearly — use the original RlPlayer for rl_games checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from gym import spaces

from deployment.rl_player_utils import read_cfg


def assert_equals(a, b) -> None:
    assert a == b, f"{a} != {b}"


class RlPlayerSimpleRL:
    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        config_path: str,
        checkpoint_path: Optional[str],
        device: str,
        num_envs: int = 1,
    ) -> None:
        # Import simple_rl after confirming it is on the path.
        # simple_rl/ lives at the repo root; adjust sys.path if needed.
        _repo_root = Path(__file__).resolve().parents[1]
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))

        from simple_rl.agent import PpoConfig
        from simple_rl.player import InferenceConfig, Player, PlayerConfig
        from simple_rl.utils.dict_to_dataclass import dict_to_dataclass
        from simple_rl.utils.network import NetworkConfig

        self.num_observations = num_observations
        self.num_actions = num_actions
        self.device = device
        self.num_envs = num_envs

        # Read the saved config (handles both simple_rl and rl_games formats
        # at the read level; we require simple_rl format below).
        cfg = read_cfg(config_path, device)

        if "train" not in cfg or "ppo" not in cfg["train"]:
            raise ValueError(
                f"Config at '{config_path}' does not look like a simple_rl config "
                f"(expected top-level keys: train.ppo, train.network). "
                f"For rl_games checkpoints, use deployment/rl_player.py instead."
            )

        train_params = cfg["train"]
        network_config = dict_to_dataclass(train_params["network"], NetworkConfig)
        ppo_config = dict_to_dataclass(train_params["ppo"], PpoConfig)
        ppo_config.device = device

        inference_config = ppo_config.to_inference_config()
        player_config = PlayerConfig(deterministic=True)

        # DummyEnv for inference (no actual simulation needed).
        # We need num_envs so Player can set up conditioning_idxs correctly for SAPG.
        from dataclasses import dataclass as _dataclass

        @_dataclass
        class _DeploymentEnv:
            """Minimal env for deployment: spaces + num_envs."""
            observation_space: spaces.Box
            action_space: spaces.Box
            num_envs: int = 1

            def get_env_info(self) -> dict:
                return {
                    "observation_space": self.observation_space,
                    "action_space": self.action_space,
                }

            def set_env_state(self, env_state) -> None:
                # No-op at deployment: env state from training is not restored.
                pass

        dummy_env = _DeploymentEnv(
            observation_space=spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(num_observations,),
                dtype=np.float32,
            ),
            action_space=spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(num_actions,),
                dtype=np.float32,
            ),
            num_envs=num_envs,
        )

        self._player = Player(
            inference_config=inference_config,
            player_config=player_config,
            network_config=network_config,
            env=dummy_env,
        )
        self._player.has_batch_dimension = True
        self._player.batch_size = num_envs  # must be set before init_rnn
        self._player.init_rnn()

        if checkpoint_path is not None:
            print(f"RlPlayerSimpleRL: loading checkpoint from {checkpoint_path}")
            self._player.restore(Path(checkpoint_path))
            print("RlPlayerSimpleRL: checkpoint loaded successfully")

        # SAPG: conditioning_idxs are built inside Player.__init__ based on
        # env.num_envs and inference_config.num_conditionings.
        # At inference time we call get_action() which does NOT append conditioning
        # automatically — we must pass obs that already includes the conditioning idx.
        # For the leader (block 0) we append integer 0.
        self._is_sapg = ppo_config.sapg is not None

    # ── Public interface (identical to RlPlayer) ──────────────────────────

    def get_normalized_action(
        self,
        obs: torch.Tensor,
        deterministic_actions: bool = True,
    ) -> torch.Tensor:
        batch_size = obs.shape[0]
        assert_equals(obs.shape, (batch_size, self.num_observations))

        if self._is_sapg:
            # Append conditioning_idx = 0 (leader block).
            # Unlike rl_games which appends 50.0, simple_rl expects the
            # integer index 0 as the last dimension.
            conditioning = torch.zeros(
                (batch_size, 1), dtype=obs.dtype, device=obs.device
            )
            obs = torch.cat([obs, conditioning], dim=1)

        action = self._player.get_action(
            obs_torch=obs,
            is_deterministic=deterministic_actions,
        )
        action = action.reshape(batch_size, self.num_actions)
        assert_equals(action.shape, (batch_size, self.num_actions))
        return action

    def reset(self) -> None:
        self._player.reset()


def main() -> None:
    """Quick smoke test: load pretrained simple_rl policy and run one forward pass."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    CONFIG_PATH = Path("pretrained_policy/config.yaml")
    CHECKPOINT_PATH = Path("pretrained_policy/model.pth")
    NUM_OBSERVATIONS = 140
    NUM_ACTIONS = 29

    player = RlPlayerSimpleRL(
        num_observations=NUM_OBSERVATIONS,
        num_actions=NUM_ACTIONS,
        config_path=str(CONFIG_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
    )

    obs = torch.zeros(1, NUM_OBSERVATIONS).to(device)
    action = player.get_normalized_action(obs=obs, deterministic_actions=True)
    print(f"obs shape: {obs.shape}  action shape: {action.shape}")
    print(f"action: {action}")


if __name__ == "__main__":
    main()

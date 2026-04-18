"""Test loading the pretrained policy (rl_games SAPG checkpoint) with simple_rl.

The pretrained policy was trained with rl_games + SAPG.
This script tests whether the checkpoint is compatible with simple_rl.Player.

Run:
  cd /home/tylerlum/github_repos/simtoolreal_private
  source .venv/bin/activate
  python test_simple_rl_load.py
"""

import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import yaml


def load_and_inspect_checkpoint(checkpoint_path: Path) -> None:
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(f"Top-level keys: {list(ckpt.keys())}")

    if 0 in ckpt:
        inner = ckpt[0]
        print(f"ckpt[0] keys: {list(inner.keys())}")
        if "model" in inner:
            model_keys = list(inner["model"].keys())
            print(f"  model state_dict keys (first 10): {model_keys[:10]}")
            print(f"  model state_dict key count: {len(model_keys)}")
    else:
        print("WARNING: checkpoint[0] not found — may not be rank-wrapped format")


def try_load_with_simple_rl(config_path: Path, checkpoint_path: Path) -> None:
    from simple_rl.player import InferenceConfig, Player, PlayerConfig
    from simple_rl.utils.dict_to_dataclass import dict_to_dataclass
    from simple_rl.utils.network import NetworkConfig
    from simple_rl.agent import PpoConfig
    from dataclasses import dataclass
    from gym import spaces
    import numpy as np

    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)

    # The pretrained policy was trained with rl_games, so config is in
    # train.params format, NOT simple_rl's train.ppo/network format.
    # We need to hand-craft the simple_rl configs to match the rl_games architecture.

    print("\nAttempting to build simple_rl Player with matching architecture...")

    # From pretrained policy config: LSTM + MLP, normalize_input=True
    # Units: [1024, 1024, 512, 512], LSTM 1024 units, before_mlp=True, layer_norm=True
    network_dict = {
        "mlp": {"units": [1024, 1024, 512, 512]},
        "rnn": {
            "name": "lstm",
            "units": 1024,
            "layers": 1,
            "before_mlp": True,
            "layer_norm": True,
        },
    }
    network_config = dict_to_dataclass(network_dict, NetworkConfig)

    NUM_OBSERVATIONS = 140
    NUM_ACTIONS = 29

    # SAPG from pretrained: fixed_sigma = 'coef_cond' in rl_games config.
    # The pretrained policy has SAPG with some num_conditionings — let's inspect.
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if 0 in ckpt and "model" in ckpt[0]:
        model_keys = list(ckpt[0]["model"].keys())
        conditioning_keys = [k for k in model_keys if "conditioning" in k.lower()]
        print(f"  Conditioning keys in checkpoint: {conditioning_keys}")
        for k in conditioning_keys:
            print(f"    {k}: shape {ckpt[0]['model'][k].shape}")

    # InferenceConfig without SAPG first (plain PPO) to check basic compatibility
    inference_config = InferenceConfig(
        normalize_input=True,
        clip_actions=True,
        device="cpu",
        conditioning_dim=None,
        num_conditionings=None,
    )
    player_config = PlayerConfig(deterministic=True)

    @dataclass
    class _DummyEnv:
        observation_space: spaces.Box
        action_space: spaces.Box
        num_envs: int = 1

        def get_env_info(self) -> dict:
            return {
                "observation_space": self.observation_space,
                "action_space": self.action_space,
            }

    dummy_env = _DummyEnv(
        observation_space=spaces.Box(-np.inf, np.inf, (NUM_OBSERVATIONS,), np.float32),
        action_space=spaces.Box(-1.0, 1.0, (NUM_ACTIONS,), np.float32),
    )

    player = Player(
        inference_config=inference_config,
        player_config=player_config,
        network_config=network_config,
        env=dummy_env,
    )
    player.has_batch_dimension = True
    player.init_rnn()

    print("\n  Attempting player.restore() ...")
    try:
        player.restore(checkpoint_path)
        print("  SUCCESS: Checkpoint loaded into simple_rl Player!")

        obs = torch.zeros(1, NUM_OBSERVATIONS)
        action = player.get_action(obs_torch=obs, is_deterministic=True)
        print(f"  get_action() OK: action shape={action.shape}, action={action[:5]}...")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print(
            "\n  (Expected if rl_games and simple_rl use different parameter name conventions)"
        )
        print("  The pretrained policy uses rl_games; load it with RlPlayer instead.")


def main() -> None:
    config_path = Path("pretrained_policy/config.yaml")
    checkpoint_path = Path("pretrained_policy/model.pth")

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run: python download_pretrained_policy.py")
        return

    load_and_inspect_checkpoint(checkpoint_path)
    try_load_with_simple_rl(config_path, checkpoint_path)


if __name__ == "__main__":
    main()

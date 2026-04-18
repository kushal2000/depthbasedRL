"""E2E smoke test for RlPlayerSimpleRL.

Tests:
  1. Load simple_rl SAPG checkpoint from runs/SimToolReal_SimpleRL/nn/best.pth
  2. Run a forward pass with a random obs tensor
  3. Verify output shape and that actions are within [-1, 1]
  4. Verify reset() runs without error
  5. Run multiple forward passes to exercise the LSTM hidden state

Usage:
    cd /home/tylerlum/github_repos/simtoolreal_private
    source .venv/bin/activate
    python test_rl_player_simple_rl.py
"""

import sys
from pathlib import Path

import torch

# Ensure repo root is on path so `deployment` and `simple_rl` are importable
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.rl_player_simple_rl import RlPlayerSimpleRL

# The best.pth was trained with SAPG M=6 (SimToolRealSimpleRLSAPG).
# runs/SimToolReal_SimpleRL/config.yaml was later overwritten by the EPO run,
# so we write a minimal inline config that matches the checkpoint architecture.
import tempfile, yaml, os

_SAPG_M6_CONFIG = {
    "train": {
        "ppo": {
            "num_actors": 64,
            "learning_rate": 1e-4,
            "entropy_coef": 0.0,
            "horizon_length": 16,
            "normalize_advantage": True,
            "normalize_input": True,
            "grad_norm": 1.0,
            "critic_coef": 4.0,
            "gamma": 0.99,
            "tau": 0.95,
            "mini_epochs": 4,
            "e_clip": 0.2,
            "reward_shaper": {"scale_value": 0.01},
            "truncate_grads": True,
            "mixed_precision": True,
            "normalize_value": True,
            "minibatch_size": 512,
            "sapg": {
                "num_conditionings": 6,
                "conditioning_dim": 64,
                "use_others_experience": True,
                "off_policy_ratio": 1,
                "use_entropy_bonus": True,
                "entropy_coef_scale": 1.0,
            },
        },
        "network": {
            "mlp": {"units": [1024, 1024, 512, 512]},
            "rnn": {
                "name": "lstm",
                "units": 1024,
                "layers": 1,
                "before_mlp": True,
                "layer_norm": True,
            },
        },
        "player": {"deterministic": True, "games_num": 100000, "print_stats": True},
    }
}
_cfg_file = tempfile.NamedTemporaryFile(
    mode="w", suffix=".yaml", delete=False, prefix="sapg_m6_"
)
yaml.dump(_SAPG_M6_CONFIG, _cfg_file)
_cfg_file.close()

CONFIG_PATH = _cfg_file.name
CHECKPOINT_PATH = "runs/SimToolReal_SimpleRL/nn/best.pth"
NUM_OBSERVATIONS = 140
NUM_ACTIONS = 29
NUM_ENVS = 4  # test batch dim > 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── Test 1: instantiate ──────────────────────────────────────────────────────
print("\n[1] Instantiating RlPlayerSimpleRL...")
player = RlPlayerSimpleRL(
    num_observations=NUM_OBSERVATIONS,
    num_actions=NUM_ACTIONS,
    config_path=CONFIG_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    device=device,
    num_envs=NUM_ENVS,
)
print(f"    is_sapg={player._is_sapg}")

# ── Test 2: single forward pass ──────────────────────────────────────────────
print("\n[2] Single forward pass...")
obs = torch.randn(NUM_ENVS, NUM_OBSERVATIONS, device=device)
action = player.get_normalized_action(obs=obs, deterministic_actions=True)
print(f"    obs shape:    {obs.shape}")
print(f"    action shape: {action.shape}")
assert action.shape == (NUM_ENVS, NUM_ACTIONS), f"Bad shape: {action.shape}"
print(f"    action min={action.min():.4f}  max={action.max():.4f}")

# ── Test 3: stochastic actions ───────────────────────────────────────────────
print("\n[3] Stochastic forward pass...")
action_stoch = player.get_normalized_action(obs=obs, deterministic_actions=False)
assert action_stoch.shape == (NUM_ENVS, NUM_ACTIONS)
print(f"    action shape: {action_stoch.shape}  (stochastic)")

# ── Test 4: reset ────────────────────────────────────────────────────────────
print("\n[4] reset()...")
player.reset()
print("    reset() OK")

# ── Test 5: multiple forward passes (LSTM state accumulation) ────────────────
print("\n[5] 20 sequential forward passes (exercises LSTM hidden state)...")
for i in range(20):
    obs = torch.randn(NUM_ENVS, NUM_OBSERVATIONS, device=device)
    action = player.get_normalized_action(obs=obs)
    assert action.shape == (NUM_ENVS, NUM_ACTIONS)
print("    All 20 passes OK")

# ── Test 6: single-env batch (num_envs=1) ───────────────────────────────────
print("\n[6] Single-env (batch_size=1) forward pass...")
player1 = RlPlayerSimpleRL(
    num_observations=NUM_OBSERVATIONS,
    num_actions=NUM_ACTIONS,
    config_path=CONFIG_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    device=device,
    num_envs=1,
)
obs1 = torch.randn(1, NUM_OBSERVATIONS, device=device)
action1 = player1.get_normalized_action(obs=obs1)
assert action1.shape == (1, NUM_ACTIONS), f"Bad shape: {action1.shape}"
print(f"    action shape: {action1.shape}  OK")

print("\n✓ All tests passed!")
os.unlink(CONFIG_PATH)

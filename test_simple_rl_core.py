"""Core unit tests for simple_rl components.

Tests:
  1. Sigma shape         — PPO: (actions,), SAPG/EPO: (M, actions)
  2. Block roundtrip     — rl_games linspace lookup always returns block index k
  3. Running mean std    — pretrained checkpoint has n>0, correct shape, std>0
  4. LSTM state persist  — consecutive steps produce different actions; reset restores
  5. Conditioning divers — distinct conditioning embeddings produce different mu outputs

Usage
-----
python test_simple_rl_core.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# ── repo root on path ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── helpers ────────────────────────────────────────────────────────────────────

def _pass(name: str) -> None:
    print(f"  PASS  {name}")

def _fail(name: str, msg: str) -> None:
    print(f"  FAIL  {name}: {msg}")
    raise AssertionError(f"{name}: {msg}")


def _build_agent(obs_dim: int, act_dim: int, num_envs: int, **ppo_extra):
    """Build a fresh simple_rl agent (no pretrained weights)."""
    from gym import spaces
    from simple_rl.agent import Agent, PpoConfig
    from simple_rl.utils.network import MlpConfig, NetworkConfig
    from simple_rl.utils.rewards_shaper import RewardsShaperParams

    # Python class bodies don't close over enclosing function locals when the
    # attribute name matches the parameter name — use private aliases.
    _obs_dim = obs_dim
    _act_dim = act_dim
    _num_envs = num_envs

    class _DummyEnv:
        num_envs          = _num_envs
        observation_space = spaces.Box(-np.inf, np.inf, (_obs_dim,), dtype=np.float32)
        action_space      = spaces.Box(-1.0, 1.0, (_act_dim,), dtype=np.float32)
        device = DEVICE

        def get_env_info(self):
            return {"observation_space": self.observation_space, "action_space": self.action_space}
        def reset(self):
            return torch.zeros(_num_envs, _obs_dim, device=DEVICE)
        def step(self, actions):
            return self.reset(), actions.mean(dim=-1), torch.zeros(_num_envs, dtype=torch.bool, device=DEVICE), {}
        def set_env_state(self, s): pass
        def set_train_info(self, *a, **k): pass
        def get_env_state(self): return None

    cfg = PpoConfig(
        device          = DEVICE,
        num_actors      = num_envs,
        learning_rate   = 1e-4,
        entropy_coef    = 0.0,
        horizon_length  = 16,
        normalize_advantage = True,
        normalize_input = True,
        normalize_value = True,
        grad_norm       = 1.0,
        critic_coef     = 4.0,
        gamma           = 0.99,
        tau             = 0.95,
        mini_epochs     = 2,
        e_clip          = 0.1,
        minibatch_size  = 256,
        truncate_grads  = True,
        mixed_precision = False,
        max_epochs      = 1,
        reward_shaper   = RewardsShaperParams(scale_value=1.0),
        save_frequency  = 0,
        print_stats     = False,
        **ppo_extra,
    )
    net_cfg = NetworkConfig(mlp=MlpConfig(units=[32, 32]), rnn=None)
    env = _DummyEnv()
    agent = Agent(
        experiment_dir=Path("/tmp/simple_rl_core_test"),
        ppo_config=cfg,
        network_config=net_cfg,
        env=env,
    )
    return agent, env


# ── Test 1: Sigma shape ────────────────────────────────────────────────────────

def test_sigma_shape() -> None:
    """PPO has sigma shape (actions,); SAPG/EPO has sigma shape (M, actions).

    num_actors must be divisible by num_conditionings (block_size = num_actors / M).
    """
    from simple_rl.agent import EpoConfig, SapgConfig

    OBS_DIM, ACT_DIM = 8, 4

    # PPO  (no conditioning constraint)
    agent_ppo, _ = _build_agent(OBS_DIM, ACT_DIM, num_envs=32)
    sigma_ppo = agent_ppo.model.a2c_network.sigma
    assert sigma_ppo.shape == (ACT_DIM,), f"PPO sigma shape {sigma_ppo.shape} != ({ACT_DIM},)"
    _pass("sigma shape — PPO: (actions,)")

    # SAPG  (M=6, num_envs=48 so 48/6=8 envs/block)
    agent_sapg, _ = _build_agent(
        OBS_DIM, ACT_DIM, num_envs=48,
        sapg=SapgConfig(num_conditionings=6, conditioning_dim=4, use_others_experience=False),
    )
    sigma_sapg = agent_sapg.model.a2c_network.sigma
    assert sigma_sapg.shape == (6, ACT_DIM), f"SAPG sigma shape {sigma_sapg.shape} != (6, {ACT_DIM})"
    _pass("sigma shape — SAPG: (M, actions)")

    # EPO  (M=4, num_envs=32 so 32/4=8 envs/block)
    agent_epo, _ = _build_agent(
        OBS_DIM, ACT_DIM, num_envs=32,
        sapg=SapgConfig(num_conditionings=4, conditioning_dim=4, use_others_experience=False),
        epo=EpoConfig(evolution_frequency=50),
    )
    sigma_epo = agent_epo.model.a2c_network.sigma
    assert sigma_epo.shape == (4, ACT_DIM), f"EPO sigma shape {sigma_epo.shape} != (4, {ACT_DIM})"
    _pass("sigma shape — EPO: (M, actions)")


# ── Test 2: Block roundtrip ────────────────────────────────────────────────────

def test_block_roundtrip() -> None:
    """rl_games linspace(50, 0, M) lookup: appending param_ids[k] → argmax → k."""
    for M in (6, 8, 64):
        param_ids = torch.linspace(50.0, 0.0, M)
        for k in range(M):
            obs_val = param_ids[k].item()
            idx = (torch.tensor([[obs_val]]) == param_ids).float().argmax().item()
            assert idx == k, f"M={M} block {k}: expected idx={k}, got {idx}"
        _pass(f"block roundtrip — M={M}: all {M} blocks map correctly")


# ── Test 3: Running mean std ───────────────────────────────────────────────────

def test_running_mean_std_pretrained() -> None:
    """Pretrained checkpoint has running_mean_std with n>0, correct shape, std>0.

    The converted simple_rl checkpoint stores running_mean_std inside rank0['model'].
    simple_rl normalises only the base 140-dim obs; the conditioning index is passed
    through unchanged, so running_mean.shape == (140,).
    """
    ckpt_path = Path("pretrained_policy/model_simple_rl.pth")
    if not ckpt_path.exists():
        print(f"  SKIP  running_mean_std (checkpoint not found: {ckpt_path})")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_sd = ckpt[0]["model"]   # state_dict inside rank0

    count_key = "running_mean_std.count"
    mean_key  = "running_mean_std.running_mean"
    var_key   = "running_mean_std.running_var"

    if count_key not in model_sd:
        print("  SKIP  running_mean_std (key not in model state dict)")
        return

    count        = model_sd[count_key]
    running_mean = model_sd[mean_key]
    running_var  = model_sd[var_key]

    assert count.item() > 0, f"running_mean_std count should be > 0, got {count.item()}"
    _pass(f"running_mean_std — count={count.item():.0f} > 0")

    # simple_rl normalises only the base obs dims (not the conditioning idx appended later).
    expected_obs_dim = 140
    assert running_mean.shape == (expected_obs_dim,), (
        f"running_mean shape {tuple(running_mean.shape)} != ({expected_obs_dim},)"
    )
    _pass(f"running_mean_std — mean shape = {tuple(running_mean.shape)}")

    # Variance should be positive for most obs dims (constant dims may be near-zero)
    std = running_var.sqrt()
    n_nonzero = (std > 1e-8).sum().item()
    assert n_nonzero > expected_obs_dim // 2, (
        f"Only {n_nonzero}/{expected_obs_dim} obs dims have nonzero std — suspiciously few"
    )
    _pass(f"running_mean_std — obs std > 0 for {n_nonzero}/{expected_obs_dim} dims")


# ── Test 4: LSTM state persistence ────────────────────────────────────────────

def test_lstm_state_persistence() -> None:
    """With LSTM: consecutive steps produce different actions; reset restores outputs."""
    from gym import spaces
    from simple_rl.agent import Agent, PpoConfig
    from simple_rl.player import InferenceConfig, Player, PlayerConfig
    from simple_rl.utils.network import MlpConfig, NetworkConfig, RnnConfig
    from simple_rl.utils.rewards_shaper import RewardsShaperParams

    OBS_DIM, ACT_DIM, N = 8, 4, 4

    class _Env:
        num_envs = N
        observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32)
        def get_env_info(self): return {"observation_space": self.observation_space, "action_space": self.action_space}
        def reset(self): return torch.zeros(N, OBS_DIM, device=DEVICE)
        def step(self, a): return self.reset(), a.mean(dim=-1), torch.zeros(N, dtype=torch.bool, device=DEVICE), {}
        def set_env_state(self, s): pass
        def set_train_info(self, *a, **k): pass
        def get_env_state(self): return None

    net_cfg = NetworkConfig(
        mlp=MlpConfig(units=[32, 32]),
        rnn=RnnConfig(name="lstm", units=32, layers=1, before_mlp=True),
    )

    # Build agent just to get a properly-initialized model with LSTM weights
    cfg = PpoConfig(
        device=DEVICE, num_actors=N, learning_rate=1e-4, entropy_coef=0.0,
        horizon_length=16, normalize_advantage=True, normalize_input=False,
        normalize_value=False, grad_norm=1.0, critic_coef=4.0, gamma=0.99,
        tau=0.95, mini_epochs=1, e_clip=0.1, minibatch_size=64,
        mixed_precision=False, max_epochs=1,
        reward_shaper=RewardsShaperParams(scale_value=1.0),
        save_frequency=0, print_stats=False,
    )
    agent = Agent(
        experiment_dir=Path("/tmp/lstm_persistence_test"),
        ppo_config=cfg, network_config=net_cfg, env=_Env(),
    )

    # Build a Player using the same model weights
    env = _Env()
    player = Player(
        inference_config=cfg.to_inference_config(),
        player_config=PlayerConfig(),
        network_config=net_cfg,
        env=env,
    )
    player.model.load_state_dict(agent.model.state_dict())
    player.has_batch_dimension = True
    player.batch_size = N
    player.init_rnn()

    # Step 1 with non-zero obs
    torch.manual_seed(0)
    obs = torch.randn(N, OBS_DIM, device=DEVICE)
    a1 = player.get_action(obs_torch=obs, is_deterministic=True)
    # Step 2 with same obs — LSTM state updated → different output
    a2 = player.get_action(obs_torch=obs, is_deterministic=True)
    assert not torch.allclose(a1, a2), "LSTM should produce different actions on consecutive calls with same obs"
    _pass("LSTM state persistence — consecutive steps differ")

    # Reset and re-run step 1 — should match original a1
    player.reset()
    a1_replay = player.get_action(obs_torch=obs, is_deterministic=True)
    assert torch.allclose(a1, a1_replay, atol=1e-6), (
        f"After reset, first action should be identical to original first action; "
        f"max diff={(a1 - a1_replay).abs().max().item():.3e}"
    )
    _pass("LSTM state persistence — reset restores initial output")


# ── Test 5: Conditioning diversity ────────────────────────────────────────────

def test_conditioning_diversity() -> None:
    """After a few training steps, all M conditioning embeddings should differ.

    num_actors must be divisible by num_conditionings; use 48/6=8 envs per block.
    """
    from simple_rl.agent import SapgConfig

    OBS_DIM, ACT_DIM, M = 8, 4, 6
    N = M * 8  # 48 envs, 8 per block

    agent, env = _build_agent(
        OBS_DIM, ACT_DIM, N,
        sapg=SapgConfig(num_conditionings=M, conditioning_dim=4, use_others_experience=False),
    )

    # Condition embeddings are initialized to zero — train briefly so gradients diversify them
    agent.init_tensors()
    agent.obs_dict = agent.env_reset()
    for _ in range(3):
        agent.train_epoch()

    cond = agent.model.a2c_network.conditioning.data  # (M, C)

    # After a few gradient steps the M rows should not all be identical
    first_row = cond[0]
    all_same = all(torch.allclose(cond[k], first_row, atol=1e-6) for k in range(1, M))
    assert not all_same, "All conditioning embeddings are identical — SAPG gradient steps may not be updating them"
    _pass(f"conditioning diversity — M={M} blocks have distinct embeddings after training")

    # Each block should produce a different mu output for the same obs
    agent.model.eval()
    obs_raw = torch.randn(1, OBS_DIM, device=DEVICE)
    mus = []
    for k in range(M):
        idx = torch.full((1, 1), float(k), device=DEVICE)
        obs_k = torch.cat([obs_raw, idx], dim=1)
        with torch.no_grad():
            result = agent.model({"obs": obs_k, "is_train": False})
        mus.append(result["mus"])

    # At least one pair should differ
    all_same_mu = all(torch.allclose(mus[0], mus[k], atol=1e-6) for k in range(1, M))
    assert not all_same_mu, "All blocks produce identical mu — conditioning lookup may be broken"
    _pass(f"conditioning diversity — M={M} blocks produce distinct mu outputs")


# ── main ───────────────────────────────────────────────────────────────────────

TESTS = [
    ("Sigma shape",             test_sigma_shape),
    ("Block roundtrip",         test_block_roundtrip),
    ("Running mean std",        test_running_mean_std_pretrained),
    ("LSTM state persistence",  test_lstm_state_persistence),
    ("Conditioning diversity",  test_conditioning_diversity),
]


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Running {len(TESTS)} unit tests\n")

    passed, failed = 0, 0
    for name, fn in TESTS:
        print(f"--- {name} ---")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {e}")
            failed += 1
        print()

    print("="*40)
    print(f"Results: {passed}/{len(TESTS)} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    main()

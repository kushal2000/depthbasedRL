"""
Multi-seed comparison: simple_rl vs rl_games across PPO, SAPG, and EPO.
Runs 10 seeds and plots mean ± std reward curves for all 8 series:
  slRL-PPO, RG-PPO,
  slRL-SAPG blk0, slRL-SAPG blk(M-1),
  RG-SAPG   blk0, RG-SAPG   blk(M-1),
  slRL-EPO  blk0, slRL-EPO  blk(M-1).
"""

import sys
sys.path.insert(0, ".")

import torch
from pathlib import Path
from typing import List, Tuple

# ── Import shared helpers from the main test ──────────────────────────────────
from test_rlgames_vs_simple_rl_dummy_env import (
    DummyEnv,
    OBS_DIM, ACT_DIM, DEVICE, HORIZON, MINIBATCH, MINI_EPOCHS, MAX_EPOCHS, E_CLIP,
    _base_ppo_cfg,
    build_agent,
    _train_and_report,
    _make_rg_vec_env,
    _rg_base_config,
    _build_rg_agent,
    _rg_train_and_report,
)
from simple_rl.agent import EpoConfig, SapgConfig

# ── env / block size constants ────────────────────────────────────────────────
PPO_ENVS   = 48          # same as SAPG for fair env-count comparison

SAPG_ENVS  = 48
SAPG_M     = 6           # num SAPG blocks
SAPG_BLOCK = 8           # 48 / 6

EPO_ENVS   = 64          # must be divisible by EPO_M
EPO_M      = 8           # num EPO blocks  (8 * 8 = 64)
EPO_FREQ   = 10          # evolutionary update every 10 epochs


# ── tiny env subclasses ───────────────────────────────────────────────────────

class _Env48(DummyEnv):
    num_envs = 48
    def reset(self):
        return torch.zeros(48, OBS_DIM, device=self.device)
    def step(self, actions):
        return (torch.zeros(48, OBS_DIM, device=self.device),
                actions.mean(dim=-1),
                torch.zeros(48, dtype=torch.bool, device=self.device), {})


class _Env64(DummyEnv):
    num_envs = 64
    def reset(self):
        return torch.zeros(64, OBS_DIM, device=self.device)
    def step(self, actions):
        return (torch.zeros(64, OBS_DIM, device=self.device),
                actions.mean(dim=-1),
                torch.zeros(64, dtype=torch.bool, device=self.device), {})


# ── per-seed runners ──────────────────────────────────────────────────────────

def run_sl_ppo(seed: int) -> List[float]:
    """simple_rl PPO (48 envs). Returns reward history."""
    torch.manual_seed(seed)
    env = _Env48(DEVICE)
    cfg = _base_ppo_cfg(num_envs=PPO_ENVS)
    agent = build_agent(env, cfg, f"sl_ppo_{seed}")
    hist, _ = _train_and_report(agent, env, f"slRL-PPO[s={seed}]")
    return hist


def run_rg_ppo(seed: int) -> List[float]:
    """rl_games PPO (48 envs). Returns reward history."""
    torch.manual_seed(seed)
    env = _Env48(DEVICE)
    vec_env = _make_rg_vec_env(env)
    config = _rg_base_config(PPO_ENVS, f"ppo_s{seed}")
    config["params"]["seed"] = seed
    agent = _build_rg_agent(vec_env, config)
    hist, _ = _rg_train_and_report(agent, f"RG-PPO[s={seed}]", is_sapg=False)
    return hist


def run_sl_sapg(seed: int) -> Tuple[List[float], List[float]]:
    """simple_rl SAPG (48 envs, M=6). Returns (hist_blk0, hist_blkM)."""
    torch.manual_seed(seed)
    env = _Env48(DEVICE)
    cfg = _base_ppo_cfg(
        num_envs=SAPG_ENVS,
        sapg=SapgConfig(
            num_conditionings      = SAPG_M,
            conditioning_dim       = 32,
            use_others_experience  = False,
            use_entropy_bonus      = True,
            entropy_coef_scale     = 0.005,
        ),
    )
    agent = build_agent(env, cfg, f"sl_sapg_{seed}")
    return _train_and_report(agent, env, f"slRL-SAPG[s={seed}]")


def run_rg_sapg(seed: int) -> Tuple[List[float], List[float]]:
    """rl_games SAPG (48 envs, M=6). Returns (hist_blk0, hist_blkM)."""
    torch.manual_seed(seed)
    env = _Env48(DEVICE)
    vec_env = _make_rg_vec_env(env)
    config = _rg_base_config(
        SAPG_ENVS, f"sapg_s{seed}",
        fixed_sigma            = "coef_cond",
        expl_type              = "mixed_expl_learn_param",
        expl_reward_coef_scale = 0.005,
        expl_reward_type       = "entropy",
        expl_coef_block_size   = SAPG_BLOCK,
    )
    config["params"]["seed"] = seed
    agent = _build_rg_agent(vec_env, config)
    return _rg_train_and_report(agent, f"RG-SAPG[s={seed}]", is_sapg=True)


def run_sl_epo(seed: int) -> Tuple[List[float], List[float]]:
    """simple_rl EPO (64 envs, M=8). Returns (hist_blk0, hist_blkM)."""
    torch.manual_seed(seed)
    env = _Env64(DEVICE)
    cfg = _base_ppo_cfg(
        num_envs=EPO_ENVS,
        sapg=SapgConfig(
            num_conditionings      = EPO_M,
            conditioning_dim       = 32,
            use_others_experience  = False,
            use_entropy_bonus      = True,
            entropy_coef_scale     = 0.005,
        ),
        epo=EpoConfig(
            evolution_frequency  = EPO_FREQ,
            evolution_kill_ratio = 0.3,
        ),
    )
    agent = build_agent(env, cfg, f"sl_epo_{seed}")
    return _train_and_report(agent, env, f"slRL-EPO[s={seed}]", epo_freq=EPO_FREQ)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Multi-seed comparison: simple_rl vs rl_games (PPO / SAPG / EPO)")
    print(f"  PPO:  {PPO_ENVS} envs")
    print(f"  SAPG: {SAPG_ENVS} envs  M={SAPG_M}  block_size={SAPG_BLOCK}")
    print(f"  EPO:  {EPO_ENVS} envs   M={EPO_M}   epo_freq={EPO_FREQ}")
    print(f"  MAX_EPOCHS={MAX_EPOCHS}  HORIZON={HORIZON}")
    print()

    CHECKPOINTS = [9, 19, 29]  # 0-indexed → epochs 10, 20, 30

    rows = []
    for seed in range(10):
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        sl_ppo           = run_sl_ppo(seed)
        rg_ppo           = run_rg_ppo(seed)
        sl_s0, sl_sfol   = run_sl_sapg(seed)
        rg_s0, rg_sfol   = run_rg_sapg(seed)
        sl_e0, sl_efol   = run_sl_epo(seed)

        rows.append(dict(
            seed=seed,
            # full histories
            sl_ppo=sl_ppo, rg_ppo=rg_ppo,
            sl_s0=sl_s0,   sl_sfol=sl_sfol,
            rg_s0=rg_s0,   rg_sfol=rg_sfol,
            sl_e0=sl_e0,   sl_efol=sl_efol,
            # absolute at checkpoints
            sl_ppo_abs=[sl_ppo[i]  for i in CHECKPOINTS],
            rg_ppo_abs=[rg_ppo[i]  for i in CHECKPOINTS],
            sl_s0_abs=[sl_s0[i]    for i in CHECKPOINTS],
            sl_sfol_abs=[sl_sfol[i] for i in CHECKPOINTS],
            rg_s0_abs=[rg_s0[i]    for i in CHECKPOINTS],
            rg_sfol_abs=[rg_sfol[i] for i in CHECKPOINTS],
            sl_e0_abs=[sl_e0[i]    for i in CHECKPOINTS],
            sl_efol_abs=[sl_efol[i] for i in CHECKPOINTS],
        ))

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "="*80)
    print("MULTI-SEED SUMMARY")
    print("="*80)

    def _avg(key, ci=None):
        if ci is None:
            return sum(r[key][-1] - r[key][0] for r in rows) / len(rows)
        return sum(r[key][ci] for r in rows) / len(rows)

    # mean absolute reward table
    print(f"\n  Mean absolute reward across {len(rows)} seeds:")
    hdr = (f"  {'Epoch':>6}  {'slRL-PPO':>10}  {'RG-PPO':>8}  "
           f"{'slRL-S0':>9}  {'slRL-Sf':>9}  {'RG-S0':>7}  {'RG-Sf':>7}  "
           f"{'slRL-E0':>9}  {'slRL-Ef':>9}")
    print(hdr)
    for ci, ep in enumerate([10, 20, 30]):
        print(
            f"  {ep:>6}  "
            f"{_avg('sl_ppo_abs', ci):>+10.3f}  {_avg('rg_ppo_abs', ci):>+8.3f}  "
            f"{_avg('sl_s0_abs',  ci):>+9.3f}  {_avg('sl_sfol_abs', ci):>+9.3f}  "
            f"{_avg('rg_s0_abs',  ci):>+7.3f}  {_avg('rg_sfol_abs', ci):>+7.3f}  "
            f"{_avg('sl_e0_abs',  ci):>+9.3f}  {_avg('sl_efol_abs', ci):>+9.3f}"
        )

    # avg improvement delta
    print(f"\n  Avg improvement delta (epoch30 − epoch1):")
    print(f"    slRL-PPO={_avg('sl_ppo'):+.3f}  RG-PPO={_avg('rg_ppo'):+.3f}  "
          f"(diff={_avg('rg_ppo')-_avg('sl_ppo'):+.3f})")
    print(f"    slRL-SAPG blk0={_avg('sl_s0'):+.3f}  RG-SAPG blk0={_avg('rg_s0'):+.3f}  "
          f"(diff={_avg('rg_s0')-_avg('sl_s0'):+.3f})")
    print(f"    slRL-SAPG blkM={_avg('sl_sfol'):+.3f}  RG-SAPG blkM={_avg('rg_sfol'):+.3f}  "
          f"(diff={_avg('rg_sfol')-_avg('sl_sfol'):+.3f})")
    print(f"    slRL-EPO  blk0={_avg('sl_e0'):+.3f}  slRL-EPO blkM={_avg('sl_efol'):+.3f}")

    _plot_reward_curves(rows)


def _plot_reward_curves(rows: list) -> None:
    """Plot mean ± 1 std reward curves for all 8 series."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n  [plot] matplotlib/numpy not available — skipping")
        return

    n_epochs = len(rows[0]["sl_ppo"])
    epochs = np.arange(1, n_epochs + 1)

    # (label, row_key, color, linestyle)
    methods = [
        ("slRL-PPO",            "sl_ppo",  "#1f77b4", "-"),
        ("RG-PPO",              "rg_ppo",  "#aec7e8", "--"),
        ("slRL-SAPG blk0",      "sl_s0",   "#ff7f0e", "-"),
        (f"slRL-SAPG blk{SAPG_M-1}", "sl_sfol", "#ffbb78", "--"),
        ("RG-SAPG blk0",        "rg_s0",   "#d62728", "-"),
        (f"RG-SAPG blk{SAPG_M-1}",   "rg_sfol", "#ff9896", "--"),
        ("slRL-EPO blk0",       "sl_e0",   "#2ca02c", "-"),
        (f"slRL-EPO blk{EPO_M-1}",   "sl_efol", "#98df8a", "--"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    for label, key, color, ls in methods:
        mat  = np.array([r[key] for r in rows])   # (n_seeds, n_epochs)
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)
        ax.plot(epochs, mean, label=label, color=color, linestyle=ls, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Estimated Reward", fontsize=12)
    ax.set_title(
        f"PPO / SAPG / EPO reward curves: simple_rl vs rl_games\n"
        f"mean ± 1 std across {len(rows)} seeds  "
        f"(solid=blk0 leader, dashed=blkM follower)",
        fontsize=11,
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = "/tmp/sapg_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  [plot] saved → {out}")


if __name__ == "__main__":
    main()

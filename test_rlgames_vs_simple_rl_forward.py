"""Tier-1 equivalence check: same pretrained weights → same actions for all SAPG blocks.

rl_games SAPG and simple_rl SAPG both load the same underlying weights
(one key rename: extra_params → conditioning).  When fed identical 140-dim
observations they should produce bit-for-bit identical actions because:

  rl_games forward:
    idxs  = argmax((obs[:,140] == param_ids).float())   # param_ids[k] matches → k
    obs   = cat([obs[:, :140], extra_params[idxs]])     # 172-dim LSTM input

  simple_rl forward:
    conditioning_idxs = obs[:, -1].long()               # k
    obs = cat([obs[:, :-1], conditioning[idxs]])        # 172-dim LSTM input

Both strip the idx dimension and replace it with the same 32-dim embedding
(extra_params[k] == conditioning[k]).  The LSTM therefore sees identical input.

Block ID mapping (M=6, SAPG):
  rl_games:   param_ids = linspace(50.0, 0.0, M) = [50.0, 40.0, 30.0, 20.0, 10.0, 0.0]
              Block k → append param_ids[k] to obs
  simple_rl:  Block k → append float(k) to obs

Both look up conditioning[k] — same weight, same output.

Usage
-----
python test_rlgames_vs_simple_rl_forward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# ── repo root on path ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deployment.rl_player import RlPlayer
from deployment.rl_player_simple_rl import RlPlayerSimpleRL

# ── config ─────────────────────────────────────────────────────────────────────
NUM_OBS   = 140
NUM_ACTS  = 29
NUM_ENVS  = 4       # test a small batch to exercise the LSTM state
NUM_STEPS = 100
M         = 6       # number of SAPG blocks
ATOL      = 1e-4    # allow for float32 / CUDA non-determinism
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

RG_CONFIG  = "pretrained_policy/config.yaml"
RG_CKPT    = "pretrained_policy/model.pth"
SL_CONFIG  = "pretrained_policy/config_simple_rl.yaml"
SL_CKPT    = "pretrained_policy/model_simple_rl.pth"

# rl_games SAPG param IDs: linspace(50.0, 0.0, M)
PARAM_IDS = torch.linspace(50.0, 0.0, M)  # [50.0, 40.0, 30.0, 20.0, 10.0, 0.0]


def run_block(
    rg: RlPlayer,
    sl: RlPlayerSimpleRL,
    block_k: int,
) -> float:
    """Run NUM_STEPS steps for SAPG block k, return worst abs diff."""
    rg_val = PARAM_IDS[block_k].item()   # e.g. 50.0 for k=0
    sl_val = float(block_k)              # e.g. 0.0 for k=0

    rg.reset()
    sl.reset()

    max_seen = 0.0
    torch.manual_seed(42)
    for step in range(NUM_STEPS):
        obs = torch.randn(NUM_ENVS, NUM_OBS, device=DEVICE)

        # Append block-specific identifier
        rg_obs = torch.cat(
            [obs, torch.full((NUM_ENVS, 1), rg_val, device=DEVICE)], dim=1
        )  # shape (B, 141)
        sl_obs = torch.cat(
            [obs, torch.full((NUM_ENVS, 1), sl_val, device=DEVICE)], dim=1
        )  # shape (B, 141)

        a_rg = rg.player.get_action(obs=rg_obs, is_deterministic=True)
        a_sl = sl._player.get_action(obs_torch=sl_obs, is_deterministic=True)

        a_rg = a_rg.reshape(NUM_ENVS, NUM_ACTS)
        a_sl = a_sl.reshape(NUM_ENVS, NUM_ACTS)

        diff = (a_rg - a_sl).abs()
        max_diff = diff.max().item()
        max_seen = max(max_seen, max_diff)

        if max_diff > ATOL:
            print(f"    Step {step:3d}: FAIL  max_diff={max_diff:.3e}")
            print(f"      a_rg[:3] = {a_rg[0, :3].tolist()}")
            print(f"      a_sl[:3] = {a_sl[0, :3].tolist()}")
            return max_seen  # non-zero means failure

        if step % 20 == 0:
            print(f"    Step {step:3d}: OK    max_diff={max_diff:.3e}")

    return max_seen


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Loading rl_games player  …  {RG_CKPT}")
    rg = RlPlayer(NUM_OBS, NUM_ACTS, RG_CONFIG, RG_CKPT, DEVICE, NUM_ENVS)

    print(f"Loading simple_rl player …  {SL_CKPT}")
    sl = RlPlayerSimpleRL(NUM_OBS, NUM_ACTS, SL_CONFIG, SL_CKPT, DEVICE, NUM_ENVS)

    print(f"\nParam IDs (rl_games): {PARAM_IDS.tolist()}")
    print(f"Block indices (simple_rl): {list(range(M))}")
    print(f"Correspondence: block k → rl_games={PARAM_IDS[0].item()}…{PARAM_IDS[-1].item()}, simple_rl=0…{M-1}\n")

    overall_worst = 0.0
    all_pass = True

    for k in range(M):
        rg_val = PARAM_IDS[k].item()
        print(f"=== Block {k}  (rl_games appends {rg_val:.1f}, simple_rl appends {float(k)}) ===")
        worst = run_block(rg, sl, k)
        overall_worst = max(overall_worst, worst)

        if worst > ATOL:
            print(f"  Block {k}: FAIL  worst_diff={worst:.3e}")
            all_pass = False
        else:
            print(f"  Block {k}: PASS  worst_diff={worst:.3e}")
        print()

    if all_pass:
        print(f"=== All {M} SAPG blocks PASS — worst diff={overall_worst:.3e} ===")
    else:
        print(f"=== FAIL — some blocks exceeded atol={ATOL:.0e}  (worst diff={overall_worst:.3e}) ===")
        sys.exit(1)


if __name__ == "__main__":
    main()

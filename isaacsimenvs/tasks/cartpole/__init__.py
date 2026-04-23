"""Cartpole task registration.

Importing this subpackage fires `gym.register`, exposing ``Isaac-Cartpole-Direct-v0``
to the gymnasium registry. Entry points cover:

- ``env_cfg_entry_point``     → CartpoleEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``→ cfg/task/Cartpole.yaml (overlay of defaults,
                                 read by our custom hydra decorator)
- ``rl_games_cfg_entry_point``→ cfg/train/CartpolePPO.yaml (rl_games algo cfg)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/CartpoleSAPG.yaml

The tuple `(CartpoleEnv, CartpoleEnvCfg)` is also re-exported for the legacy
`isaacsim_task_map` path (kept alive until Phase 2 of the @hydra_task_config
migration deletes it).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .cartpole_env import CartpoleEnv, CartpoleEnvCfg

__all__ = ["CartpoleEnv", "CartpoleEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaac-Cartpole-Direct-v0",
    entry_point="isaacsimenvs.tasks.cartpole.cartpole_env:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cartpole.cartpole_env:CartpoleEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "Cartpole.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "CartpolePPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "CartpoleSAPG.yaml"),
    },
)

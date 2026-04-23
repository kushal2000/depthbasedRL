"""SimToolReal task registration.

Registers ``Isaacsimenvs-SimToolReal-Direct-v0`` with the gymnasium registry
so the Cartpole play/train pattern extends uniformly to this task. The env
class is currently a stub (see ``simtoolreal_env.py``); the pretrained-policy
rollout runs via ``play_simtoolreal.py``, not ``gym.make``.

Entry points:
- ``env_cfg_entry_point``           → SimToolRealEnvCfg
- ``env_cfg_yaml_entry_point``      → cfg/task/SimToolReal.yaml overlay
- ``rl_games_cfg_entry_point``      → cfg/train/SimToolRealPPO.yaml (for future
                                       training; matches the architecture of the
                                       pretrained checkpoint).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .simtoolreal_env import SimToolRealEnv, SimToolRealEnvCfg

__all__ = ["SimToolRealEnv", "SimToolRealEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaacsimenvs-SimToolReal-Direct-v0",
    entry_point="isaacsimenvs.tasks.simtoolreal.simtoolreal_env:SimToolRealEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.simtoolreal.simtoolreal_env:SimToolRealEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "SimToolReal.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealPPO.yaml"),
    },
)

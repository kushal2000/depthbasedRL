"""Peg-in-hole-fixtured task registration.

Parallel to ``isaacsimenvs.tasks.peg_in_hole`` but spawns a second hole-fixture
body (``peg_fixture``) on the table. Intended for rendering snapshots where the
peg is propped upright in the peg_fixture before insertion into the goal hole.

Reuses the same RL cfg entry points as ``peg_in_hole`` so policies trained on
the standard env load with no obs/action layout change.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .peg_in_hole_fixtured_env import PegInHoleFixturedEnv
from .peg_in_hole_fixtured_env_cfg import PegInHoleFixturedEnvCfg


__all__ = ["PegInHoleFixturedEnv", "PegInHoleFixturedEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaacsimenvs-PegInHoleFixtured-Direct-v0",
    entry_point="isaacsimenvs.tasks.peg_in_hole_fixtured.peg_in_hole_fixtured_env:PegInHoleFixturedEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.peg_in_hole_fixtured.peg_in_hole_fixtured_env_cfg:PegInHoleFixturedEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "PegInHole.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealSAPG.yaml"),
    },
)

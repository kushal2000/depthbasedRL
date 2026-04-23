"""Task registry for isaacsimenvs.

Two parallel registration paths during the hydra_task_config migration:

- ``isaacsim_task_map`` (legacy) — string → (env_cls, cfg_cls) dict consumed by
  the current `train.py` and `play_video.py`. Mirrors `isaacgymenvs.tasks.isaacgym_task_map`.
- ``gym.register`` (new) — importing each task subpackage fires the registration,
  exposing typed entry points for the upcoming hydra-driven trainer.

Both paths coexist so Phase 1 lands without breaking anything. Phase 2 deletes
the legacy map once the hydra trainer is validated.
"""

from . import cartpole  # side effect: gym.register("Isaac-Cartpole-Direct-v0", ...)
from .cartpole.cartpole_env import CartpoleEnv, CartpoleEnvCfg

isaacsim_task_map: dict[str, tuple[type, type]] = {
    "Cartpole": (CartpoleEnv, CartpoleEnvCfg),
}

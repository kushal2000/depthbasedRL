"""Task registry for isaacsimenvs.

Each entry is `(env_class, env_cfg_class)`. Mirrors `isaacgymenvs.tasks.isaacgym_task_map`.
"""

from .cartpole.cartpole_env import CartpoleEnv, CartpoleEnvCfg

isaacsim_task_map: dict[str, tuple[type, type]] = {
    "Cartpole": (CartpoleEnv, CartpoleEnvCfg),
}

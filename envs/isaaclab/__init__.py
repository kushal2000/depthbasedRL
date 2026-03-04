"""Isaac Lab environment implementations for SimToolReal."""

import gymnasium

from .sim_tool_real_cfg import SimToolRealEnvCfg

# Register the environment so it can be created via gymnasium.make()
gymnasium.register(
    id="SimToolReal-Direct-v0",
    entry_point="envs.isaaclab.sim_tool_real_env:SimToolRealEnv",
    disable_env_checker=True,
    kwargs={"cfg": SimToolRealEnvCfg},
)

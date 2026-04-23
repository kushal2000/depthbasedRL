"""Glue between Isaac Lab DirectRLEnv and our vendored `./rl_games/`.

Uses `isaaclab_rl.RlGamesVecEnvWrapper` as the env-side adapter (it handles obs/
action clipping, sim↔rl device bridging, obs-group routing, gym↔gymnasium
coercion). The training algorithm/Runner/PPO/SAPG code still comes from
`./rl_games/`.
"""

from __future__ import annotations

from rl_games.common.algo_observer import AlgoObserver


class MultiObserver(AlgoObserver):
    """Fan out every `AlgoObserver` callback to a list of observers.

    Local copy of `isaacgymenvs.utils.rlgames_utils.MultiObserver` — we avoid
    importing from isaacgymenvs because that package's `__init__.py` pulls in
    `isaacgym`, which is absent from `.venv_isaacsim`.
    """

    def __init__(self, observers):
        super().__init__()
        self.observers = list(observers)

    def _call_multi(self, method, *args, **kwargs):
        ret = None
        for o in self.observers:
            fn = getattr(o, method, None)
            if fn is None:
                continue
            result = fn(*args, **kwargs)
            if result is not None:
                ret = result
        return ret

    def before_init(self, base_name, config, experiment_name):
        self._call_multi("before_init", base_name, config, experiment_name)

    def after_init(self, algo):
        self._call_multi("after_init", algo)

    def process_infos(self, infos, done_indices, **kwargs):
        self._call_multi("process_infos", infos, done_indices, **kwargs)

    def after_steps(self):
        return self._call_multi("after_steps")

    def after_clear_stats(self):
        self._call_multi("after_clear_stats")

    def after_print_stats(self, frame, epoch_num, total_time):
        self._call_multi("after_print_stats", frame, epoch_num, total_time)


def register_rlgames_env(
    env,
    rl_device: str = "cuda:0",
    clip_obs: float = 1e6,
    clip_actions: float = 1e6,
    name: str = "rlgpu",
):
    """Wrap an Isaac Lab env for rl_games and register under `name` ("rlgpu")."""
    from rl_games.common import env_configurations, vecenv
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    wrapped = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw),
    )
    env_configurations.register(
        name,
        {
            "vecenv_type": "IsaacRlgWrapper",
            "env_creator": lambda **kw: wrapped,
        },
    )
    return wrapped

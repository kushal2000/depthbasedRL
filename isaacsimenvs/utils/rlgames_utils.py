"""Glue between Isaac Lab DirectRLEnv and our vendored `./rl_games/`.

Uses `isaaclab_rl.RlGamesVecEnvWrapper` as the env-side adapter (it handles obs/
action clipping, sim↔rl device bridging, obs-group routing, gym↔gymnasium
coercion). The training algorithm/Runner/PPO/SAPG code still comes from
`./rl_games/`.
"""

from __future__ import annotations

from omegaconf import DictConfig, ListConfig, OmegaConf

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


# Keys in the task YAML that are NOT configclass attributes — they're
# consumed elsewhere (train.py, metadata) so the walker skips them.
_TASK_YAML_SKIP_KEYS = {"name", "clip_observations", "clip_actions"}


def apply_task_overrides(env_cfg, task_yaml: DictConfig, num_envs: str | int = "") -> None:
    """Push YAML values onto an Isaac Lab env_cfg (configclass).

    The task YAML's key paths must mirror the configclass attribute tree
    (snake_case). This walker recurses into nested configclasses (e.g. `sim`,
    `scene`) and `setattr`'s every leaf. Unknown keys are warned about but
    not errored.

    `num_envs` is a special top-level override (from `cfg.num_envs` on the
    CLI) — wins over `task_yaml.scene.num_envs` if set.
    """
    if num_envs not in ("", None):
        env_cfg.scene.num_envs = int(num_envs)
    _walk(env_cfg, task_yaml, path="task")


def _walk(target, src: DictConfig, path: str) -> None:
    for key, value in src.items():
        full = f"{path}.{key}"
        if key in _TASK_YAML_SKIP_KEYS:
            continue
        if not hasattr(target, key):
            print(f"[apply_task_overrides] skipping unknown key: {full}")
            continue
        sub_target = getattr(target, key)
        if isinstance(value, DictConfig) and hasattr(sub_target, "__dataclass_fields__"):
            # Nested section whose target is a configclass → recurse.
            _walk(sub_target, value, full)
            continue
        # Unwrap OmegaConf containers (ListConfig / DictConfig without a
        # configclass target) to plain Python — Isaac Lab's `configclass.validate`
        # chokes on OmegaConf types because they are dict-like AND contain
        # themselves at times, which makes `_validate` recurse forever.
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)
        setattr(target, key, value)

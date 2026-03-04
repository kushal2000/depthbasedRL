from omegaconf import OmegaConf

# OmegaConf custom resolvers used by Hydra configs
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)
OmegaConf.register_new_resolver('eval', lambda x: eval(x))

# Lazy task map — SimToolReal import is deferred until first access
# to avoid triggering heavy imports (isaacgym, pytorch3d, dextoolbench)
# when only OmegaConf resolvers or lightweight utils are needed.
_isaacgym_task_map = None


def __getattr__(name):
    if name == "isaacgym_task_map":
        global _isaacgym_task_map
        if _isaacgym_task_map is None:
            from .env import SimToolReal
            _isaacgym_task_map = {"SimToolReal": SimToolReal}
        return _isaacgym_task_map
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

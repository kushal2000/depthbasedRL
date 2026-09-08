"""Whole-script wall-clock benchmark for legacy isaacgymenvs SimToolReal.

Mirrors ``physics_dump_isaacgym.py`` for env construction (Hydra → cfg →
isaacgym_task_map["SimToolReal"](...)) but skips the actor dump: just records
how long env creation takes and exits. Used as a north-star for the
optimized isaacsim port.

    .venv/bin/python debug_differences/bench_isaacgym_env_creation.py \
        --num_envs 256 --num_per_type 25
"""

from __future__ import annotations

# isort: off
from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401
import torch  # noqa: F401
# isort: on

import argparse
import time
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_per_type", type=int, default=25)
    return parser


def _build_cfg(args) -> dict:
    cfg_dir = str((REPO_ROOT / "isaacgymenvs" / "cfg").resolve())
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=SimToolReal",
                f"task.env.numEnvs={args.num_envs}",
                f"task.env.numAssetsPerType={args.num_per_type}",
                # All DR off — match the isaacsim bench (we want creation
                # time, not training-time noise).
                "task.env.resetPositionNoiseX=0.0",
                "task.env.resetPositionNoiseY=0.0",
                "task.env.resetPositionNoiseZ=0.0",
                "task.env.randomizeObjectRotation=False",
                "task.env.resetDofPosRandomIntervalFingers=0.0",
                "task.env.resetDofPosRandomIntervalArm=0.0",
                "task.env.resetDofVelRandomInterval=0.0",
                "task.env.tableResetZRange=0.0",
                "task.env.forceScale=0.0",
                "task.env.torqueScale=0.0",
                "task.env.useObsDelay=False",
                "task.env.useActionDelay=False",
                "task.env.useObjectStateDelayNoise=False",
                "task.env.objectScaleNoiseMultiplierRange=[1.0,1.0]",
                "task.env.capture_viewer=False",
                "task.env.capture_video=False",
                "task.env.episodeLength=600",
                "task.env.resetWhenDropped=False",
                "task.env.useFixedGoalStates=False",
                "task.env.fixedGoalStatesJsonPath=null",
            ],
        )
    return omegaconf_to_dict(cfg.task)


def main() -> None:
    args = _build_parser().parse_args()
    print(f"[bench-gym] num_envs={args.num_envs} num_per_type={args.num_per_type}")

    t_cfg0 = time.perf_counter()
    cfg_dict = _build_cfg(args)
    t_cfg = time.perf_counter() - t_cfg0
    print(f"[bench-gym] cfg compose: {t_cfg:.2f}s")

    t_env0 = time.perf_counter()
    env = isaacgym_task_map["SimToolReal"](
        cfg=cfg_dict,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )
    t_env = time.perf_counter() - t_env0
    print(f"[bench-gym] env construction: {t_env:.2f}s")

    # First step / reset is sometimes lazy-init in legacy gym. Time it too
    # so we report a comparable "fully ready to train" number.
    t_reset0 = time.perf_counter()
    env.reset()
    t_reset = time.perf_counter() - t_reset0
    print(f"[bench-gym] first reset: {t_reset:.2f}s")

    print(f"[bench-gym] TOTAL (cfg+env+reset): {t_cfg + t_env + t_reset:.2f}s")


if __name__ == "__main__":
    main()

"""Verify Isaac Lab's collision-property warnings claim — read USD attributes
back from the live stage after SimToolReal scene setup and compare against
the values our scene_utils.py asks for.

Builds a 2-env env (cheap), then walks the relevant prims and prints:
  - whether the prim is instanced (read-only)
  - the actual ``physics:collisionEnabled`` value (UsdPhysics.CollisionAPI)
  - the actual ``physxCollision:contactOffset`` value (PhysxSchema.PhysxCollisionAPI)
  - the actual ``physxCollision:restOffset`` value
  - whether the prim has a `UsdPhysics.CollisionAPI` schema attached at all

If a configured override is silently dropped, the actual value will either
be the PhysX default (typically ``contact_offset = 0.02 m``,
``rest_offset = 0.0 m``, ``collision_enabled = True``) or the attribute
won't be authored on any reachable prim.

    .venv_isaacsim/bin/python debug_differences/verify_collision_props.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _launch_app():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    return AppLauncher(args).app


_app = _launch_app()


def _walk_authored_attr(stage, root_path: str, attr_name: str):
    """Walk every prim under ``root_path`` and yield
    (path, is_instance, has_collision_api, attribute_value_or_None).

    Mirrors Isaac Lab's ``apply_nested`` walker except we don't stop on
    success — we want the full picture of what's authored where.
    """
    from pxr import Usd, UsdPhysics, PhysxSchema  # noqa: F401
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        yield (root_path, False, False, None, "INVALID PRIM")
        return

    queue = [root]
    while queue:
        prim = queue.pop(0)
        path = prim.GetPath().pathString
        is_instance = prim.IsInstance()

        has_col_api = bool(UsdPhysics.CollisionAPI(prim))
        has_physx_col_api = bool(PhysxSchema.PhysxCollisionAPI(prim))

        attr = prim.GetAttribute(attr_name)
        if attr.IsValid() and attr.HasAuthoredValue():
            authored = attr.Get()
        else:
            authored = None

        yield (path, is_instance, has_col_api or has_physx_col_api, authored,
               "instance(read-only)" if is_instance else "ok")

        if not is_instance:
            queue.extend(prim.GetChildren())


def main() -> None:
    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from omni.usd import get_context
    from pxr import Usd, UsdPhysics, PhysxSchema  # noqa: F401

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = 2  # smallest count that exercises per-env spawn
    cfg.assets.num_assets_per_type = 1  # tiny pool — fast URDF→USD conversion

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None

    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(
        torch.zeros((cfg.scene.num_envs, cfg.action_space), device=inner.device)
    )

    stage = get_context().get_stage()

    print("\n" + "=" * 100)
    print("CONFIGURED values (from scene_utils.py):")
    print("  Robot:    contact_offset=0.002, rest_offset=0.0")
    print("  Table:    contact_offset=0.002, rest_offset=0.0")
    print("  Object:   contact_offset=0.002, rest_offset=0.0")
    print("  GoalViz:  collision_enabled=False, contact_offset=0.002, rest_offset=0.0")
    print()
    print("PhysX DEFAULTS (when an override is dropped):")
    print("  collision_enabled = True")
    print("  contact_offset    = 0.02   (2 cm)")
    print("  rest_offset       = 0.0")
    print("=" * 100)

    targets = [
        ("/World/envs/env_0/Robot", "Robot (env_0)"),
        ("/World/envs/env_0/Table", "Table (env_0)"),
        ("/World/envs/env_0/Object", "Object (env_0)"),
        ("/World/envs/env_0/GoalViz", "GoalViz (env_0)"),
    ]
    attrs_to_check = [
        ("physics:collisionEnabled", "collision_enabled"),
        ("physxCollision:contactOffset", "contact_offset"),
        ("physxCollision:restOffset", "rest_offset"),
    ]

    for root_path, label in targets:
        print(f"\n--- {label} : {root_path} ---")
        for usd_attr_name, friendly in attrs_to_check:
            print(f"\n  attribute: {friendly}  ({usd_attr_name})")
            authored_count = 0
            instance_count = 0
            api_count = 0
            for (path, is_instance, has_api, value, status) in _walk_authored_attr(
                stage, root_path, usd_attr_name
            ):
                if is_instance:
                    instance_count += 1
                if has_api:
                    api_count += 1
                if value is not None:
                    authored_count += 1
                    rel = path.replace(root_path, "")
                    print(f"    AUTHORED at .{rel} = {value!r}")
                # only print non-authored prims for the first attr to keep output short
            print(f"    (subtree summary: {instance_count} instance prims skipped, "
                  f"{api_count} prims with CollisionAPI, "
                  f"{authored_count} prims with authored value for this attr)")

    print("\n" + "=" * 100)
    print("Also: enumerate which descendants of each target are USD instances")
    print("(these are the prims Isaac Lab's apply_nested walker bails on):")
    print("=" * 100)
    for root_path, label in targets:
        instances = []
        prim = stage.GetPrimAtPath(root_path)
        if not prim.IsValid():
            print(f"  {label}: INVALID")
            continue
        queue = [prim]
        while queue:
            cur = queue.pop(0)
            if cur.IsInstance():
                instances.append(cur.GetPath().pathString)
                continue
            queue.extend(cur.GetChildren())
        print(f"\n  {label}: {len(instances)} instanced prim(s)")
        for p in instances[:8]:
            print(f"    - {p}")
        if len(instances) > 8:
            print(f"    ... ({len(instances) - 8} more)")

    print("\n" + "=" * 100)
    print("DONE")


if __name__ == "__main__":
    import os
    import sys
    main()
    del _app
    sys.stdout.flush()
    os._exit(0)

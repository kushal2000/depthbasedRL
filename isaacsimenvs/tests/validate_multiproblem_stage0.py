#!/usr/bin/env python3
"""Stage 0 validation for the multi-problem refactor.

Two modes, covering the paths the peg bolted eval does not reach:

  --mode dump      Construct PegInHoleEnv, seed, step with seeded random
                   actions, and dump the tensors that the refactor could
                   plausibly corrupt. Run this on the pre-refactor commit and
                   on HEAD, then `--mode compare` the two dumps. Point it at a
                   transportPreInsertFinal problem to exercise the prelude
                   path, _prelude_lift_off_env, _prelude_pose_world sizing and
                   the coarse-tolerance branch.

  --mode fixtured  Construct PegInHoleFixturedEnv and assert its goal budget
                   came through _override_goal_counts. That env rewrites the
                   goal counts from scenes.npz; before the hook existed it
                   wrote the scalars directly, which Phase C would have ignored
                   -- giving wrong env_max_goals with no error anywhere.

  --mode compare   torch.equal every tensor in two dumps.

Compare needs no GPU and no Isaac Sim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Tensors a per-env-promotion bug would show up in. goal_viz pose is the most
# sensitive: it is the end product of _write_goal_pose, so a wrong prelude
# count, a wrong tail index or a bad gather all land here.
_DUMP_KEYS = (
    "env_max_goals", "successes", "goal_pos", "goal_quat",
    "object_pos", "keypoints_max_dist", "reward",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dump", "fixtured", "compare"], required=True)
    p.add_argument("--task", default="Isaacsimenvs-PegInHole-Direct-v0")
    p.add_argument("--problem", default="Lpeg_dense_traj.tol0p5mm")
    p.add_argument("--goal-mode", default="transportPreInsertFinal")
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="dump path (.pt)")
    p.add_argument("--a", default=None, help="compare: first dump")
    p.add_argument("--b", default=None, help="compare: second dump")
    return p.parse_args()


def _compare(a_path: str, b_path: str) -> int:
    import torch
    a, b = torch.load(a_path), torch.load(b_path)
    ok = True
    print(f"comparing\n  A {a_path}\n  B {b_path}")
    for k in _DUMP_KEYS:
        if k not in a or k not in b:
            print(f"  {k:22s} MISSING"); ok = False; continue
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape:
            print(f"  {k:22s} SHAPE {tuple(ta.shape)} vs {tuple(tb.shape)}")
            ok = False; continue
        if torch.equal(ta, tb):
            print(f"  {k:22s} identical  {tuple(ta.shape)}")
        else:
            d = (ta.float() - tb.float()).abs()
            n = int((d > 0).sum())
            print(f"  {k:22s} DIFFERS on {n}/{d.numel()} elems, max |d| = {d.max():.3e}")
            ok = False
    print("==> IDENTICAL" if ok else "==> MISMATCH")
    return 0 if ok else 1


def main() -> int:
    args = _parse_args()
    if args.mode == "compare":
        return _compare(args.a, args.b)

    import os
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    largs, _ = lp.parse_known_args([])
    largs.headless = True
    largs.enable_cameras = False
    app = AppLauncher(largs).app

    import torch
    import isaacsimenvs  # noqa: F401

    # The venv installs the repo editable, pointing at the MAIN tree. When this
    # runs from a git worktree we rely on sys.path[0] (the script's own repo
    # root) winning over the .pth entry. Verify it: if the import silently
    # resolved to the main tree, a pre-vs-post comparison would be comparing
    # HEAD against itself and would "pass" while testing nothing.
    _got = Path(isaacsimenvs.__file__).resolve().parent.parent
    if _got != REPO_ROOT:
        raise SystemExit(
            f"isaacsimenvs resolved to {_got}, expected {REPO_ROOT}. The "
            "editable install shadowed the worktree; the comparison would be "
            "vacuous. Refusing to run."
        )
    print(f"[validate] isaacsimenvs <- {_got}", flush=True)

    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides, _instantiate_env, _load_env_cfg,
    )

    task = args.task
    if args.mode == "fixtured":
        task = "Isaacsimenvs-PegInHoleFixtured-Direct-v0"

    cfg = _load_env_cfg(task)
    _apply_env_overrides(
        cfg,
        problem=args.problem,
        goal_mode=args.goal_mode,
        random_goal_fraction=0.0,
        insertion_success_tolerance=0.010,
        retract_success_tolerance=0.005,
        num_envs=int(args.num_envs),
        sim_device="cuda:0",
        sdf=False,
        keep_dr=False,          # DR off: we are testing goal plumbing, not noise
        extra_overrides={},
    )
    env = _instantiate_env(task, cfg)

    if args.mode == "fixtured":
        # The whole point of the _override_goal_counts hook: the fixtured env
        # sources its trajectory from scenes.npz, so its goal budget must be
        # _scenes_max_traj_len -- not whatever the Problem implied.
        want = int(env._scenes_max_traj_len)
        got_scalar = int(env._num_total_insertion_goals)
        got_table = int(env._num_total_goals_p[0].item())
        got_env = env._num_total_goals_env
        print(f"scenes_max_traj_len       = {want}")
        print(f"_num_total_insertion_goals= {got_scalar}")
        print(f"_num_total_goals_p[0]     = {got_table}")
        print(f"_num_total_goals_env      : min={int(got_env.min())} max={int(got_env.max())}")
        print(f"env_max_goals             : min={int(env.env_max_goals.min())} "
              f"max={int(env.env_max_goals.max())}")
        bad = []
        if got_scalar != want:
            bad.append("scalar _num_total_insertion_goals")
        if got_table != want:
            bad.append("Phase C table _num_total_goals_p (the hook did not take)")
        if int(got_env.min()) != want or int(got_env.max()) != want:
            bad.append("per-env _num_total_goals_env")
        if bad:
            print("==> FAIL: " + "; ".join(bad))
            return 1
        print("==> PASS: goal budget propagated through _override_goal_counts")
        return 0

    # dump mode
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env.reset()
    torch.manual_seed(args.seed)
    act_dim = env.action_space.shape[-1] if hasattr(env.action_space, "shape") else env.cfg.action_space
    frames = {k: [] for k in _DUMP_KEYS}
    origins = env.scene.env_origins
    for _ in range(int(args.steps)):
        a = (torch.rand(env.num_envs, int(act_dim), device=env.device) * 2.0 - 1.0)
        env.step(a)
        frames["env_max_goals"].append(env.env_max_goals.clone())
        frames["successes"].append(env._successes.clone())
        frames["goal_pos"].append((env.goal_viz.data.root_pos_w - origins).clone())
        frames["goal_quat"].append(env.goal_viz.data.root_quat_w.clone())
        frames["object_pos"].append((env.object.data.root_pos_w - origins).clone())
        frames["keypoints_max_dist"].append(env._keypoints_max_dist.clone())
        frames["reward"].append(env.reward_buf.clone())
    out = {k: torch.stack(v).cpu() for k, v in frames.items()}
    torch.save(out, args.out)
    print(f"=> wrote {args.out}")
    for k, v in out.items():
        finite = bool(torch.isfinite(v.float()).all())
        print(f"   {k:22s} {tuple(v.shape)}  finite={finite}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

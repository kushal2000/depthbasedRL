"""Smoke test: run a few iterations of PPO (or SAPG with --sapg) on Cartpole.

Budget is kept small (~2 PPO iters, a handful of envs) so the test validates
the pipeline end-to-end without caring about convergence.

Invokes `isaacsimenvs/train.py` via subprocess so Hydra's `config_path="./cfg"`
resolves correctly (calling in-process confuses Hydra's config lookup).

    python isaacsimenvs/tests/test_cartpole_train_smoke.py           # PPO
    python isaacsimenvs/tests/test_cartpole_train_smoke.py --sapg    # SAPG
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "isaacsimenvs" / "train.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sapg", action="store_true", help="Use CartpoleSAPG train config")
    parser.add_argument("--num_envs", type=int, default=None, help="Override task.env.numEnvs")
    parser.add_argument("--max_iterations", type=int, default=2)
    my_args = parser.parse_args()

    if my_args.sapg:
        train_cfg = "CartpoleSAPG"
        default_num_envs = 2048  # num_blocks=2 → block_size=1024
        # batch_size = num_envs * horizon_length = 2048 * 32 = 65536
        minibatch_size = 16384
    else:
        train_cfg = "CartpolePPO"
        default_num_envs = 64
        # batch_size = 64 * 32 = 2048 — must divide, pick 512.
        minibatch_size = 512

    num_envs = my_args.num_envs if my_args.num_envs is not None else default_num_envs

    run_dir = REPO_ROOT / "runs" / f"cartpole_smoke_{'sapg' if my_args.sapg else 'ppo'}"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        f"train={train_cfg}",
        f"num_envs={num_envs}",
        f"max_iterations={my_args.max_iterations}",
        "headless=true",
        "wandb_activate=false",
        f"train.params.config.minibatch_size={minibatch_size}",
        f"hydra.run.dir={run_dir}",
    ]
    env = {**os.environ}
    env.pop("PYTHONPATH", None)
    print(f"[smoke test] running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    print(f"[smoke test] exit={result.returncode}", flush=True)

    # Kit's atexit handler sometimes masks Python-level exceptions to exit 0.
    # Cross-check by grepping the captured output for Traceback before any rl_games
    # log line confirming an actual train iter happened.
    stdout = result.stdout + result.stderr
    if "Traceback" in stdout and "Error executing job" in stdout:
        print(stdout[-3000:])
        raise AssertionError("train.py hit a Python exception (see tail above)")

    assert result.returncode == 0, f"train.py exited {result.returncode}"
    assert run_dir.exists(), f"run_dir missing: {run_dir}"

    exp_name = "0_cartpole_sapg" if my_args.sapg else "0_cartpole_direct"
    rlg_summaries = REPO_ROOT / "runs" / exp_name / "summaries"
    assert rlg_summaries.exists(), f"rl_games wrote no summaries at {rlg_summaries}"
    event_files = list(rlg_summaries.glob("events.out.tfevents.*"))
    assert event_files, f"no tfevents under {rlg_summaries}"

    # rl_games prints `fps step:` once per training iter; at least one must
    # appear to confirm training actually ran.
    fps_lines = [ln for ln in stdout.splitlines() if "fps step" in ln]
    assert fps_lines, "no 'fps step' lines — rl_games never completed an iter"
    print(f"[smoke test] OK — {len(fps_lines)} iter(s), summaries at {rlg_summaries}")


if __name__ == "__main__":
    main()

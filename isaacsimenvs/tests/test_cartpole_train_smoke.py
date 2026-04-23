"""Smoke test: run a few iterations of PPO (or SAPG with --sapg) on Cartpole.

Budget is kept small (~2 iters, a handful of envs) so the test validates
the pipeline end-to-end without caring about convergence.

Invokes ``isaacsimenvs/train.py`` via subprocess so Hydra's ConfigStore is
clean for each run (calling in-process leaves stale state between tests).

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
    parser.add_argument("--sapg", action="store_true", help="Use SAPG agent config")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=2)
    my_args = parser.parse_args()

    if my_args.sapg:
        agent = "rl_games_sapg_cfg_entry_point"
        exp_name = "0_cartpole_sapg"
        default_num_envs = 2048  # num_blocks=2 → block_size=1024
        minibatch_size = 16384   # num_envs * horizon_length = 65536
    else:
        agent = "rl_games_cfg_entry_point"
        exp_name = "0_cartpole_direct"
        default_num_envs = 64
        minibatch_size = 512     # 64 * 32 = 2048; pick smaller divisor

    num_envs = my_args.num_envs if my_args.num_envs is not None else default_num_envs

    run_dir = REPO_ROOT / "runs" / f"cartpole_smoke_{'sapg' if my_args.sapg else 'ppo'}"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "--task", "Isaacsimenvs-Cartpole-Direct-v0",
        "--agent", agent,
        "--headless",
        f"env.scene.num_envs={num_envs}",
        f"agent.params.config.max_epochs={my_args.max_iterations}",
        f"agent.params.config.minibatch_size={minibatch_size}",
        f"hydra.run.dir={run_dir}",
    ]
    env = {**os.environ}
    env.pop("PYTHONPATH", None)
    print(f"[smoke test] running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    print(f"[smoke test] exit={result.returncode}", flush=True)

    stdout = result.stdout + result.stderr
    if "Traceback" in stdout and ("Error executing job" in stdout or result.returncode != 0):
        print(stdout[-4000:])
        raise AssertionError("train.py hit a Python exception (see tail above)")

    assert result.returncode == 0, f"train.py exited {result.returncode}"
    assert run_dir.exists(), f"run_dir missing: {run_dir}"

    # train.py sets rl_games' train_dir to hydra_run_dir, so summaries land at
    # <hydra_run_dir>/<exp_name>/summaries/.
    rlg_summaries = run_dir / exp_name / "summaries"
    assert rlg_summaries.exists(), f"rl_games wrote no summaries at {rlg_summaries}\nstdout tail:\n{stdout[-2000:]}"
    event_files = list(rlg_summaries.glob("events.out.tfevents.*"))
    assert event_files, f"no tfevents under {rlg_summaries}"

    # rl_games prints `fps step:` once per training iter; at least one must
    # appear to confirm training actually ran.
    fps_lines = [ln for ln in stdout.splitlines() if "fps step" in ln]
    assert fps_lines, f"no 'fps step' lines — rl_games never completed an iter\nstdout tail:\n{stdout[-2000:]}"
    print(f"[smoke test] OK — {len(fps_lines)} iter(s), summaries at {rlg_summaries}")


if __name__ == "__main__":
    main()

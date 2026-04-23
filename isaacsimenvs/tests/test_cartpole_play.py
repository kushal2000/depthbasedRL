"""Smoke test for the `test=true` (play/eval) code path.

Trains for 2 iters (to produce a checkpoint) then runs a short rollout in play
mode against that checkpoint. Validates that `train.py test=true checkpoint=...`
does not crash and that rl_games loads the saved weights.
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


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ}
    env.pop("PYTHONPATH", None)
    print(f"[play test] running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    print(f"[play test] exit={result.returncode}", flush=True)
    out = result.stdout + result.stderr
    if "Traceback" in out and "Error executing job" in out:
        print(out[-3000:])
        raise AssertionError("train.py hit a Python exception")
    assert result.returncode == 0, f"subprocess exited {result.returncode}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--train_iters", type=int, default=2)
    my_args = parser.parse_args()

    run_dir_train = REPO_ROOT / "runs" / "cartpole_play_train"
    run_dir_play = REPO_ROOT / "runs" / "cartpole_play_eval"
    exp_dir = REPO_ROOT / "runs" / "0_cartpole_direct"
    for d in (run_dir_train, run_dir_play, exp_dir):
        if d.exists():
            shutil.rmtree(d)

    # --- 1. train a checkpoint ---
    train_cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "train=CartpolePPO",
        f"num_envs={my_args.num_envs}",
        f"max_iterations={my_args.train_iters}",
        "headless=true",
        "wandb_activate=false",
        "train.params.config.minibatch_size=512",
        f"hydra.run.dir={run_dir_train}",
    ]
    _run(train_cmd)

    # rl_games writes checkpoints under runs/<exp_name>/nn/last_*.pth
    nn_dir = exp_dir / "nn"
    ckpts = sorted(nn_dir.glob("last_*.pth"))
    assert ckpts, f"no last_*.pth under {nn_dir}"
    checkpoint = ckpts[-1]
    print(f"[play test] checkpoint: {checkpoint}")

    # --- 2. play from that checkpoint ---
    play_cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "train=CartpolePPO",
        "test=true",
        f"checkpoint={checkpoint}",
        f"num_envs={my_args.num_envs}",
        "headless=true",
        "wandb_activate=false",
        "train.params.config.minibatch_size=512",
        "train.params.config.player.games_num=4",
        f"hydra.run.dir={run_dir_play}",
    ]
    result = _run(play_cmd)

    # rl_games' player prints "=> loading checkpoint '<path>'" on successful load
    assert f"loading checkpoint" in result.stdout + result.stderr, (
        "rl_games player did not log a checkpoint-load message"
    )
    print("[play test] OK — checkpoint loaded and play completed without error")


if __name__ == "__main__":
    main()

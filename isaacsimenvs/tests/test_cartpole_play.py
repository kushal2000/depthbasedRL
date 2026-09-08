"""Smoke test for the ``--test`` (play/eval) code path.

Trains for 2 iters (to produce a checkpoint) then runs a short rollout in play
mode against that checkpoint. Validates that ``train.py --test --checkpoint=...``
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
EXP_NAME = "0_cartpole_direct"  # matches CartpolePPO.yaml params.config.name


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ}
    env.pop("PYTHONPATH", None)
    print(f"[play test] running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    print(f"[play test] exit={result.returncode}", flush=True)
    out = result.stdout + result.stderr
    if "Traceback" in out and ("Error executing job" in out or result.returncode != 0):
        print(out[-4000:])
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
    for d in (run_dir_train, run_dir_play):
        if d.exists():
            shutil.rmtree(d)

    # --- 1. train a checkpoint ---
    train_cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "--task", "Isaacsimenvs-Cartpole-Direct-v0",
        "--agent", "rl_games_cfg_entry_point",
        "--headless",
        f"env.scene.num_envs={my_args.num_envs}",
        f"agent.params.config.max_epochs={my_args.train_iters}",
        "agent.params.config.minibatch_size=512",
        f"hydra.run.dir={run_dir_train}",
    ]
    _run(train_cmd)

    # rl_games writes checkpoints under <train_dir>/<exp_name>/nn/last_*.pth
    nn_dir = run_dir_train / EXP_NAME / "nn"
    ckpts = sorted(nn_dir.glob("last_*.pth"))
    assert ckpts, f"no last_*.pth under {nn_dir}"
    checkpoint = ckpts[-1]
    print(f"[play test] checkpoint: {checkpoint}")

    # --- 2. play from that checkpoint ---
    play_cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "--task", "Isaacsimenvs-Cartpole-Direct-v0",
        "--agent", "rl_games_cfg_entry_point",
        "--test",
        "--checkpoint", str(checkpoint),
        "--headless",
        f"env.scene.num_envs={my_args.num_envs}",
        "agent.params.config.minibatch_size=512",
        "agent.params.config.player.games_num=4",
        f"hydra.run.dir={run_dir_play}",
    ]
    result = _run(play_cmd)

    # rl_games' player prints "=> loading checkpoint '<path>'" on successful load
    assert "loading checkpoint" in (result.stdout + result.stderr), (
        "rl_games player did not log a checkpoint-load message"
    )
    print("[play test] OK — checkpoint loaded and play completed without error")


if __name__ == "__main__":
    main()

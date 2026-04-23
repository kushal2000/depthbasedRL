"""Smoke test for play_simtoolreal.py — the pretrained-policy rollout path.

Runs a short rollout (240 sim steps, roughly 4s sim time) against the beam-2
fabrica assembly with the pretrained policy. Asserts:
  - exit code 0
  - mp4 produced
  - trajectory.npz produced
  - at least some steps were logged

Does NOT assert goal progress — that's a regression test the user can add once
the port settles; overnight we just verify the pipeline completes.

    python isaacsimenvs/tests/test_simtoolreal_play.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAY_SCRIPT = REPO_ROOT / "isaacsimenvs" / "tasks" / "simtoolreal" / "play_simtoolreal.py"


def main() -> None:
    video_dir = REPO_ROOT / "isaacsimenvs" / "rollout_videos_smoke"
    if video_dir.exists():
        shutil.rmtree(video_dir)

    cmd = [
        sys.executable,
        "-u",
        str(PLAY_SCRIPT),
        "--assembly", "beam",
        "--part_id", "2",
        "--collision_method", "coacd",
        "--max_steps", "240",
        "--video_dir", str(video_dir),
        "--headless",
        "--enable_cameras",
    ]
    env = {**os.environ}
    env.pop("PYTHONPATH", None)  # must rely on `uv pip install -e . --no-deps`
    print(f"[simtoolreal play test] running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    print(f"[simtoolreal play test] exit={result.returncode}", flush=True)

    stdout = result.stdout + result.stderr
    if "Traceback" in stdout and result.returncode != 0:
        print(stdout[-4000:])
        raise AssertionError("play_simtoolreal.py hit a Python exception (tail above)")

    assert result.returncode == 0, f"play script exited {result.returncode}"

    mp4 = video_dir / "rollout_beam_2.mp4"
    traj = video_dir / "trajectory.npz"
    assert mp4.exists(), f"no mp4 at {mp4}\nstdout tail:\n{stdout[-2000:]}"
    assert traj.exists(), f"no trajectory.npz at {traj}"
    mp4_size = mp4.stat().st_size
    assert mp4_size > 1024, f"mp4 too small ({mp4_size} bytes) — likely empty capture"

    step_lines = [ln for ln in stdout.splitlines() if ln.startswith("[IsaacSimEnv] [step")]
    assert step_lines, f"no [step …] log lines — rollout loop may not have run\nstdout tail:\n{stdout[-2000:]}"

    print(f"[simtoolreal play test] OK — mp4={mp4} ({mp4_size} bytes), {len(step_lines)} step log(s)")


if __name__ == "__main__":
    main()

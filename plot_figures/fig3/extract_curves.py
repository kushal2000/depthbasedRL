"""Extract success-rate-vs-wallclock curves from fig3 ablation finetuning runs.

Walks train_dir/fig3_ablations/<task>/<axis>/<level>/seed*/ (including the
play2win/seed0 symlink to the reused fig4 wrench teacher), reads each run's
tensorboard event file, and writes a per-task JSON of smoothed, uniformly
resampled curves for plotting.

Primary metric: `all_goals_hit_ratio` — fraction of episodes that hit ALL
goals (pre-insert + final in preInsertAndFinal mode), i.e. full task success.
Plotted UN-normalized for every line: per-run fall-normalization (dividing each
curve by its own 1-done_fall) makes lines incomparable, since a policy that
drops the part more often gets a friendlier denominator. The smoothed
done_fall series is stored per run so any normalization applied later can use
one shared quantity across all lines, computed explicitly at plot time.

For reference the JSON also stores `episode_final/success_ratio` (raw).

Two data sources:
  --source wandb (default): pulls just the needed metric keys from the W&B API
    (server-side downsampling, ~2 s/run). Needs network + wandb auth.
  --source tensorboard: streams the local event files (~30 s/run, no network).
    Decodes every protobuf record since event files have no per-tag index.

Usage:
    .venv/bin/python plot_figures/fig3/extract_curves.py                 # all seeds found
    .venv/bin/python plot_figures/fig3/extract_curves.py --seeds 0 1     # restrict seeds

Writes:
    plot_figures/fig3/outputs/<task>_curves.json
    {axis: {level: {"seed0": {"t": [...], "feasible": [...], "raw": [...]}, ...}}}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event

REPO = "/share/portal/kk837/depthbasedRL"
ABLATIONS_ROOT = os.path.join(REPO, "train_dir", "fig3_ablations")
OUT_DIR = os.path.join(REPO, "plot_figures", "fig3", "outputs")

AGH_TAG = "all_goals_hit_ratio"          # primary: full task success
SUCCESS_TAG = "episode_final/success_ratio"  # reference: any-goal success
FALL_TAG = "episode_final/done_fall"

WANDB_ENTITY = "kk837"
WANDB_PROJECT = "fig3_ablations"
WANDB_SAMPLES = 1500  # server-side downsampling budget per run

# run-name slug -> (axis, level); names look like <task>__<slug>__seed<N>_<datetime>...
SLUG_MAP = {
    "play2win": ("play2win", "play2win"),
    "objdiv_obj10": ("object_diversity", "obj10"),
    "objdiv_obj100": ("object_diversity", "obj100"),
    "prec_5cm": ("goal_precision", "prec5cm"),
    "prec_10cm": ("goal_precision", "prec10cm"),
    "traj_10": ("trajectory_diversity", "traj10"),
    "traj_100": ("trajectory_diversity", "traj100"),
    "obj_rotation": ("training_objective", "rotation"),
    "obj_grasp": ("training_objective", "grasp"),
}

# Reused teacher runs that live in other W&B projects: task -> (project, name
# prefix, axis, level, seed). Mirrors REUSE in experiments/generate_subs.py.
WANDB_REUSE = {
    "furniture_bench": [
        ("fig4", "furniture_bench_leg4_200mm_finetune_rgf10_dr_wrench_2026-05-25_22-16-15",
         "play2win", "play2win", 0),
    ],
    "beam_part0": [
        ("fig4", "beam_3x_part_0_finetune_rgf0_dr_wrench_2026-05-25_05-40-24",
         "play2win", "play2win", 0),
    ],
    "beam_part2": [
        ("fig4", "beam_3x_part_2_finetune_rgf0_dr_wrench_2026-05-25_05-40-24",
         "play2win", "play2win", 0),
    ],
}

MAX_HOURS = 24.0
GRID_N = 201          # uniform output grid over [0, MAX_HOURS]
SMOOTH_WIN = 51       # centered rolling mean over raw points (~12 s cadence -> ~10 min)


def find_runs(task: str) -> list[tuple[str, str, int, str]]:
    """Yield (axis, level, seed, run_dir). play2win has no <level> tier."""
    runs = []
    task_root = os.path.join(ABLATIONS_ROOT, task)
    for seed_dir in sorted(glob.glob(os.path.join(task_root, "*", "*", "seed*"))) + sorted(
        glob.glob(os.path.join(task_root, "play2win", "seed*"))
    ):
        rel = os.path.relpath(seed_dir, task_root)
        parts = rel.split(os.sep)
        if parts[0] == "play2win":
            axis, level, seed_name = "play2win", "play2win", parts[1]
        else:
            axis, level, seed_name = parts[0], parts[1], parts[2]
        m = re.match(r"seed(\d+)", seed_name)
        if not m:
            continue
        runs.append((axis, level, int(m.group(1)), seed_dir))
    return runs


def find_event_file(run_dir: str) -> str | None:
    evs = glob.glob(os.path.join(run_dir, "0_*", "summaries", "events.out.tfevents.*"))
    # the reused fig4 teacher names its agent dir 0_<original tag>, same glob works
    return max(evs, key=os.path.getsize) if evs else None


def rolling_mean(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(y) < win:
        return y
    kernel = np.ones(win) / win
    pad = win // 2
    ypad = np.pad(y, pad, mode="edge")
    return np.convolve(ypad, kernel, mode="valid")[: len(y)]


def build_curve(t_a, v_a, t_s, v_s, t_f, v_f, smooth_win: int) -> dict:
    """Smooth + resample (time arrays in hours from run start)."""
    agh_s = rolling_mean(v_a, smooth_win)
    raw_s = rolling_mean(v_s, smooth_win)
    fall_s = rolling_mean(np.interp(t_s, t_f, v_f), smooth_win)

    grid = np.linspace(0.0, MAX_HOURS, GRID_N)
    cov = min(t_a[-1], MAX_HOURS)
    # interp clamps beyond data end; mask the grid to actual coverage
    mask = grid <= cov + 1e-9
    return {
        "t": grid[mask].round(4).tolist(),
        "agh": np.interp(grid[mask], t_a, agh_s).round(4).tolist(),
        "raw": np.interp(grid[mask], t_s, raw_s).round(4).tolist(),
        "fall": np.interp(grid[mask], t_s, fall_s).round(4).tolist(),
        "hours_covered": round(float(t_a[-1]), 2),
        "final_agh": round(float(agh_s[-1]), 4),
        "max_agh": round(float(agh_s.max()), 4),
    }


def extract_run_wandb(run) -> dict | None:
    # The metrics are logged at different steps, and wandb's sampled history
    # only returns rows containing ALL requested keys — so fetch per key.
    def fetch(key, attempts=4):
        last = None
        for i in range(attempts):
            try:
                rows = list(run.history(keys=[key, "_runtime"],
                                        samples=WANDB_SAMPLES, pandas=False))
                break
            except Exception as e:  # transient wandb API timeout/comm error
                last = e
                print(f"    retry {i+1}/{attempts} for {key} on {run.name}: {e}",
                      flush=True)
        else:
            raise last
        rows.sort(key=lambda r: r["_runtime"])
        t = np.array([r["_runtime"] for r in rows], dtype=float)
        v = np.array([r[key] for r in rows], dtype=float)
        return t, v

    t_a, v_a = fetch(AGH_TAG)
    t_s, v_s = fetch(SUCCESS_TAG)
    t_f, v_f = fetch(FALL_TAG)
    if len(t_a) == 0 or len(t_s) == 0 or len(t_f) == 0:
        return None
    t0 = t_a[0]
    # rows are already server-side downsampled (~8.5k -> 1.5k); scale the
    # smoothing window to keep the same ~10 min effective span
    win = max(5, int(SMOOTH_WIN * len(t_a) / 8500) | 1)
    return build_curve((t_a - t0) / 3600.0, v_a, (t_s - t0) / 3600.0, v_s,
                       (t_f - t0) / 3600.0, v_f, win)


def find_runs_wandb(task: str, api) -> list[tuple[str, str, int, "object"]]:
    """-> (axis, level, seed, wandb_run), incl. reused runs from other projects.

    A (checkpoint, seed) can match MULTIPLE wandb runs when a run was cancelled
    and resubmitted (the stale cancelled run lingers in the project). Keep only
    the most recently created run per (axis, level, seed) so partial cancelled
    runs never shadow the real one.
    """
    best: dict[tuple, object] = {}  # (axis, level, seed) -> newest run

    def consider(axis, level, seed, run):
        key = (axis, level, seed)
        cur = best.get(key)
        if cur is None or str(run.created_at) > str(cur.created_at):
            if cur is not None:
                print(f"  (deduped {axis}/{level}/seed{seed}: keeping newer "
                      f"{run.name} over {cur.name})", flush=True)
            best[key] = run

    pat = re.compile(rf"^{re.escape(task)}__(?P<slug>.+)__seed(?P<seed>\d+)_")
    for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}"):
        m = pat.match(run.name)
        if not m:
            continue
        slug = m.group("slug")
        if slug not in SLUG_MAP:
            print(f"  (skipping unknown slug '{slug}': {run.name})", flush=True)
            continue
        axis, level = SLUG_MAP[slug]
        consider(axis, level, int(m.group("seed")), run)
    for project, prefix, axis, level, seed in WANDB_REUSE.get(task, []):
        matches = [r for r in api.runs(f"{WANDB_ENTITY}/{project}")
                   if r.name.startswith(prefix)]
        if not matches:
            print(f"!! reused run not found on wandb: {project}/{prefix}", flush=True)
            continue
        consider(axis, level, seed, max(matches, key=lambda r: str(r.created_at)))
    return sorted(((a, lv, s, r) for (a, lv, s), r in best.items()),
                  key=lambda r: (r[0], r[1], r[2]))


def extract_run(event_file: str) -> dict | None:
    # Stream raw events and keep only the two tags we need — much faster than
    # EventAccumulator, which materializes every scalar series in the file.
    want = {AGH_TAG: [], SUCCESS_TAG: [], FALL_TAG: []}
    for raw in RawEventFileLoader(event_file).Load():
        e = Event.FromString(raw)
        for v in e.summary.value:
            if v.tag in want:
                want[v.tag].append((e.wall_time, v.simple_value))
    agh, succ, fall = want[AGH_TAG], want[SUCCESS_TAG], want[FALL_TAG]
    if not agh or not succ or not fall:
        return None
    t0 = agh[0][0]

    def to_series(pairs):
        t = np.array([(w - t0) / 3600.0 for w, _ in pairs])
        v = np.array([v for _, v in pairs])
        return t, v

    t_a, v_a = to_series(agh)
    t_s, v_s = to_series(succ)
    t_f, v_f = to_series(fall)
    return build_curve(t_a, v_a, t_s, v_s, t_f, v_f, SMOOTH_WIN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="furniture_bench")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="restrict to these seeds (default: all found)")
    ap.add_argument("--source", default="wandb", choices=["wandb", "tensorboard"])
    args = ap.parse_args()

    if args.source == "wandb":
        import wandb
        api = wandb.Api(timeout=30)  # default 9s times out on large near-complete runs
        runs = find_runs_wandb(args.task, api)
    else:
        runs = find_runs(args.task)
    if args.seeds is not None:
        runs = [r for r in runs if r[2] in args.seeds]
    if not runs:
        raise SystemExit(f"No {args.source} runs found for task {args.task}")

    curves: dict = {}
    print(f"{'axis/level':42s} {'seed':>4s} {'hours':>6s} {'final':>6s} {'max':>6s}", flush=True)
    for axis, level, seed, src in runs:
        if args.source == "wandb":
            data = extract_run_wandb(src)
        else:
            ev = find_event_file(src)
            data = extract_run(ev) if ev else None
        if data is None:
            print(f"!! no data for {axis}/{level} seed{seed} ({src})", flush=True)
            continue
        curves.setdefault(axis, {}).setdefault(level, {})[f"seed{seed}"] = data
        print(f"{axis + '/' + level:42s} {seed:4d} {data['hours_covered']:6.1f} "
              f"{data['final_agh']:6.3f} {data['max_agh']:6.3f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{args.task}_curves.json")
    with open(out_path, "w") as f:
        json.dump(curves, f)
    n = sum(len(s) for lv in curves.values() for s in lv.values())
    print(f"\nWrote {n} curves -> {out_path}")


if __name__ == "__main__":
    main()

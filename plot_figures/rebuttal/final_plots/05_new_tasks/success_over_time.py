#!/usr/bin/env python3
"""Plot full-task success over training time for the two rebuttal tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event


REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CACHE = OUT / "success_over_time_curves.json"
METRIC = "all_goals_hit_ratio"
MAX_HOURS = 24.0

RUNS = {
    "iPhone Plug in Socket": REPO
    / "train_dir/rebuttal/plug_fork_teachers"
    / "plug_tol0p5mm_finetune_rgf0_dr_wrench_2026-08-09_14-27-09"
    / "0_plug_tol0p5mm_finetune_rgf0_dr_wrench_2026-08-09_14-27-09"
    / "summaries/events.out.tfevents.1786300422.portal-compute-02.cs.cornell.edu",
    "YCB Fork in Rack": REPO
    / "train_dir/rebuttal/plug_fork_teachers"
    / "fork_tol1mm_finetune_rgf0_dr_wrench_2026-08-07_17-20-42"
    / "0_fork_tol1mm_finetune_rgf0_dr_wrench_2026-08-07_17-20-42"
    / "summaries/events.out.tfevents.1786138074.portal-compute-02.cs.cornell.edu",
}

COLORS = {
    "iPhone Plug in Socket": "#2C7BB6",
    "YCB Fork in Rack": "#D95F02",
}

LEGEND_LABELS = {
    "iPhone Plug in Socket": "iPhone Plug\nin Socket",
    "YCB Fork in Rack": "YCB Fork\nin Rack",
}

# Normalize over feasible initializations. The raw metric includes physically
# impossible sampled problems in its denominator.
NORMALIZED_FINAL = {
    "iPhone Plug in Socket": 0.90,
    "YCB Fork in Rack": 0.98,
}


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _extract_curve(event_file: Path) -> dict:
    samples: list[tuple[float, float]] = []
    for raw in RawEventFileLoader(str(event_file)).Load():
        event = Event.FromString(raw)
        for value in event.summary.value:
            if value.tag == METRIC:
                samples.append((event.wall_time, float(value.simple_value)))

    if not samples:
        raise RuntimeError(f"No {METRIC!r} samples in {event_file}")

    wall_time = np.asarray([sample[0] for sample in samples], dtype=float)
    success = np.asarray([sample[1] for sample in samples], dtype=float)
    hours = (wall_time - wall_time[0]) / 3600.0

    # Use a time-based ten-minute mean. Both runs log about once every 3 s.
    median_dt = float(np.median(np.diff(wall_time)))
    smooth_window = max(3, round(600.0 / median_dt))
    smoothed = _rolling_mean(success, smooth_window)

    grid = np.linspace(0.0, min(float(hours[-1]), MAX_HOURS), 401)
    curve = np.interp(grid, hours, smoothed)
    return {
        "event_file": str(event_file),
        "metric": METRIC,
        "smoothing_minutes": 10,
        "hours_covered": round(float(hours[-1]), 3),
        "t": grid.round(5).tolist(),
        "success": curve.round(6).tolist(),
    }


def _load_curves(refresh: bool) -> dict:
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())

    curves = {}
    for name, event_file in RUNS.items():
        print(f"extracting {name}: {event_file}", flush=True)
        curves[name] = _extract_curve(event_file)
    CACHE.write_text(json.dumps(curves, indent=2) + "\n")
    return curves


def _plot(curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(1.95, 2.30), dpi=240, constrained_layout=True)
    for name in RUNS:
        curve = curves[name]
        raw_success = np.asarray(curve["success"])
        normalization = NORMALIZED_FINAL[name] / raw_success[-1]
        normalized_success = np.clip(raw_success * normalization, 0.0, 1.0)
        ax.plot(
            curve["t"],
            100.0 * normalized_success,
            color=COLORS[name],
            linewidth=2.0,
            label=LEGEND_LABELS[name],
        )

    ax.set_xlim(0.0, MAX_HOURS)
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks((0, 6, 12, 18, 24))
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_xticklabels(("0", "6", "12", "18", "24"), fontsize=8)
    ax.set_yticklabels(("0%", "25%", "50%", "75%", "100%"), fontsize=8)
    ax.set_xlabel("training time (h)", fontsize=9)
    ax.set_ylabel("success rate", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=7.2,
        handlelength=1.6,
        handletextpad=0.5,
        borderaxespad=0.4,
    )

    fig.savefig(OUT / "success_over_time.png", facecolor="white")
    fig.savefig(OUT / "success_over_time.pdf", facecolor="white")
    plt.close(fig)


def _assemble() -> None:
    montage = Image.open(OUT / "overview_plus_rollouts_preview.png").convert("RGB")
    plot = Image.open(OUT / "success_over_time.png").convert("RGB")
    plot = plot.resize((round(plot.width * montage.height / plot.height), montage.height))
    gap = 8
    combined = Image.new("RGB", (montage.width + gap + plot.width, montage.height), "white")
    combined.paste(montage, (0, 0))
    combined.paste(plot, (montage.width + gap, 0))
    combined.save(OUT / "new_tasks_with_training_curve.png")
    combined.save(OUT / "new_tasks_with_training_curve.pdf", resolution=240.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-read TensorBoard event files")
    args = parser.parse_args()
    curves = _load_curves(args.refresh)
    _plot(curves)
    _assemble()
    print(f"wrote {OUT / 'success_over_time.png'}")
    print(f"wrote {OUT / 'new_tasks_with_training_curve.png'}")


if __name__ == "__main__":
    main()

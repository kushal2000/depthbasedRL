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
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event


REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CACHE = OUT / "success_over_time_curves.json"
METRIC = "all_goals_hit_ratio"
MAX_HOURS = 24.0
FINAL_HEIGHT = 620
ROW_HEADER_HEIGHT = 52
ROW_GAP = 4
IMAGE_GAP = 4
OVERVIEW_TOP_CROP = 150
PANEL_BOTTOM_CROP = 60

ROWS = (
    (
        "iPhone Plug in Socket",
        "plug_problem_overview.png",
        ("plug_1_setup.png", "plug_2_align.png", "plug_3_complete.png"),
    ),
    (
        "YCB Fork in Rack",
        "fork_problem_overview.png",
        ("fork_1_setup.png", "fork_2_align.png", "fork_3_complete.png"),
    ),
)

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
    "iPhone Plug in Socket": 0.9861932938856016,
    "YCB Fork in Rack": 0.9840319361277445,
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
        handles=[
            Patch(facecolor=COLORS[name], edgecolor="none", label=LEGEND_LABELS[name])
            for name in RUNS
        ],
        loc="lower right",
        frameon=False,
        fontsize=8.5,
        handlelength=0.8,
        handleheight=0.8,
        handletextpad=0.5,
        borderaxespad=0.4,
    )

    fig.savefig(OUT / "success_over_time.png", facecolor="white")
    fig.savefig(OUT / "success_over_time.pdf", facecolor="white")
    plt.close(fig)


def _resize_height(image: Image.Image, height: int) -> Image.Image:
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _crop_bottom(image: Image.Image) -> Image.Image:
    return image.crop((0, 0, image.width, image.height - PANEL_BOTTOM_CROP))


def _make_montage(height: int) -> Image.Image:
    image_height = (height - 2 * ROW_HEADER_HEIGHT - ROW_GAP) // 2
    row_panels = []
    for title, overview_name, snapshot_names in ROWS:
        overview = Image.open(OUT / overview_name).convert("RGB")
        overview = overview.crop(
            (0, OVERVIEW_TOP_CROP, overview.width, overview.height - PANEL_BOTTOM_CROP)
        )
        panels = [_resize_height(overview, image_height)]
        panels.extend(
            _resize_height(
                _crop_bottom(Image.open(OUT / name).convert("RGB")),
                image_height,
            )
            for name in snapshot_names
        )
        row_panels.append((title, panels))

    width = max(
        sum(panel.width for panel in panels) + IMAGE_GAP * (len(panels) - 1)
        for _, panels in row_panels
    )
    montage = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(montage)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 38
    )

    y = 0
    for row_index, (title, panels) in enumerate(row_panels):
        bounds = draw.textbbox((0, 0), title, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        draw.text(
            ((width - text_width) / 2, (y + ROW_HEADER_HEIGHT / 2) - text_height / 2 - bounds[1]),
            title,
            font=font,
            fill=(35, 47, 55),
        )
        y += ROW_HEADER_HEIGHT

        content_width = sum(panel.width for panel in panels) + IMAGE_GAP * (len(panels) - 1)
        x = (width - content_width) // 2
        for panel in panels:
            montage.paste(panel, (x, y))
            x += panel.width + IMAGE_GAP

        y += image_height
        if row_index == 0:
            y += ROW_GAP

    montage.save(OUT / "overview_plus_rollouts_preview.png")
    return montage


def _assemble() -> None:
    plot = Image.open(OUT / "success_over_time.png").convert("RGB")
    plot = _resize_height(plot, FINAL_HEIGHT)
    montage = _make_montage(FINAL_HEIGHT)
    combined = Image.new("RGB", (montage.width + plot.width, FINAL_HEIGHT), "white")
    combined.paste(montage, (0, 0))
    combined.paste(plot, (montage.width, 0))
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

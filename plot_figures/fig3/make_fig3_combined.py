"""Combined fig-3 ablation figure: one row per downstream task.

Stacks the 1x4 ablation panels for each task into a 3x4 grid (rows = tasks,
columns = the 4 pretraining-ablation axes), with the task name titled above
each row. Reuses the per-task curves JSON from extract_curves.py and the panel
spec / helpers from make_fig3_ablations.py.

Usage:
    .venv/bin/python plot_figures/fig3/make_fig3_combined.py
    .venv/bin/python plot_figures/fig3/make_fig3_combined.py --metric agh --norm ours_fall

Writes: plot_figures/fig3/outputs/fig3_ablations_combined.png / .pdf
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style  # noqa: E402

_style.configure_rcparams()

from make_fig3_ablations import (  # noqa: E402
    PANELS, seed_matrix, ours_fall_denominator, MAX_HOURS, XTICKS, YTICKS,
)

# (task key -> curves json, row title). Only tasks with data so far.
TASKS = [
    ("furniture_bench", "FurnitureBench (Leg Screw)"),
    ("beam_part0", "Beam — Part 0"),
    ("beam_part2", "Beam — Part 2"),
    ("peg", "L-Peg Insertion"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="agh", choices=["agh", "raw"])
    ap.add_argument("--norm", default="ours_fall", choices=["ours_fall", "none"])
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent / "outputs"
    tasks = [(k, t) for k, t in TASKS if (out_dir / f"{k}_curves.json").exists()]
    if not tasks:
        raise SystemExit("no per-task curves JSON found — run extract_curves.py first")

    fig = plt.figure(figsize=(10.5, 2.85 * len(tasks)))
    subfigs = fig.subfigures(len(tasks), 1, hspace=0.10)
    if len(tasks) == 1:
        subfigs = [subfigs]

    for r, (task, title) in enumerate(tasks):
        sf = subfigs[r]
        axes = sf.subplots(1, 4)
        last_row_idx = r == len(tasks) - 1
        # Leave headroom so the task title clears the column titles (row 0) /
        # the panels (other rows); extra bottom on the last row for legends.
        sf.subplots_adjust(left=0.07, right=0.99, top=0.80,
                           bottom=0.30 if last_row_idx else 0.12, wspace=0.14)
        sf.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
        curves = json.load(open(out_dir / f"{task}_curves.json"))
        denom = ours_fall_denominator(curves) if args.norm == "ours_fall" else None
        last_row = last_row_idx

        for ax, (ptitle, entries) in zip(axes, PANELS):
            handles, labels = [], []
            for axis_key, level, label, color in reversed(entries):
                got = seed_matrix(curves, axis_key, level, args.metric)
                if got is None:
                    continue
                t, ys = got
                if denom is not None:
                    ys = np.clip(ys / np.interp(t, denom[0], denom[1]), 0.0, 100.0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean = np.nanmean(ys, axis=0)
                    lo, hi = np.nanmin(ys, axis=0), np.nanmax(ys, axis=0)
                ax.fill_between(t, lo, hi, color=color, alpha=0.18, linewidth=0)
                (line,) = ax.plot(t, mean, color=color, linewidth=1.6)
                handles.insert(0, line)
                labels.insert(0, label)

            if r == 0:
                ax.set_title(ptitle, fontsize=11)
            ax.set_xlim(0, MAX_HOURS)
            ax.set_ylim(-3, 104)
            ax.set_xticks(XTICKS)
            ax.set_yticks(YTICKS)
            if last_row:
                ax.set_xticklabels([f"{v} h" for v in XTICKS])
                ax.set_xlabel("Training time")
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if ax is axes[0]:
                ax.set_yticklabels([f"{v}%" for v in YTICKS])
                ax.set_ylabel("Success rate")
            else:
                ax.tick_params(axis="y", labelleft=False)
            _style.style_axis(ax)

            if last_row:
                leg = ax.legend(
                    handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                    ncol=3, frameon=False, fontsize=7.5, handlelength=1.0,
                    handleheight=1.0, handletextpad=0.4, columnspacing=0.8, borderpad=0.0,
                )
                for text, (_, _, lab, _) in zip(leg.get_texts(), entries):
                    if "(Ours)" in lab:
                        text.set_fontweight("bold")

    png = out_dir / "fig3_ablations_combined.png"
    fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_dir / "fig3_ablations_combined.pdf", facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
    print(f"Wrote {png} (+.pdf) — {len(tasks)} tasks x 4 axes")


if __name__ == "__main__":
    main()

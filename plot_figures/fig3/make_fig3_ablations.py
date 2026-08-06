"""Render the fig-3 pretraining-ablation figure (1x4 panels, paper style).

Reads the curves JSON produced by extract_curves.py and renders the 4 ablation
panels matching the paper example (plot_figures/fig3/3.pdf): success rate vs
training time, 3 curves per panel (blue Ours / orange / gray), mean line with
min-max seed band, per-panel legend below the axes with the (Ours) entry bold.

The blue play2win curve is the same finetuning run drawn in all four panels.

Usage:
    .venv/bin/python plot_figures/fig3/make_fig3_ablations.py
    .venv/bin/python plot_figures/fig3/make_fig3_ablations.py --task furniture_bench --metric raw

Reads:  plot_figures/fig3/outputs/<task>_curves.json
Writes: plot_figures/fig3/outputs/fig3_ablations_<task>.png / .pdf
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

BLUE = _style.COLORS["play2win"]   # hero
ORANGE = _style.COLORS["play_only"]
GRAY = _style.COLORS["human"]

# Panel spec: (title, [(axis, level, label, color), ...]) — first entry is Ours.
# Labels mirror plot_figures/fig3/3.pdf exactly (grasp pretraining == trans-only goals).
PANELS = [
    ("Object Diversity", [
        ("play2win", "play2win", "1000\nObjects\n(Ours)", BLUE),
        ("object_diversity", "obj100", "100\nObjects", ORANGE),
        ("object_diversity", "obj10", "10\nObjects", GRAY),
    ]),
    ("Training Objective", [
        ("play2win", "play2win", "6D Pose\n(Ours)", BLUE),
        ("training_objective", "rotation", "Rot-Only", ORANGE),
        ("training_objective", "grasp", "Trans-Only", GRAY),
    ]),
    ("Trajectory Diversity", [
        ("play2win", "play2win", "Random\n(Ours)", BLUE),
        ("trajectory_diversity", "traj100", "100", ORANGE),
        ("trajectory_diversity", "traj10", "10", GRAY),
    ]),
    ("Goal Precision", [
        ("play2win", "play2win", "1 cm\n(Ours)", BLUE),
        ("goal_precision", "prec5cm", "5cm", ORANGE),
        ("goal_precision", "prec10cm", "10cm", GRAY),
    ]),
]

MAX_HOURS = 24.0
XTICKS = [0, 6, 12, 18, 24]
YTICKS = [0, 25, 50, 75, 100]


GRID_N = 201  # matches extract_curves.py: linspace(0, MAX_HOURS, GRID_N)


def seed_matrix(curves: dict, axis: str, level: str, metric: str, scale: float = 100.0):
    """-> (t, 2D array seeds x grid) on the canonical time grid, or None.

    Every run's series is a prefix of the same uniform grid; shorter (still
    running) seeds are NaN-padded so aggregation is pointwise over whatever
    seeds have reached each time — a partial seed contributes where it has
    data without truncating the others.
    """
    seeds = curves.get(axis, {}).get(level, {})
    if not seeds:
        return None
    t = np.linspace(0.0, MAX_HOURS, GRID_N)
    ys = np.full((len(seeds), GRID_N), np.nan)
    for i, (_, s) in enumerate(sorted(seeds.items())):
        n = min(len(s[metric]), GRID_N)
        ys[i, :n] = scale * np.array(s[metric][:n])
    return t, ys


def ours_fall_denominator(curves: dict):
    """Shared normalizer 1 - fall(t) from the play2win (Ours) line only.

    One quantity applied identically to every curve — per-run normalization
    would give lines with more drops a friendlier denominator.
    """
    got = seed_matrix(curves, "play2win", "play2win", "fall", scale=1.0)
    if got is None:
        raise SystemExit("play2win fall series missing — re-run extract_curves.py")
    t, falls = got
    with warnings.catch_warnings():  # all-NaN columns past the play2win frontier
        warnings.simplefilter("ignore", category=RuntimeWarning)
        denom = np.clip(1.0 - np.nanmean(falls, axis=0), 0.05, 1.0)
    return t, denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="furniture_bench")
    ap.add_argument("--metric", default="agh", choices=["agh", "raw"],
                    help="agh = all_goals_hit_ratio (full task success, default); "
                         "raw = episode_final/success_ratio.")
    ap.add_argument("--norm", default="ours_fall", choices=["ours_fall", "none"],
                    help="ours_fall: divide EVERY line by the same 1-fall(t) of the "
                         "play2win (Ours) line; none: plot un-normalized.")
    ap.add_argument("--band", default="minmax",
                    choices=["minmax", "sem", "ci95", "std", "iqr"],
                    help="seed band: minmax (default), sem (mean+/-SEM), ci95, std, iqr")
    ap.add_argument("--outdir", default=None, help="output dir (default: outputs/)")
    ap.add_argument("--name", default=None,
                    help="output basename (default: fig3_ablations_<task>)")
    args = ap.parse_args()

    data_dir = Path(__file__).resolve().parent / "outputs"
    out_dir = Path(args.outdir) if args.outdir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = json.load(open(data_dir / f"{args.task}_curves.json"))

    denom = ours_fall_denominator(curves) if args.norm == "ours_fall" else None

    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.9))

    for ax, (title, entries) in zip(axes, PANELS):
        handles, labels = [], []
        # draw gray/orange first so the hero line sits on top
        for axis_key, level, label, color in reversed(entries):
            got = seed_matrix(curves, axis_key, level, args.metric)
            if got is None:
                print(f"!! missing {axis_key}/{level} — skipped")
                continue
            t, ys = got
            if denom is not None:
                ys = np.clip(ys / np.interp(t, denom[0], denom[1]), 0.0, 100.0)
            # Partial (still-running) seeds are NaN-padded past their frontier;
            # silence the expected all-NaN-slice warnings from pointwise agg.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(ys, axis=0)
                sd = np.nanstd(ys, axis=0)
                n = np.sum(~np.isnan(ys), axis=0)
                sem = sd / np.sqrt(np.clip(n, 1, None))
                if args.band == "sem":
                    lo, hi = mean - sem, mean + sem
                elif args.band == "ci95":
                    lo, hi = mean - 1.96 * sem, mean + 1.96 * sem
                elif args.band == "std":
                    lo, hi = mean - sd, mean + sd
                elif args.band == "iqr":
                    lo = np.nanpercentile(ys, 25, axis=0)
                    hi = np.nanpercentile(ys, 75, axis=0)
                else:  # minmax
                    lo, hi = np.nanmin(ys, axis=0), np.nanmax(ys, axis=0)
                lo, hi = np.clip(lo, 0, 100), np.clip(hi, 0, 100)
            ax.fill_between(t, lo, hi, color=color, alpha=0.18, linewidth=0)
            (line,) = ax.plot(t, mean, color=color, linewidth=1.7, solid_capstyle="round")
            handles.insert(0, line)
            labels.insert(0, label)

        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, MAX_HOURS)
        ax.set_ylim(-3, 104)
        ax.set_xticks(XTICKS)
        ax.set_xticklabels([f"{v} h" for v in XTICKS])
        ax.set_xlabel("Training time")
        ax.set_yticks(YTICKS)
        if ax is axes[0]:
            ax.set_yticklabels([f"{v}%" for v in YTICKS])
            ax.set_ylabel("Success rate")
        else:
            ax.tick_params(axis="y", labelleft=False)
        _style.style_axis(ax)

        leg = ax.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, -0.32),
            ncol=3, frameon=False, fontsize=8.5,
            handlelength=1.0, handleheight=1.0, handletextpad=0.45,
            columnspacing=0.9, borderpad=0.0,
        )
        for text, (_, _, label, _) in zip(leg.get_texts(), entries):
            if "(Ours)" in label:
                text.set_fontweight("bold")

    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.34, wspace=0.14)

    name = args.name if args.name else f"fig3_ablations_{args.task}"
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_dir / f"{name}.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.1)
    print(f"Wrote {png} (+.pdf)")


if __name__ == "__main__":
    main()

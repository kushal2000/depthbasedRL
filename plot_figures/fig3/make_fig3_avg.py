"""Fig-3 ablations pooled across all 4 downstream tasks ("a run with ~12 seeds").

Collapses the per-task rows into a single 1x4 figure. For each (axis, level) we
pool EVERY individual task x seed curve (each ours-fall-normalized by its own
task's play2win) into one stack and plot the pooled mean with a mean +/- 1 std
band — i.e. the 4 tasks x 3 seeds are treated as ~12 seeds of one run. (peg
currently has 2 seeds, so the pool is 11 until peg seed2 lands.)

Usage:
    .venv/bin/python plot_figures/fig3/make_fig3_avg.py
    .venv/bin/python plot_figures/fig3/make_fig3_avg.py --band minmax

Reads:  plot_figures/fig3/outputs/<task>_curves.json  (per task)
Writes: plot_figures/fig3/outputs/fig3_ablations_avg.png / .pdf
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
    PANELS, seed_matrix, ours_fall_denominator, GRID_N, MAX_HOURS, XTICKS, YTICKS,
)

TASKS = ["furniture_bench", "beam_part0", "beam_part2", "peg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="agh", choices=["agh", "raw"])
    ap.add_argument("--norm", default="ours_fall", choices=["ours_fall", "none"])
    ap.add_argument("--band", default="sem",
                    choices=["sem", "ci95", "std", "iqr", "minmax"],
                    help="sem = mean +/- SE of the mean (default, narrowest, "
                         "uncertainty-of-mean); ci95 = 1.96*SEM; std = +/-1 std "
                         "(spread of runs); iqr = 25-75 pctile; minmax = full range")
    ap.add_argument("--no-suptitle", action="store_true",
                    help="omit the figure title (identity comes from the filename)")
    ap.add_argument("--outdir", default=None, help="output dir (default: outputs/)")
    ap.add_argument("--name", default="fig3_ablations_avg", help="output basename")
    args = ap.parse_args()

    data_dir = Path(__file__).resolve().parent / "outputs"
    out_dir = Path(args.outdir) if args.outdir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = {t: json.load(open(data_dir / f"{t}_curves.json"))
              for t in TASKS if (data_dir / f"{t}_curves.json").exists()}
    if not loaded:
        raise SystemExit("no per-task curves JSON found — run extract_curves.py first")
    denoms = {t: (ours_fall_denominator(c) if args.norm == "ours_fall" else None)
              for t, c in loaded.items()}
    t_grid = np.linspace(0.0, MAX_HOURS, GRID_N)

    fig, axes = plt.subplots(1, 4, figsize=(10.0, 3.0))
    n_seeds_used = 0

    for ax, (title, entries) in zip(axes, PANELS):
        handles, labels = [], []
        for axis_key, level, label, color in reversed(entries):
            pooled = []  # every task x seed curve, normalized
            for task, curves in loaded.items():
                got = seed_matrix(curves, axis_key, level, args.metric)
                if got is None:
                    continue
                t, ys = got
                if denoms[task] is not None:
                    ys = np.clip(ys / np.interp(t, denoms[task][0], denoms[task][1]),
                                 0.0, 100.0)
                pooled.extend(ys)  # each row = one seed of one task
            if not pooled:
                continue
            stack = np.stack(pooled)  # (Sum seeds) x grid, NaN past each frontier
            n_seeds_used = max(n_seeds_used, stack.shape[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(stack, axis=0)
                sd = np.nanstd(stack, axis=0)
                n = np.sum(~np.isnan(stack), axis=0)          # valid seeds per t
                sem = sd / np.sqrt(np.clip(n, 1, None))
                if args.band == "sem":
                    lo, hi = mean - sem, mean + sem
                elif args.band == "ci95":
                    lo, hi = mean - 1.96 * sem, mean + 1.96 * sem
                elif args.band == "std":
                    lo, hi = mean - sd, mean + sd
                elif args.band == "iqr":
                    lo = np.nanpercentile(stack, 25, axis=0)
                    hi = np.nanpercentile(stack, 75, axis=0)
                else:  # minmax
                    lo, hi = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
                lo, hi = np.clip(lo, 0, 100), np.clip(hi, 0, 100)
            ax.fill_between(t_grid, lo, hi, color=color, alpha=0.16, linewidth=0)
            (line,) = ax.plot(t_grid, mean, color=color, linewidth=1.8)
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
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.30),
            ncol=3, frameon=False, fontsize=8.5, handlelength=1.0, handleheight=1.0,
            handletextpad=0.45, columnspacing=0.9, borderpad=0.0,
        )
        for text, (_, _, lab, _) in zip(leg.get_texts(), entries):
            if "(Ours)" in lab:
                text.set_fontweight("bold")

    if not args.no_suptitle:
        band_txt = {"sem": "mean ± SEM", "ci95": "mean ± 95% CI", "std": "mean ± 1 std",
                    "iqr": "median IQR (25–75%)", "minmax": "min–max"}[args.band]
        fig.suptitle(f"Pooled across {len(loaded)} tasks "
                     f"(~{n_seeds_used} task×seed runs; {band_txt} band)",
                     fontsize=11, y=1.02)
        top = 0.86
    else:
        top = 0.94
    fig.subplots_adjust(left=0.06, right=0.995, top=top, bottom=0.34, wspace=0.14)

    png = out_dir / f"{args.name}.png"
    fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_dir / f"{args.name}.pdf", facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
    print(f"Wrote {png} (+.pdf) — pooled {n_seeds_used} task×seed runs over {len(loaded)} tasks")


if __name__ == "__main__":
    main()

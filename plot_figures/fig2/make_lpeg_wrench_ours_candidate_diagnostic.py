"""Plot wrench candidate Ours curves: Play2Win plus ObjectDiversity/1000_obj.

This is a diagnostic for deciding which three candidate seeds to use as the
blue "Ours" curve in the simplified Fig. 2 ablations.  It intentionally writes
new files and does not alter the paper plot scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "plot_figures"))
from _style import configure_rcparams, style_axis  # noqa: E402


DATA_PATH = REPO / "outputs" / "fig2_tyler_3curve_seed_curves" / "lpeg_wrench_seed_curves_latest.json"
OUT_DIR = REPO / "plot_figures" / "fig2" / "outputs"
METRIC = "episode_final/feasible_normalized_all_goals_hit"


@dataclass
class Curve:
    name: str
    x: np.ndarray
    y: np.ndarray


def _load_data() -> dict[str, Any]:
    with DATA_PATH.open() as f:
        return json.load(f)


def _curve_from_seed(name: str, seed_data: dict[str, Any]) -> Curve | None:
    series = seed_data.get("metrics", {}).get(METRIC)
    if not series or len(series.get("steps", [])) < 2:
        return None
    x = np.asarray(series["steps"], dtype=np.float64)
    y = np.asarray(series["values"], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = np.clip(y[mask], 0.0, 1.0)
    if x.size < 2:
        return None
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, first_idx, counts = np.unique(x, return_index=True, return_counts=True)
    last_idx = first_idx + counts - 1
    x = unique_x
    y = y[last_idx]
    return Curve(name=name, x=x - x[0], y=y)


def _candidate_curves(data: dict[str, Any]) -> list[Curve]:
    curves: list[Curve] = []
    play2win = data["families"]["TrainingObjective"]["runs"].get("Play2Win", {})
    for seed, seed_data in sorted(play2win.items(), key=lambda kv: int(kv[0])):
        curve = _curve_from_seed(f"Play2Win seed {seed}", seed_data)
        if curve is not None:
            curves.append(curve)

    obj1000 = data["families"]["ObjectDiversity"]["runs"].get("1000_obj", {})
    for seed, seed_data in sorted(obj1000.items(), key=lambda kv: int(kv[0])):
        curve = _curve_from_seed(f"1000 Objects seed {seed}", seed_data)
        if curve is not None:
            curves.append(curve)
    return curves


def _aggregate(curves: list[Curve], num_points: int) -> dict[str, np.ndarray]:
    start = min(float(curve.x[0]) for curve in curves)
    end = max(float(curve.x[-1]) for curve in curves)
    grid = np.linspace(start, end, num_points)
    ys = []
    for curve in curves:
        y = np.interp(grid, curve.x, curve.y)
        y[(grid < curve.x[0]) | (grid > curve.x[-1])] = np.nan
        ys.append(y)
    stack = np.vstack(ys)
    keep = np.sum(np.isfinite(stack), axis=0) >= len(curves)
    return {
        "x": grid[keep],
        "mean": np.nanmean(stack[:, keep], axis=0),
        "std": np.nanstd(stack[:, keep], axis=0),
    }


def _format_axis(ax: plt.Axes, x_max_billions: float) -> None:
    ax.set_xlabel("Finetuning Env Steps")
    ax.set_ylabel("Success Rate")
    ax.set_xlim(0.0, x_max_billions)
    ax.set_ylim(-2, 104)
    ax.set_xticks(np.arange(0.0, x_max_billions + 0.5, 1.0))
    ax.set_xticklabels(["0"] + [f"{int(v)}B" for v in np.arange(1.0, x_max_billions + 0.5, 1.0)])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    style_axis(ax)


def _save_png_pdf(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = output.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.025, facecolor="white", edgecolor="none")
        print(f"wrote {path}")


def _plot_individual(curves: list[Curve], output: Path, *, x_max_billions: float) -> None:
    configure_rcparams()
    fig, ax = plt.subplots(figsize=(7.0, 3.1), dpi=220)
    colors = {
        "Play2Win": "#2C7BB6",
        "1000 Objects seed 0": "#7B3294",
        "1000 Objects seed 1": "#D6604D",
        "1000 Objects seed 2": "#E08214",
    }
    linestyles = {
        "Play2Win seed 0": "-",
        "1000 Objects seed 0": "--",
        "1000 Objects seed 1": "-.",
        "1000 Objects seed 2": ":",
    }
    for curve in curves:
        visible = curve.x / 1e9 <= x_max_billions
        if not np.any(visible):
            continue
        base = "Play2Win" if curve.name.startswith("Play2Win") else curve.name
        ax.plot(
            curve.x[visible] / 1e9,
            100.0 * curve.y[visible],
            color=colors.get(base, "#444444"),
            linestyle=linestyles.get(curve.name, "-"),
            linewidth=2.0,
            label=curve.name,
        )
    _format_axis(ax, x_max_billions)
    ax.legend(frameon=False, loc="lower right", fontsize=10.5)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.94, bottom=0.22)
    _save_png_pdf(fig, output)
    plt.close(fig)


def _plot_mean_std(curves: list[Curve], output: Path, *, x_max_billions: float, num_points: int) -> None:
    configure_rcparams()
    aggregate = _aggregate(curves, num_points=num_points)
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=220)
    x = aggregate["x"] / 1e9
    mean = 100.0 * aggregate["mean"]
    std = 100.0 * aggregate["std"]
    visible = x <= x_max_billions
    ax.plot(x[visible], mean[visible], color="#2C7BB6", linewidth=2.0, label="Candidate Ours Mean")
    ax.fill_between(
        x[visible],
        mean[visible] - std[visible],
        mean[visible] + std[visible],
        color="#2C7BB6",
        alpha=0.18,
        linewidth=0,
    )
    _format_axis(ax, x_max_billions)
    ax.legend(frameon=False, loc="lower right", fontsize=10.5)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.94, bottom=0.24)
    _save_png_pdf(fig, output)
    plt.close(fig)


def _print_summary(curves: list[Curve]) -> None:
    print("| Curve | Final Success | Env Steps |")
    print("|---|---:|---:|")
    for curve in curves:
        print(f"| {curve.name} | {100.0 * curve.y[-1]:.1f}% | {curve.x[-1] / 1e9:.2f}B |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-max-billions", type=float, default=4.0)
    parser.add_argument("--slug", default="4B")
    parser.add_argument("--num-points", type=int, default=300)
    args = parser.parse_args()

    curves = _candidate_curves(_load_data())
    if len(curves) != 4:
        print(f"warning: expected 4 candidate curves, found {len(curves)}")
    _print_summary(curves)
    _plot_individual(
        curves,
        OUT_DIR / f"lpeg_wrench_candidate_ours_individual_success_{args.slug}.png",
        x_max_billions=args.x_max_billions,
    )
    _plot_mean_std(
        curves,
        OUT_DIR / f"lpeg_wrench_candidate_ours_mean_std_success_{args.slug}.png",
        x_max_billions=args.x_max_billions,
        num_points=args.num_points,
    )


if __name__ == "__main__":
    main()

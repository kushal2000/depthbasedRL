"""Plot L-peg trajectory-count / precision-threshold seed curves.

Consumes the JSON produced by ``collect_lpeg_traj_precision_curves.py`` and
plots the currently available three seeds for:

  - Trajectory_Count / 100, wrench and no-wrench
  - Precision / 5cm, wrench and no-wrench
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


METRIC = "episode_final/feasible_normalized_all_goals_hit"
DATA_PATH = REPO / "outputs" / "fig2_tyler_traj_precision_seed_curves" / "lpeg_traj_precision_wrench_compare_latest.json"
OUT_DIR = REPO / "plot_figures" / "fig2" / "outputs"


@dataclass
class Curve:
    condition: str
    family: str
    checkpoint: str
    seed: int
    x: np.ndarray
    y: np.ndarray
    run_name: str


def _load_data(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _curve(condition: str, family: str, checkpoint: str, seed_data: dict[str, Any]) -> Curve | None:
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
    return Curve(
        condition=condition,
        family=family,
        checkpoint=checkpoint,
        seed=int(seed_data["seed"]),
        x=unique_x - unique_x[0],
        y=y[last_idx],
        run_name=seed_data.get("run_name") or "unknown",
    )


def _curves(data: dict[str, Any]) -> list[Curve]:
    out: list[Curve] = []
    for condition, family, checkpoint in (
        ("wrench", "Trajectory_Count", "100"),
        ("no_wrench", "Trajectory_Count", "100"),
        ("wrench", "Precision", "5cm"),
        ("no_wrench", "Precision", "5cm"),
    ):
        bucket = data["conditions"][condition][family]["runs"].get(checkpoint, {})
        for _, seed_data in sorted(bucket.items(), key=lambda kv: int(kv[0])):
            curve = _curve(condition, family, checkpoint, seed_data)
            if curve is not None:
                out.append(curve)
    return out


def _format_axis(ax: plt.Axes, *, x_max_billions: float, ylabel: bool) -> None:
    ax.set_xlim(0.0, x_max_billions)
    ax.set_ylim(-2.0, 104.0)
    ax.set_xticks(np.arange(0.0, x_max_billions + 0.5, 1.0))
    ax.set_xticklabels(["0"] + [f"{int(v)}B" for v in np.arange(1.0, x_max_billions + 0.5, 1.0)])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    ax.set_xlabel("Finetuning Env Steps")
    if ylabel:
        ax.set_ylabel("Success Rate")
    style_axis(ax)


def _save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = output.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white", edgecolor="none")
        print(f"wrote {path}")


def _plot_2x2(curves: list[Curve], output: Path, *, x_max_billions: float) -> None:
    configure_rcparams()
    specs = [
        ("wrench", "Trajectory_Count", "100", "Wrench / 100 Trajectories", "#2C7BB6"),
        ("no_wrench", "Trajectory_Count", "100", "No Wrench / 100 Trajectories", "#2C7BB6"),
        ("wrench", "Precision", "5cm", "Wrench / 5cm Threshold", "#D6604D"),
        ("no_wrench", "Precision", "5cm", "No Wrench / 5cm Threshold", "#D6604D"),
    ]
    linestyles = {0: "-", 1: "--", 2: "-.", 3: ":"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.7), dpi=220, sharex=True, sharey=True)
    for ax, (condition, family, checkpoint, title, color) in zip(axes.reshape(-1), specs, strict=True):
        subset = [
            curve
            for curve in curves
            if curve.condition == condition and curve.family == family and curve.checkpoint == checkpoint
        ]
        for curve in subset:
            visible = curve.x / 1e9 <= x_max_billions
            if not np.any(visible):
                continue
            ax.plot(
                curve.x[visible] / 1e9,
                100.0 * curve.y[visible],
                color=color,
                linestyle=linestyles.get(curve.seed, "-"),
                linewidth=2.0,
                label=f"seed {curve.seed}",
            )
        ax.set_title(title, fontsize=12)
        _format_axis(ax, x_max_billions=x_max_billions, ylabel=ax in axes[:, 0])
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.10, hspace=0.32, wspace=0.16)
    _save(fig, output)
    plt.close(fig)


def _plot_traj10_overlay(data: dict[str, Any], output: Path, *, x_max_billions: float) -> None:
    configure_rcparams()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "mathtext.fontset": "dejavusans",
        }
    )
    colors = {"wrench": "#E08214", "no_wrench": "#8C8C8C"}
    condition_labels = {"wrench": "Wrench", "no_wrench": "No Wrench"}
    linestyles = {0: "-", 1: "--", 2: "-.", 3: ":"}

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.0), dpi=240)
    for condition in ("wrench", "no_wrench"):
        bucket = data["conditions"][condition]["Trajectory_Count"]["runs"].get("10", {})
        for seed_key, seed_data in sorted(bucket.items(), key=lambda kv: int(kv[0])):
            seed = int(seed_key)
            if seed not in (0, 1, 2):
                continue
            curve = _curve(condition, "Trajectory_Count", "10", seed_data)
            if curve is None:
                continue
            x_b = curve.x / 1e9
            visible = x_b <= x_max_billions
            if not np.any(visible):
                continue
            ax.plot(
                x_b[visible],
                100.0 * curve.y[visible],
                color=colors[condition],
                linestyle=linestyles.get(seed, "-"),
                linewidth=1.8,
                alpha=0.95,
                label=f"{condition_labels[condition]} seed {seed}",
            )

    ax.set_title("10 Trajectory Seeds", fontsize=12)
    ax.set_xlim(0.0, x_max_billions)
    ax.set_ylim(-3.0, 104.0)
    xticks = np.arange(0.0, x_max_billions + 0.5, 1.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0"] + [f"{int(v)}B" for v in xticks[1:]], fontsize=9)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]], fontsize=9)
    ax.set_xlabel("Env steps", fontsize=10)
    ax.set_ylabel("Success rate", fontsize=10)
    style_axis(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.16)
    _save(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--x-max-billions", type=float, default=4.0)
    parser.add_argument("--slug", default="4B")
    parser.add_argument("--traj10-overlay", action="store_true")
    args = parser.parse_args()

    data = _load_data(args.data)
    if args.traj10_overlay:
        _plot_traj10_overlay(
            data,
            OUT_DIR / f"lpeg_traj10_wrench_vs_no_wrench_seed_overlay_{args.slug}.png",
            x_max_billions=args.x_max_billions,
        )
        return

    curves = _curves(data)
    _plot_2x2(
        curves,
        OUT_DIR / f"lpeg_traj_precision_seed_traces_2x2_{args.slug}.png",
        x_max_billions=args.x_max_billions,
    )


if __name__ == "__main__":
    main()

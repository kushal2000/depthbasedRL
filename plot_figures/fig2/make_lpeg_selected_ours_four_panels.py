"""Create the current four L-peg ablation panels with a shared selected-Ours curve.

This script intentionally writes new filenames and does not overwrite the older
Fig. 2 ablation outputs.  The shared "Ours" aggregate is fixed to:

  - wrench ObjectDiversity / 1000_obj / seed 0
  - wrench ObjectDiversity / 1000_obj / seed 1
  - wrench ObjectDiversity / 1000_obj / seed 2

Panels:

  1. Object Diversity: Ours vs 100/10 objects, wrench seeds
  2. Training Objective: Ours vs Rotation-/Translation-Only, no-wrench seeds
  3. Trajectory Diversity: Ours vs 100 trajectories with and without wrench
  4. Play Precision: Ours vs 5 cm with and without wrench

Example:
    .venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_selected_ours_four_panels.py
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
from _style import STYLES, configure_rcparams, style_axis  # noqa: E402

sys.path.insert(0, str(REPO / "plot_figures" / "fig2"))
from collect_lpeg_traj_precision_curves import collect as collect_traj_precision  # noqa: E402
from make_lpeg_wrench_objectdiversity_plot import _collect as collect_wrench  # noqa: E402


METRIC = "episode_final/feasible_normalized_all_goals_hit"

NO_WRENCH_DATA_DIR = REPO / "outputs" / "fig2_tyler_seed_curves"
WRENCH_DATA_PATH = REPO / "outputs" / "fig2_tyler_3curve_seed_curves" / "lpeg_wrench_seed_curves_latest.json"
TRAJ_PRECISION_DATA_PATH = (
    REPO / "outputs" / "fig2_tyler_traj_precision_seed_curves" / "lpeg_traj_precision_wrench_compare_latest.json"
)
OUT_DIR = REPO / "plot_figures" / "fig2" / "outputs"

SELECTED_OURS = (
    ("ObjectDiversity", "1000_obj", 0),
    ("ObjectDiversity", "1000_obj", 1),
    ("ObjectDiversity", "1000_obj", 2),
)

COLORS = {
    "ours": "#2C7BB6",
    "secondary": STYLES["RotationOnly"][0],
    "tertiary": "#D6604D",
}


@dataclass
class Curve:
    name: str
    seed: int
    x: np.ndarray
    y: np.ndarray
    run_name: str


@dataclass
class SeriesSpec:
    label: str
    color: str
    curves: list[Curve]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {path}")


def _series_to_curve(name: str, seed_data: dict[str, Any]) -> Curve | None:
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
    return Curve(
        name=name,
        seed=int(seed_data["seed"]),
        x=x - x[0],
        y=y,
        run_name=seed_data.get("run_name") or seed_data.get("run_dir") or "unknown",
    )


def _curves_from_wrench(
    wrench: dict[str, Any],
    *,
    family: str,
    checkpoint: str,
    seeds: tuple[int, ...],
    name: str,
) -> list[Curve]:
    bucket = wrench["families"][family]["runs"].get(checkpoint, {})
    curves: list[Curve] = []
    for seed in seeds:
        seed_data = bucket.get(str(seed))
        if seed_data is None:
            print(f"warning: missing wrench {family}/{checkpoint} seed {seed}")
            continue
        curve = _series_to_curve(name, seed_data)
        if curve is not None:
            curves.append(curve)
    return curves


def _curves_from_no_wrench(
    data: dict[str, Any],
    *,
    checkpoint: str,
    seeds: tuple[int, ...],
    name: str,
) -> list[Curve]:
    bucket = data["runs"].get(checkpoint, {})
    curves: list[Curve] = []
    for seed in seeds:
        seed_data = bucket.get(str(seed))
        if seed_data is None:
            print(f"warning: missing no-wrench {data.get('family')}/{checkpoint} seed {seed}")
            continue
        curve = _series_to_curve(name, seed_data)
        if curve is not None:
            curves.append(curve)
    return curves


def _curves_from_traj_precision(
    data: dict[str, Any],
    *,
    condition: str,
    family: str,
    checkpoint: str,
    seeds: tuple[int, ...],
    name: str,
) -> list[Curve]:
    bucket = data["conditions"][condition][family]["runs"].get(checkpoint, {})
    curves: list[Curve] = []
    for seed in seeds:
        seed_data = bucket.get(str(seed))
        if seed_data is None:
            print(f"warning: missing {condition} {family}/{checkpoint} seed {seed}")
            continue
        curve = _series_to_curve(name, seed_data)
        if curve is not None:
            curves.append(curve)
    return curves


def _selected_ours_curves(wrench: dict[str, Any]) -> list[Curve]:
    curves: list[Curve] = []
    for family, checkpoint, seed in SELECTED_OURS:
        curves.extend(
            _curves_from_wrench(
                wrench,
                family=family,
                checkpoint=checkpoint,
                seeds=(seed,),
                name=f"selected_ours_{family}_{checkpoint}_seed{seed}",
            )
        )
    return curves


def _aggregate(curves: list[Curve], *, num_points: int) -> dict[str, np.ndarray] | None:
    if not curves:
        return None
    end = max(float(curve.x[-1]) for curve in curves)
    if end <= 0.0:
        return None
    grid = np.linspace(0.0, end, num_points)
    ys = []
    for curve in curves:
        y = np.interp(grid, curve.x, curve.y)
        y[(grid < curve.x[0]) | (grid > curve.x[-1])] = np.nan
        ys.append(y)
    stack = np.vstack(ys)
    keep = np.sum(np.isfinite(stack), axis=0) >= len(curves)
    if not np.any(keep):
        return None
    return {
        "x": grid[keep],
        "mean": np.nanmean(stack[:, keep], axis=0),
        "std": np.nanstd(stack[:, keep], axis=0),
    }


def _format_axis(ax: plt.Axes, x_max_billions: float) -> None:
    ax.set_xlim(0.0, x_max_billions)
    ax.set_ylim(-2.0, 104.0)
    ax.set_xticks(np.arange(0.0, x_max_billions + 0.5, 1.0))
    ax.set_xticklabels(["0"] + [f"{int(v)}B" for v in np.arange(1.0, x_max_billions + 0.5, 1.0)])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    ax.set_xlabel("Finetuning Env Steps")
    ax.set_ylabel("Success Rate")
    style_axis(ax)


def _save(fig: plt.Figure, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = output_png.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.035, facecolor="white", edgecolor="none")
        print(f"wrote {path}")


def _plot_panel(
    title: str,
    series: list[SeriesSpec],
    output_png: Path,
    *,
    x_max_billions: float,
    num_points: int,
) -> None:
    configure_rcparams()
    fig, ax = plt.subplots(figsize=(6.4, 3.35), dpi=220)
    for item in series:
        aggregate = _aggregate(item.curves, num_points=num_points)
        if aggregate is None:
            print(f"warning: no aggregate for {item.label}")
            continue
        x = aggregate["x"] / 1e9
        mean = 100.0 * aggregate["mean"]
        std = 100.0 * aggregate["std"]
        visible = x <= x_max_billions
        if not np.any(visible):
            print(f"warning: no visible points for {item.label}")
            continue
        ax.plot(x[visible], mean[visible], color=item.color, linewidth=1.9, label=item.label)
        ax.fill_between(
            x[visible],
            mean[visible] - std[visible],
            mean[visible] + std[visible],
            color=item.color,
            alpha=0.17,
            linewidth=0,
        )
    _format_axis(ax, x_max_billions)
    ax.set_title(title, fontsize=15, pad=6)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(axis="both", which="major", labelsize=11)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=3,
        fontsize=12,
        columnspacing=1.1,
        handlelength=1.25,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.15, right=0.985, top=0.82, bottom=0.36)
    _save(fig, output_png)
    plt.close(fig)


def _plot_review_grid(paths: list[Path], output_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.7), dpi=180)
    for ax, path in zip(axes.reshape(-1), paths, strict=True):
        ax.imshow(plt.imread(path))
        ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.02, hspace=0.04)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.015, facecolor="white", edgecolor="none")
    print(f"wrote {output_png}")
    plt.close(fig)


def _summary(label: str, curves: list[Curve]) -> str:
    aggregate = _aggregate(curves, num_points=300)
    if aggregate is None:
        return f"| {label} | n/a | n/a | n/a |"
    seeds = ",".join(str(c.seed) for c in curves)
    return f"| {label} | {seeds} | {100.0 * float(aggregate['mean'][-1]):.1f}% | {float(aggregate['x'][-1]) / 1e9:.2f}B |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-max-billions", type=float, default=4.0)
    parser.add_argument("--slug", default="selected_ours_4B")
    parser.add_argument("--num-points", type=int, default=500)
    parser.add_argument("--wandb-samples", type=int, default=7000)
    parser.add_argument("--refresh-wrench", action="store_true")
    parser.add_argument("--refresh-traj-precision", action="store_true")
    args = parser.parse_args()

    if args.refresh_wrench or not WRENCH_DATA_PATH.exists():
        _write_json(WRENCH_DATA_PATH, collect_wrench(samples=args.wandb_samples))
    if args.refresh_traj_precision or not TRAJ_PRECISION_DATA_PATH.exists():
        _write_json(TRAJ_PRECISION_DATA_PATH, collect_traj_precision(samples=args.wandb_samples))

    wrench = _load_json(WRENCH_DATA_PATH)
    traj_precision = _load_json(TRAJ_PRECISION_DATA_PATH)
    no_training = _load_json(NO_WRENCH_DATA_DIR / "lpeg_trainingobjective_seed_curves.json")

    ours = _selected_ours_curves(wrench)
    if len(ours) != len(SELECTED_OURS):
        raise SystemExit(f"Expected {len(SELECTED_OURS)} selected Ours curves, got {len(ours)}")

    panels: list[tuple[str, str, list[SeriesSpec]]] = [
        (
            "Object Diversity",
            "object_diversity",
            [
                SeriesSpec("1000 Objects (Ours)", COLORS["ours"], ours),
                SeriesSpec(
                    "100 Objects",
                    COLORS["secondary"],
                    _curves_from_wrench(
                        wrench, family="ObjectDiversity", checkpoint="100_obj", seeds=(0, 1, 2), name="wrench_100_obj"
                    ),
                ),
                SeriesSpec(
                    "10 Objects",
                    COLORS["tertiary"],
                    _curves_from_wrench(
                        wrench, family="ObjectDiversity", checkpoint="10_obj", seeds=(0, 1, 2), name="wrench_10_obj"
                    ),
                ),
            ],
        ),
        (
            "Training Objective",
            "training_objective",
            [
                SeriesSpec("Full Pose (Ours)", COLORS["ours"], ours),
                SeriesSpec(
                    "Rotation-Only",
                    COLORS["secondary"],
                    _curves_from_no_wrench(
                        no_training, checkpoint="RotationOnly", seeds=(0, 1, 3), name="no_wrench_rotation_only"
                    ),
                ),
                SeriesSpec(
                    "Translation-Only",
                    COLORS["tertiary"],
                    _curves_from_no_wrench(
                        no_training, checkpoint="TranslationOnly", seeds=(0, 1, 3), name="no_wrench_translation_only"
                    ),
                ),
            ],
        ),
        (
            "Trajectory Diversity",
            "trajectory_diversity",
            [
                SeriesSpec("Random Trajectories (Ours)", COLORS["ours"], ours),
                SeriesSpec(
                    "100 Trajectories (Wrench)",
                    COLORS["secondary"],
                    _curves_from_traj_precision(
                        traj_precision,
                        condition="wrench",
                        family="Trajectory_Count",
                        checkpoint="100",
                        seeds=(0, 1, 2),
                        name="wrench_100_traj",
                    ),
                ),
                SeriesSpec(
                    "100 Trajectories (No Wrench)",
                    COLORS["tertiary"],
                    _curves_from_traj_precision(
                        traj_precision,
                        condition="no_wrench",
                        family="Trajectory_Count",
                        checkpoint="100",
                        seeds=(0, 1, 2),
                        name="no_wrench_100_traj",
                    ),
                ),
            ],
        ),
        (
            "Play Precision",
            "play_precision",
            [
                SeriesSpec("1 cm (Ours)", COLORS["ours"], ours),
                SeriesSpec(
                    "5 cm (Wrench)",
                    COLORS["secondary"],
                    _curves_from_traj_precision(
                        traj_precision,
                        condition="wrench",
                        family="Precision",
                        checkpoint="5cm",
                        seeds=(0, 1, 2),
                        name="wrench_5cm",
                    ),
                ),
                SeriesSpec(
                    "5 cm (No Wrench)",
                    COLORS["tertiary"],
                    _curves_from_traj_precision(
                        traj_precision,
                        condition="no_wrench",
                        family="Precision",
                        checkpoint="5cm",
                        seeds=(0, 1, 2),
                        name="no_wrench_5cm",
                    ),
                ),
            ],
        ),
    ]

    output_paths = []
    print("| Panel Curve | Seeds | Final Mean | Max Common Env Steps |")
    print("|---|---:|---:|---:|")
    for title, slug, series in panels:
        for item in series:
            print(_summary(f"{title}: {item.label}", item.curves))
        output = OUT_DIR / f"lpeg_final_{slug}_{args.slug}.png"
        _plot_panel(title, series, output, x_max_billions=args.x_max_billions, num_points=args.num_points)
        output_paths.append(output)

    _plot_review_grid(output_paths, OUT_DIR / f"lpeg_final_four_panel_review_{args.slug}.png")


if __name__ == "__main__":
    main()

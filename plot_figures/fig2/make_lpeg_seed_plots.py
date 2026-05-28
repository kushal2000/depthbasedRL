"""Render L-peg seed-mean/std ablation plots from collected TensorBoard data.

Run after ``collect_lpeg_seed_curves.py``:

    .venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_seed_plots.py
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
from matplotlib.ticker import FuncFormatter
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "plot_figures"))
from _style import STYLES, configure_rcparams, save_figure, style_axis  # noqa: E402


DEFAULT_DATA_DIR = REPO / "outputs" / "fig2_tyler_seed_curves"
DEFAULT_OUT_DIR = REPO / "plot_figures" / "fig2" / "outputs"

FAMILY_FILES = {
    "ObjectDiversity": "lpeg_objectdiversity_seed_curves.json",
    "TrainingObjective": "lpeg_trainingobjective_seed_curves.json",
}

PLOT_FAMILIES = ("ObjectDiversity", "TrainingObjective", "ObjectDiversityPlay2Win")

COMPARISON_COLORS = {
    "Play2Win": "#2C7BB6",
    "1000_obj": "#2C7BB6",
    "RotationOnly": STYLES["RotationOnly"][0],
    "100_obj": STYLES["RotationOnly"][0],
    "SingleGoal": STYLES["SingleGoal"][0],
    "10_obj": STYLES["SingleGoal"][0],
    "TranslationOnly": "#E08214",
    "1_obj": "#E08214",
}

DISPLAY = {
    "ObjectDiversity": {
        "title": "L-peg: object-diversity pretraining",
        "ylabel": "Success Rate (%)",
        "checkpoint_labels": {
            "1000_obj": "1000 Objects",
            "100_obj": "100 Objects",
            "10_obj": "10 Objects",
            "1_obj": "1 Object",
        },
        "colors": COMPARISON_COLORS,
    },
    "ObjectDiversityPlay2Win": {
        "title": "L-peg: Object Diversity, 1000 Objects (Ours) Replacing 1000 Objects",
        "ylabel": "Success Rate (%)",
        "checkpoint_labels": {
            "Play2Win": "1000 Objects (Ours)",
            "100_obj": "100 Objects",
            "10_obj": "10 Objects",
            "1_obj": "1 Object",
        },
        "colors": COMPARISON_COLORS,
    },
    "TrainingObjective": {
        "title": "L-peg: pretraining objective ablation",
        "ylabel": "Success Rate (%)",
        "checkpoint_labels": {
            "Play2Win": "Full Pose (Ours)",
            "RotationOnly": "Rotation Only",
            "SingleGoal": "Single Goal",
            "TranslationOnly": "Translation Only",
        },
        "colors": COMPARISON_COLORS,
    },
}


@dataclass
class Curve:
    seed: int
    x: np.ndarray
    y: np.ndarray
    run_dir: str


SEED_STYLES = {
    0: "-",
    1: "--",
    2: "-.",
    3: ":",
}


def _load_family(data_dir: Path, family: str) -> dict[str, Any]:
    if family == "ObjectDiversityPlay2Win":
        object_data = _load_family(data_dir, "ObjectDiversity")
        objective_data = _load_family(data_dir, "TrainingObjective")
        play2win_runs = objective_data.get("runs", {}).get("Play2Win")
        if not play2win_runs:
            raise RuntimeError("TrainingObjective JSON has no Play2Win runs for hybrid ObjectDiversity plot.")

        runs = {
            "Play2Win": play2win_runs,
            "100_obj": object_data.get("runs", {}).get("100_obj", {}),
            "10_obj": object_data.get("runs", {}).get("10_obj", {}),
            "1_obj": object_data.get("runs", {}).get("1_obj", {}),
        }
        return {
            "family": family,
            "checkpoint_order": ["Play2Win", "100_obj", "10_obj", "1_obj"],
            "metric_default": object_data.get("metric_default"),
            "raw_success_metric": object_data.get("raw_success_metric"),
            "runs": runs,
            "metadata": {
                "source": "ObjectDiversity with TrainingObjective/Play2Win replacing ObjectDiversity/1000_obj",
                "object_diversity_metadata": object_data.get("metadata", {}),
                "training_objective_metadata": objective_data.get("metadata", {}),
            },
        }

    path = data_dir / FAMILY_FILES[family]
    with path.open() as f:
        return json.load(f)


def _curve_from_seed(seed_data: dict[str, Any], metric: str, x_axis: str) -> Curve | None:
    series = seed_data.get("metrics", {}).get(metric)
    if not series:
        return None
    x = np.asarray(series["steps"], dtype=np.float64)
    y = np.asarray(series["values"], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return None
    order = np.argsort(x)
    x = x[order]
    if x_axis == "relative_env_frames":
        x = x - x[0]
    return Curve(
        seed=int(seed_data["seed"]),
        x=x,
        y=np.clip(y[order], 0.0, 1.0),
        run_dir=seed_data["run_dir"],
    )


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size < window:
        return y
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _aggregate(
    curves: list[Curve],
    num_points: int,
    smooth_window: int,
    min_seeds: int,
) -> dict[str, np.ndarray] | None:
    if not curves:
        return None
    start = min(float(c.x[0]) for c in curves)
    end = max(float(c.x[-1]) for c in curves)
    if end <= start:
        return None

    grid = np.linspace(start, end, num_points)
    ys = []
    for curve in curves:
        y = _moving_average(curve.y, smooth_window)
        interpolated = np.interp(grid, curve.x, y)
        outside = (grid < curve.x[0]) | (grid > curve.x[-1])
        interpolated[outside] = np.nan
        ys.append(interpolated)
    y_stack = np.vstack(ys)
    counts = np.sum(np.isfinite(y_stack), axis=0)
    if min_seeds <= 0:
        min_seeds = len(curves)
    keep = counts >= min_seeds
    if not np.any(keep):
        return None

    return {
        "x": grid[keep],
        "mean": np.nanmean(y_stack[:, keep], axis=0),
        "std": np.nanstd(y_stack[:, keep], axis=0),
        "n": counts[keep].astype(np.float64),
    }


def _format_billions(x: np.ndarray) -> np.ndarray:
    return x / 1e9


def _format_env_step_axis(ax: plt.Axes, *, x_max_billions: float | None) -> None:
    ax.set_xlabel("Finetuning Env Steps")
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _pos: "0" if abs(x) < 1e-9 else f"{x:g}B")
    )
    if x_max_billions is not None:
        ax.set_xlim(0.0, x_max_billions)
        if abs(x_max_billions - round(x_max_billions)) < 1e-9 and x_max_billions <= 8:
            ax.set_xticks(np.arange(0.0, x_max_billions + 0.5, 1.0))


def _seed_slug(seeds: set[int] | None) -> str:
    if not seeds:
        return "allseeds"
    return "seeds" + "-".join(str(seed) for seed in sorted(seeds))


def _family_slug(family: str) -> str:
    if family == "ObjectDiversityPlay2Win":
        return "objectdiversity_play2win"
    return family.lower()


def _plot_family(
    family_data: dict[str, Any],
    family: str,
    metric: str,
    x_axis: str,
    out_dir: Path,
    num_points: int,
    smooth_window: int,
    min_seeds: int,
    show_seeds: bool,
    seeds: set[int] | None,
    x_max_billions: float | None,
    no_title: bool,
) -> Path:
    configure_rcparams()
    fig, ax = plt.subplots(figsize=(5.4, 3.0))

    display = DISPLAY[family]
    checkpoint_order = [ckpt for ckpt in display["checkpoint_labels"] if ckpt in family_data["runs"]]
    any_curve = False
    x_plot_max = 0.0

    for checkpoint in checkpoint_order:
        seed_bucket = family_data.get("runs", {}).get(checkpoint, {})
        curves = []
        for _, seed_data in sorted(seed_bucket.items(), key=lambda kv: int(kv[0])):
            if seeds is not None and int(seed_data["seed"]) not in seeds:
                continue
            curve = _curve_from_seed(seed_data, metric, x_axis=x_axis)
            if curve is not None:
                curves.append(curve)

        aggregate = _aggregate(
            curves,
            num_points=num_points,
            smooth_window=smooth_window,
            min_seeds=min_seeds,
        )
        if aggregate is None:
            print(f"Skipping {family}/{checkpoint}: no aggregateable curves")
            continue

        any_curve = True
        x_plot_max = max(x_plot_max, float(np.nanmax(aggregate["x"])))
        color = display["colors"][checkpoint]
        label = display["checkpoint_labels"][checkpoint]

        if show_seeds:
            agg_min = float(np.nanmin(aggregate["x"]))
            agg_max = float(np.nanmax(aggregate["x"]))
            for curve in curves:
                visible = (curve.x >= agg_min) & (curve.x <= agg_max)
                if not np.any(visible):
                    continue
                ax.plot(
                    _format_billions(curve.x[visible]),
                    100.0 * curve.y[visible],
                    color=color,
                    linewidth=0.8,
                    alpha=0.22,
                )

        x = _format_billions(aggregate["x"])
        mean = 100.0 * aggregate["mean"]
        std = 100.0 * aggregate["std"]
        ax.plot(x, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)

    if not any_curve:
        raise RuntimeError(f"No curves found for {family} metric={metric}")

    if not no_title:
        ax.set_title(display["title"], fontsize=12)
    _format_env_step_axis(ax, x_max_billions=x_max_billions)
    ax.set_ylabel(display["ylabel"])
    ax.set_ylim(-2, 104)
    if x_max_billions is None:
        ax.set_xlim(0.0, _format_billions(np.asarray([x_plot_max]))[0] * 1.05)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    style_axis(ax)

    metric_slug = metric.replace("/", "_")
    x_slug = "relative_frames" if x_axis == "relative_env_frames" else "global_step"
    summary_slug = "mean_std_with_seed_overlay" if show_seeds else "mean_std_clean"
    name = f"lpeg_{_family_slug(family)}_{metric_slug}_{x_slug}_{_seed_slug(seeds)}_{summary_slug}"
    return save_figure(fig, name=name, out_dir=out_dir)


def _plot_family_individual(
    family_data: dict[str, Any],
    family: str,
    metric: str,
    x_axis: str,
    out_dir: Path,
    seeds: set[int] | None,
    x_max_billions: float | None,
    no_title: bool,
) -> Path:
    configure_rcparams()
    fig, ax = plt.subplots(figsize=(5.6, 3.1))

    display = DISPLAY[family]
    checkpoint_order = [ckpt for ckpt in display["checkpoint_labels"] if ckpt in family_data["runs"]]
    x_plot_max = 0.0
    line_handles = []
    label_handles = []

    for checkpoint in checkpoint_order:
        seed_bucket = family_data.get("runs", {}).get(checkpoint, {})
        color = display["colors"][checkpoint]
        checkpoint_label = display["checkpoint_labels"][checkpoint]
        first_handle = None

        for _, seed_data in sorted(seed_bucket.items(), key=lambda kv: int(kv[0])):
            if seeds is not None and int(seed_data["seed"]) not in seeds:
                continue
            curve = _curve_from_seed(seed_data, metric, x_axis=x_axis)
            if curve is None:
                continue
            x_plot_max = max(x_plot_max, float(np.nanmax(curve.x)))
            linestyle = SEED_STYLES.get(curve.seed, "-")
            handle, = ax.plot(
                _format_billions(curve.x),
                100.0 * curve.y,
                color=color,
                linestyle=linestyle,
                linewidth=1.45,
                alpha=0.90,
                label=f"{checkpoint_label}, seed {curve.seed}",
            )
            if first_handle is None:
                first_handle = handle

        if first_handle is not None:
            line_handles.append(first_handle)
            label_handles.append(checkpoint_label)

    if not line_handles:
        raise RuntimeError(f"No individual curves found for {family} metric={metric}")

    if not no_title:
        ax.set_title(f"{display['title']} (individual seeds)", fontsize=12)
    _format_env_step_axis(ax, x_max_billions=x_max_billions)
    ax.set_ylabel(display["ylabel"])
    ax.set_ylim(-2, 104)
    if x_max_billions is None:
        ax.set_xlim(0.0, _format_billions(np.asarray([x_plot_max]))[0] * 1.05)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    style_axis(ax)

    color_legend = ax.legend(line_handles, label_handles, frameon=False, loc="lower right", fontsize=8.3)
    ax.add_artist(color_legend)

    seed_handles = []
    seed_labels = []
    for seed, linestyle in sorted(SEED_STYLES.items()):
        if seeds is not None and seed not in seeds:
            continue
        if any(str(seed) in family_data.get("runs", {}).get(ckpt, {}) for ckpt in checkpoint_order):
            h, = ax.plot([], [], color="#444444", linestyle=linestyle, linewidth=1.4)
            seed_handles.append(h)
            seed_labels.append(f"seed {seed}")
    ax.legend(seed_handles, seed_labels, frameon=False, loc="center right", fontsize=7.8)

    metric_slug = metric.replace("/", "_")
    x_slug = "relative_frames" if x_axis == "relative_env_frames" else "global_step"
    name = f"lpeg_{_family_slug(family)}_{metric_slug}_{x_slug}_{_seed_slug(seeds)}_individual_seeds"
    return save_figure(fig, name=name, out_dir=out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--metric",
        default="episode_final/feasible_normalized_all_goals_hit",
        help="Metric key from the collected JSON.",
    )
    parser.add_argument("--num-points", type=int, default=300)
    parser.add_argument(
        "--x-axis",
        choices=["relative_env_frames", "global_step"],
        default="relative_env_frames",
        help="Use relative finetune frames by default; some old W&B runs have huge absolute global_step offsets.",
    )
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=0,
        help="Only plot aggregate points with at least this many seeds contributing. 0 means all available seeds for that curve.",
    )
    parser.add_argument("--show-seeds", action="store_true", help="Overlay faint individual seed curves.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional seed filter for plotting.")
    parser.add_argument(
        "--x-max-billions",
        type=float,
        default=None,
        help="Optional x-axis cap in billions of finetuning environment steps.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Render individual seed curves instead of mean/std aggregate curves.",
    )
    parser.add_argument("--no-title", action="store_true", help="Do not draw axis titles.")
    args = parser.parse_args()
    seed_filter = set(args.seeds) if args.seeds else None

    for family in PLOT_FAMILIES:
        family_data = _load_family(args.data_dir, family)
        if args.individual:
            out = _plot_family_individual(
                family_data=family_data,
                family=family,
                metric=args.metric,
                x_axis=args.x_axis,
                out_dir=args.out_dir,
                seeds=seed_filter,
                x_max_billions=args.x_max_billions,
                no_title=args.no_title,
            )
        else:
            out = _plot_family(
                family_data=family_data,
                family=family,
                metric=args.metric,
                x_axis=args.x_axis,
                out_dir=args.out_dir,
                num_points=args.num_points,
                smooth_window=args.smooth_window,
                min_seeds=args.min_seeds,
                show_seeds=args.show_seeds,
                seeds=seed_filter,
                x_max_billions=args.x_max_billions,
                no_title=args.no_title,
            )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

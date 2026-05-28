"""Create simplified 3-curve L-peg ablation plots.

This is intentionally separate from the existing 4-curve plotting scripts so
the old figures remain reproducible.  It creates:

  1. no-wrench TrainingObjective: Full Pose / Rotation-Only / Translation-Only
  2. no-wrench ObjectDiversity: 1000 / 100 / 10 objects
  3. wrench ObjectDiversity: 1000 / 100 / 10 objects, refreshed from W&B

Example:
    .venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_3curve_plots.py
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
from make_lpeg_wrench_objectdiversity_plot import _collect as _collect_wrench  # noqa: E402


NO_WRENCH_DATA_DIR = REPO / "outputs" / "fig2_tyler_seed_curves"
THREE_CURVE_DATA_DIR = REPO / "outputs" / "fig2_tyler_3curve_seed_curves"
OUT_DIR = REPO / "plot_figures" / "fig2" / "outputs"

METRIC = "episode_final/feasible_normalized_all_goals_hit"

COLORS = {
    "Play2Win": "#2C7BB6",
    "1000_obj": "#2C7BB6",
    "RotationOnly": STYLES["RotationOnly"][0],
    "100_obj": STYLES["RotationOnly"][0],
    "10_obj": STYLES["SingleGoal"][0],
    "TranslationOnly": "#E08214",
}

PANEL_SPECS = {
    "trainingobjective_no_wrench": {
        "title": "(b) Training Objective",
        "source": "no_wrench",
        "family": "TrainingObjective",
        "checkpoints": ("Play2Win", "RotationOnly", "TranslationOnly"),
        "labels": {
            "Play2Win": "Full Pose (Ours)",
            "RotationOnly": "Rotation-Only",
            "TranslationOnly": "Translation-Only",
        },
    },
    "objectdiversity_no_wrench": {
        "title": "(a) Object Diversity",
        "source": "no_wrench",
        "family": "ObjectDiversity",
        "checkpoints": ("1000_obj", "100_obj", "10_obj"),
        "labels": {
            "1000_obj": "1000 Objects (Ours)",
            "100_obj": "100 Objects",
            "10_obj": "10 Objects",
        },
    },
    "objectdiversity_play2win_no_wrench": {
        "title": "(a) Object Diversity",
        "source": "hybrid_no_wrench",
        "family": "ObjectDiversity",
        "checkpoints": ("Play2Win", "100_obj", "10_obj"),
        "labels": {
            "Play2Win": "1000 Objects (Ours)",
            "100_obj": "100 Objects",
            "10_obj": "10 Objects",
        },
    },
    "objectdiversity_wrench": {
        "title": "(a) Object Diversity",
        "source": "wrench",
        "family": "ObjectDiversity",
        "checkpoints": ("1000_obj", "100_obj", "10_obj"),
        "labels": {
            "1000_obj": "1000 Objects (Ours)",
            "100_obj": "100 Objects",
            "10_obj": "10 Objects",
        },
    },
    "objectdiversity_play2win_wrench": {
        "title": "(a) Object Diversity",
        "source": "hybrid_wrench",
        "family": "ObjectDiversity",
        "checkpoints": ("Play2Win", "100_obj", "10_obj"),
        "labels": {
            "Play2Win": "1000 Objects (Ours)",
            "100_obj": "100 Objects",
            "10_obj": "10 Objects",
        },
    },
}


@dataclass
class Curve:
    seed: int
    x: np.ndarray
    y: np.ndarray
    run_name: str


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _load_no_wrench_family(family: str) -> dict[str, Any]:
    file_name = {
        "ObjectDiversity": "lpeg_objectdiversity_seed_curves.json",
        "TrainingObjective": "lpeg_trainingobjective_seed_curves.json",
    }[family]
    return _load_json(NO_WRENCH_DATA_DIR / file_name)


def _load_or_collect_wrench(*, refresh: bool, samples: int) -> dict[str, Any]:
    THREE_CURVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = THREE_CURVE_DATA_DIR / "lpeg_wrench_seed_curves_latest.json"
    if refresh or not out_path.exists():
        data = _collect_wrench(samples=samples)
        with out_path.open("w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"wrote {out_path}")
    return _load_json(out_path)


def _curve_from_seed(seed_data: dict[str, Any]) -> Curve | None:
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
    x = x - x[0]
    return Curve(
        seed=int(seed_data["seed"]),
        x=x,
        y=y,
        run_name=seed_data.get("run_name") or seed_data.get("run_dir") or "unknown",
    )


def _aggregate(curves: list[Curve], num_points: int) -> dict[str, np.ndarray] | None:
    if not curves:
        return None
    start = min(float(curve.x[0]) for curve in curves)
    end = max(float(curve.x[-1]) for curve in curves)
    if end <= start:
        return None
    grid = np.linspace(start, end, num_points)
    ys = []
    for curve in curves:
        y = np.interp(grid, curve.x, curve.y)
        y[(grid < curve.x[0]) | (grid > curve.x[-1])] = np.nan
        ys.append(y)
    stack = np.vstack(ys)
    # Use the common support for the selected seeds of each curve, matching the
    # existing paper-style mean/std plots.
    keep = np.sum(np.isfinite(stack), axis=0) >= len(curves)
    if not np.any(keep):
        return None
    return {
        "x": grid[keep],
        "mean": np.nanmean(stack[:, keep], axis=0),
        "std": np.nanstd(stack[:, keep], axis=0),
    }


def _group_curves(source_data: dict[str, Any], spec: dict[str, Any]) -> list[tuple[str, str, str, list[Curve]]]:
    if spec["source"] == "wrench":
        family_data = source_data["families"][spec["family"]]
    elif spec["source"] == "hybrid_wrench":
        family_data = {
            "runs": {
                "Play2Win": source_data["families"]["TrainingObjective"]["runs"].get("Play2Win", {}),
                "100_obj": source_data["families"]["ObjectDiversity"]["runs"].get("100_obj", {}),
                "10_obj": source_data["families"]["ObjectDiversity"]["runs"].get("10_obj", {}),
            }
        }
    elif spec["source"] == "hybrid_no_wrench":
        family_data = {
            "runs": {
                "Play2Win": source_data["TrainingObjective"]["runs"].get("Play2Win", {}),
                "100_obj": source_data["ObjectDiversity"]["runs"].get("100_obj", {}),
                "10_obj": source_data["ObjectDiversity"]["runs"].get("10_obj", {}),
            }
        }
    else:
        family_data = source_data

    grouped = []
    for checkpoint in spec["checkpoints"]:
        curves = []
        seed_bucket = family_data.get("runs", {}).get(checkpoint, {})
        for _, seed_data in sorted(seed_bucket.items(), key=lambda kv: int(kv[0])):
            curve = _curve_from_seed(seed_data)
            if curve is not None:
                curves.append(curve)
        if curves:
            grouped.append((checkpoint, spec["labels"][checkpoint], COLORS[checkpoint], curves))
        else:
            print(f"warning: no curves for {spec['family']}/{checkpoint}")
    return grouped


def _format_env_axis(ax: plt.Axes, x_max_billions: float) -> None:
    ax.set_xlabel("Finetuning Env Steps")
    ax.set_xlim(0.0, x_max_billions)
    ax.set_xticks(np.arange(0.0, x_max_billions + 0.5, 1.0))
    ax.set_xticklabels(["0"] + [f"{int(v)}B" for v in np.arange(1.0, x_max_billions + 0.5, 1.0)])


def _save_png_pdf(fig: plt.Figure, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = output_png.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.025, facecolor="white", edgecolor="none")
        print(f"wrote {path}")


def _plot_grouped(
    grouped: list[tuple[str, str, str, list[Curve]]],
    output_png: Path,
    *,
    x_max_billions: float,
    title: str | None,
    num_points: int,
    legend_mode: str,
) -> None:
    configure_rcparams()
    label_size = 14
    tick_size = 11
    title_size = 15
    fig_height = 3.28 if title else 3.0
    fig, ax = plt.subplots(figsize=(6.4, fig_height), dpi=220)

    for _checkpoint, label, color, curves in grouped:
        aggregate = _aggregate(curves, num_points=num_points)
        if aggregate is None:
            print(f"warning: no aggregate for {label}")
            continue
        x = aggregate["x"] / 1e9
        mean = 100.0 * aggregate["mean"]
        std = 100.0 * aggregate["std"]
        visible = x <= x_max_billions
        if not np.any(visible):
            print(f"warning: no visible points for {label}")
            continue
        ax.plot(x[visible], mean[visible], color=color, linewidth=1.8, label=label)
        ax.fill_between(
            x[visible],
            mean[visible] - std[visible],
            mean[visible] + std[visible],
            color=color,
            alpha=0.17,
            linewidth=0,
        )

    _format_env_axis(ax, x_max_billions)
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-2, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    if title:
        ax.set_title(title, fontsize=title_size, pad=6)
    style_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    if legend_mode == "below_1row":
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.050 if not title else 0.045),
            ncol=3,
            fontsize=label_size,
            columnspacing=1.15,
            handlelength=1.35,
            handletextpad=0.35,
        )
        fig.subplots_adjust(left=0.15, right=0.985, top=0.82 if title else 0.91, bottom=0.38 if not title else 0.36)
    elif legend_mode == "inside":
        ax.legend(
            handles,
            labels,
            frameon=False,
            loc="lower right",
            fontsize=11.0,
            handlelength=1.35,
            handletextpad=0.35,
        )
        fig.subplots_adjust(left=0.15, right=0.985, top=0.82 if title else 0.91, bottom=0.22 if not title else 0.24)
    else:
        raise ValueError(f"Unexpected legend_mode={legend_mode}")
    _save_png_pdf(fig, output_png)
    plt.close(fig)


def _summarize_grouped(name: str, grouped: list[tuple[str, str, str, list[Curve]]]) -> None:
    print(f"\n{name}")
    print("| Curve | Seeds | Final Mean | Max Common Env Steps |")
    print("|---|---:|---:|---:|")
    for _checkpoint, label, _color, curves in grouped:
        aggregate = _aggregate(curves, num_points=300)
        if aggregate is None:
            continue
        final_mean = 100.0 * float(aggregate["mean"][-1])
        max_steps = float(aggregate["x"][-1]) / 1e9
        seeds = ",".join(str(curve.seed) for curve in curves)
        print(f"| {label} | {seeds} | {final_mean:.1f}% | {max_steps:.2f}B |")


def _review_image(
    paths_by_panel: dict[str, dict[str, Path]],
    output: Path,
) -> None:
    rows = [
        ("Training Objective, no wrench", "trainingobjective_no_wrench"),
        ("Object Diversity, no wrench", "objectdiversity_no_wrench"),
        ("Object Diversity, wrench", "objectdiversity_wrench"),
    ]
    cols = [
        ("Legend below, one row", "below_1row"),
        ("Legend inside", "inside"),
    ]

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(11.8, 7.2), dpi=180)
    for row_idx, (row_title, panel_name) in enumerate(rows):
        for col_idx, (col_title, legend_mode) in enumerate(cols):
            ax = axes[row_idx, col_idx]
            image = plt.imread(paths_by_panel[panel_name][legend_mode])
            ax.imshow(image)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_title, fontsize=13, pad=8)
            if col_idx == 0:
                ax.text(
                    -0.02,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=12,
                )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.94, bottom=0.02, wspace=0.02, hspace=0.10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.025, facecolor="white", edgecolor="none")
    print(f"wrote {output}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-max-billions", type=float, default=4.0)
    parser.add_argument("--slug", default="4B")
    parser.add_argument("--num-points", type=int, default=300)
    parser.add_argument("--wandb-samples", type=int, default=5000)
    parser.add_argument(
        "--no-refresh-wrench",
        action="store_true",
        help="Use cached 3-curve wrench JSON instead of refreshing W&B.",
    )
    parser.add_argument("--with-titles", action="store_true", help="Render panel titles.")
    parser.add_argument(
        "--legend-mode",
        choices=["below_1row", "inside", "both"],
        default="both",
        help="Legend placement for the new 3-curve outputs.",
    )
    args = parser.parse_args()

    no_wrench = {
        "TrainingObjective": _load_no_wrench_family("TrainingObjective"),
        "ObjectDiversity": _load_no_wrench_family("ObjectDiversity"),
    }
    wrench = _load_or_collect_wrench(refresh=not args.no_refresh_wrench, samples=args.wandb_samples)

    base_outputs = {
        "trainingobjective_no_wrench": "lpeg_3curve_trainingobjective_no_wrench",
        "objectdiversity_no_wrench": "lpeg_3curve_objectdiversity_no_wrench",
        "objectdiversity_play2win_no_wrench": "lpeg_3curve_objectdiversity_play2win_no_wrench",
        "objectdiversity_wrench": "lpeg_3curve_objectdiversity_wrench",
        "objectdiversity_play2win_wrench": "lpeg_3curve_objectdiversity_play2win_wrench",
    }

    legend_modes = ("below_1row", "inside") if args.legend_mode == "both" else (args.legend_mode,)
    paths_by_panel: dict[str, dict[str, Path]] = {}

    for panel_name, spec in PANEL_SPECS.items():
        if spec["source"] in {"wrench", "hybrid_wrench"}:
            source_data = wrench
        elif spec["source"] == "hybrid_no_wrench":
            source_data = no_wrench
        else:
            source_data = no_wrench[spec["family"]]
        grouped = _group_curves(source_data, spec)
        _summarize_grouped(panel_name, grouped)
        paths_by_panel[panel_name] = {}
        for legend_mode in legend_modes:
            title = spec["title"] if args.with_titles else None
            output = OUT_DIR / f"{base_outputs[panel_name]}_{legend_mode}_{args.slug}.png"
            _plot_grouped(
                grouped,
                output,
                x_max_billions=args.x_max_billions,
                title=title,
                num_points=args.num_points,
                legend_mode=legend_mode,
            )
            paths_by_panel[panel_name][legend_mode] = output

    if args.legend_mode == "both":
        _review_image(paths_by_panel, OUT_DIR / f"lpeg_3curve_legend_review_{args.slug}.png")


if __name__ == "__main__":
    main()

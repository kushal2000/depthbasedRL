"""Create one 3x3 overview image for the L-peg seed-ablation plots."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "plot_figures" / "fig2"))

from make_lpeg_seed_plots import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    DISPLAY,
    PLOT_FAMILIES,
    SEED_STYLES,
    _aggregate,
    _curve_from_seed,
    _format_billions,
    _format_env_step_axis,
    _load_family,
)

sys.path.insert(0, str(REPO / "plot_figures"))
from _style import configure_rcparams, style_axis  # noqa: E402


METRIC = "episode_final/feasible_normalized_all_goals_hit"
X_AXIS = "relative_env_frames"
SUMMARY_X_MAX_BILLIONS = 3.0
FAMILY_SEED_FILTERS = {
    "ObjectDiversity": {0, 1, 3},
}
CHECKPOINT_SEED_FILTERS = {
    ("ObjectDiversityPlay2Win", "Play2Win"): {0, 1, 3},
    ("ObjectDiversityPlay2Win", "100_obj"): {0, 1, 3},
    ("ObjectDiversityPlay2Win", "10_obj"): {0, 1, 3},
    ("ObjectDiversityPlay2Win", "1_obj"): {0, 1, 3},
}

ROW_TITLES = {
    "TrainingObjective": "Training objective",
    "ObjectDiversity": "Object diversity",
    "ObjectDiversityPlay2Win": "Object diversity, Play2Perfect as 1000-object baseline",
}

COLUMN_TITLES = (
    "Individual seeds",
    "Mean +/- std + seed traces",
    "Mean +/- std",
)


def _iter_curves(family_data: dict, family: str) -> list[tuple[str, str, str, list]]:
    display = DISPLAY[family]
    checkpoint_order = [ckpt for ckpt in display["checkpoint_labels"] if ckpt in family_data["runs"]]
    seed_filter = FAMILY_SEED_FILTERS.get(family)
    grouped = []
    for checkpoint in checkpoint_order:
        curves = []
        checkpoint_seed_filter = CHECKPOINT_SEED_FILTERS.get((family, checkpoint), seed_filter)
        for _, seed_data in sorted(family_data["runs"].get(checkpoint, {}).items(), key=lambda kv: int(kv[0])):
            if checkpoint_seed_filter is not None and int(seed_data["seed"]) not in checkpoint_seed_filter:
                continue
            curve = _curve_from_seed(seed_data, METRIC, x_axis=X_AXIS)
            if curve is not None:
                curves.append(curve)
        if curves:
            grouped.append((checkpoint, display["checkpoint_labels"][checkpoint], display["colors"][checkpoint], curves))
    return grouped


def _style_common_axis(ax: plt.Axes, *, x_max_billions: float | None, show_ylabel: bool) -> None:
    _format_env_step_axis(ax, x_max_billions=x_max_billions)
    ax.set_ylim(-2, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    if show_ylabel:
        ax.set_ylabel("Success Rate (%)")
    else:
        ax.set_ylabel("")
    style_axis(ax)


def _plot_individual(ax: plt.Axes, grouped: list[tuple[str, str, str, list]]) -> None:
    x_plot_max = 0.0
    color_handles = []
    color_labels = []

    for _checkpoint, label, color, curves in grouped:
        first_handle = None
        for curve in curves:
            x_plot_max = max(x_plot_max, float(np.nanmax(curve.x)))
            handle, = ax.plot(
                _format_billions(curve.x),
                100.0 * curve.y,
                color=color,
                linestyle=SEED_STYLES.get(curve.seed, "-"),
                linewidth=1.15,
                alpha=0.9,
            )
            if first_handle is None:
                first_handle = handle
        if first_handle is not None:
            color_handles.append(first_handle)
            color_labels.append(label)

    ax.set_xlim(0.0, _format_billions(np.asarray([x_plot_max]))[0] * 1.04)
    _style_common_axis(ax, x_max_billions=SUMMARY_X_MAX_BILLIONS, show_ylabel=True)
    color_legend = ax.legend(color_handles, color_labels, frameon=False, loc="lower right", fontsize=7.0)
    ax.add_artist(color_legend)

    seed_handles = []
    seed_labels = []
    used_seeds = sorted({curve.seed for _ckpt, _label, _color, curves in grouped for curve in curves})
    for seed in used_seeds:
        h, = ax.plot([], [], color="#444444", linestyle=SEED_STYLES.get(seed, "-"), linewidth=1.2)
        seed_handles.append(h)
        seed_labels.append(f"seed {seed}")
    ax.legend(seed_handles, seed_labels, frameon=False, loc="center right", fontsize=6.8)


def _plot_summary(
    ax: plt.Axes,
    grouped: list[tuple[str, str, str, list]],
    *,
    show_seed_traces: bool,
) -> None:
    for _checkpoint, label, color, curves in grouped:
        aggregate = _aggregate(curves, num_points=300, smooth_window=1, min_seeds=0)
        if aggregate is None:
            continue

        if show_seed_traces:
            for curve in curves:
                visible = _format_billions(curve.x) <= SUMMARY_X_MAX_BILLIONS
                if np.any(visible):
                    ax.plot(
                        _format_billions(curve.x[visible]),
                        100.0 * curve.y[visible],
                        color=color,
                        linewidth=0.6,
                        alpha=0.20,
                    )

        x = _format_billions(aggregate["x"])
        mean = 100.0 * aggregate["mean"]
        std = 100.0 * aggregate["std"]
        ax.plot(x, mean, color=color, linewidth=1.8, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.17, linewidth=0)

    _style_common_axis(ax, x_max_billions=SUMMARY_X_MAX_BILLIONS, show_ylabel=False)
    ax.legend(frameon=False, loc="lower right", fontsize=7.0)


def main() -> None:
    configure_rcparams()
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 9.6), dpi=220)

    for col_idx, title in enumerate(COLUMN_TITLES):
        axes[0, col_idx].set_title(title, fontsize=13, pad=10)

    for row_idx, family in enumerate(("TrainingObjective", "ObjectDiversity", "ObjectDiversityPlay2Win")):
        family_data = _load_family(DEFAULT_DATA_DIR, family)
        grouped = _iter_curves(family_data, family)
        axes[row_idx, 0].text(
            -0.22,
            0.5,
            ROW_TITLES[family],
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
        )
        _plot_individual(axes[row_idx, 0], grouped)
        _plot_summary(axes[row_idx, 1], grouped, show_seed_traces=True)
        _plot_summary(axes[row_idx, 2], grouped, show_seed_traces=False)

    fig.tight_layout(rect=(0.025, 0.0, 1.0, 1.0), h_pad=1.2, w_pad=0.9)

    output = DEFAULT_OUT_DIR / "lpeg_seed_ablation_summary_grid_3x3_relative_frames_objectdiv_seeds0-1-3_summary3B.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

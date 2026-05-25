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
AXIS_LABEL_FONT_SIZE = 12
TICK_LABEL_FONT_SIZE = 10
TITLE_FONT_SIZE = 13
FAMILY_SEED_FILTERS = {
    "ObjectDiversity": {0, 2, 3},
}
CHECKPOINT_SEED_FILTERS = {
    ("ObjectDiversityPlay2Win", "Play2Win"): {0, 1, 3},
    ("ObjectDiversityPlay2Win", "100_obj"): {0, 2, 3},
    ("ObjectDiversityPlay2Win", "10_obj"): {0, 2, 3},
    ("ObjectDiversityPlay2Win", "1_obj"): {0, 2, 3},
}

ROW_TITLES = {
    "TrainingObjective": "Training objective",
    "ObjectDiversity": "Object diversity",
    "ObjectDiversityPlay2Win": "Object diversity + ours baseline",
}

COLUMN_TITLES = (
    "Individual seeds",
    "Mean +/- std + seed traces",
    "Mean +/- std",
)


def _configure_big_text() -> None:
    configure_rcparams()
    matplotlib.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": TICK_LABEL_FONT_SIZE,
            "legend.fontsize": AXIS_LABEL_FONT_SIZE,
        }
    )


def _save_png_pdf(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = output.with_suffix(suffix)
        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor="white",
            edgecolor="none",
        )
        print(f"wrote {path}")


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
        ax.set_ylabel("Success Rate")
    else:
        ax.set_ylabel("")
    style_axis(ax)


def _plot_individual(ax: plt.Axes, grouped: list[tuple[str, str, str, list]], *, draw_legend: bool = True) -> None:
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
    if draw_legend:
        color_legend = ax.legend(color_handles, color_labels, frameon=False, loc="lower right", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.add_artist(color_legend)

        seed_handles = []
        seed_labels = []
        used_seeds = sorted({curve.seed for _ckpt, _label, _color, curves in grouped for curve in curves})
        for seed in used_seeds:
            h, = ax.plot([], [], color="#444444", linestyle=SEED_STYLES.get(seed, "-"), linewidth=1.2)
            seed_handles.append(h)
            seed_labels.append(f"seed {seed}")
        ax.legend(seed_handles, seed_labels, frameon=False, loc="center right", fontsize=AXIS_LABEL_FONT_SIZE)


def _plot_summary(
    ax: plt.Axes,
    grouped: list[tuple[str, str, str, list]],
    *,
    show_seed_traces: bool,
    show_ylabel: bool = False,
    draw_legend: bool = True,
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

    _style_common_axis(ax, x_max_billions=SUMMARY_X_MAX_BILLIONS, show_ylabel=show_ylabel)
    if draw_legend:
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.58),
            ncol=2,
            fontsize=AXIS_LABEL_FONT_SIZE,
            columnspacing=1.4,
            handlelength=1.6,
            handletextpad=0.4,
        )


def _standalone_summary_name(family: str) -> str:
    if family == "TrainingObjective":
        return "lpeg_top_right_trainingobjective_mean_std_clean_3B.png"
    if family == "ObjectDiversity":
        return "lpeg_middle_right_objectdiversity_1000obj_mean_std_clean_3B.png"
    if family == "ObjectDiversityPlay2Win":
        return "lpeg_bottom_right_objectdiversity_play2perfect_mean_std_clean_3B.png"
    raise ValueError(f"Unexpected family: {family}")


def _save_standalone_clean_summary(family: str, grouped: list[tuple[str, str, str, list]]) -> None:
    # Keep the standalone panels close to the paper-style reference crop:
    # compact axes, two-row legend underneath, and a small but readable gap
    # between the x-axis label and legend.
    standalone_label_size = 14
    standalone_tick_size = 11
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=220)
    _plot_summary(ax, grouped, show_seed_traces=False, show_ylabel=True, draw_legend=False)
    ax.xaxis.label.set_size(standalone_label_size)
    ax.yaxis.label.set_size(standalone_label_size)
    ax.tick_params(axis="both", which="major", labelsize=standalone_tick_size)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        fontsize=standalone_label_size,
        columnspacing=1.5,
        handlelength=1.6,
        handletextpad=0.4,
    )
    output = DEFAULT_OUT_DIR / _standalone_summary_name(family)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.91, bottom=0.465)
    _save_png_pdf(fig, output)
    plt.close(fig)


def main() -> None:
    _configure_big_text()
    fig, axes = plt.subplots(3, 3, figsize=(14.1, 10.7), dpi=220)

    for col_idx, title in enumerate(COLUMN_TITLES):
        axes[0, col_idx].set_title(title, fontsize=TITLE_FONT_SIZE, pad=8)

    for row_idx, family in enumerate(("TrainingObjective", "ObjectDiversity", "ObjectDiversityPlay2Win")):
        family_data = _load_family(DEFAULT_DATA_DIR, family)
        grouped = _iter_curves(family_data, family)
        axes[row_idx, 0].text(
            0.02,
            0.97,
            ROW_TITLES[family],
            transform=axes[row_idx, 0].transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.5},
        )
        _plot_individual(axes[row_idx, 0], grouped, draw_legend=False)
        _plot_summary(axes[row_idx, 1], grouped, show_seed_traces=True, draw_legend=False)
        _plot_summary(axes[row_idx, 2], grouped, show_seed_traces=False, draw_legend=False)
        _save_standalone_clean_summary(family, grouped)

    fig.tight_layout(rect=(0.01, 0.0, 1.0, 1.0), h_pad=1.2, w_pad=0.55)

    output = DEFAULT_OUT_DIR / "lpeg_seed_ablation_summary_grid_3x3_relative_frames_objectdiv_seeds0-2-3_summary3B.png"
    _save_png_pdf(fig, output)
    plt.close(fig)


if __name__ == "__main__":
    main()

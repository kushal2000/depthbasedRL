"""Create titled standalone L-peg ablation panels without overwriting clean panels."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "plot_figures" / "fig2"))

from make_lpeg_seed_summary_grid import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    _configure_big_text,
    _dummy_grouped_from_play2win,
    _iter_curves,
    _load_family,
    _plot_summary,
    _save_png_pdf,
)


TITLE_FONT_SIZE = 15
AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 11


PANEL_SPECS = (
    (
        "ObjectDiversityPlay2Win",
        "(a) Object Diversity",
        "lpeg_panel_a_object_diversity_titled_4B.png",
    ),
    (
        "TrainingObjective",
        "(b) Training Objective",
        "lpeg_panel_b_training_objective_titled_4B.png",
    ),
    (
        "DummyNumTrajectories",
        "(c) Trajectory Diversity",
        "lpeg_panel_c_trajectory_diversity_titled_4B.png",
    ),
    (
        "DummySuccessTolerance",
        "(d) Play Precision",
        "lpeg_panel_d_play_precision_titled_4B.png",
    ),
)


def _grouped_for_family(family: str):
    if family.startswith("Dummy"):
        return _dummy_grouped_from_play2win(family)
    return _iter_curves(_load_family(DEFAULT_DATA_DIR, family), family)


def _save_titled_panel(family: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.28), dpi=220)
    _plot_summary(ax, _grouped_for_family(family), show_seed_traces=False, show_ylabel=True, draw_legend=False)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=6)
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.040),
        ncol=2,
        fontsize=AXIS_LABEL_FONT_SIZE,
        columnspacing=1.5,
        handlelength=1.6,
        handletextpad=0.4,
    )

    fig.subplots_adjust(left=0.15, right=0.985, top=0.82, bottom=0.43)
    _save_png_pdf(fig, DEFAULT_OUT_DIR / filename)
    plt.close(fig)


def main() -> None:
    _configure_big_text()
    for family, title, filename in PANEL_SPECS:
        _save_titled_panel(family, title, filename)


if __name__ == "__main__":
    main()

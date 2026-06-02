"""Render the four selected L-peg ablation panels in finalized Fig. 2 style.

This is a style-exploration script.  It reuses the selected-Ours data and panel
definitions from ``make_lpeg_selected_ours_one_row_narrow.py`` but writes new
filenames so the older/wider plots remain reproducible.

Styling follows ``origin/2026_05_28_finalize_figs:plot_figures/fig2``:
small 1x4 row, serif fonts, success-rate ticks on every axis, and patch-square
legends with the Ours entry semibold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "plot_figures" / "fig2"))
import make_lpeg_selected_ours_four_panels as base  # noqa: E402
import make_lpeg_selected_ours_one_row_narrow as narrow  # noqa: E402


OUT_DIR = REPO / "plot_figures" / "fig2" / "outputs"
FIG2_COLORS = {
    "ours": "#2C7BB6",
    "secondary": "#E08214",
    "tertiary": "#8C8C8C",
    "quaternary": "#D6604D",
}


def _apply_fig2_palette(panels: list[narrow.PanelSpec]) -> None:
    """Use the finalized Fig. 2 blue/orange/gray palette in every panel."""
    for panel in panels:
        if len(panel.series) >= 1:
            panel.series[0].color = FIG2_COLORS["ours"]
        if len(panel.series) >= 2:
            panel.series[1].color = FIG2_COLORS["secondary"]
        if len(panel.series) >= 3:
            panel.series[2].color = FIG2_COLORS["tertiary"]
        if len(panel.series) >= 4:
            panel.series[3].color = FIG2_COLORS["quaternary"]


def _apply_compact_wrapped_labels(panels: list[narrow.PanelSpec]) -> None:
    """Use shorter labels that fit a single-row per-panel legend."""
    replacements = {
        "1000 Objects (Ours)": "1000\nObjects\n(Ours)",
        "100 Objects": "100\nObjects",
        "10 Objects": "10\nObjects",
        "Full Pose (Ours)": "6D Pose\n(Ours)",
        "Random Traj. (Ours)": "Random\n(Ours)",
        "1 cm (Ours)": "1 cm\n(Ours)",
        "Rotation-only": "Rot-Only",
        "Translation-only": "Trans-Only",
    }
    for panel in panels:
        for series in panel.series:
            series.label = replacements.get(series.label, series.label)


def _use_wrench_training_objective(panels: list[narrow.PanelSpec], wrench: dict) -> None:
    """Swap Training Objective baselines from no-wrench to wrench curves."""
    ours = panels[1].series[0].curves
    panels[1] = narrow.PanelSpec(
        "Training Objective",
        [
            base.SeriesSpec("Full Pose (Ours)", FIG2_COLORS["ours"], ours),
            base.SeriesSpec(
                "Rotation-only",
                FIG2_COLORS["secondary"],
                base._curves_from_wrench(
                    wrench,
                    family="TrainingObjective",
                    checkpoint="RotationOnly",
                    seeds=(0, 1, 2),
                    name="wrench_rotation_only",
                ),
            ),
            base.SeriesSpec(
                "Translation-only",
                FIG2_COLORS["tertiary"],
                base._curves_from_wrench(
                    wrench,
                    family="TrainingObjective",
                    checkpoint="TranslationOnly",
                    seeds=(0, 1, 2),
                    name="wrench_translation_only",
                ),
            ),
        ],
    )


def _set_trajectory_10_condition(
    panels: list[narrow.PanelSpec],
    traj_precision: dict,
    *,
    condition: str,
) -> None:
    """Swap the Trajectory Diversity 10-trajectory curve condition."""
    if condition == "wrench":
        return
    panels[2].series[2] = base.SeriesSpec(
        "10",
        FIG2_COLORS["tertiary"],
        base._curves_from_traj_precision(
            traj_precision,
            condition=condition,
            family="Trajectory_Count",
            checkpoint="10",
            seeds=(0, 1, 2),
            name=f"{condition}_10_traj",
        ),
    )


def _add_temporary_fourth_curves(
    panels: list[narrow.PanelSpec],
    *,
    wrench: dict,
    traj_precision: dict,
    no_training: dict,
) -> None:
    """Add temporary fourth curves requested for visual comparison only."""
    no_object = base._load_json(base.NO_WRENCH_DATA_DIR / "lpeg_objectdiversity_seed_curves.json")

    panels[0].series.append(
        base.SeriesSpec(
            "1 Object",
            FIG2_COLORS["quaternary"],
            base._curves_from_no_wrench(
                no_object,
                checkpoint="1_obj",
                seeds=(0, 2, 3),
                name="no_wrench_1_obj_worst3",
            ),
        )
    )
    panels[1].series.append(
        base.SeriesSpec(
            "Single Goal",
            FIG2_COLORS["quaternary"],
            base._curves_from_no_wrench(
                no_training,
                checkpoint="SingleGoal",
                seeds=(0, 1, 3),
                name="no_wrench_single_goal",
            ),
        )
    )
    panels[2].series.append(
        base.SeriesSpec(
            "1",
            FIG2_COLORS["quaternary"],
            base._curves_from_traj_precision(
                traj_precision,
                condition="wrench",
                family="Trajectory_Count",
                checkpoint="1",
                seeds=(0, 1, 2),
                name="wrench_1_traj",
            ),
        )
    )
    panels[3].series.append(
        base.SeriesSpec(
            "2.5cm",
            FIG2_COLORS["quaternary"],
            base._curves_from_traj_precision(
                traj_precision,
                condition="wrench",
                family="Precision",
                checkpoint="2p5cm",
                seeds=(0, 1, 2),
                name="wrench_2p5cm",
            ),
        )
    )


def _plot_series(
    ax: plt.Axes,
    series: list[base.SeriesSpec],
    *,
    x_max_billions: float,
    num_points: int,
    x_axis_mode: str,
    hours_at_xmax: float,
) -> None:
    for item in series:
        aggregate = base._aggregate(item.curves, num_points=num_points)
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
        plot_x = x
        if x_axis_mode == "hours":
            plot_x = x / x_max_billions * hours_at_xmax
        line_x = plot_x[visible]
        ax.plot(line_x, mean[visible], color=item.color, linewidth=1.5, zorder=3)
        ax.fill_between(
            line_x,
            mean[visible] - std[visible],
            mean[visible] + std[visible],
            color=item.color,
            alpha=0.15,
            linewidth=0,
            zorder=2,
        )


def _style_axes(ax: plt.Axes, *, x_max_billions: float, x_axis_mode: str, hours_at_xmax: float) -> None:
    x_limit = x_max_billions if x_axis_mode == "env_steps" else hours_at_xmax
    ax.set_xlim(0.0, x_limit)
    ax.set_ylim(-3.0, 104.0)
    if x_axis_mode == "env_steps":
        xticks = np.arange(0.0, x_max_billions + 0.5, 1.0)
        ax.set_xticks(xticks)
        ax.set_xticklabels(["0"] + [f"{int(v)}B" for v in xticks[1:]], fontsize=7)
        ax.set_xlabel("Env steps", fontsize=8.5, labelpad=1.5)
    else:
        xticks = np.arange(0.0, hours_at_xmax + 0.5, 6.0)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(v)} h" for v in xticks], fontsize=7)
        ax.set_xlabel("Training time", fontsize=8.5, labelpad=1.5)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7)
    ax.set_ylabel("Success rate", fontsize=8.5, labelpad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#333333")
    ax.tick_params(axis="both", which="major", length=3, width=0.8, colors="#333333", pad=1.0)


def _legend_handles(series: list[base.SeriesSpec]) -> list[Patch]:
    return [Patch(facecolor=item.color, edgecolor="#333333", linewidth=0.7) for item in series]


def _emphasize_ours(legend, series: list[base.SeriesSpec]) -> None:
    for text, item in zip(legend.get_texts(), series, strict=True):
        if "(Ours)" in item.label:
            text.set_fontweight(600)


def _add_patch_legend(
    ax: plt.Axes,
    series: list[base.SeriesSpec],
    *,
    layout: str,
    legend_font_size: float,
) -> None:
    handles = _legend_handles(series)
    labels = [item.label for item in series]
    common = dict(
        frameon=False,
        fontsize=legend_font_size,
        handlelength=0.95,
        handleheight=0.95,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    if layout == "one_row":
        legend = ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.34),
            ncol=len(handles),
            columnspacing=0.38,
            labelspacing=0.1,
            **common,
        )
        for text in legend.get_texts():
            text.set_multialignment("center")
        _emphasize_ours(legend, series)
    elif layout == "ours_first":
        ours = ax.legend(
            handles[:1],
            labels[:1],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.34),
            ncol=1,
            **common,
        )
        _emphasize_ours(ours, series[:1])
        ax.add_artist(ours)
        if len(handles) > 3:
            middle = ax.legend(
                handles[1:3],
                labels[1:3],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.52),
                ncol=2,
                columnspacing=0.75,
                **common,
            )
            _emphasize_ours(middle, series[1:3])
            ax.add_artist(middle)
            rest = ax.legend(
                handles[3:],
                labels[3:],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.70),
                ncol=1,
                columnspacing=0.75,
                **common,
            )
            _emphasize_ours(rest, series[3:])
        else:
            rest = ax.legend(
                handles[1:],
                labels[1:],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.52),
                ncol=2,
                columnspacing=0.75,
                **common,
            )
            _emphasize_ours(rest, series[1:])
    elif layout == "three_row":
        legend = ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.34),
            ncol=1,
            labelspacing=0.12,
            **common,
        )
        _emphasize_ours(legend, series)
    else:
        raise ValueError(f"unknown legend layout: {layout}")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        out = path.with_suffix(suffix)
        fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white", edgecolor="none")
        print(f"wrote {out}")


def _plot_one_row(
    panels: list[narrow.PanelSpec],
    output: Path,
    *,
    legend_layout: str,
    x_max_billions: float,
    num_points: int,
    font_style: str,
    legend_font_size: float,
    x_axis_mode: str,
    hours_at_xmax: float,
) -> Path:
    base.configure_rcparams()
    if font_style == "sans":
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
                "mathtext.fontset": "dejavusans",
            }
        )
    has_four_curves = any(len(panel.series) > 3 for panel in panels)
    fig_width = 7.0
    fig_height = {"one_row": 2.30, "ours_first": 2.56, "three_row": 2.76}[legend_layout]
    if legend_layout == "one_row" and font_style == "sans":
        fig_width = 7.6
        fig_height = 2.76
    if legend_layout == "ours_first" and has_four_curves:
        fig_height = 2.88
    fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height), dpi=240)
    for ax, panel in zip(axes, panels, strict=True):
        _plot_series(
            ax,
            panel.series,
            x_max_billions=x_max_billions,
            num_points=num_points,
            x_axis_mode=x_axis_mode,
            hours_at_xmax=hours_at_xmax,
        )
        _style_axes(ax, x_max_billions=x_max_billions, x_axis_mode=x_axis_mode, hours_at_xmax=hours_at_xmax)
        ax.set_title(panel.title, fontsize=8, pad=2.0)
        _add_patch_legend(ax, panel.series, layout=legend_layout, legend_font_size=legend_font_size)

    bottom = {"one_row": 0.34, "ours_first": 0.43, "three_row": 0.47}[legend_layout]
    if legend_layout == "one_row" and font_style == "sans":
        bottom = 0.47
    if legend_layout == "ours_first" and has_four_curves:
        bottom = 0.50
    fig.subplots_adjust(left=0.065, right=0.995, top=0.83, bottom=bottom, wspace=0.45)
    _save(fig, output)
    plt.close(fig)
    return output.with_suffix(".png")


def _plot_review(paths: list[Path], output: Path) -> None:
    fig, axes = plt.subplots(len(paths), 1, figsize=(8.0, 2.25 * len(paths)), dpi=180)
    if len(paths) == 1:
        axes = [axes]
    for ax, path in zip(axes, paths, strict=True):
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem, fontsize=8, pad=2)
        ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01, hspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.02, facecolor="white", edgecolor="none")
    print(f"wrote {output}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-max-billions", type=float, default=4.0)
    parser.add_argument("--slug", default="selected_ours_4B_fig2style")
    parser.add_argument("--num-points", type=int, default=500)
    parser.add_argument("--wandb-samples", type=int, default=7000)
    parser.add_argument("--refresh-wrench", action="store_true")
    parser.add_argument("--refresh-traj-precision", action="store_true")
    parser.add_argument(
        "--font-style",
        choices=["serif", "sans"],
        default="serif",
        help="Use the original serif styling or a compact sans-serif variant.",
    )
    parser.add_argument("--legend-font-size", type=float, default=6.0)
    parser.add_argument("--x-axis-mode", choices=["env_steps", "hours"], default="env_steps")
    parser.add_argument(
        "--hours-at-xmax",
        type=float,
        default=24.0,
        help="When --x-axis-mode=hours, map --x-max-billions to this many hours.",
    )
    parser.add_argument(
        "--compact-wrapped-labels",
        action="store_true",
        help="Shorten/wrap labels for a one-row legend.",
    )
    parser.add_argument(
        "--training-objective-condition",
        choices=["no_wrench", "wrench"],
        default="no_wrench",
        help="Whether the RotationOnly/TranslationOnly baselines use no-wrench or wrench curves.",
    )
    parser.add_argument(
        "--trajectory-10-condition",
        choices=["wrench", "no_wrench"],
        default="wrench",
        help="Whether the Trajectory Diversity 10-trajectory curve uses wrench or no-wrench runs.",
    )
    parser.add_argument(
        "--legend-layouts",
        nargs="+",
        default=["ours_first", "one_row", "three_row"],
        choices=["ours_first", "one_row", "three_row"],
    )
    parser.add_argument(
        "--include-temp-fourth-curves",
        action="store_true",
        help="Temporarily add 1 Object, Single Goal, 1 trajectory, and 2.5cm curves.",
    )
    args = parser.parse_args()

    wrench, traj_precision, no_training = narrow._load_or_collect(
        refresh_wrench=args.refresh_wrench,
        refresh_traj_precision=args.refresh_traj_precision,
        wandb_samples=args.wandb_samples,
    )
    panels = narrow._build_panels(wrench, traj_precision, no_training)
    if args.training_objective_condition == "wrench":
        _use_wrench_training_objective(panels, wrench)
    _set_trajectory_10_condition(panels, traj_precision, condition=args.trajectory_10_condition)
    if args.include_temp_fourth_curves:
        _add_temporary_fourth_curves(
            panels,
            wrench=wrench,
            traj_precision=traj_precision,
            no_training=no_training,
        )
    _apply_fig2_palette(panels)
    if args.compact_wrapped_labels:
        _apply_compact_wrapped_labels(panels)

    generated: list[Path] = []
    for layout in args.legend_layouts:
        output = OUT_DIR / f"lpeg_final_one_row_fig2style_{layout}_{args.slug}.png"
        generated.append(
            _plot_one_row(
                panels,
                output,
                legend_layout=layout,
                x_max_billions=args.x_max_billions,
                num_points=args.num_points,
                font_style=args.font_style,
                legend_font_size=args.legend_font_size,
                x_axis_mode=args.x_axis_mode,
                hours_at_xmax=args.hours_at_xmax,
            )
        )

    if len(generated) > 1:
        _plot_review(generated, OUT_DIR / f"lpeg_final_one_row_fig2style_legend_review_{args.slug}.png")


if __name__ == "__main__":
    main()

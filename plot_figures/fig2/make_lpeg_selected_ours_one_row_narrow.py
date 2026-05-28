"""Create narrow one-row variants of the four L-peg ablation panels.

This is a layout exploration script only.  It intentionally does not overwrite
the existing standalone/wide panels from ``make_lpeg_selected_ours_four_panels``.

Outputs:
    plot_figures/fig2/outputs/lpeg_final_one_row_narrow_*_4B.{png,pdf}

Example:
    .venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_selected_ours_one_row_narrow.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "plot_figures" / "fig2"))
import make_lpeg_selected_ours_four_panels as base  # noqa: E402


OUT_DIR = REPO / "plot_figures" / "fig2" / "outputs"


@dataclass
class PanelSpec:
    title: str
    series: list[base.SeriesSpec]


def _load_or_collect(*, refresh_wrench: bool, refresh_traj_precision: bool, wandb_samples: int):
    if refresh_wrench or not base.WRENCH_DATA_PATH.exists():
        base._write_json(base.WRENCH_DATA_PATH, base.collect_wrench(samples=wandb_samples))
    if refresh_traj_precision or not base.TRAJ_PRECISION_DATA_PATH.exists():
        base._write_json(
            base.TRAJ_PRECISION_DATA_PATH,
            base.collect_traj_precision(samples=wandb_samples),
        )
    wrench = base._load_json(base.WRENCH_DATA_PATH)
    traj_precision = base._load_json(base.TRAJ_PRECISION_DATA_PATH)
    no_training = base._load_json(base.NO_WRENCH_DATA_DIR / "lpeg_trainingobjective_seed_curves.json")
    return wrench, traj_precision, no_training


def _build_panels(wrench: dict, traj_precision: dict, no_training: dict) -> list[PanelSpec]:
    ours = base._selected_ours_curves(wrench)
    if len(ours) != len(base.SELECTED_OURS):
        raise SystemExit(f"Expected {len(base.SELECTED_OURS)} selected Ours curves, got {len(ours)}")

    colors = base.COLORS
    return [
        PanelSpec(
            "Object Diversity",
            [
                base.SeriesSpec("1000 Objects (Ours)", colors["ours"], ours),
                base.SeriesSpec(
                    "100 Objects",
                    colors["secondary"],
                    base._curves_from_wrench(
                        wrench,
                        family="ObjectDiversity",
                        checkpoint="100_obj",
                        seeds=(0, 1, 2),
                        name="wrench_100_obj",
                    ),
                ),
                base.SeriesSpec(
                    "10 Objects",
                    colors["tertiary"],
                    base._curves_from_wrench(
                        wrench,
                        family="ObjectDiversity",
                        checkpoint="10_obj",
                        seeds=(0, 1, 2),
                        name="wrench_10_obj",
                    ),
                ),
            ],
        ),
        PanelSpec(
            "Training Objective",
            [
                base.SeriesSpec("Full Pose (Ours)", colors["ours"], ours),
                base.SeriesSpec(
                    "Rotation-only",
                    colors["secondary"],
                    base._curves_from_no_wrench(
                        no_training,
                        checkpoint="RotationOnly",
                        seeds=(0, 1, 3),
                        name="no_wrench_rotation_only",
                    ),
                ),
                base.SeriesSpec(
                    "Translation-only",
                    colors["tertiary"],
                    base._curves_from_no_wrench(
                        no_training,
                        checkpoint="TranslationOnly",
                        seeds=(0, 1, 3),
                        name="no_wrench_translation_only",
                    ),
                ),
            ],
        ),
        PanelSpec(
            "Trajectory Diversity",
            [
                base.SeriesSpec("Random Traj. (Ours)", colors["ours"], ours),
                base.SeriesSpec(
                    "100",
                    colors["secondary"],
                    base._curves_from_traj_precision(
                        traj_precision,
                        condition="wrench",
                        family="Trajectory_Count",
                        checkpoint="100",
                        seeds=(0, 1, 2),
                        name="wrench_100_traj",
                    ),
                ),
                base.SeriesSpec(
                    "10",
                    colors["tertiary"],
                    base._curves_from_traj_precision(
                        traj_precision,
                        condition="wrench",
                        family="Trajectory_Count",
                        checkpoint="10",
                        seeds=(0, 1, 2),
                        name="wrench_10_traj",
                    ),
                ),
            ],
        ),
        PanelSpec(
            "Goal Precision",
            [
                base.SeriesSpec("1 cm (Ours)", colors["ours"], ours),
                base.SeriesSpec(
                    "5cm",
                    colors["secondary"],
                    base._curves_from_traj_precision(
                        traj_precision,
                        condition="wrench",
                        family="Precision",
                        checkpoint="5cm",
                        seeds=(0, 1, 2),
                        name="wrench_5cm",
                    ),
                ),
                base.SeriesSpec(
                    "10cm",
                    colors["tertiary"],
                    base._curves_from_traj_precision(
                        traj_precision,
                        condition="wrench",
                        family="Precision",
                        checkpoint="10cm",
                        seeds=(0, 1, 2),
                        name="wrench_10cm",
                    ),
                ),
            ],
        ),
    ]


def _apply_small_multiples_axis(ax: plt.Axes, *, idx: int, x_max_billions: float, show_y_label: bool) -> None:
    ax.set_xlim(0.0, x_max_billions)
    ax.set_ylim(-3.0, 104.0)
    ax.set_xticks(np.arange(0.0, x_max_billions + 0.5, 1.0))
    ax.set_xticklabels(["0"] + [f"{int(v)}B" for v in np.arange(1.0, x_max_billions + 0.5, 1.0)])
    ax.set_yticks([0, 50, 100])
    if idx == 0 or not show_y_label:
        ax.set_yticklabels(["0%", "50%", "100%"])
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("Env Steps", fontsize=7.8, labelpad=1.5)
    base.style_axis(ax)
    ax.tick_params(axis="both", which="major", labelsize=6.8, pad=1.2)


def _plot_series(ax: plt.Axes, series: list[base.SeriesSpec], *, x_max_billions: float, num_points: int):
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
        ax.plot(x[visible], mean[visible], color=item.color, linewidth=1.35, label=item.label, zorder=3)
        ax.fill_between(
            x[visible],
            mean[visible] - std[visible],
            mean[visible] + std[visible],
            color=item.color,
            alpha=0.15,
            linewidth=0,
            zorder=2,
        )


def _add_panel_legend(ax: plt.Axes, *, layout: str) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    common = dict(frameon=False, handlelength=1.0, handletextpad=0.25, borderaxespad=0.0)
    if layout == "one_row":
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.27),
            ncol=3,
            fontsize=5.1,
            columnspacing=0.28,
            **common,
        )
    elif layout == "ours_first":
        ours = ax.legend(
            handles[:1],
            labels[:1],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.27),
            ncol=1,
            fontsize=5.8,
            **common,
        )
        ax.add_artist(ours)
        ax.legend(
            handles[1:],
            labels[1:],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.42),
            ncol=2,
            fontsize=5.5,
            columnspacing=0.42,
            **common,
        )
    elif layout == "three_row":
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.27),
            ncol=1,
            fontsize=5.5,
            labelspacing=0.18,
            **common,
        )
    else:
        raise ValueError(f"unknown legend layout: {layout}")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        out = path.with_suffix(suffix)
        fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white", edgecolor="none")
        print(f"wrote {out}")


def _plot_one_row(
    panels: list[PanelSpec],
    output: Path,
    *,
    legend_layout: str,
    x_max_billions: float,
    num_points: int,
) -> Path:
    base.configure_rcparams()
    fig_height = {"one_row": 2.25, "ours_first": 2.55, "three_row": 2.75}[legend_layout]
    fig, axes = plt.subplots(1, 4, figsize=(7.15, fig_height), dpi=240)
    for idx, (ax, panel) in enumerate(zip(axes, panels, strict=True)):
        _plot_series(ax, panel.series, x_max_billions=x_max_billions, num_points=num_points)
        _apply_small_multiples_axis(ax, idx=idx, x_max_billions=x_max_billions, show_y_label=True)
        ax.set_title(panel.title, fontsize=8.8, pad=3.0)
        _add_panel_legend(ax, layout=legend_layout)

    bottom = {"one_row": 0.30, "ours_first": 0.37, "three_row": 0.43}[legend_layout]
    fig.text(0.012, 0.58, "Success Rate", va="center", rotation=90, fontsize=8.5)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.84, bottom=bottom, wspace=0.42)
    _save(fig, output)
    plt.close(fig)
    return output.with_suffix(".png")


def _plot_review(paths: list[Path], output: Path) -> None:
    fig, axes = plt.subplots(len(paths), 1, figsize=(8.0, 2.3 * len(paths)), dpi=180)
    if len(paths) == 1:
        axes = [axes]
    for ax, path in zip(axes, paths, strict=True):
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem, fontsize=9, pad=2)
        ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01, hspace=0.09)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.02, facecolor="white", edgecolor="none")
    print(f"wrote {output}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-max-billions", type=float, default=4.0)
    parser.add_argument("--slug", default="selected_ours_4B")
    parser.add_argument("--num-points", type=int, default=500)
    parser.add_argument("--wandb-samples", type=int, default=7000)
    parser.add_argument("--refresh-wrench", action="store_true")
    parser.add_argument("--refresh-traj-precision", action="store_true")
    parser.add_argument(
        "--legend-layouts",
        nargs="+",
        default=["ours_first", "one_row", "three_row"],
        choices=["ours_first", "one_row", "three_row"],
    )
    args = parser.parse_args()

    wrench, traj_precision, no_training = _load_or_collect(
        refresh_wrench=args.refresh_wrench,
        refresh_traj_precision=args.refresh_traj_precision,
        wandb_samples=args.wandb_samples,
    )
    panels = _build_panels(wrench, traj_precision, no_training)

    generated: list[Path] = []
    for layout in args.legend_layouts:
        output = OUT_DIR / f"lpeg_final_one_row_narrow_{layout}_{args.slug}.png"
        generated.append(
            _plot_one_row(
                panels,
                output,
                legend_layout=layout,
                x_max_billions=args.x_max_billions,
                num_points=args.num_points,
            )
        )

    if len(generated) > 1:
        _plot_review(generated, OUT_DIR / f"lpeg_final_one_row_narrow_legend_review_{args.slug}.png")


if __name__ == "__main__":
    main()

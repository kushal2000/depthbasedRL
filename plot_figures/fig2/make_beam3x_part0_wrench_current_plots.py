"""Plot the current beam_3x part-0 wrench finetuning runs.

This is intentionally separate from the L-peg plotting scripts and writes a
beam-specific filename so the existing Fig. 2/L-peg outputs remain untouched.

The script reads local TensorBoard event files from the 18 trusted runs in:

    /move/u/tylerlum/github_repos/depthbasedRL/train_dir/fig2/
    panel_a_teachers_tyler_beam3x_part0_wrench

It uses the same feasible-normalized success metric as the L-peg plots:

    episode_final/all_goals_hit / (1 - episode_final/done_fall)

Example:
    /move/u/tylerlum/github_repos/depthbasedRL/.venv-isaacsim-py311/bin/python \
        plot_figures/fig2/make_beam3x_part0_wrench_current_plots.py
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "plot_figures" / "fig2"))
from collect_lpeg_seed_curves import EVENT_TAGS, _add_derived_metrics, _scalar_value  # noqa: E402


RUN_ROOT = Path(
    "/move/u/tylerlum/github_repos/depthbasedRL/train_dir/fig2/"
    "panel_a_teachers_tyler_beam3x_part0_wrench"
)
OUT_DIR = REPO / "plot_figures" / "fig2" / "outputs"
DATA_OUT_DIR = REPO / "outputs" / "fig2_tyler_beam3x_part0_wrench"
METRIC = "episode_final/feasible_normalized_all_goals_hit"
MAX_SCALAR_RECORD_BYTES = 1_000_000

COLORS = {
    "ours": "#2C7BB6",
    "secondary": "#E08214",
    "tertiary": "#8C8C8C",
}


@dataclass(frozen=True)
class RunSpec:
    family: str
    checkpoint: str
    seed: int
    run_name: str


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


@dataclass
class PanelSpec:
    title: str
    series: list[SeriesSpec]


# Use only the known-good active/retry run dirs.  The root also contains failed
# full-scale A5000 attempts and half-env attempts that hit PhysX CUDA code 2.
TRUSTED_RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        "ObjectDiversity",
        "100_obj",
        0,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed0_ObjectDiversity_100_obj_2026-06-03_03-42-55",
    ),
    RunSpec(
        "ObjectDiversity",
        "100_obj",
        1,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed1_ObjectDiversity_100_obj_2026-06-03_03-42-55",
    ),
    RunSpec(
        "ObjectDiversity",
        "10_obj",
        0,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed0_a5000_quarterenv_physxretry_ObjectDiversity_10_obj_2026-06-03_11-16-05",
    ),
    RunSpec(
        "ObjectDiversity",
        "10_obj",
        1,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed1_a5000_halfenv_ObjectDiversity_10_obj_2026-06-03_03-51-32",
    ),
    RunSpec(
        "TrainingObjective",
        "Play2Win",
        0,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed0_TrainingObjective_Play2Win_2026-06-03_03-42-55",
    ),
    RunSpec(
        "TrainingObjective",
        "Play2Win",
        1,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed1_TrainingObjective_Play2Win_2026-06-03_03-42-55",
    ),
    RunSpec(
        "TrainingObjective",
        "RotationOnly",
        0,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed0_TrainingObjective_RotationOnly_2026-06-03_03-42-55",
    ),
    RunSpec(
        "TrainingObjective",
        "RotationOnly",
        1,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed1_TrainingObjective_RotationOnly_2026-06-03_03-42-55",
    ),
    RunSpec(
        "TrainingObjective",
        "TranslationOnly",
        0,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed0_a5000_halfenv_TrainingObjective_TranslationOnly_2026-06-03_03-51-54",
    ),
    RunSpec(
        "TrainingObjective",
        "TranslationOnly",
        1,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed1_a5000_halfenv_TrainingObjective_TranslationOnly_2026-06-03_03-51-55",
    ),
    RunSpec(
        "Trajectory_Count",
        "100",
        0,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed0_Trajectory_Count_100_2026-06-03_03-40-55",
    ),
    RunSpec(
        "Trajectory_Count",
        "100",
        1,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed1_Trajectory_Count_100_2026-06-03_03-42-55",
    ),
    RunSpec(
        "Trajectory_Count",
        "10",
        0,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed0_a5000_halfenv_Trajectory_Count_10_2026-06-03_03-51-55",
    ),
    RunSpec(
        "Trajectory_Count",
        "10",
        1,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed1_a5000_halfenv_Trajectory_Count_10_2026-06-03_03-51-54",
    ),
    RunSpec(
        "Precision",
        "10cm",
        0,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed0_a5000_halfenv_Precision_10cm_2026-06-03_03-51-55",
    ),
    RunSpec(
        "Precision",
        "10cm",
        1,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed1_a5000_halfenv_Precision_10cm_2026-06-03_03-51-54",
    ),
    RunSpec(
        "Precision",
        "5cm",
        0,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed0_a5000_quarterenv_physxretry_Precision_5cm_2026-06-03_11-16-08",
    ),
    RunSpec(
        "Precision",
        "5cm",
        1,
        "beam_3x_part_0_finetune_rgf0_dr_wrench_seed1_a5000_halfenv_Precision_5cm_2026-06-03_03-51-54",
    ),
)


def _event_file_for_run(run_name: str) -> Path:
    event_files = sorted(
        (RUN_ROOT / run_name).glob("*/summaries/events.out.tfevents*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found for {RUN_ROOT / run_name}")
    return event_files[-1]


def _read_scalars_fast(event_file: Path, tags: list[str]) -> dict[str, dict[str, list[float]]]:
    """Read TensorBoard scalar summaries while skipping large media records.

    The generic TensorBoard loader parses image/video summaries too, which is
    unnecessarily slow for these event files. Scalar summaries are tiny, so any
    very large TFRecord is skipped before protobuf parsing.
    """
    from tensorboard.compat.proto import event_pb2

    tag_set = set(tags)
    data: dict[str, dict[str, list[float]]] = {
        tag: {"steps": [], "values": [], "wall_times": []} for tag in tags
    }
    skipped_large = 0
    with event_file.open("rb") as f:
        while True:
            header = f.read(8)
            if not header:
                break
            if len(header) != 8:
                break
            (length,) = struct.unpack("<Q", header)
            f.seek(4, 1)  # masked CRC of length
            if length > MAX_SCALAR_RECORD_BYTES:
                f.seek(length + 4, 1)  # payload + masked CRC of payload
                skipped_large += 1
                continue
            payload = f.read(length)
            if len(payload) != length:
                break
            f.seek(4, 1)

            event = event_pb2.Event()
            event.ParseFromString(payload)
            if not event.summary.value:
                continue
            for value in event.summary.value:
                if value.tag not in tag_set:
                    continue
                scalar = _scalar_value(value)
                if scalar is None:
                    continue
                data[value.tag]["steps"].append(int(event.step))
                data[value.tag]["values"].append(float(scalar))

    if skipped_large:
        print(f"  skipped {skipped_large} large non-scalar records")
    return {tag: series for tag, series in data.items() if series["steps"]}


def _series_to_curve(name: str, spec: RunSpec, seed_data: dict[str, Any]) -> Curve | None:
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
    return Curve(name=name, seed=spec.seed, x=x - x[0], y=y, run_name=spec.run_name)


def _summarize_metrics(metrics: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for tag, series in metrics.items():
        steps = series.get("steps", [])
        values = series.get("values", [])
        if not steps or not values:
            continue
        summary[tag] = {
            "num_points": len(values),
            "first_step": int(steps[0]),
            "last_step": int(steps[-1]),
            "relative_env_steps": int(steps[-1] - steps[0]),
            "first": float(values[0]),
            "last": float(values[-1]),
            "max": float(np.nanmax(values)),
        }
    return summary


def _collect_runs() -> dict[tuple[str, str, int], dict[str, Any]]:
    collected: dict[tuple[str, str, int], dict[str, Any]] = {}
    for idx, spec in enumerate(TRUSTED_RUNS, start=1):
        event_file = _event_file_for_run(spec.run_name)
        size_mb = event_file.stat().st_size / (1024.0 * 1024.0)
        print(
            f"[{idx:02d}/{len(TRUSTED_RUNS)}] reading {spec.family}/{spec.checkpoint} "
            f"seed{spec.seed}: {size_mb:.1f} MB"
        )
        raw = _read_scalars_fast(event_file, EVENT_TAGS)
        metrics = _add_derived_metrics(raw)
        seed_data = {
            "family": spec.family,
            "checkpoint": spec.checkpoint,
            "seed": spec.seed,
            "run_name": spec.run_name,
            "run_dir": str(RUN_ROOT / spec.run_name),
            "event_file": str(event_file),
            "metrics": metrics,
            "summary": _summarize_metrics(metrics),
        }
        curve = _series_to_curve(f"{spec.family}_{spec.checkpoint}_seed{spec.seed}", spec, seed_data)
        if curve is None:
            print(f"warning: no usable curve for {spec.run_name}")
        seed_data["curve"] = curve
        collected[(spec.family, spec.checkpoint, spec.seed)] = seed_data
    return collected


def _curve(
    collected: dict[tuple[str, str, int], dict[str, Any]],
    family: str,
    checkpoint: str,
    seed: int,
) -> Curve | None:
    data = collected.get((family, checkpoint, seed))
    if data is None:
        print(f"warning: missing {family}/{checkpoint} seed {seed}")
        return None
    curve = data.get("curve")
    if curve is None:
        print(f"warning: no curve {family}/{checkpoint} seed {seed}")
    return curve


def _curves(
    collected: dict[tuple[str, str, int], dict[str, Any]],
    family: str,
    checkpoint: str,
    seeds: tuple[int, ...] = (0, 1),
) -> list[Curve]:
    out: list[Curve] = []
    for seed in seeds:
        curve = _curve(collected, family, checkpoint, seed)
        if curve is not None:
            out.append(curve)
    return out


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
    # Keep only common support so short retry runs do not leave one-seed tails.
    keep = np.sum(np.isfinite(stack), axis=0) >= len(curves)
    if not np.any(keep):
        return None
    return {
        "x": grid[keep],
        "mean": np.nanmean(stack[:, keep], axis=0),
        "std": np.nanstd(stack[:, keep], axis=0),
    }


def _build_panels(collected: dict[tuple[str, str, int], dict[str, Any]]) -> list[PanelSpec]:
    ours = _curves(collected, "TrainingObjective", "Play2Win")
    if len(ours) != 2:
        print(f"warning: expected 2 Play2Win/Ours curves, got {len(ours)}")
    return [
        PanelSpec(
            "Object Diversity",
            [
                SeriesSpec("Play2Win\n(Ours)", COLORS["ours"], ours),
                SeriesSpec("100\nObjects", COLORS["secondary"], _curves(collected, "ObjectDiversity", "100_obj")),
                SeriesSpec("10\nObjects", COLORS["tertiary"], _curves(collected, "ObjectDiversity", "10_obj")),
            ],
        ),
        PanelSpec(
            "Training Objective",
            [
                SeriesSpec("6D Pose\n(Ours)", COLORS["ours"], ours),
                SeriesSpec("Rot-Only", COLORS["secondary"], _curves(collected, "TrainingObjective", "RotationOnly")),
                SeriesSpec("Trans-Only", COLORS["tertiary"], _curves(collected, "TrainingObjective", "TranslationOnly")),
            ],
        ),
        PanelSpec(
            "Trajectory Diversity",
            [
                SeriesSpec("Random\n(Ours)", COLORS["ours"], ours),
                SeriesSpec("100", COLORS["secondary"], _curves(collected, "Trajectory_Count", "100")),
                SeriesSpec("10", COLORS["tertiary"], _curves(collected, "Trajectory_Count", "10")),
            ],
        ),
        PanelSpec(
            "Goal Precision",
            [
                SeriesSpec("1 cm\n(Ours)", COLORS["ours"], ours),
                SeriesSpec("10 cm", COLORS["secondary"], _curves(collected, "Precision", "10cm")),
                SeriesSpec("5 cm", COLORS["tertiary"], _curves(collected, "Precision", "5cm")),
            ],
        ),
    ]


def _configure_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_series(
    ax: plt.Axes,
    series: list[SeriesSpec],
    *,
    x_max_billions: float,
    num_points: int,
    x_axis_mode: str,
    hours_at_xmax: float,
) -> None:
    for item in series:
        aggregate = _aggregate(item.curves, num_points=num_points)
        if aggregate is None:
            print(f"warning: no aggregate for {item.label}")
            continue
        x = aggregate["x"] / 1e9
        plot_x = x if x_axis_mode == "env_steps" else x / x_max_billions * hours_at_xmax
        mean = 100.0 * aggregate["mean"]
        std = 100.0 * aggregate["std"]
        visible = x <= x_max_billions
        if not np.any(visible):
            print(f"warning: no visible points for {item.label}")
            continue
        ax.plot(plot_x[visible], mean[visible], color=item.color, linewidth=1.5, zorder=3)
        ax.fill_between(
            plot_x[visible],
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


def _add_legend(ax: plt.Axes, series: list[SeriesSpec], *, legend_font_size: float) -> None:
    handles = [Patch(facecolor=item.color, edgecolor="#333333", linewidth=0.7) for item in series]
    labels = [item.label for item in series]
    legend = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=len(handles),
        frameon=False,
        fontsize=legend_font_size,
        handlelength=0.95,
        handleheight=0.95,
        handletextpad=0.35,
        borderaxespad=0.0,
        columnspacing=0.38,
        labelspacing=0.1,
    )
    for text, item in zip(legend.get_texts(), series, strict=True):
        text.set_multialignment("center")
        if "(Ours)" in item.label:
            text.set_fontweight(600)


def _save_outputs(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white", edgecolor="none")
        print(f"wrote {path}")


def _plot(
    panels: list[PanelSpec],
    output: Path,
    *,
    x_max_billions: float,
    num_points: int,
    x_axis_mode: str,
    hours_at_xmax: float,
) -> None:
    _configure_rcparams()
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.76), dpi=240)
    for ax, panel in zip(axes, panels, strict=True):
        _plot_series(
            ax,
            panel.series,
            x_max_billions=x_max_billions,
            num_points=num_points,
            x_axis_mode=x_axis_mode,
            hours_at_xmax=hours_at_xmax,
        )
        _style_axes(
            ax,
            x_max_billions=x_max_billions,
            x_axis_mode=x_axis_mode,
            hours_at_xmax=hours_at_xmax,
        )
        ax.set_title(panel.title, fontsize=8, pad=2.0)
        _add_legend(ax, panel.series, legend_font_size=6.0)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.83, bottom=0.47, wspace=0.45)
    _save_outputs(fig, output)
    plt.close(fig)


def _write_data_json(collected: dict[tuple[str, str, int], dict[str, Any]], path: Path) -> None:
    serializable: dict[str, Any] = {
        "run_root": str(RUN_ROOT),
        "metric": METRIC,
        "notes": [
            "Uses trusted active/retry run directories only.",
            "Env-step x-axis is relative finetuning steps: event_step - first_event_step.",
            "The 10_obj seed0 and 5cm seed0 curves use quarter-env PhysX retry runs.",
        ],
        "runs": [],
    }
    for spec in TRUSTED_RUNS:
        data = collected[(spec.family, spec.checkpoint, spec.seed)]
        serializable["runs"].append(
            {
                "family": spec.family,
                "checkpoint": spec.checkpoint,
                "seed": spec.seed,
                "run_name": spec.run_name,
                "run_dir": data["run_dir"],
                "event_file": data["event_file"],
                "summary": data["summary"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(serializable, f, indent=2)
    print(f"wrote {path}")


def _print_summary(collected: dict[tuple[str, str, int], dict[str, Any]]) -> None:
    print("\nBeam3x part0 current wrench data:")
    print("| Family | Checkpoint | Seed | Final norm success | Env steps | Run |")
    print("|---|---|---:|---:|---:|---|")
    for spec in TRUSTED_RUNS:
        summary = collected[(spec.family, spec.checkpoint, spec.seed)]["summary"].get(METRIC, {})
        final = summary.get("last", float("nan"))
        steps = summary.get("relative_env_steps", 0)
        print(
            f"| {spec.family} | {spec.checkpoint} | {spec.seed} | "
            f"{100.0 * final:.1f}% | {steps / 1e9:.2f}B | {spec.run_name} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-max-billions", type=float, default=4.0)
    parser.add_argument("--num-points", type=int, default=500)
    parser.add_argument("--slug", default="current_4B")
    parser.add_argument("--x-axis-mode", choices=["env_steps", "hours"], default="env_steps")
    parser.add_argument(
        "--hours-at-xmax",
        type=float,
        default=24.0,
        help="When --x-axis-mode=hours, map --x-max-billions to this many hours.",
    )
    args = parser.parse_args()

    collected = _collect_runs()
    _write_data_json(collected, DATA_OUT_DIR / f"beam3x_part0_wrench_{args.slug}_runs.json")
    panels = _build_panels(collected)
    _plot(
        panels,
        OUT_DIR / f"beam3x_part0_wrench_one_row_{args.slug}.png",
        x_max_billions=args.x_max_billions,
        num_points=args.num_points,
        x_axis_mode=args.x_axis_mode,
        hours_at_xmax=args.hours_at_xmax,
    )
    _print_summary(collected)


if __name__ == "__main__":
    main()

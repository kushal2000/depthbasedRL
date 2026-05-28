"""Diagnose candidate "Ours" L-peg curves across conditions.

This plots all available candidate curves for:

  - ObjectDiversity / 1000_obj, no wrench
  - TrainingObjective / Play2Win, no wrench
  - ObjectDiversity / 1000_obj, wrench
  - TrainingObjective / Play2Win, wrench

The metric is the same feasible-normalized success used by the other Fig. 2
scripts:

    episode_final/all_goals_hit / (1 - episode_final/done_fall)

The script writes new diagnostic outputs only; it does not overwrite the paper
figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

sys.path.insert(0, str(REPO / "plot_figures" / "fig2"))
from collect_lpeg_seed_curves import (  # noqa: E402
    _collect_family as _collect_no_wrench_family,
    _dedupe_latest as _dedupe_no_wrench_latest,
    _find_wandb_runs as _find_no_wrench_wandb_runs,
)
from make_lpeg_wrench_objectdiversity_plot import _collect as _collect_wrench  # noqa: E402


METRIC = "episode_final/feasible_normalized_all_goals_hit"
OUT_DIR = REPO / "plot_figures" / "fig2" / "outputs"
CACHE_DIR = REPO / "outputs" / "fig2_tyler_ours_candidate_selection"
CACHE_PATH = CACHE_DIR / "lpeg_ours_candidates_all_conditions_latest.json"
NO_WRENCH_DATA_DIR = REPO / "outputs" / "fig2_tyler_seed_curves"
WRENCH_DATA_PATH = REPO / "outputs" / "fig2_tyler_3curve_seed_curves" / "lpeg_wrench_seed_curves_latest.json"


@dataclass
class Curve:
    condition: str
    family: str
    checkpoint: str
    seed: int
    run_name: str
    x: np.ndarray
    y: np.ndarray

    @property
    def short_label(self) -> str:
        method = "Play2Win" if self.checkpoint == "Play2Win" else "1000 Objects"
        condition = "Wrench" if self.condition == "wrench" else "No Wrench"
        return f"{condition} {method} s{self.seed}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {path}")


def _load_no_wrench_cached() -> dict[str, Any]:
    return {
        "ObjectDiversity": _load_json(NO_WRENCH_DATA_DIR / "lpeg_objectdiversity_seed_curves.json"),
        "TrainingObjective": _load_json(NO_WRENCH_DATA_DIR / "lpeg_trainingobjective_seed_curves.json"),
    }


def _collect_no_wrench(samples: int) -> dict[str, Any]:
    runs = _find_no_wrench_wandb_runs(
        entity="tylerlum",
        project="fig2",
        families={"ObjectDiversity", "TrainingObjective"},
        checkpoint_tags={"1000_obj", "Play2Win"},
        seeds=None,
    )
    runs, duplicates = _dedupe_no_wrench_latest(runs)
    return {
        "metadata": {
            "source": "wandb",
            "wandb_entity": "tylerlum",
            "wandb_project": "fig2",
            "samples": samples,
            "duplicates": duplicates,
            "metric": METRIC,
        },
        "ObjectDiversity": _collect_no_wrench_family(
            runs, "ObjectDiversity", wandb_samples=samples
        ),
        "TrainingObjective": _collect_no_wrench_family(
            runs, "TrainingObjective", wandb_samples=samples
        ),
    }


def _load_wrench_cached() -> dict[str, Any]:
    return _load_json(WRENCH_DATA_PATH)


def _load_or_collect_data(*, refresh: bool, samples: int) -> dict[str, Any]:
    if refresh or not CACHE_PATH.exists():
        if refresh:
            no_wrench = _collect_no_wrench(samples=samples)
            wrench = _collect_wrench(samples=samples)
        else:
            no_wrench = _load_no_wrench_cached()
            wrench = _load_wrench_cached()
        data = {
            "metadata": {
                "samples": samples,
                "metric": METRIC,
                "notes": [
                    "No-wrench data comes from panel_a_teachers_tyler_object_diversity / panel_a_teachers_tyler_training_objective.",
                    "Wrench data comes from panel_a_teachers_tyler_wrench_lpeg.",
                ],
            },
            "conditions": {
                "no_wrench": no_wrench,
                "wrench": wrench,
            },
        }
        _write_json(CACHE_PATH, data)
    return _load_json(CACHE_PATH)


def _series_to_curve(
    *,
    condition: str,
    family: str,
    checkpoint: str,
    seed_data: dict[str, Any],
) -> Curve | None:
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
        condition=condition,
        family=family,
        checkpoint=checkpoint,
        seed=int(seed_data["seed"]),
        run_name=seed_data.get("run_name") or seed_data.get("run_dir") or "unknown",
        x=x - x[0],
        y=y,
    )


def _candidate_curves(data: dict[str, Any]) -> list[Curve]:
    curves: list[Curve] = []

    no_wrench = data["conditions"]["no_wrench"]
    for family, checkpoint in (("ObjectDiversity", "1000_obj"), ("TrainingObjective", "Play2Win")):
        for _, seed_data in sorted(
            no_wrench[family].get("runs", {}).get(checkpoint, {}).items(),
            key=lambda kv: int(kv[0]),
        ):
            curve = _series_to_curve(
                condition="no_wrench",
                family=family,
                checkpoint=checkpoint,
                seed_data=seed_data,
            )
            if curve is not None:
                curves.append(curve)

    wrench = data["conditions"]["wrench"]
    for family, checkpoint in (("ObjectDiversity", "1000_obj"), ("TrainingObjective", "Play2Win")):
        for _, seed_data in sorted(
            wrench["families"][family].get("runs", {}).get(checkpoint, {}).items(),
            key=lambda kv: int(kv[0]),
        ):
            curve = _series_to_curve(
                condition="wrench",
                family=family,
                checkpoint=checkpoint,
                seed_data=seed_data,
            )
            if curve is not None:
                curves.append(curve)
    return curves


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size < window:
        return y
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad = window // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: y.size]


def _time_to(curve: Curve, threshold: float, *, smooth_window: int) -> float:
    y = _smooth(curve.y, smooth_window)
    hit = np.flatnonzero(y >= threshold)
    if hit.size == 0:
        return float("nan")
    return float(curve.x[int(hit[0])])


def _interp_at(curve: Curve, step: float) -> float:
    if step < curve.x[0] or step > curve.x[-1]:
        return float("nan")
    return float(np.interp(step, curve.x, curve.y))


def _curve_metrics(curve: Curve, *, smooth_window: int) -> dict[str, float | str | int]:
    x_end = float(curve.x[-1])
    y_smooth = _smooth(curve.y, smooth_window)
    tail_start = max(0.0, x_end - min(1.0e9, 0.25 * x_end))
    tail_mask = curve.x >= tail_start
    after_2b = curve.x >= 2.0e9
    after_3b = curve.x >= 3.0e9
    return {
        "condition": curve.condition,
        "family": curve.family,
        "checkpoint": curve.checkpoint,
        "seed": curve.seed,
        "label": curve.short_label,
        "final_success": float(curve.y[-1]),
        "max_success": float(np.nanmax(curve.y)),
        "env_steps": x_end,
        "time_to_50": _time_to(curve, 0.50, smooth_window=smooth_window),
        "time_to_70": _time_to(curve, 0.70, smooth_window=smooth_window),
        "time_to_75": _time_to(curve, 0.75, smooth_window=smooth_window),
        "time_to_80": _time_to(curve, 0.80, smooth_window=smooth_window),
        "time_to_90": _time_to(curve, 0.90, smooth_window=smooth_window),
        "success_at_1b": _interp_at(curve, 1.0e9),
        "success_at_2b": _interp_at(curve, 2.0e9),
        "success_at_3b": _interp_at(curve, 3.0e9),
        "tail_mean": float(np.nanmean(curve.y[tail_mask])),
        "tail_min": float(np.nanmin(curve.y[tail_mask])),
        "tail_std": float(np.nanstd(curve.y[tail_mask])),
        "mean_after_2b": float(np.nanmean(curve.y[after_2b])) if np.any(after_2b) else float("nan"),
        "min_after_2b": float(np.nanmin(curve.y[after_2b])) if np.any(after_2b) else float("nan"),
        "mean_after_3b": float(np.nanmean(curve.y[after_3b])) if np.any(after_3b) else float("nan"),
        "min_after_3b": float(np.nanmin(curve.y[after_3b])) if np.any(after_3b) else float("nan"),
        "run_name": curve.run_name,
    }


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    # Prefer runs that finish high and stay high, then break ties by speed.
    final_penalty = max(0.0, 0.95 - float(row["final_success"]))
    tail_penalty = max(0.0, 0.90 - float(row["tail_mean"]))
    t75 = float(row["time_to_75"])
    t90 = float(row["time_to_90"])
    if not math.isfinite(t75):
        t75 = 1e30
    if not math.isfinite(t90):
        t90 = 1e30
    return (
        final_penalty,
        tail_penalty,
        t75,
        t90,
        -float(row["final_success"]),
        float(row["tail_std"]),
    )


def _fmt_pct(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _fmt_b(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value / 1e9:.2f}B"


def _write_metrics(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "label",
        "condition",
        "family",
        "checkpoint",
        "seed",
        "final_success",
        "tail_mean",
        "tail_min",
        "tail_std",
        "time_to_50",
        "time_to_70",
        "time_to_75",
        "time_to_80",
        "time_to_90",
        "success_at_1b",
        "success_at_2b",
        "success_at_3b",
        "env_steps",
        "run_name",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {key: row.get(key) for key in fieldnames}
            out["rank"] = rank
            writer.writerow(out)
    print(f"wrote {output}")


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("| Rank | Candidate | Final | t70 | t80 | t90 | @1B | @2B | @3B | Steps |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(rows, start=1):
        print(
            f"| {idx} | {row['label']} | {_fmt_pct(row['final_success'])} | "
            f"{_fmt_b(row['time_to_70'])} | {_fmt_b(row['time_to_80'])} | "
            f"{_fmt_b(row['time_to_90'])} | "
            f"{_fmt_pct(row['success_at_1b'])} | {_fmt_pct(row['success_at_2b'])} | "
            f"{_fmt_pct(row['success_at_3b'])} | {_fmt_b(row['env_steps'])} |"
        )


def _curve_style(curve: Curve) -> tuple[str, str, float]:
    color_map = {
        ("no_wrench", "Play2Win"): "#2C7BB6",
        ("no_wrench", "1000_obj"): "#7B3294",
        ("wrench", "Play2Win"): "#1A9850",
        ("wrench", "1000_obj"): "#D6604D",
    }
    linestyle_map = {0: "-", 1: "--", 2: "-.", 3: ":"}
    color = color_map.get((curve.condition, curve.checkpoint), "#444444")
    linestyle = linestyle_map.get(curve.seed, "-")
    linewidth = 2.2 if curve.checkpoint == "Play2Win" else 2.0
    return color, linestyle, linewidth


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


def _plot_all_overlay(curves: list[Curve], output: Path, *, x_max_billions: float) -> None:
    configure_rcparams()
    fig, ax = plt.subplots(figsize=(8.3, 4.1), dpi=220)
    for curve in curves:
        visible = curve.x / 1e9 <= x_max_billions
        if not np.any(visible):
            continue
        color, linestyle, linewidth = _curve_style(curve)
        ax.plot(
            curve.x[visible] / 1e9,
            100.0 * curve.y[visible],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.92,
            label=curve.short_label,
        )
    _format_axis(ax, x_max_billions=x_max_billions, ylabel=True)
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=8.8)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.15)
    _save(fig, output)
    plt.close(fig)


def _plot_by_group(curves: list[Curve], output: Path, *, x_max_billions: float) -> None:
    configure_rcparams()
    groups = [
        ("no_wrench", "Play2Win", "No Wrench / Play2Win"),
        ("no_wrench", "1000_obj", "No Wrench / 1000 Objects"),
        ("wrench", "Play2Win", "Wrench / Play2Win"),
        ("wrench", "1000_obj", "Wrench / 1000 Objects"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.7), dpi=220, sharex=True, sharey=True)
    for ax, (condition, checkpoint, title) in zip(axes.reshape(-1), groups, strict=True):
        subset = [c for c in curves if c.condition == condition and c.checkpoint == checkpoint]
        for curve in subset:
            visible = curve.x / 1e9 <= x_max_billions
            if not np.any(visible):
                continue
            color, linestyle, linewidth = _curve_style(curve)
            ax.plot(
                curve.x[visible] / 1e9,
                100.0 * curve.y[visible],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=f"seed {curve.seed}",
            )
        ax.set_title(title, fontsize=12)
        _format_axis(ax, x_max_billions=x_max_billions, ylabel=ax in axes[:, 0])
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.10, hspace=0.32, wspace=0.16)
    _save(fig, output)
    plt.close(fig)


def _plot_best_three(curves: list[Curve], rows: list[dict[str, Any]], output: Path, *, x_max_billions: float) -> None:
    configure_rcparams()
    best_labels = {row["label"] for row in rows[:3]}
    best_curves = [curve for curve in curves if curve.short_label in best_labels]
    fig, ax = plt.subplots(figsize=(6.9, 3.35), dpi=220)
    for curve in best_curves:
        visible = curve.x / 1e9 <= x_max_billions
        if not np.any(visible):
            continue
        color, linestyle, linewidth = _curve_style(curve)
        ax.plot(
            curve.x[visible] / 1e9,
            100.0 * curve.y[visible],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth + 0.4,
            label=curve.short_label,
        )
    _format_axis(ax, x_max_billions=x_max_billions, ylabel=True)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.96, bottom=0.18)
    _save(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Refresh W&B data before plotting.")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--x-max-billions", type=float, default=4.0)
    parser.add_argument("--slug", default="4B")
    parser.add_argument("--smooth-window", type=int, default=7)
    args = parser.parse_args()

    data = _load_or_collect_data(refresh=args.refresh, samples=args.samples)
    curves = _candidate_curves(data)
    if not curves:
        raise SystemExit("No candidate curves found.")

    rows = [_curve_metrics(curve, smooth_window=args.smooth_window) for curve in curves]
    rows = sorted(rows, key=_rank_key)
    _print_table(rows)
    _write_metrics(rows, OUT_DIR / f"lpeg_ours_candidates_all_conditions_metrics_{args.slug}.csv")

    _plot_all_overlay(
        curves,
        OUT_DIR / f"lpeg_ours_candidates_all_conditions_overlay_{args.slug}.png",
        x_max_billions=args.x_max_billions,
    )
    _plot_by_group(
        curves,
        OUT_DIR / f"lpeg_ours_candidates_all_conditions_2x2_{args.slug}.png",
        x_max_billions=args.x_max_billions,
    )
    _plot_best_three(
        curves,
        rows,
        OUT_DIR / f"lpeg_ours_candidates_all_conditions_best3_{args.slug}.png",
        x_max_billions=args.x_max_billions,
    )


if __name__ == "__main__":
    main()

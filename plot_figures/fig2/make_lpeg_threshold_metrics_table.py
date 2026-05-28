"""Summarize L-peg success curves by threshold-crossing metrics.

This script reads the cached L-peg curve JSONs and writes:

  - per-seed metrics for every available condition/checkpoint/seed
  - mean-curve metrics for every condition/checkpoint group

Times are relative finetuning environment steps, i.e. first logged global_step
is subtracted before computing threshold crossings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "plot_figures" / "fig2" / "outputs"

METRIC = "episode_final/feasible_normalized_all_goals_hit"
THRESHOLDS = (0.50, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

NO_WRENCH_OBJECT = REPO / "outputs" / "fig2_tyler_seed_curves" / "lpeg_objectdiversity_seed_curves.json"
NO_WRENCH_TRAINING = REPO / "outputs" / "fig2_tyler_seed_curves" / "lpeg_trainingobjective_seed_curves.json"
WRENCH_OBJECT_TRAINING = (
    REPO / "outputs" / "fig2_tyler_3curve_seed_curves" / "lpeg_wrench_seed_curves_latest.json"
)
TRAJ_PRECISION = (
    REPO / "outputs" / "fig2_tyler_traj_precision_seed_curves" / "lpeg_traj_precision_wrench_compare_latest.json"
)

FAMILY_ORDER = {
    "ObjectDiversity": 0,
    "TrainingObjective": 1,
    "Trajectory_Count": 2,
    "Precision": 3,
}
CHECKPOINT_ORDER = {
    "ObjectDiversity": ["1000_obj", "100_obj", "10_obj", "1_obj"],
    "TrainingObjective": ["Play2Win", "RotationOnly", "SingleGoal", "TranslationOnly"],
    "Trajectory_Count": ["100", "10", "1"],
    "Precision": ["2p5cm", "5cm", "10cm"],
}
CONDITION_ORDER = {"wrench": 0, "no_wrench": 1}


@dataclass
class Curve:
    condition: str
    family: str
    checkpoint: str
    seed: int
    x: np.ndarray
    y: np.ndarray
    run_name: str
    wandb_state: str | None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _curve_from_seed(
    *,
    condition: str,
    family: str,
    checkpoint: str,
    seed_key: str,
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
    x = x - x[0]
    return Curve(
        condition=condition,
        family=family,
        checkpoint=checkpoint,
        seed=int(seed_data.get("seed", seed_key)),
        x=x,
        y=y,
        run_name=seed_data.get("run_name") or seed_data.get("run_dir") or "unknown",
        wandb_state=seed_data.get("wandb_state"),
    )


def _iter_curves() -> list[Curve]:
    curves: list[Curve] = []

    for path, family in ((NO_WRENCH_OBJECT, "ObjectDiversity"), (NO_WRENCH_TRAINING, "TrainingObjective")):
        data = _load_json(path)
        for checkpoint, seeds in data.get("runs", {}).items():
            for seed_key, seed_data in seeds.items():
                curve = _curve_from_seed(
                    condition="no_wrench",
                    family=family,
                    checkpoint=checkpoint,
                    seed_key=seed_key,
                    seed_data=seed_data,
                )
                if curve is not None:
                    curves.append(curve)

    data = _load_json(WRENCH_OBJECT_TRAINING)
    for family, family_data in data.get("families", {}).items():
        for checkpoint, seeds in family_data.get("runs", {}).items():
            for seed_key, seed_data in seeds.items():
                curve = _curve_from_seed(
                    condition="wrench",
                    family=family,
                    checkpoint=checkpoint,
                    seed_key=seed_key,
                    seed_data=seed_data,
                )
                if curve is not None:
                    curves.append(curve)

    data = _load_json(TRAJ_PRECISION)
    for condition, condition_data in data.get("conditions", {}).items():
        for family, family_data in condition_data.items():
            for checkpoint, seeds in family_data.get("runs", {}).items():
                for seed_key, seed_data in seeds.items():
                    curve = _curve_from_seed(
                        condition=condition,
                        family=family,
                        checkpoint=checkpoint,
                        seed_key=seed_key,
                        seed_data=seed_data,
                    )
                    if curve is not None:
                        curves.append(curve)

    return sorted(curves, key=_curve_sort_key)


def _curve_sort_key(curve: Curve) -> tuple[Any, ...]:
    checkpoint_order = CHECKPOINT_ORDER.get(curve.family, [])
    try:
        checkpoint_idx = checkpoint_order.index(curve.checkpoint)
    except ValueError:
        checkpoint_idx = 999
    return (
        FAMILY_ORDER.get(curve.family, 999),
        checkpoint_idx,
        CONDITION_ORDER.get(curve.condition, 999),
        curve.seed,
    )


def _format_b(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v / 1e9:.2f}B"


def _format_pct(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{100.0 * v:.1f}%"


def _first_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    above = np.flatnonzero(y >= threshold)
    if above.size == 0:
        return None
    idx = int(above[0])
    if idx == 0:
        return float(x[0])
    x0, x1 = float(x[idx - 1]), float(x[idx])
    y0, y1 = float(y[idx - 1]), float(y[idx])
    if not math.isfinite(y0) or not math.isfinite(y1) or abs(y1 - y0) < 1e-12:
        return x1
    frac = min(max((threshold - y0) / (y1 - y0), 0.0), 1.0)
    return x0 + frac * (x1 - x0)


def _metrics_for_curve(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "final_success": float(y[-1]) if y.size else None,
        "peak_success": float(np.nanmax(y)) if y.size else None,
        "env_steps": float(x[-1]) if x.size else None,
    }
    for threshold in THRESHOLDS:
        metrics[f"t{int(threshold * 100)}"] = _first_crossing(x, y, threshold)
    return metrics


def _aggregate_mean_curve(curves: list[Curve], *, num_points: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not curves:
        return None
    end = max(float(curve.x[-1]) for curve in curves)
    if end <= 0.0:
        return None
    grid = np.linspace(0.0, end, num_points)
    ys = []
    for curve in curves:
        interp = np.interp(grid, curve.x, curve.y)
        interp[(grid < curve.x[0]) | (grid > curve.x[-1])] = np.nan
        ys.append(interp)
    stack = np.vstack(ys)
    keep = np.sum(np.isfinite(stack), axis=0) >= len(curves)
    if not np.any(keep):
        return None
    return grid[keep], np.nanmean(stack[:, keep], axis=0)


def _row_from_metrics(base: dict[str, Any], metrics: dict[str, float | None]) -> dict[str, Any]:
    row = dict(base)
    row["peak_success"] = metrics["peak_success"]
    row["final_success"] = metrics["final_success"]
    row["env_steps"] = metrics["env_steps"]
    for threshold in THRESHOLDS:
        row[f"t{int(threshold * 100)}"] = metrics[f"t{int(threshold * 100)}"]
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col)
            if col in {"peak_success", "final_success"}:
                vals.append(_format_pct(value))
            elif col.startswith("t") or col == "env_steps":
                vals.append(_format_b(value))
            elif isinstance(value, tuple):
                vals.append(",".join(str(v) for v in value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-points", type=int, default=1000)
    parser.add_argument("--slug", default="latest")
    args = parser.parse_args()

    curves = _iter_curves()
    per_seed_rows: list[dict[str, Any]] = []
    for curve in curves:
        metrics = _metrics_for_curve(curve.x, curve.y)
        per_seed_rows.append(
            _row_from_metrics(
                {
                    "family": curve.family,
                    "checkpoint": curve.checkpoint,
                    "condition": curve.condition,
                    "seed": curve.seed,
                    "wandb_state": curve.wandb_state or "",
                    "run_name": curve.run_name,
                },
                metrics,
            )
        )

    grouped: dict[tuple[str, str, str], list[Curve]] = {}
    for curve in curves:
        grouped.setdefault((curve.family, curve.checkpoint, curve.condition), []).append(curve)

    group_rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda k: _curve_sort_key(grouped[k][0])):
        family, checkpoint, condition = key
        group_curves = sorted(grouped[key], key=lambda c: c.seed)
        agg = _aggregate_mean_curve(group_curves, num_points=args.num_points)
        if agg is None:
            continue
        x, y = agg
        metrics = _metrics_for_curve(x, y)
        group_rows.append(
            _row_from_metrics(
                {
                    "family": family,
                    "checkpoint": checkpoint,
                    "condition": condition,
                    "n_seeds": len(group_curves),
                    "seeds": tuple(curve.seed for curve in group_curves),
                    "states": ",".join(sorted({curve.wandb_state or "" for curve in group_curves})),
                },
                metrics,
            )
        )

    csv_per_seed = OUT_DIR / f"lpeg_threshold_metrics_per_seed_{args.slug}.csv"
    csv_group = OUT_DIR / f"lpeg_threshold_metrics_group_mean_curve_{args.slug}.csv"
    _write_csv(csv_per_seed, per_seed_rows)
    _write_csv(csv_group, group_rows)

    columns_group = [
        "family",
        "checkpoint",
        "condition",
        "n_seeds",
        "seeds",
        "peak_success",
        "final_success",
        "env_steps",
        "t50",
        "t70",
        "t75",
        "t80",
        "t85",
        "t90",
        "t95",
    ]
    columns_seed = [
        "family",
        "checkpoint",
        "condition",
        "seed",
        "peak_success",
        "final_success",
        "env_steps",
        "t50",
        "t70",
        "t75",
        "t80",
        "t85",
        "t90",
        "t95",
        "wandb_state",
    ]

    md = OUT_DIR / f"lpeg_threshold_metrics_summary_{args.slug}.md"
    with md.open("w") as f:
        f.write("# L-peg Threshold Metrics\n\n")
        f.write("Metric: `episode_final/feasible_normalized_all_goals_hit`.\n\n")
        f.write("Env steps are relative finetuning env steps: first logged `global_step` is subtracted.\n\n")
        f.write("Threshold times are first linearly interpolated crossings of the mean/raw curve. Blank means not reached.\n\n")
        f.write("## Mean Curve By Checkpoint / Wrench Setting\n\n")
        f.write(_markdown_table(group_rows, columns_group))
        f.write("\n\n## Per Seed\n\n")
        f.write(_markdown_table(per_seed_rows, columns_seed))
        f.write("\n")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()

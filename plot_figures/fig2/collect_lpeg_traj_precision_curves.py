"""Collect L-peg trajectory-count and precision-threshold curves from W&B.

This is a preparation step for the trajectory-diversity and play-precision
panels.  It collects the current runs from:

    tylerlum/fig2, group panel_a_teachers_tyler_traj_precision_wrench_compare

and writes a compact JSON with both raw success/fall metrics and the derived
feasible-normalized success metric.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "outputs" / "fig2_tyler_traj_precision_seed_curves"
WANDB_ENTITY = "tylerlum"
WANDB_PROJECT = "fig2"
WANDB_GROUP = "panel_a_teachers_tyler_traj_precision_wrench_compare"

RAW_SUCCESS = "episode_final/all_goals_hit"
DONE_FALL = "episode_final/done_fall"
METRIC = "episode_final/feasible_normalized_all_goals_hit"
EVENT_TAGS = [RAW_SUCCESS, DONE_FALL]

FAMILIES = ("Trajectory_Count", "Precision")
CHECKPOINT_ORDER = {
    "Trajectory_Count": ["100", "10", "1"],
    "Precision": ["5cm", "10cm", "2p5cm"],
}

RUN_RE = re.compile(
    r"^lpeg_tol0p5mm_finetune_rgf0_dr_"
    r"(?P<condition>wrench|no_wrench)"
    r"_seed(?P<seed>\d+)"
    r"_(?P<family>Trajectory_Count|Precision)"
    r"_(?P<tag>.+?)"
    r"_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"(?:_\d{8}_\d{6})?$"
)


@dataclass(frozen=True)
class RunInfo:
    condition: str
    family: str
    checkpoint_tag: str
    seed: int
    timestamp: str
    run_name: str
    wandb_path: str
    wandb_state: str
    wandb_run: Any


def _parse_run(run: Any) -> RunInfo | None:
    match = RUN_RE.match(run.name)
    if match is None:
        return None
    family = match.group("family")
    checkpoint_tag = match.group("tag")
    if checkpoint_tag not in CHECKPOINT_ORDER[family]:
        return None
    return RunInfo(
        condition=match.group("condition"),
        family=family,
        checkpoint_tag=checkpoint_tag,
        seed=int(match.group("seed")),
        timestamp=match.group("timestamp"),
        run_name=run.name,
        wandb_path=f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run.id}",
        wandb_state=run.state,
        wandb_run=run,
    )


def _find_wandb_runs() -> list[RunInfo]:
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit("wandb is required to collect trajectory/precision curves.") from exc

    api = wandb.Api(timeout=45)
    filters = {
        "group": WANDB_GROUP,
        "display_name": {"$regex": r"^lpeg_tol0p5mm_finetune_rgf0_dr_(wrench|no_wrench)"},
    }
    runs = []
    for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters, per_page=100):
        info = _parse_run(run)
        if info is not None:
            runs.append(info)
    return runs


def _dedupe_latest(runs: list[RunInfo]) -> list[RunInfo]:
    by_key: dict[tuple[str, str, str, int], list[RunInfo]] = {}
    for run in runs:
        by_key.setdefault((run.condition, run.family, run.checkpoint_tag, run.seed), []).append(run)
    selected = []
    for vals in by_key.values():
        selected.append(sorted(vals, key=lambda r: r.timestamp)[-1])
    return sorted(
        selected,
        key=lambda r: (
            r.condition,
            FAMILIES.index(r.family),
            CHECKPOINT_ORDER[r.family].index(r.checkpoint_tag),
            r.seed,
        ),
    )


def _read_wandb_scalars(run: Any, samples: int) -> dict[str, dict[str, list[float]]]:
    data: dict[str, dict[str, list[float]]] = {}
    for tag in EVENT_TAGS:
        steps: list[int] = []
        values: list[float] = []
        rows = run.history(samples=samples, keys=["global_step", tag], pandas=False, x_axis="global_step")
        for row in rows:
            if tag not in row or row[tag] is None:
                continue
            step = row.get("global_step", row.get("_step"))
            if step is None:
                continue
            steps.append(int(step))
            values.append(float(row[tag]))
        if steps:
            order = np.argsort(np.asarray(steps))
            data[tag] = {
                "steps": [steps[i] for i in order],
                "values": [values[i] for i in order],
            }
    return data


def _series_by_step(series: dict[str, list[float]]) -> dict[int, float]:
    return {int(step): float(value) for step, value in zip(series["steps"], series["values"], strict=True)}


def _safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den <= 1e-6:
        return float("nan")
    return min(max(num / den, 0.0), 1.0)


def _add_derived_metrics(raw: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
    derived = dict(raw)
    if RAW_SUCCESS not in raw or DONE_FALL not in raw:
        return derived
    success = _series_by_step(raw[RAW_SUCCESS])
    done_fall = _series_by_step(raw[DONE_FALL])
    common_steps = sorted(set(success) & set(done_fall))
    derived[METRIC] = {
        "steps": common_steps,
        "values": [_safe_ratio(success[s], 1.0 - done_fall[s]) for s in common_steps],
    }
    return derived


def _summarize(metrics: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for tag, series in metrics.items():
        steps = series.get("steps", [])
        values = series.get("values", [])
        if not steps or not values:
            continue
        summary[tag] = {
            "num_points": len(values),
            "first_step": steps[0],
            "last_step": steps[-1],
            "first": values[0],
            "last": values[-1],
            "max": max(values),
        }
    return summary


def collect(samples: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "metadata": {
            "wandb_entity": WANDB_ENTITY,
            "wandb_project": WANDB_PROJECT,
            "wandb_group": WANDB_GROUP,
            "samples": samples,
            "metric": METRIC,
        },
        "conditions": {
            "wrench": {},
            "no_wrench": {},
        },
    }
    for condition in out["conditions"]:
        for family in FAMILIES:
            out["conditions"][condition][family] = {
                "checkpoint_order": CHECKPOINT_ORDER[family],
                "runs": {},
            }

    for run in _dedupe_latest(_find_wandb_runs()):
        raw = _read_wandb_scalars(run.wandb_run, samples=samples)
        metrics = _add_derived_metrics(raw)
        bucket = out["conditions"][run.condition][run.family]["runs"].setdefault(run.checkpoint_tag, {})
        bucket[str(run.seed)] = {
            "seed": run.seed,
            "run_name": run.run_name,
            "wandb_path": run.wandb_path,
            "wandb_state": run.wandb_state,
            "timestamp": run.timestamp,
            "metrics": metrics,
            "summary": _summarize(metrics),
        }
    return out


def _print_summary(data: dict[str, Any]) -> None:
    print("| Condition | Family | Checkpoint | Seed | State | Final Norm Success | Env Steps |")
    print("|---|---|---|---:|---|---:|---:|")
    for condition, condition_data in data["conditions"].items():
        for family, family_data in condition_data.items():
            for checkpoint, seed_bucket in family_data.get("runs", {}).items():
                for seed, seed_data in sorted(seed_bucket.items(), key=lambda kv: int(kv[0])):
                    summary = seed_data.get("summary", {}).get(METRIC, {})
                    final = summary.get("last", float("nan"))
                    steps = summary.get("last_step", float("nan"))
                    print(
                        f"| {condition} | {family} | {checkpoint} | {seed} | "
                        f"{seed_data.get('wandb_state')} | {100.0 * final:.1f}% | {steps / 1e9:.2f}B |"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "lpeg_traj_precision_wrench_compare_latest.json")
    args = parser.parse_args()

    data = collect(samples=args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {args.out}")
    _print_summary(data)


if __name__ == "__main__":
    main()

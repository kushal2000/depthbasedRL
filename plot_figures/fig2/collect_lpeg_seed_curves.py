"""Collect L-peg Fig. 2 seed curves from TensorBoard event files.

This intentionally does not modify or depend on the existing static Fig. 2
JSONs under ``outputs/fig2_panel_bcd``.  The current Fig. 2 scripts consume
those JSONs directly; this script creates separate Tyler seed-aggregation JSONs
from local training runs.

Example:
    .venv-isaacsim-py311/bin/python plot_figures/fig2/collect_lpeg_seed_curves.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = Path("/move/u/tylerlum/github_repos/depthbasedRL/train_dir/fig2")
DEFAULT_OUT_DIR = REPO / "outputs" / "fig2_tyler_seed_curves"
DEFAULT_WANDB_ENTITY = "tylerlum"
DEFAULT_WANDB_PROJECT = "fig2"
WANDB_GROUPS = {
    "ObjectDiversity": "panel_a_teachers_tyler_object_diversity",
    "TrainingObjective": "panel_a_teachers_tyler_training_objective",
}

FAMILIES = ("ObjectDiversity", "TrainingObjective")
CHECKPOINT_ORDER: dict[str, list[str]] = {
    "ObjectDiversity": ["1000_obj", "100_obj", "10_obj", "1_obj"],
    "TrainingObjective": ["Play2Win", "RotationOnly", "SingleGoal", "TranslationOnly"],
}

EVENT_TAGS = [
    "episode_final/all_goals_hit",
    "episode_final/success_ratio",
    "episode_final/successes",
    "episode_final/done_fall",
    "episode_lengths/step",
]

RUN_RE = re.compile(
    r"^(?P<task>lpeg_tol0p5mm_finetune_rgf0_dr)"
    r"(?:_seed(?P<seed>\d+))?"
    r"_(?P<family>ObjectDiversity|TrainingObjective)"
    r"_(?P<tag>.+?)"
    r"_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"(?:_\d{8}_\d{6})?$"
)


@dataclass(frozen=True)
class RunInfo:
    family: str
    checkpoint_tag: str
    seed: int
    timestamp: str
    run_name: str
    run_dir: str | None = None
    event_file: Path | None = None
    wandb_path: str | None = None
    wandb_state: str | None = None
    wandb_run: Any | None = None


def _event_file_loader_cls():
    try:
        from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorBoard is required to read event files. Run with the Isaac Sim env, e.g.:\n"
            "  .venv-isaacsim-py311/bin/python plot_figures/fig2/collect_lpeg_seed_curves.py"
        ) from exc
    return EventFileLoader


def _scalar_value(summary_value: Any) -> float | None:
    if summary_value.HasField("simple_value"):
        return float(summary_value.simple_value)

    # Newer TensorBoard scalar summaries may be stored as TensorProto scalars.
    tensor = summary_value.tensor
    if tensor is None:
        return None
    if tensor.float_val:
        return float(tensor.float_val[0])
    if tensor.double_val:
        return float(tensor.double_val[0])
    if tensor.int_val:
        return float(tensor.int_val[0])
    if tensor.int64_val:
        return float(tensor.int64_val[0])
    if tensor.tensor_content:
        try:
            from tensorboard.util import tensor_util

            arr = tensor_util.make_ndarray(tensor)
            if arr.size:
                return float(arr.reshape(-1)[0])
        except Exception:
            return None
    return None


def _parse_name(
    name: str,
    *,
    run_dir: str | None = None,
    event_file: Path | None = None,
    wandb_path: str | None = None,
    wandb_state: str | None = None,
    wandb_run: Any | None = None,
) -> RunInfo | None:
    match = RUN_RE.match(name)
    if match is None:
        return None

    family = match.group("family")
    checkpoint_tag = match.group("tag")
    if checkpoint_tag not in CHECKPOINT_ORDER[family]:
        return None

    seed = int(match.group("seed") or 0)
    return RunInfo(
        family=family,
        checkpoint_tag=checkpoint_tag,
        seed=seed,
        timestamp=match.group("timestamp"),
        run_name=name,
        run_dir=run_dir,
        event_file=event_file,
        wandb_path=wandb_path,
        wandb_state=wandb_state,
        wandb_run=wandb_run,
    )


def _parse_tensorboard_run(event_file: Path) -> RunInfo | None:
    # events are under <run_dir>/0_<run_name>/summaries/events.out.tfevents.*
    if len(event_file.parents) < 3:
        return None
    run_dir = event_file.parents[2]
    return _parse_name(str(run_dir.name), run_dir=str(run_dir), event_file=event_file)


def _find_runs(
    run_root: Path,
    families: set[str] | None = None,
    checkpoint_tags: set[str] | None = None,
    seeds: set[int] | None = None,
) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for event_file in sorted(run_root.glob("**/events.out.tfevents*")):
        info = _parse_tensorboard_run(event_file)
        if info is None:
            continue
        if families is not None and info.family not in families:
            continue
        if checkpoint_tags is not None and info.checkpoint_tag not in checkpoint_tags:
            continue
        if seeds is not None and info.seed not in seeds:
            continue
        runs.append(info)
    return runs


def _find_wandb_runs(
    *,
    entity: str,
    project: str,
    families: set[str],
    checkpoint_tags: set[str] | None,
    seeds: set[int] | None,
) -> list[RunInfo]:
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "wandb is required for --source wandb. Run with the Isaac Sim env, e.g.:\n"
            "  .venv-isaacsim-py311/bin/python plot_figures/fig2/collect_lpeg_seed_curves.py\n"
            "or use --source tensorboard."
        ) from exc

    api = wandb.Api(timeout=30)
    runs: list[RunInfo] = []
    for family in sorted(families):
        group = WANDB_GROUPS[family]
        filters = {
            "group": group,
            "display_name": {"$regex": r"^lpeg_tol0p5mm_finetune_rgf0_dr"},
        }
        for run in api.runs(f"{entity}/{project}", filters=filters, per_page=100):
            info = _parse_name(
                run.name,
                wandb_path=f"{entity}/{project}/{run.id}",
                wandb_state=run.state,
                wandb_run=run,
            )
            if info is None:
                continue
            if checkpoint_tags is not None and info.checkpoint_tag not in checkpoint_tags:
                continue
            if seeds is not None and info.seed not in seeds:
                continue
            runs.append(info)
    return runs


def _dedupe_latest(runs: list[RunInfo]) -> tuple[list[RunInfo], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, int], list[RunInfo]] = {}
    for run in runs:
        by_key.setdefault((run.family, run.checkpoint_tag, run.seed), []).append(run)

    selected: list[RunInfo] = []
    duplicates: list[dict[str, Any]] = []
    for key, vals in sorted(by_key.items()):
        vals = sorted(vals, key=lambda r: (r.timestamp, str(r.run_dir)))
        selected.append(vals[-1])
        if len(vals) > 1:
            duplicates.append(
                {
                    "key": {"family": key[0], "checkpoint_tag": key[1], "seed": key[2]},
                    "selected": vals[-1].run_dir or vals[-1].wandb_path or vals[-1].run_name,
                    "ignored": [v.run_dir or v.wandb_path or v.run_name for v in vals[:-1]],
                }
            )
    return selected, duplicates


def _read_scalars(event_file: Path, tags: list[str]) -> dict[str, dict[str, list[float]]]:
    EventFileLoader = _event_file_loader_cls()
    tag_set = set(tags)
    data: dict[str, dict[str, list[float]]] = {
        tag: {"steps": [], "values": [], "wall_times": []} for tag in tags
    }

    for event in EventFileLoader(str(event_file)).Load():
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
            # Wall-time is intentionally not stored; env frames are the fair
            # x-axis across heterogeneous GPUs.

    data = {tag: series for tag, series in data.items() if series["steps"]}
    return data


def _read_wandb_scalars(wandb_run: Any, tags: list[str], samples: int) -> dict[str, dict[str, list[float]]]:
    data: dict[str, dict[str, list[float]]] = {}
    for tag in tags:
        steps: list[int] = []
        values: list[float] = []
        wall_times: list[float] = []
        # W&B's sampled history is much faster than scan_history for these long
        # runs. Query one scalar at a time; requesting all scalar keys together
        # can return an empty table when the backend cannot find rows with all
        # keys materialized.
        rows = wandb_run.history(
            samples=samples,
            keys=["global_step", tag],
            pandas=False,
            x_axis="global_step",
        )
        for row in rows:
            if tag not in row or row[tag] is None:
                continue
            step = row.get("global_step", row.get("_step"))
            if step is None:
                continue
            steps.append(int(step))
            values.append(float(row[tag]))
            # Wall-time is intentionally not stored; env frames are the fair
            # x-axis across heterogeneous GPUs.
        if steps:
            data[tag] = {"steps": steps, "values": values, "wall_times": wall_times}
    return data


def _series_by_step(series: dict[str, list[float]]) -> dict[int, float]:
    return {int(step): float(value) for step, value in zip(series["steps"], series["values"], strict=True)}


def _safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den <= 1e-6:
        return float("nan")
    return min(max(num / den, 0.0), 1.0)


def _add_derived_metrics(raw: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
    derived = dict(raw)
    success_tag = "episode_final/all_goals_hit"
    done_fall_tag = "episode_final/done_fall"
    if success_tag not in raw or done_fall_tag not in raw:
        return derived

    success = _series_by_step(raw[success_tag])
    done_fall = _series_by_step(raw[done_fall_tag])
    common_steps = sorted(set(success) & set(done_fall))
    feasible_values = [_safe_ratio(success[s], 1.0 - done_fall[s]) for s in common_steps]
    derived["episode_final/feasible_normalized_all_goals_hit"] = {
        "steps": common_steps,
        "values": feasible_values,
        "wall_times": [],
    }
    return derived


def _summarize_run(metrics: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for tag, series in metrics.items():
        values = series.get("values", [])
        steps = series.get("steps", [])
        if values and steps:
            summary[tag] = {
                "num_points": len(values),
                "first_step": steps[0],
                "last_step": steps[-1],
                "first": values[0],
                "last": values[-1],
                "max": max(values),
            }
    return summary


def _collect_family(runs: list[RunInfo], family: str, wandb_samples: int) -> dict[str, Any]:
    family_runs = [run for run in runs if run.family == family]
    out: dict[str, Any] = {
        "family": family,
        "checkpoint_order": CHECKPOINT_ORDER[family],
        "metric_default": "episode_final/feasible_normalized_all_goals_hit",
        "raw_success_metric": "episode_final/all_goals_hit",
        "runs": {},
    }

    for run in sorted(family_runs, key=lambda r: (CHECKPOINT_ORDER[family].index(r.checkpoint_tag), r.seed)):
        if run.wandb_run is not None:
            raw = _read_wandb_scalars(run.wandb_run, EVENT_TAGS, samples=wandb_samples)
        else:
            assert run.event_file is not None
            raw = _read_scalars(run.event_file, EVENT_TAGS)
        metrics = _add_derived_metrics(raw)
        checkpoint_bucket = out["runs"].setdefault(run.checkpoint_tag, {})
        checkpoint_bucket[str(run.seed)] = {
            "seed": run.seed,
            "run_name": run.run_name,
            "run_dir": run.run_dir,
            "event_file": str(run.event_file) if run.event_file is not None else None,
            "wandb_path": run.wandb_path,
            "wandb_state": run.wandb_state,
            "timestamp": run.timestamp,
            "metrics": metrics,
            "summary": _summarize_run(metrics),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["wandb", "tensorboard"], default="wandb")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument(
        "--wandb-samples",
        type=int,
        default=2000,
        help="Sample count per metric when --source=wandb. Higher is smoother but slower.",
    )
    parser.add_argument("--families", nargs="*", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument(
        "--checkpoint-tags",
        nargs="*",
        default=None,
        help="Optional checkpoint tag filter, e.g. Play2Win 1000_obj.",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="Keep duplicate run dirs for the same family/checkpoint/seed. Default keeps the latest.",
    )
    args = parser.parse_args()

    families = set(args.families)
    checkpoint_tags = set(args.checkpoint_tags) if args.checkpoint_tags else None
    seeds = set(args.seeds) if args.seeds else None
    if args.source == "wandb":
        runs = _find_wandb_runs(
            entity=args.wandb_entity,
            project=args.wandb_project,
            families=families,
            checkpoint_tags=checkpoint_tags,
            seeds=seeds,
        )
    else:
        runs = _find_runs(args.run_root, families=families, checkpoint_tags=checkpoint_tags, seeds=seeds)
    duplicates: list[dict[str, Any]] = []
    if not args.include_duplicates:
        runs, duplicates = _dedupe_latest(runs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": args.source,
        "run_root": str(args.run_root),
        "wandb_entity": args.wandb_entity,
        "wandb_project": args.wandb_project,
        "num_runs": len(runs),
        "duplicates": duplicates,
        "notes": [
            "Existing Fig. 2 JSONs were committed as source-of-truth data; this file is regenerated from TensorBoard.",
            "Default metric matches old Fig. 2 feasible-normalized success: all_goals_hit / (1 - done_fall).",
        ],
    }

    for family in FAMILIES:
        if family not in families:
            continue
        family_data = _collect_family(runs, family, wandb_samples=args.wandb_samples)
        family_data["metadata"] = metadata
        out_path = args.out_dir / f"lpeg_{family.lower()}_seed_curves.json"
        with out_path.open("w") as f:
            json.dump(family_data, f, separators=(",", ":"))
        print(f"wrote {out_path}")

    manifest = {
        "metadata": metadata,
        "families": {
            family: str(args.out_dir / f"lpeg_{family.lower()}_seed_curves.json")
            for family in FAMILIES
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

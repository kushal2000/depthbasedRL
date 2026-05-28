"""Collect and plot L-peg wrench-disturbance ablation curves from W&B.

This intentionally writes separate wrench-specific JSON and figure files so the
existing non-wrench L-peg plots remain reproducible and untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
from _style import STYLES, configure_rcparams, style_axis  # noqa: E402


WANDB_ENTITY = "tylerlum"
WANDB_PROJECT = "fig2"
WANDB_GROUP = "panel_a_teachers_tyler_wrench_lpeg"
DATA_OUT_DIR = REPO / "outputs" / "fig2_tyler_wrench_seed_curves"
PLOT_OUT_DIR = REPO / "plot_figures" / "fig2" / "outputs"

METRIC = "episode_final/feasible_normalized_all_goals_hit"
RAW_SUCCESS = "episode_final/all_goals_hit"
DONE_FALL = "episode_final/done_fall"
EVENT_TAGS = [RAW_SUCCESS, DONE_FALL]

FAMILIES = ("TrainingObjective", "ObjectDiversity")
CHECKPOINT_ORDER = {
    "TrainingObjective": ["Play2Win", "RotationOnly", "SingleGoal", "TranslationOnly"],
    "ObjectDiversity": ["1000_obj", "100_obj", "10_obj", "1_obj"],
}

RUN_RE = re.compile(
    r"^lpeg_tol0p5mm_finetune_rgf0_dr_wrench"
    r"_seed(?P<seed>\d+)"
    r"_(?P<family>ObjectDiversity|TrainingObjective)"
    r"_(?P<tag>.+?)"
    r"_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"(?:_\d{8}_\d{6})?$"
)

COLORS = {
    "Play2Win": "#2C7BB6",
    "1000_obj": "#2C7BB6",
    "RotationOnly": STYLES["RotationOnly"][0],
    "100_obj": STYLES["RotationOnly"][0],
    "SingleGoal": STYLES["SingleGoal"][0],
    "10_obj": STYLES["SingleGoal"][0],
    "TranslationOnly": "#E08214",
    "1_obj": "#E08214",
}
LABELS_OBJECT = {
    "1000_obj": "1000 Objects",
    "100_obj": "100 Objects",
    "10_obj": "10 Objects",
    "1_obj": "1 Object",
}
LABELS_HYBRID = {
    "Play2Win": "1000 Objects (Ours)",
    "100_obj": "100 Objects",
    "10_obj": "10 Objects",
    "1_obj": "1 Object",
}
LABELS_TRAINING = {
    "Play2Win": "Full Pose (Ours)",
    "RotationOnly": "Rotation Only",
    "SingleGoal": "Single Goal",
    "TranslationOnly": "Translation Only",
}
DUMMY_STANDALONE_SPECS = {
    "DummySuccessTolerance": (
        ("Success Threshold = 1 cm (Ours)", "#2C7BB6", "play2win"),
        ("2.5 cm", STYLES["RotationOnly"][0], "zero"),
        ("5 cm", STYLES["SingleGoal"][0], "zero"),
        ("10 cm", "#E08214", "zero"),
    ),
    "DummyNumTrajectories": (
        ("Random Trajectories (Ours)", "#2C7BB6", "play2win"),
        ("100 Trajectories", STYLES["RotationOnly"][0], "zero"),
        ("10 Trajectories", STYLES["SingleGoal"][0], "zero"),
        ("1 Trajectory", "#E08214", "zero"),
    ),
}


@dataclass(frozen=True)
class RunInfo:
    family: str
    checkpoint_tag: str
    seed: int
    timestamp: str
    run_name: str
    wandb_path: str
    wandb_state: str
    wandb_run: Any


@dataclass
class Curve:
    seed: int
    x: np.ndarray
    y: np.ndarray
    run_name: str


def _parse_run(run: Any) -> RunInfo | None:
    match = RUN_RE.match(run.name)
    if match is None:
        return None
    family = match.group("family")
    tag = match.group("tag")
    if tag not in CHECKPOINT_ORDER[family]:
        return None
    return RunInfo(
        family=family,
        checkpoint_tag=tag,
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
        raise SystemExit("wandb is required to collect wrench plots.") from exc

    api = wandb.Api(timeout=45)
    runs = []
    filters = {
        "group": WANDB_GROUP,
        "display_name": {"$regex": r"^lpeg_tol0p5mm_finetune_rgf0_dr_wrench"},
    }
    for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters, per_page=100):
        info = _parse_run(run)
        if info is not None:
            runs.append(info)
    return runs


def _dedupe_latest(runs: list[RunInfo]) -> list[RunInfo]:
    by_key: dict[tuple[str, str, int], list[RunInfo]] = {}
    for run in runs:
        by_key.setdefault((run.family, run.checkpoint_tag, run.seed), []).append(run)
    selected = []
    for vals in by_key.values():
        selected.append(sorted(vals, key=lambda r: r.timestamp)[-1])
    return sorted(
        selected,
        key=lambda r: (
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


def _collect(samples: int) -> dict[str, Any]:
    runs = _dedupe_latest(_find_wandb_runs())
    out = {
        "metadata": {
            "wandb_entity": WANDB_ENTITY,
            "wandb_project": WANDB_PROJECT,
            "wandb_group": WANDB_GROUP,
            "samples": samples,
            "metric": METRIC,
        },
        "families": {},
    }
    for family in FAMILIES:
        out["families"][family] = {
            "checkpoint_order": CHECKPOINT_ORDER[family],
            "runs": {},
        }
    for run in runs:
        raw = _read_wandb_scalars(run.wandb_run, samples=samples)
        metrics = _add_derived_metrics(raw)
        bucket = out["families"][run.family]["runs"].setdefault(run.checkpoint_tag, {})
        bucket[str(run.seed)] = {
            "seed": run.seed,
            "run_name": run.run_name,
            "wandb_path": run.wandb_path,
            "wandb_state": run.wandb_state,
            "timestamp": run.timestamp,
            "metrics": metrics,
        }
    return out


def _curve_from_seed(seed_data: dict[str, Any]) -> Curve | None:
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
    x = x - x[0]
    return Curve(seed=int(seed_data["seed"]), x=x, y=y, run_name=seed_data["run_name"])


def _aggregate(curves: list[Curve], num_points: int = 300) -> dict[str, np.ndarray] | None:
    if not curves:
        return None
    start = min(float(c.x[0]) for c in curves)
    end = max(float(c.x[-1]) for c in curves)
    if end <= start:
        return None
    grid = np.linspace(start, end, num_points)
    ys = []
    for curve in curves:
        interp = np.interp(grid, curve.x, curve.y)
        interp[(grid < curve.x[0]) | (grid > curve.x[-1])] = np.nan
        ys.append(interp)
    stack = np.vstack(ys)
    keep = np.sum(np.isfinite(stack), axis=0) >= len(curves)
    if not np.any(keep):
        return None
    return {
        "x": grid[keep],
        "mean": np.nanmean(stack[:, keep], axis=0),
        "std": np.nanstd(stack[:, keep], axis=0),
    }


def _format_env_axis(ax: plt.Axes, x_max_billions: float) -> None:
    ax.set_xlabel("Finetuning Env Steps")
    ax.set_xlim(0.0, x_max_billions)
    ax.set_xticks(np.arange(0.0, x_max_billions + 0.5, 1.0))
    ax.set_xticklabels(["0"] + [f"{int(v)}B" for v in np.arange(1.0, x_max_billions + 0.5, 1.0)])


def _plot_grouped(
    grouped: list[tuple[str, str, str, list[Curve]]],
    output: Path,
    *,
    x_max_billions: float,
    title: str | None,
) -> None:
    configure_rcparams()
    label_size = 14
    tick_size = 11
    title_size = 15
    fig_height = 3.28 if title else 3.0
    fig, ax = plt.subplots(figsize=(6.4, fig_height), dpi=220)

    for _key, label, color, curves in grouped:
        agg = _aggregate(curves)
        if agg is None:
            continue
        x = agg["x"] / 1e9
        mean = 100.0 * agg["mean"]
        std = 100.0 * agg["std"]
        visible = x <= x_max_billions
        if not np.any(visible):
            continue
        ax.plot(x[visible], mean[visible], color=color, linewidth=1.8, label=label)
        ax.fill_between(
            x[visible],
            mean[visible] - std[visible],
            mean[visible] + std[visible],
            color=color,
            alpha=0.17,
            linewidth=0,
        )

    _format_env_axis(ax, x_max_billions=x_max_billions)
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-2, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    if title:
        ax.set_title(title, fontsize=title_size, pad=6)
    style_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045 if not title else 0.040),
        ncol=2,
        fontsize=label_size,
        columnspacing=1.5,
        handlelength=1.6,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.15, right=0.985, top=0.82 if title else 0.91, bottom=0.465 if not title else 0.43)

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = output.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.025, facecolor="white", edgecolor="none")
        print(f"wrote {path}")
    plt.close(fig)


def _group_object_diversity(data: dict[str, Any]) -> list[tuple[str, str, str, list[Curve]]]:
    family = data["families"]["ObjectDiversity"]
    grouped = []
    for checkpoint in CHECKPOINT_ORDER["ObjectDiversity"]:
        curves = []
        for _, seed_data in sorted(family["runs"].get(checkpoint, {}).items(), key=lambda kv: int(kv[0])):
            curve = _curve_from_seed(seed_data)
            if curve is not None:
                curves.append(curve)
        if curves:
            grouped.append((checkpoint, LABELS_OBJECT[checkpoint], COLORS[checkpoint], curves))
    return grouped


def _group_hybrid(data: dict[str, Any]) -> list[tuple[str, str, str, list[Curve]]]:
    grouped = []
    objective = data["families"]["TrainingObjective"]
    objects = data["families"]["ObjectDiversity"]
    specs = [
        ("Play2Win", LABELS_HYBRID["Play2Win"], COLORS["Play2Win"], objective["runs"].get("Play2Win", {})),
        ("100_obj", LABELS_HYBRID["100_obj"], COLORS["100_obj"], objects["runs"].get("100_obj", {})),
        ("10_obj", LABELS_HYBRID["10_obj"], COLORS["10_obj"], objects["runs"].get("10_obj", {})),
        ("1_obj", LABELS_HYBRID["1_obj"], COLORS["1_obj"], objects["runs"].get("1_obj", {})),
    ]
    for checkpoint, label, color, seed_bucket in specs:
        curves = []
        for _, seed_data in sorted(seed_bucket.items(), key=lambda kv: int(kv[0])):
            curve = _curve_from_seed(seed_data)
            if curve is not None:
                curves.append(curve)
        if curves:
            grouped.append((checkpoint, label, color, curves))
    return grouped


def _group_training_objective(data: dict[str, Any]) -> list[tuple[str, str, str, list[Curve]]]:
    family = data["families"]["TrainingObjective"]
    grouped = []
    for checkpoint in CHECKPOINT_ORDER["TrainingObjective"]:
        curves = []
        for _, seed_data in sorted(family["runs"].get(checkpoint, {}).items(), key=lambda kv: int(kv[0])):
            curve = _curve_from_seed(seed_data)
            if curve is not None:
                curves.append(curve)
        if curves:
            grouped.append((checkpoint, LABELS_TRAINING[checkpoint], COLORS[checkpoint], curves))
    return grouped


def _zero_curves_like(label: str, curves: list[Curve]) -> list[Curve]:
    return [
        Curve(
            seed=curve.seed,
            x=curve.x.copy(),
            y=np.zeros_like(curve.y),
            run_name=f"dummy_zero/{label}/seed_{curve.seed}",
        )
        for curve in curves
    ]


def _group_dummy_from_play2win(data: dict[str, Any], family: str) -> list[tuple[str, str, str, list[Curve]]]:
    play2win_bucket = data["families"]["TrainingObjective"]["runs"].get("Play2Win", {})
    play2win_curves = []
    for _, seed_data in sorted(play2win_bucket.items(), key=lambda kv: int(kv[0])):
        curve = _curve_from_seed(seed_data)
        if curve is not None:
            play2win_curves.append(curve)
    if not play2win_curves:
        raise RuntimeError("Could not find wrench Play2Win curves for dummy panel.")

    grouped = []
    for idx, (label, color, source) in enumerate(DUMMY_STANDALONE_SPECS[family]):
        if source == "play2win":
            curves = play2win_curves
        elif source == "zero":
            curves = _zero_curves_like(label, play2win_curves)
        else:
            raise ValueError(f"Unexpected dummy source: {source}")
        grouped.append((f"{family}_{idx}", label, color, curves))
    return grouped


def _latest_row(seed_data: dict[str, Any]) -> tuple[float, float, str]:
    series = seed_data.get("metrics", {}).get(METRIC)
    if not series or not series.get("values"):
        return float("nan"), float("nan"), "unknown"
    steps = series["steps"]
    values = series["values"]
    env_steps = (float(steps[-1]) - float(steps[0])) / 1e9
    running = "yes" if seed_data.get("wandb_state", "").lower() == "running" else "no"
    return 100.0 * float(values[-1]), env_steps, running


def _print_tables(data: dict[str, Any]) -> None:
    for family in FAMILIES:
        fam = data["families"][family]
        rows = []
        for checkpoint in CHECKPOINT_ORDER[family]:
            for seed, seed_data in sorted(fam["runs"].get(checkpoint, {}).items(), key=lambda kv: int(kv[0])):
                success, env_steps, running = _latest_row(seed_data)
                rows.append((checkpoint, int(seed), success, env_steps, running))
        seeds = sorted({row[1] for row in rows})
        print(f"\n{family}, {len(CHECKPOINT_ORDER[family])} x {len(seeds)}")
        print("| Checkpoint | Seed | Norm Success | Env Steps | Still Training |")
        print("|---|---:|---:|---:|---|")
        for checkpoint, seed, success, env_steps, running in rows:
            print(f"| {checkpoint} | {seed} | {success:.1f}% | {env_steps:.2f}B | {running} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-samples", type=int, default=5000)
    parser.add_argument("--x-max-billions", type=float, default=4.0)
    parser.add_argument("--slug", default="4B")
    args = parser.parse_args()

    data = _collect(samples=args.wandb_samples)
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_OUT_DIR / "lpeg_wrench_seed_curves.json"
    with data_path.open("w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {data_path}")

    _print_tables(data)

    _plot_grouped(
        _group_object_diversity(data),
        PLOT_OUT_DIR / f"lpeg_wrench_objectdiversity_1000obj_mean_std_clean_{args.slug}.png",
        x_max_billions=args.x_max_billions,
        title=None,
    )
    _plot_grouped(
        _group_hybrid(data),
        PLOT_OUT_DIR / f"lpeg_wrench_objectdiversity_ours_mean_std_clean_{args.slug}.png",
        x_max_billions=args.x_max_billions,
        title=None,
    )
    _plot_grouped(
        _group_hybrid(data),
        PLOT_OUT_DIR / f"lpeg_wrench_panel_a_object_diversity_titled_{args.slug}.png",
        x_max_billions=args.x_max_billions,
        title="(a) Object Diversity",
    )
    _plot_grouped(
        _group_training_objective(data),
        PLOT_OUT_DIR / f"lpeg_wrench_trainingobjective_seed0_mean_std_clean_{args.slug}.png",
        x_max_billions=args.x_max_billions,
        title=None,
    )
    _plot_grouped(
        _group_training_objective(data),
        PLOT_OUT_DIR / f"lpeg_wrench_panel_b_training_objective_titled_{args.slug}.png",
        x_max_billions=args.x_max_billions,
        title="(b) Training Objective",
    )
    _plot_grouped(
        _group_dummy_from_play2win(data, "DummyNumTrajectories"),
        PLOT_OUT_DIR / f"lpeg_wrench_panel_c_trajectory_diversity_titled_{args.slug}.png",
        x_max_billions=args.x_max_billions,
        title="(c) Trajectory Diversity",
    )
    _plot_grouped(
        _group_dummy_from_play2win(data, "DummySuccessTolerance"),
        PLOT_OUT_DIR / f"lpeg_wrench_panel_d_play_precision_titled_{args.slug}.png",
        x_max_billions=args.x_max_billions,
        title="(d) Play Precision",
    )


if __name__ == "__main__":
    main()

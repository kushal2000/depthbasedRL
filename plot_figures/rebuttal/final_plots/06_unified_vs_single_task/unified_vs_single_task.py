#!/usr/bin/env python3
"""Grouped success bars: task-specific teachers versus one unified teacher.

The plotted metric is the reportable early-drop-filtered insertion rate from the
offline-evaluation JSONs listed in ``unified_vs_single_task_data.json``.

Writes: unified_vs_single_task.{png,pdf}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "plot_figures"))
from _style import configure_rcparams  # noqa: E402

DATA = HERE / "unified_vs_single_task_data.json"
C_SINGLE = "#777777"
C_UNIFIED = "#2a78d6"
INK = "#333333"


def _rate(path: str) -> tuple[float, int, int]:
    source = (HERE / path).resolve()
    blob = json.loads(source.read_text())
    result = blob["results"][0]
    filtered = result["early_drop_filtered"]
    return (
        100.0 * float(filtered["insertion_rate"]),
        int(filtered["n"]),
        int(result["unfinished_envs"]),
    )


def main() -> None:
    configure_rcparams()
    cfg = json.loads(DATA.read_text())
    tasks = cfg["tasks"]
    single = [_rate(task["single_task"]) for task in tasks]
    unified = [_rate(task["unified"]) for task in tasks]

    x = 0.70 * np.arange(len(tasks), dtype=float)
    width = 0.27
    gap = 0.04
    fig, ax = plt.subplots(figsize=(4.33, 3.25))
    ax.set_axisbelow(True)

    for values, offset, color in (
        (single, -(width + gap) / 2, C_SINGLE),
        (unified, +(width + gap) / 2, C_UNIFIED),
    ):
        bars = ax.bar(
            x + offset,
            [v[0] for v in values],
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for bar, (value, _, _) in zip(bars, values):
            ax.annotate(
                f"{value:.0f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=INK,
            )

    ax.set_ylabel("Success Rate", fontsize=13, labelpad=2)
    ax.set_ylim(0, 108)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels([task["label"] for task in tasks], fontsize=10.5)
    ax.set_xlim(-0.42, x[-1] + 0.42)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9a9a")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="#9a9a9a", labelcolor=INK, length=3, width=0.8)

    fig.legend(
        handles=[
            Patch(facecolor=C_SINGLE, label="Task-Specific\nPolicies"),
            Patch(facecolor=C_UNIFIED, label="Single Unified\nPolicy"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
        fontsize=13,
        handlelength=1.1,
        handletextpad=0.45,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.18, right=0.99, top=0.84, bottom=0.14)

    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"unified_vs_single_task.{ext}", dpi=600,
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {HERE / 'unified_vs_single_task.png'}")
    for task, a, b in zip(tasks, single, unified):
        name = task["label"].replace("\n", " ")
        print(f"  {name:28s} task-specific={a[0]:6.2f}%  unified={b[0]:6.2f}%")


if __name__ == "__main__":
    main()

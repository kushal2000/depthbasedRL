#!/usr/bin/env python3
"""Grouped success bars for solo policies trained in matched fixture regimes.

Bars report early-drop-filtered insertion success. Bolted bars use the
completed legacy solo-policy evals. An unbolted result is
loaded from the newest file matching its configured glob; until one exists the
figure draws a labelled pending outline instead of mixing in zero-shot data.

Writes: bolted_vs_unbolted.{png,pdf}
"""

from __future__ import annotations

import glob
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

DATA = HERE / "bolted_vs_unbolted_data.json"
C_BOLTED = "#777777"
C_UNBOLTED = "#eb6834"
INK = "#333333"
MUTED = "#777777"


def _rate(path: Path) -> tuple[float, int, int]:
    blob = json.loads(path.read_text())
    result = blob["results"][0]
    filtered = result["early_drop_filtered"]
    return (
        100.0 * float(filtered["insertion_rate"]),
        int(filtered["n"]),
        int(result["unfinished_envs"]),
    )


def _latest(pattern: str) -> Path | None:
    absolute_pattern = str((HERE / pattern).resolve())
    matches = [Path(path) for path in glob.glob(absolute_pattern)]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def main() -> None:
    configure_rcparams()
    cfg = json.loads(DATA.read_text())
    tasks = cfg["tasks"]
    bolted = [_rate((HERE / task["bolted"]).resolve()) for task in tasks]
    unbolted_paths = [_latest(task["unbolted_glob"]) for task in tasks]
    unbolted = [_rate(path) if path else None for path in unbolted_paths]

    x = 0.70 * np.arange(len(tasks), dtype=float)
    width = 0.27
    gap = 0.04
    left = x - (width + gap) / 2
    right = x + (width + gap) / 2
    fig, ax = plt.subplots(figsize=(4.33, 3.25))
    ax.set_axisbelow(True)

    bars = ax.bar(left, [value[0] for value in bolted], width=width,
                  color=C_BOLTED, edgecolor="white", linewidth=0.7, zorder=3)
    for bar, (value, _, _) in zip(bars, bolted):
        ax.annotate(f"{value:.0f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, color=INK)

    for xpos, value in zip(right, unbolted):
        if value is None:
            ax.bar(xpos, 4, width=width, color="none", edgecolor=C_UNBOLTED,
                   linewidth=1.0, hatch="////", alpha=0.65, zorder=2)
            ax.text(xpos, 7, "pending", ha="center", va="bottom",
                    rotation=90, fontsize=8.5, color=MUTED, style="italic")
            continue
        bar = ax.bar(xpos, value[0], width=width, color=C_UNBOLTED,
                     edgecolor="white", linewidth=0.7, zorder=3)[0]
        ax.annotate(f"{value[0]:.0f}",
                    (bar.get_x() + bar.get_width() / 2, value[0]),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, color=INK)

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
            Patch(facecolor=C_BOLTED, label="Fixed Fixture\nTraining"),
            Patch(facecolor=C_UNBOLTED, label="Free Fixture\nTraining"),
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
        fig.savefig(HERE / f"bolted_vs_unbolted.{ext}", dpi=600,
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {HERE / 'bolted_vs_unbolted.png'}")
    for task, b, u, path in zip(tasks, bolted, unbolted, unbolted_paths):
        name = task["label"].replace("\n", " ")
        rhs = f"{u[0]:6.2f}% ({path.name})" if u and path else "pending"
        print(f"  {name:28s} bolted={b[0]:6.2f}%  unbolted={rhs}")


if __name__ == "__main__":
    main()

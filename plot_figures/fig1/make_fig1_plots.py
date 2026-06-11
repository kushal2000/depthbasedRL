"""Fig 1 right-column plots: success rate in Sim (line) + Real (bar).

Vertically stacked. fig2-matching style:
  * Patch-swatch legend with thin dark border, ours first semibold.
  * Same blue (#2C7BB6) for Play2Perfect, orange (#E08214) for Play-only.
  * Tolerance reported as diametral clearance (2 * per-side from the JSON).

Real-world success counts (operator-supplied, 2026-05-27):
    10  mm  :  Play2Perfect 10/10,  Play-only 6/10
     2  mm  :  Play2Perfect  9/10,  Play-only 2/10
     0.5 mm :  Play2Perfect  6/10,  Play-only 0/10
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUT = ROOT / "outputs"

COLORS = {
    "Play2Perfect": "#2C7BB6",
    "Play-only":    "#E08214",
}

# JSON method-key mapping; "play2win" is the historical key for Play2Perfect.
JSON_METHOD_KEY = {
    "Play2Perfect": "play2win",
    "Play-only":    "play_only",
}

# per-side -> diametral
TOLERANCE_PAPER_SCALE = 2.0

# Real-world rollout counts (successes, trials), keyed by diametral clearance.
REAL_RESULTS = {
    10.0: {"Play2Perfect": (10, 10), "Play-only": (6, 10)},
    2.0:  {"Play2Perfect": (9, 10),  "Play-only": (2, 10)},
    0.5:  {"Play2Perfect": (6, 10),  "Play-only": (0, 10)},
}


def load_sim() -> dict:
    raw = json.loads((INPUTS / "sim_results.json").read_text())
    tols_per_side = list(raw.get("tolerances_mm") or [])
    tols = [TOLERANCE_PAPER_SCALE * float(t) for t in tols_per_side]
    out: dict = {"tolerances_mm": tols}
    for label, key in JSON_METHOD_KEY.items():
        block = raw.get(key) or {}
        # Prefer the filtered series (drops init-fall artifact envs).
        src = block.get("filtered", block) if isinstance(block, dict) else {}
        sr = src.get("success_rate") or []
        kept_t, kept_s = [], []
        for t, s in zip(tols, sr):
            if s is None:
                continue
            kept_t.append(float(t))
            kept_s.append(float(s))
        out[label] = (np.asarray(kept_t), np.asarray(kept_s))
    return out


def style_axes(ax: plt.Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


X_TICKS = [40.0, 20.0, 10.0, 4.0, 2.0, 1.0, 0.4, 0.2]
X_TICK_LABELS = ["40", "20", "10", "4", "2", "1", "0.4", "0.2"]


def main() -> None:
    sim = load_sim()

    fig, (ax_sim, ax_real) = plt.subplots(
        2, 1, figsize=(2.9, 6.0),
        gridspec_kw={"hspace": 0.55},
    )
    lw_line = 1.5

    # === Top: Sim line plot ===
    for method in ("Play-only", "Play2Perfect"):
        t, s = sim[method]
        ax_sim.plot(
            t, 100.0 * s, color=COLORS[method], linewidth=lw_line,
            marker="o", markersize=3.8, markerfacecolor="white",
            markeredgecolor=COLORS[method], markeredgewidth=1.2,
        )
    ax_sim.set_xscale("log")
    ax_sim.set_xticks(X_TICKS)
    ax_sim.set_xticklabels(X_TICK_LABELS, fontsize=9)
    ax_sim.xaxis.set_minor_formatter(NullFormatter())
    ax_sim.set_xlim(50, 0.15)  # easier -> harder
    ax_sim.set_ylim(-2, 104)
    ax_sim.set_yticks([0, 25, 50, 75, 100])
    ax_sim.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=9)
    ax_sim.set_xlabel("Clearance (mm)", fontsize=10)
    ax_sim.set_ylabel("Success rate", fontsize=10)
    ax_sim.set_title("Success Rate (Sim)", fontsize=10)
    ax_sim.set_box_aspect(0.85)  # slightly wider than tall
    style_axes(ax_sim)

    # === Bottom: Real bar plot ===
    # Style cues from the CHORUS reference bar chart: thick darker outline in
    # the same color family as the fill, visible gap between paired bars,
    # no grid.
    tols = sorted(REAL_RESULTS.keys(), reverse=True)  # 10, 2, 0.5 — easier to harder
    x_pos = np.arange(len(tols))
    bar_w = 0.34
    pair_gap = 0.06  # gap between Play-only and Play2Perfect within a group
    methods = ("Play2Perfect", "Play-only")
    # Darker-shade edge colors (multiply fill by 0.40 in RGB).
    import matplotlib.colors as mcolors
    edge = {m: tuple(c * 0.40 for c in mcolors.to_rgb(COLORS[m])) for m in methods}
    for i, method in enumerate(methods):
        sr = [100.0 * REAL_RESULTS[t][method][0] / REAL_RESULTS[t][method][1] for t in tols]
        offset = (i - 0.5) * (bar_w + pair_gap)
        ax_real.bar(
            x_pos + offset, sr, bar_w,
            color=COLORS[method], edgecolor=edge[method], linewidth=2.0,
        )
    ax_real.set_xticks(x_pos)
    ax_real.set_xticklabels([f"{t:g}" for t in tols], fontsize=9)
    ax_real.set_xlabel("Clearance (mm)", fontsize=10)
    ax_real.set_ylim(0, 104)
    ax_real.set_yticks([0, 25, 50, 75, 100])
    ax_real.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=9)
    ax_real.set_ylabel("Success rate", fontsize=10)
    ax_real.set_title("Success Rate (Real)", fontsize=10)
    ax_real.set_box_aspect(0.85)  # slightly wider than tall
    style_axes(ax_real)

    # === Shared legend (fig2 style) ===
    legend_entries = [
        ("Play2Perfect", COLORS["Play2Perfect"], True),
        ("Play-only",    COLORS["Play-only"],    False),
    ]
    legend_handles = [
        Patch(facecolor=c, edgecolor="#333333", linewidth=0.7) for _, c, _ in legend_entries
    ]
    legend_labels = [lbl for lbl, _, _ in legend_entries]
    leg = fig.legend(
        legend_handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=2, frameon=False, fontsize=10,
        handlelength=1.2, handleheight=1.2, handletextpad=0.6, columnspacing=2.5,
    )
    for text, (_, _, emphasize) in zip(leg.get_texts(), legend_entries):
        if emphasize:
            text.set_fontweight(600)

    fig.subplots_adjust(left=0.26, right=0.96, top=0.90, bottom=0.06, hspace=0.45)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "fig1_plots.png"
    fig.savefig(out_path, dpi=240, facecolor="white")
    print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

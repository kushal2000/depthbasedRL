"""Combined panel (b+c) of fig 2 in panel-a layout.

Single 1x4 row matching make_panel_a.py's format:
    [0] Training-time success curves (was panel b)
    [1] Force-perturbation robustness sweep (was panel c)
    [2] Whitespace placeholder (image goes here)
    [3] Whitespace placeholder (image goes here)

Shared 3-entry patch-swatch legend below, ours first semibold, matching panel a.

Reads:
    outputs/fig2_panel_bcd/panel_b_curves.json
    outputs/fig2_panel_bcd/panel_c_{play2win,scratch_multistage}_fs{N}.json

Writes:
    plot_figures/fig2/outputs/panel_bc_draft.png
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "outputs" / "fig2_panel_bcd"
OUT = REPO / "plot_figures" / "fig2" / "outputs"

COLORS = {
    "Play2Win":      "#2C7BB6",
    "Scratch_dense": "#E08214",
    "Scratch_task":  "#8C8C8C",
}

LABEL_PB = {
    "Play2Win":           ("Play2Perfect (sparse reward)", COLORS["Play2Win"]),
    "Scratch_multistage": ("Scratch (dense reward)",       COLORS["Scratch_dense"]),
    "Scratch_task":       ("Scratch (sparse reward)",      COLORS["Scratch_task"]),
}
LABEL_PC = {
    "play2win":           ("Play2Perfect (sparse reward)", COLORS["Play2Win"]),
    "scratch_multistage": ("Scratch (dense reward)",       COLORS["Scratch_dense"]),
}


def load_panel_c(method):
    pts = []
    for p in sorted(
        DATA_DIR.glob(f"panel_c_{method}_fs*.json"),
        key=lambda q: int(re.search(r"fs(\d+)", q.stem).group(1)),
    ):
        d = json.load(open(p))
        fs = int(re.search(r"fs(\d+)", p.stem).group(1))
        sr = next(iter(d["results"].values()))["success_rate"] * 100
        pts.append((fs, sr))
    return pts


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    panel_b = json.load(open(DATA_DIR / "panel_b_curves.json"))

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2))
    lw = 1.5

    # === [0] Training-time curves (panel b) ===
    ax_b = axes[0]
    X_MAX_B = 140.0
    target_ratio = panel_b["Scratch_multistage"][-1][1]
    x_cross = next(t for t, s in panel_b["Play2Win"] if s >= target_ratio)
    speedup = panel_b["Scratch_multistage"][-1][0] / x_cross
    y_cross = target_ratio * 100

    for key, (_label, c) in LABEL_PB.items():
        pts = panel_b[key]
        xs = [p[0] for p in pts]
        ys = [100 * p[1] for p in pts]
        ax_b.plot(xs, ys, color=c, linewidth=lw)
        if key == "Play2Win" and xs[-1] < X_MAX_B:
            ax_b.plot([xs[-1], X_MAX_B], [ys[-1], ys[-1]], color=c, linewidth=lw)

    ax_b.plot([x_cross, x_cross], [-2, y_cross], color="#555555", linestyle="--", linewidth=0.8)
    ax_b.text(
        0.37, 0.13,
        f"{speedup:.0f}× faster\n({x_cross:.1f}h)",
        fontsize=7, color="#333333", ha="center", va="center", multialignment="center",
        transform=ax_b.transAxes,
    )
    ax_b.set_title("Training efficiency", fontsize=8)
    ax_b.set_xlim(0, X_MAX_B)
    ax_b.set_ylim(-2, 104)
    ax_b.set_xticks([0, 70, 140])
    ax_b.set_xticklabels([f"{v} h" for v in (0, 70, 140)], fontsize=7)
    ax_b.set_yticks([0, 25, 50, 75, 100])
    ax_b.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=7)
    ax_b.set_xlabel("Training time", fontsize=8.5)
    ax_b.set_ylabel("Success rate", fontsize=8.5)
    style_axes(ax_b)

    # === [1] Force-perturbation sweep (panel c) ===
    ax_c = axes[1]
    for key, (_label, c) in LABEL_PC.items():
        pts = load_panel_c(key)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax_c.plot(
            xs, ys, color=c, linewidth=1.5, marker="o", markersize=3.8,
            markerfacecolor="white", markeredgecolor=c, markeredgewidth=1.2,
        )
    ax_c.set_title("Force robustness", fontsize=8)
    ax_c.set_xlim(-3, 53)
    ax_c.set_ylim(-2, 104)
    ax_c.set_xticks([0, 25, 50])
    ax_c.set_xticklabels([f"{v} N" for v in (0, 25, 50)], fontsize=7)
    ax_c.set_yticks([0, 25, 50, 75, 100])
    ax_c.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=7)
    ax_c.set_xlabel("Force perturbation", fontsize=8.5)
    ax_c.set_ylabel("Success rate", fontsize=8.5)
    style_axes(ax_c)

    # === [2], [3] Whitespace placeholders for images ===
    for ax in axes[2:]:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # === Shared legend (same style as panel a) ===
    legend_entries = [
        ("Play2Perfect (sparse reward)",  COLORS["Play2Win"],      True),
        ("Scratch (dense reward)",        COLORS["Scratch_dense"], False),
        ("Scratch (sparse reward)",       COLORS["Scratch_task"],  False),
    ]
    legend_handles = [
        Patch(facecolor=c, edgecolor="#333333", linewidth=0.7) for _, c, _ in legend_entries
    ]
    legend_labels = [lbl for lbl, _, _ in legend_entries]
    leg = fig.legend(
        legend_handles, legend_labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.04),
        ncol=3, frameon=False, fontsize=8,
        handlelength=1.1, handleheight=1.1, handletextpad=0.6, columnspacing=2.2,
    )
    for text, (_, _, emphasize) in zip(leg.get_texts(), legend_entries):
        if emphasize:
            text.set_fontweight(600)

    plt.tight_layout(w_pad=0.3, rect=[0, 0.10, 1, 1])
    # No bbox_inches="tight" — that trims the empty right slots reserved for images.
    plt.savefig(OUT / "panel_bc_draft.png", dpi=240, facecolor="white")
    plt.close()
    print(f"wrote {OUT / 'panel_bc_draft.png'}")


if __name__ == "__main__":
    main()

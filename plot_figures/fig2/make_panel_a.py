"""Render panel (a) of fig 2: 1x4 row of training-curve mini-plots.

Tasks (left to right):
    Peg-In-Hole, Asm-Pillar, Asm-Beam, Screw-Leg

Each subplot shows:
    Play2Perfect (sparse reward) (blue, solid)
    Scratch (dense reward) (orange, solid at 0%)
    Scratch (sparse reward) (red, solid at 2% with a small offset so both show in legend)

A task-config inset is embedded in the upper-right of each subplot, read from
plot_figures/fig2/inputs/task_inset_{...}.png.

The Play2Perfect curves are normalized by (1 - terminal done_fall) — i.e. success
conditional on feasible initialization. A shared 3-entry legend sits below the
row.

Reads:
    outputs/fig2_panel_bcd/panel_a_curves.json
    plot_figures/fig2/inputs/task_inset_{peg_in_hole,asm_pillar,asm_beam,screw_leg}.png

Writes:
    plot_figures/fig2/outputs/panel_a_draft.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "outputs" / "fig2_panel_bcd"
INPUTS_DIR = REPO / "plot_figures" / "fig2" / "inputs"
OUT = REPO / "plot_figures" / "fig2" / "outputs"

COLORS = {
    "Play2Win": "#2C7BB6",
    "Scratch_dense": "#E08214",
    "Scratch_task": "#8C8C8C",
}
# (JSON key, inset image relative to inputs/, display title)
TASKS = [
    ("Peg-In-Hole", "new_inputs_assembly/tight_insertion.png", "Tight-Insertion"),
    ("Asm-Pillar",  "new_inputs_assembly/assemble_beam_1.png", "Assemble-Beam (Step 1)"),
    ("Asm-Beam",    "new_inputs_assembly/assemble_beam_2.png", "Assemble-Beam (Step 2)"),
    ("Screw-Leg",   "new_inputs_assembly/screw_leg.png",       "Screw-Leg"),
]


def main():
    panel_a = json.load(open(DATA_DIR / "panel_a_curves.json"))

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2))
    xticks = [0, 6, 12, 18, 24]
    yticks = [0, 25, 50, 75, 100]
    lw_main, lw_base = 1.5, 1.3

    handles = None
    for i, (task, inset_rel, title) in enumerate(TASKS):
        ax = axes[i]
        pts = panel_a[task]["pts"]
        xs = [p[0] for p in pts]
        ys = [100 * p[1] for p in pts]
        h0, = ax.plot(xs, ys, color=COLORS["Play2Win"], linewidth=lw_main, label="Play2Perfect (sparse reward)")
        if xs[-1] < 24.0:
            ax.plot([xs[-1], 24.0], [ys[-1], ys[-1]], color=COLORS["Play2Win"], linewidth=lw_main)
        # Dense drawn visually above sparse: sparse at y=0, dense at y=2.
        h2, = ax.plot(
            [0, 24.0], [0.0, 0.0], color=COLORS["Scratch_task"], linewidth=lw_base,
            label="Scratch (sparse reward)",
        )
        h1, = ax.plot(
            [0, 24.0], [2.0, 2.0], color=COLORS["Scratch_dense"], linewidth=lw_base,
            label="Scratch (dense reward)",
        )
        if handles is None:
            handles = [h0, h1, h2]  # kept for backward-compat; not used for the legend
        ax.set_title(title, fontsize=8)
        ax.set_xlim(0, 24)
        ax.set_ylim(-3, 104)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{v} h" for v in xticks], fontsize=7)
        ax.set_xlabel("Training time", fontsize=8.5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v}%" for v in yticks], fontsize=7)
        ax.set_ylabel("Success rate", fontsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        # Task-config inset in upper-right corner (axes-relative coords).
        inset_path = INPUTS_DIR / inset_rel
        if inset_path.exists():
            inset = ax.inset_axes([0.48, 0.35, 0.55, 0.55])
            inset.imshow(Image.open(inset_path))
            inset.axis("off")

    # Patch-swatch legend (baseline -> baseline -> ours). Square swatches with a
    # thin dark border, matching the reference screenshot style. "Ours" label is
    # semibold (weight 600) rather than full bold for a softer emphasis.
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
            text.set_fontweight(600)  # semibold, softer than 'bold' (700)

    # w_pad reduced from 1.0 -> 0.3 to shrink the gap between subplots.
    plt.tight_layout(w_pad=0.3, rect=[0, 0.10, 1, 1])
    plt.savefig(OUT / "panel_a_draft.png", dpi=240, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT / 'panel_a_draft.png'}")


if __name__ == "__main__":
    main()

"""Thesis-PRESENTATION variants of fig2 panel (a). Not for the paper.

Differences from make_panel_a.py:
  * The two beam-assembly tasks (Asm-Pillar = beam part 0, Asm-Beam = beam
    part 2) are AVERAGED into a single "Beam Assembly" panel, giving a 1x3 row:
        Peg-In-Hole | Beam Assembly (avg) | Screw-Leg
  * Three reveal variants for a slide build-up (--variant), ending on the
    Play2Perfect punchline:
        1_scratch_sparse       : Scratch (sparse) only
        2_scratch_sparse_dense : Scratch (sparse) + Scratch (dense)
        3_all                  : Scratch (sparse) + Scratch (dense) + Play2Perfect

Scratch baselines are flat lines (sparse=0%, dense=2%) — these are accurate:
the from-scratch runs converged there, they are not placeholders.

Play2Perfect curves come from outputs/fig2_panel_bcd/panel_a_curves.json
(success conditional on feasible init). The averaged beam curve interpolates
both beam curves onto a common time grid and means them.

Usage:
    .venv/bin/python plot_figures/fig2/make_panel_a_thesis.py            # all 3 variants
    .venv/bin/python plot_figures/fig2/make_panel_a_thesis.py --variant 3_all

Writes:
    plot_figures/fig2/outputs/thesis_presentation/panel_a_<variant>.png / .pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "outputs" / "fig2_panel_bcd"
INPUTS_DIR = REPO / "plot_figures" / "fig2" / "inputs"
OUT = REPO / "plot_figures" / "fig2" / "outputs" / "thesis_presentation"

COLORS = {
    "Play2Win": "#2C7BB6",
    "Scratch_dense": "#E08214",
    "Scratch_task": "#8C8C8C",
}
MAX_H = 24.0


def curve(panel_a, key):
    pts = panel_a[key]["pts"]
    return np.array([p[0] for p in pts]), np.array([100 * p[1] for p in pts])


def beam_avg(panel_a):
    """Mean of the two beam-assembly Play2Perfect curves on a common grid."""
    xa, ya = curve(panel_a, "Asm-Pillar")
    xb, yb = curve(panel_a, "Asm-Beam")
    hi = min(xa[-1], xb[-1])
    grid = np.linspace(0.0, hi, 203)
    return grid, 0.5 * (np.interp(grid, xa, ya) + np.interp(grid, xb, yb))


# variant -> (show scratch_sparse, show scratch_dense, show play2perfect)
VARIANTS = {
    "1_scratch_sparse":       (True, False, False),
    "2_scratch_sparse_dense": (True, True,  False),
    "3_all":                  (True, True,  True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    args = ap.parse_args()
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]

    panel_a = json.load(open(DATA_DIR / "panel_a_curves.json"))
    OUT.mkdir(parents=True, exist_ok=True)

    # (display title, inset, xs, ys) — beam panel is the averaged curve.
    xs_peg, ys_peg = curve(panel_a, "Peg-In-Hole")
    xs_beam, ys_beam = beam_avg(panel_a)
    xs_screw, ys_screw = curve(panel_a, "Screw-Leg")
    panels = [
        ("Tight-Insertion", "new_inputs_assembly/tight_insertion.png", xs_peg, ys_peg),
        ("Beam Assembly",   "new_inputs_assembly/assemble_beam_2.png", xs_beam, ys_beam),
        ("Screw-Leg",       "new_inputs_assembly/screw_leg.png",       xs_screw, ys_screw),
    ]

    for variant in variants:
        show_sparse, show_dense, show_p2p = VARIANTS[variant]
        fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.3))
        xticks = [0, 6, 12, 18, 24]
        yticks = [0, 25, 50, 75, 100]
        lw_main, lw_base = 1.6, 1.3

        for ax, (title, inset_rel, xs, ys) in zip(axes, panels):
            # Scratch baselines: sparse flat at 0%, dense flat at 2%.
            if show_sparse:
                ax.plot([0, MAX_H], [0.0, 0.0], color=COLORS["Scratch_task"], linewidth=lw_base)
            if show_dense:
                ax.plot([0, MAX_H], [2.0, 2.0], color=COLORS["Scratch_dense"], linewidth=lw_base)
            if show_p2p:  # the reveal — drawn on top
                ax.plot(xs, ys, color=COLORS["Play2Win"], linewidth=lw_main)
                if xs[-1] < MAX_H:  # extend flat tail to 24h
                    ax.plot([xs[-1], MAX_H], [ys[-1], ys[-1]],
                            color=COLORS["Play2Win"], linewidth=lw_main)

            ax.set_title(title, fontsize=8)
            ax.set_xlim(0, MAX_H)
            ax.set_ylim(-3, 104)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{v} h" for v in xticks], fontsize=7)
            ax.set_xlabel("Training time", fontsize=8.5)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{v}%" for v in yticks], fontsize=7)
            ax.set_ylabel("Success rate", fontsize=8.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

            inset_path = INPUTS_DIR / inset_rel
            if inset_path.exists():
                inset = ax.inset_axes([0.48, 0.35, 0.55, 0.55])
                inset.imshow(Image.open(inset_path))
                inset.axis("off")

        legend_entries = []
        if show_p2p:
            legend_entries.append(("Play2Perfect (sparse reward)", COLORS["Play2Win"], True))
        if show_dense:
            legend_entries.append(("Scratch (dense reward)", COLORS["Scratch_dense"], False))
        if show_sparse:
            legend_entries.append(("Scratch (sparse reward)", COLORS["Scratch_task"], False))

        legend_handles = [Patch(facecolor=c, edgecolor="#333333", linewidth=0.7)
                          for _, c, _ in legend_entries]
        leg = fig.legend(
            legend_handles, [lbl for lbl, _, _ in legend_entries],
            loc="lower center", bbox_to_anchor=(0.5, 0.02),
            ncol=len(legend_entries), frameon=False, fontsize=8,
            handlelength=1.1, handleheight=1.1, handletextpad=0.6, columnspacing=1.8,
        )
        for text, (_, _, emphasize) in zip(leg.get_texts(), legend_entries):
            if emphasize:
                text.set_fontweight(600)

        plt.tight_layout(w_pad=0.3, rect=[0, 0.12, 1, 1])
        for ext in ("png", "pdf"):
            p = OUT / f"panel_a_{variant}.{ext}"
            plt.savefig(p, dpi=240, facecolor="white", bbox_inches="tight")
        plt.close()
        print(f"wrote {OUT / f'panel_a_{variant}.png'} (+.pdf)")


if __name__ == "__main__":
    main()

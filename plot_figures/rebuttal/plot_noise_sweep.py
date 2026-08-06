"""lpeg object-pose observation-noise sweep: success rate vs noise level.

Translational and rotational noise are swept jointly, so every x tick carries
both values (cm on the top line, degrees on the bottom).

Reads  : lpeg_noise_sweep.json  (offline_eval_obj_pose_noise_sweep.py output)
Writes : plot_figures/rebuttal/outputs/lpeg_noise_sweep.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "plot_figures"))
from _style import configure_rcparams  # noqa: E402

# Single series -> one hue, no legend (the title names it). Slot 1 of the
# validated categorical theme; passes lightness band, chroma floor and 4.30:1
# contrast vs a light surface.
C_LINE = "#2a78d6"
INK = "#333333"
MUTED = "#777777"

TRAIN_XYZ_M, TRAIN_ROT_DEG = 0.01, 5.0  # training-time object-state noise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "outputs"))
    args = ap.parse_args()

    rows = json.load(open(args.json))["results"]
    xyz_cm = [100 * r["obj_xyz_noise_std_m"] for r in rows]
    rot = [r["obj_rot_noise_deg"] for r in rows]
    succ = [100 * r["early_drop_filtered"]["retract_rate"] for r in rows]

    configure_rcparams()
    fig, ax = plt.subplots(figsize=(6.8, 3.6))

    # The collapse happens between samples 2 and 3 -- mark it as unresolved
    # rather than implying the line between them is measured.
    ax.axvspan(xyz_cm[1], xyz_cm[2], color="#000000", alpha=0.05, linewidth=0)
    ax.text((xyz_cm[1] + xyz_cm[2]) / 2, 55, "unresolved\ntransition", fontsize=8,
            color=MUTED, ha="center", va="center", multialignment="center")

    ax.axvline(100 * TRAIN_XYZ_M, color=MUTED, linestyle=":", linewidth=1.0)
    ax.text(100 * TRAIN_XYZ_M + 0.8, 30,
            f"training σ\n({100 * TRAIN_XYZ_M:.0f} cm / {TRAIN_ROT_DEG:.0f}°)",
            fontsize=8, color=MUTED, ha="left", va="center")

    ax.plot(xyz_cm, succ, color=C_LINE, linewidth=2.0, marker="o", markersize=5,
            markerfacecolor=C_LINE, markeredgecolor="white", markeredgewidth=1.0,
            clip_on=False, zorder=3, solid_capstyle="round")

    for (x, y), off in zip(zip(xyz_cm[:3], succ[:3]), ((9, -3), (8, 9), (9, 7))):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=off,
                    fontsize=8.5, color=INK)

    ax.set_ylabel("Success rate")
    ax.set_xlabel("Object-pose observation noise      "
                  "translational σ (cm)  /  rotational σ (deg)")
    ax.set_ylim(-3, 108)
    ax.set_xlim(-2, 52)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)])
    # Every swept point labelled with BOTH magnitudes, since they move together.
    ax.set_xticks(xyz_cm)
    ax.set_xticklabels([f"{c:.1f}\n{d:.0f}°" for c, d in zip(xyz_cm, rot)], fontsize=8)
    ax.set_title("Peg teacher: success rate vs object-pose observation noise",
                 loc="left", fontsize=10.5, pad=8)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelcolor=INK)

    fig.tight_layout()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"lpeg_noise_sweep.{ext}", dpi=600,
                    facecolor="white", bbox_inches="tight", pad_inches=0.06)
    print(f"wrote {out / 'lpeg_noise_sweep.png'}")


if __name__ == "__main__":
    main()

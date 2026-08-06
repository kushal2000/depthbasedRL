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
# Categorical slots 1-2 of the validated theme. Validated on all pairs, light:
#   CVD worst protan/deutan dE 9.2 (target 8), normal-vision 24.0 (floor 15).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
C_LINE = PALETTE[0]
TRAIN_XYZ_CM, TRAIN_ROT = 1.0, 5.0  # training sigmas
INK = "#333333"
MUTED = "#777777"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, action="append",
                    help="sweep JSON; repeat to overlay several series")
    ap.add_argument("--label", action="append", default=[],
                    help="series label, one per --json")
    ap.add_argument("--name", default="lpeg_noise_sweep", help="output basename")
    ap.add_argument("--title", default=None)
    ap.add_argument("--x-mode", choices=["xyz", "multiple"], default="xyz",
                    help="'xyz': x = translational sigma (cm). 'multiple': x = "
                         "multiple of the training sigma (1 cm / 5 deg), which "
                         "is the only shared axis when each sweep varies a "
                         "different quantity.")
    ap.add_argument("--per-condition-filter", action="store_true",
                    help="filter early drops per condition instead of using the "
                         "zero-noise baseline count as a fixed denominator")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "outputs"))
    args = ap.parse_args()

    series = []
    for k, path in enumerate(args.json):
        blob = json.load(open(path))
        rows = blob["results"]
        # Fixed denominator: exclude only the number of early drops seen at the
        # ZERO-noise point. Unstable initial placement is a property of the reset
        # distribution (identical init states at every level), so that count is
        # the legitimate exclusion. Filtering per-condition instead would also
        # discard noise-induced failures -- and would silently drop envs censored
        # by the step budget, which must count as failures, not vanish.
        n_env = blob["num_envs"]
        denom = n_env - rows[0]["n_dropped_early"]
        if args.per_condition_filter:
            succ = [100 * r["early_drop_filtered"]["retract_rate"] for r in rows]
        else:
            succ = [100 * r["early_drop_filtered"]["retracted"] / denom for r in rows]
        xyz_cm = [100 * r["obj_xyz_noise_std_m"] for r in rows]
        rot = [r["obj_rot_noise_deg"] for r in rows]
        # If only rotation varies, the x axis IS rotation (absolute degrees);
        # otherwise it is translational sigma in cm.
        varies_rot = len(set(rot)) > 1 and len(set(xyz_cm)) == 1
        x = rot if varies_rot else xyz_cm
        series.append({
            "label": args.label[k] if k < len(args.label) else Path(path).stem,
            "xyz_cm": x, "raw_cm": xyz_cm, "rot": rot, "succ": succ,
        })
    # Tick positions come from the longest series; all sweeps share a grid.
    ref = max(series, key=lambda s: len(s["xyz_cm"]))
    xyz_cm, rot, succ = ref["xyz_cm"], ref["rot"], ref["succ"]

    configure_rcparams()
    fig, ax = plt.subplots(figsize=(6.8, 3.6))

    # Data-driven limits: the sweep range varies between runs (and partial runs
    # are plotted mid-sweep), so nothing here may assume a fixed span.
    span = max(xyz_cm) - min(xyz_cm)
    pad = 0.06 * span if span > 0 else 1.0
    ax.set_xlim(min(xyz_cm) - pad, max(xyz_cm) + pad)

    for k, s in enumerate(series):
        c = PALETTE[k % len(PALETTE)]
        ax.plot(s["xyz_cm"], s["succ"], color=c, linewidth=2.0, marker="o",
                markersize=5, markerfacecolor=c, markeredgecolor="white",
                markeredgewidth=1.0, clip_on=False, zorder=3 + k,
                solid_capstyle="round", label=s["label"])

    if len(series) > 1:
        # >= 2 series: a legend is mandatory so identity is never colour-alone.
        ax.legend(loc="lower left", frameon=False, fontsize=9,
                  handlelength=1.4, handletextpad=0.5, borderaxespad=0.4)
    else:
        # Single series: no legend (the title names it); label the first point
        # and any >5-point drop. A number on every marker is noise, and a label
        # on a flat tail just repeats the one before it.
        label_idx = {0}
        label_idx |= {i for i in range(1, len(succ)) if succ[i - 1] - succ[i] > 5.0}
        for i in sorted(label_idx):
            # Place below only when falling AND with room; near the floor a
            # below-label collides with the x tick labels.
            below = i > 0 and succ[i] < succ[i - 1] - 5.0 and succ[i] > 12.0
            at_right = xyz_cm[i] > min(xyz_cm) + 0.85 * span
            ax.annotate(f"{succ[i]:.1f}%", (xyz_cm[i], succ[i]),
                        textcoords="offset points",
                        xytext=(-34 if at_right else 6, -12 if below else 9),
                        fontsize=8.5, color=INK)

    ax.set_ylabel("Success rate")
    if len(set(rot)) > 1 and len(set(ref["raw_cm"])) == 1:
        _xlab = "Rotational observation noise σ (deg)"
    elif len(set(rot)) == 1:
        _xlab = "Translational observation noise σ (cm)"
    else:
        _xlab = ("Object-pose observation noise      "
                 "translational σ (cm)  /  rotational σ (deg)")
    ax.set_xlabel(_xlab)
    ax.set_ylim(-3, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)])
    # Every swept point labelled with BOTH magnitudes, since they move together.
    # Thin the ticks if the points are too dense to read.
    step = 1 if len(xyz_cm) <= 12 else 2
    ax.set_xticks(xyz_cm[::step])
    if len(set(rot)) > 1 and len(set(ref["raw_cm"])) == 1:
        ax.set_xticklabels([f"{v:.1f}" for v in xyz_cm[::step]], fontsize=8.5)
    elif len(set(rot)) == 1:
        # Rotation is held constant (the title says at what) -- repeating it on
        # every tick is noise.
        ax.set_xticklabels([f"{c:.1f}" for c in xyz_cm[::step]], fontsize=8.5)
    else:
        ax.set_xticklabels(
            [f"{c:.1f}\n{d:.0f}°" for c, d in zip(xyz_cm[::step], rot[::step])],
            fontsize=8)
    ax.set_title(args.title or "Success rate vs object-pose observation noise",
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
        fig.savefig(out / f"{args.name}.{ext}", dpi=600,
                    facecolor="white", bbox_inches="tight", pad_inches=0.06)
    print(f"wrote {out / f'{args.name}.png'}")


if __name__ == "__main__":
    main()

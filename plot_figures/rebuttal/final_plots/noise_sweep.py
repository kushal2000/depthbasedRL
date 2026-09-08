"""Final rebuttal figure: success rate vs object-pose noise, all 4 tasks.

Reads the four *_ratio_sweep.json files from experiments/outputs (real re-run
data, not the digitised PNG) and renders one overlay with the fig1-style
patch-swatch legend below the axes.

Writes: final_plots/01_noise_sweep/noise_sweep_all_tasks.{png,pdf}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "plot_figures"))
from _style import configure_rcparams  # noqa: E402

EXP = HERE.parent / "experiments" / "outputs"
# (json, legend label) — Step number on its own line per the rebuttal layout.
SERIES = [
    (EXP / "peg_ratio_sweep.json", "Tight Insertion"),
    (EXP / "beam1_ratio_sweep.json", "Assemble-Beam\n(Step 1)"),
    (EXP / "beam2_ratio_sweep.json", "Assemble-Beam\n(Step 2)"),
    (EXP / "screw_leg_ratio_sweep.json", "Screw-Leg"),
]

# 4-series set, validated on ALL pairs (light): CVD worst protan/deutan dE 9.2
# (target 8), normal-vision worst 16.3 (floor 15).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
TRAIN_XYZ_CM = 1.0  # training-time object-state noise sigma (1 cm : 5 deg)
INK = "#333333"
MUTED = "#777777"


def load(path: Path) -> tuple[list[float], list[float], list[float]]:
    blob = json.load(open(path))
    rows = blob["results"]
    # Fixed denominator: exclude only the zero-noise early drops (property of
    # the reset distribution); noise-induced failures must count as failures.
    denom = blob["num_envs"] - rows[0]["n_dropped_early"]
    succ = [100 * r["early_drop_filtered"]["retracted"] / denom for r in rows]
    xyz_cm = [100 * r["obj_xyz_noise_std_m"] for r in rows]
    rot = [r["obj_rot_noise_deg"] for r in rows]
    return xyz_cm, rot, succ


def main() -> None:
    configure_rcparams()
    # Title-less canvas: the title band is trimmed off the height.
    fig, ax = plt.subplots(figsize=(4.6, 3.22))
    ax.set_axisbelow(True)

    xyz_cm = rot = None
    for k, (path, label) in enumerate(SERIES):
        xyz_cm, rot, succ = load(path)
        c = PALETTE[k]
        ax.plot(xyz_cm, succ, color=c, linewidth=2.2, marker="o",
                markersize=6.0, markerfacecolor=c, markeredgecolor="white",
                markeredgewidth=1.0, clip_on=False, zorder=3 + k,
                solid_capstyle="round", solid_joinstyle="round")

    span = max(xyz_cm) - min(xyz_cm)
    ax.set_xlim(min(xyz_cm), max(xyz_cm) + 0.015 * span)

    # Training-noise reference: the sigma the policies were trained with.
    # Two lines, low and just right of the line: the bottom-left corner stays
    # curve-free out to x~4.5, and stacking keeps the text clear of the ticks.
    ax.axvline(TRAIN_XYZ_CM, color=MUTED, linestyle=":", linewidth=1.2, zorder=1)
    ax.text(TRAIN_XYZ_CM + 0.03 * span, 6, "training\nnoise level",
            fontsize=11.5, color=MUTED, ha="left", va="bottom", linespacing=1.25)

    ax.set_ylabel("Success rate", fontsize=14)
    ax.set_xlabel("Object Pose Noise Level", fontsize=14)
    ax.set_ylim(-2, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=11.5)
    # Both magnitudes on each tick (they sweep jointly), thinned to fit.
    # Two-line ticks (cm over deg): five "x cm / y°" one-liners collide on a
    # square canvas.
    TICK_IDX = [1, 3, 5, 7, 9]
    ax.set_xticks([xyz_cm[i] for i in TICK_IDX])
    ax.set_xticklabels(
        [f"{xyz_cm[i]:.1f} cm\n/ {rot[i]:.0f}°" for i in TICK_IDX],
        fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a9a")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors="#9a9a9a", labelcolor=INK, length=3, width=0.8)

    # Right-align the last label, then nudge it right so it does not collide
    # with the 7.8 cm tick label on the fixed-aspect canvas. This must run
    # after tick_params(), which updates tick-label transforms.
    last_tick_label = ax.get_xticklabels()[-1]
    last_tick_label.set_ha("right")
    last_tick_label.set_transform(
        last_tick_label.get_transform()
        + ScaledTranslation(8 / 72, 0, fig.dpi_scale_trans)
    )

    # fig1-style legend: filled square swatches with a thin dark border,
    # frameless, one row centred below the axes.
    # Legend order [0, 1, 3, 2]: ncol=2 fills column-major, so this puts the
    # one-liners (Tight Insertion, Screw-Leg) on row 1 and the two Assemble-
    # Beam entries together on row 2.
    order = [0, 1, 3, 2]
    handles = [Patch(facecolor=PALETTE[k], edgecolor="#333333", linewidth=0.7)
               for k in order]
    labels = [SERIES[k][1] for k in order]
    # Single-column legend in the right-hand strip, vertically centred on the
    # axes.
    # 2x2 below the axes: on the 4.6 in-wide canvas two columns of these
    # labels fit at ~10.5 pt (on the old 3.5 in canvas they capped at 9 pt).
    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, frameon=False, fontsize=12.5,
               handlelength=1.0, handleheight=1.0, handletextpad=0.35,
               columnspacing=1.6, labelspacing=0.5,
               borderaxespad=0.0, borderpad=0.2)

    fig.subplots_adjust(left=0.18, right=0.985, top=0.972, bottom=0.452)
    out = HERE / "01_noise_sweep"
    out.mkdir(exist_ok=True)
    # No tight bbox: the canvas IS the deliverable, at exactly 1.5:1.
    for ext in ("png", "pdf"):
        fig.savefig(out / f"noise_sweep_all_tasks.{ext}", dpi=600,
                    facecolor="white")
    print(f"wrote {out / 'noise_sweep_all_tasks.png'}")


if __name__ == "__main__":
    main()

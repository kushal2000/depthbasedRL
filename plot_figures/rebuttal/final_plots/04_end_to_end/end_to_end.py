"""Final rebuttal figure: end-to-end multi-part beam assembly, 20 real trials.

Left: 2x2 rollout frames from inputs/slowmo_multi_part.mp4 (row 1 = Step 1
grasp-and-insert, row 2 = Step 2 place-on-top + final state). Right: a narrow
success-rate table. The paper evaluated the two steps separately; these are
sequential end-to-end trials: Step 1 succeeded 16/20, Step 2 succeeded on 12
of those 16, so 12/20 end-to-end.

Writes: 04_end_to_end/end_to_end.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "plot_figures"))
from _style import configure_rcparams  # noqa: E402

# Row-major: Step 1 (start state with both parts on the table -> proves the
# trial is end-to-end -- then insert), Step 2 (place on top, final state).
# ONE shared crop window for every frame: per-frame windows shift the view
# and make the fixed camera look like it moved between tiles.
FRAMES = [("frame_14.png", 72), ("frame_120.png", 72),
          ("frame_300.png", 72), ("frame_412.png", 72)]

# Thin spaces around the solidus keep the fractions legible at small sizes.
# The dagger on Step 2 points to the footnote explaining its denominator.
ROWS = [
    ("Step 1", "16 / 20", "80%", False),
    ("Step 2†", "12 / 16", "75%", False),
    ("Overall", "12 / 20", "60%", True),
]
INK = "#333333"


def main() -> None:
    configure_rcparams()
    # Height matches panels 01-03; 1.55:1 wide.
    FIG_H = 3.07
    top_in, bottom_in, gap_in = 0.03, 0.03, 0.045
    # 720px frames cropped 10% top / 10% bottom and 40px per side (the
    # orange parts span x=56..671 at the extremes) -> ~1.11:1 tiles.
    CROP_TOP, CROP_BOT = 72, 72
    CROP_L, CROP_R = 40, 40
    TILE_AR = (720 - CROP_L - CROP_R) / (720 - CROP_TOP - CROP_BOT)
    tile = (FIG_H - top_in - bottom_in - gap_in) / 2
    tile_w = tile * TILE_AR
    side_in = 0.03
    grid_w = 2 * tile_w + gap_in
    # Width DERIVES from the content (photo grid + table strip): no forced
    # aspect, no dead margins.
    table_in, table_pad_in = 1.52, 0.14
    FIG_W = side_in + grid_w + table_pad_in + table_in + table_pad_in

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    pos = [(side_in, bottom_in + tile + gap_in),
           (side_in + tile_w + gap_in, bottom_in + tile + gap_in),
           (side_in, bottom_in),
           (side_in + tile_w + gap_in, bottom_in)]
    for k, ((fname, y0), (x_in, y_in)) in enumerate(zip(FRAMES, pos)):
        ax = fig.add_axes([x_in / FIG_W, y_in / FIG_H,
                           tile_w / FIG_W, tile / FIG_H])
        img = Image.open(HERE / fname)
        win_h = img.height - CROP_TOP - CROP_BOT
        ax.imshow(img.crop((CROP_L, y0, img.width - CROP_R, y0 + win_h)))
        ax.set_axis_off()
        # Small sequence number: orders the frames as ONE continuous rollout
        # (step labels here made the trial read as two separate demos).
        ax.text(0.055, 0.905, str(k + 1), transform=ax.transAxes,
                fontsize=10.5, color=INK, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.25", facecolor="white",
                          edgecolor="none", alpha=0.85))

    # Success-rate table: a narrow single column filling the strip's height.
    # Each stage is a label-over-value block; Overall is set off by a rule.
    tx0 = (side_in + grid_w + table_pad_in) / FIG_W
    tax = fig.add_axes([tx0, bottom_in / FIG_H, table_in / FIG_W,
                        (FIG_H - top_in - bottom_in) / FIG_H])
    tax.set_axis_off()
    tax.set_xlim(0, 1)
    tax.set_ylim(0, 1)
    # booktabs-style: heavy top/bottom rules, light mid rules, uniform row
    # rhythm. The stage shares the fraction's baseline; the percentage is a
    # smaller muted second line (bold ink for the Overall row).
    MUTED = "#777777"

    # Rules overhang the text margins slightly (booktabs \tabcolsep air) and
    # sit at symmetric insets so they anchor to the photo grid's edges.
    def rule(y, heavy):
        tax.plot([-0.03, 1.03], [y, y], color=INK if heavy else "#bbbbbb",
                 linewidth=1.4 if heavy else 0.8, clip_on=False)

    rule(0.985, True)
    tax.text(0.5, 0.93, "Success Rate", fontsize=14, color=INK,
             fontweight="bold", ha="center", va="center")
    rule(0.865, False)
    # Constant 0.225 block pitch; the Overall separator sits midway between
    # Step 2's percentage line and the Overall label so no gap collapses.
    blocks = [0.74, 0.515, 0.29]
    for (stage, frac, pct, bold), y in zip(ROWS, blocks):
        w = "bold" if bold else "normal"
        c2 = INK if bold else MUTED
        if bold:
            rule(y + 0.075, False)
        tax.text(0.0, y, stage, fontsize=13.5, color=INK,
                 ha="left", va="center", fontweight=w)
        tax.text(1.0, y, frac, fontsize=13.5, color=INK,
                 ha="right", va="center", fontweight=w)
        tax.text(1.0, y - 0.075, f"({pct})", fontsize=12, color=c2,
                 ha="right", va="center", fontweight=w)
    rule(0.115, True)
    tax.text(0.0, 0.08, "† attempted on the\n16 Step-1 successes",
             fontsize=9, color=MUTED, ha="left", va="top",
             style="italic", linespacing=1.25)

    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"end_to_end.{ext}", dpi=600, facecolor="white")
    print(f"wrote {HERE / 'end_to_end.png'}")


if __name__ == "__main__":
    main()

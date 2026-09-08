"""Final rebuttal figure: real-world failure breakdown pie (14 failures).

From-scratch replacement for the pie cropped out of fig1_noise_and_failures_v4:
category names move from beside the pie into a 2x2 swatch legend below, so the
pie itself can fill the panel width. Styled to sit next to 01_noise_sweep
(same serif style, title size, legend aesthetic, palette).

Writes: 02_failure_analysis/failure_analysis.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "plot_figures"))
from _style import configure_rcparams  # noqa: E402

# Clockwise order chosen so reorientation's arc crosses 12 o'clock and
# insertion's crosses 6 o'clock (their labels sit above/below in one line),
# with pose tracking on the right and timeout on the left. Hue-to-category
# mapping matches the v4 pie.
N = 14
CATEGORIES = [
    ("Drop during reorientation", 3, "#1baf7a"),
    ("Pose tracking failure", 2, "#4a3aa7"),
    ("Drop during insertion", 4, "#eb6834"),
    ("Timeout", 5, "#2a78d6"),
]
# Centres the reorientation wedge on 12 o'clock: 90 + (3/14 * 360) / 2.
START_ANGLE = 90 + 0.5 * (3 / N) * 360


def main() -> None:
    configure_rcparams()
    # Title-less canvas: the title band is trimmed off the height.
    fig, ax = plt.subplots(figsize=(4.6, 3.13))

    counts = [c for _, c, _ in CATEGORIES]
    colors = [col for _, _, col in CATEGORIES]
    wedges, _ = ax.pie(
        counts, colors=colors, startangle=START_ANGLE, counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 2.0},
    )
    # Percentage + raw count inside each wedge; the category name lives in the
    # legend, so the wedge text stays short enough to fit even the 14% slice.
    for w, cnt in zip(wedges, counts):
        ang = 0.5 * (w.theta1 + w.theta2)
        import math
        # The 2/14 wedge is too narrow at r=0.62 for two lines of text --
        # push its label outward (wedges widen with r) and shrink it a step.
        r, fs = (0.68, 11) if cnt == 2 else (0.62, 12.5)
        x, y = r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang))
        ax.text(x, y, f"{100 * cnt / N:.1f}%\n({cnt}/{N})",
                ha="center", va="center", fontsize=fs, color="white",
                fontweight="bold", linespacing=1.3)

    # Category names sit beside their wedges in the wedge colour, like the
    # original fig1 pie -- no legend. Positions are hand-placed per wedge.
    # Each label sits on its wedge's mid-angle direction so it reads as
    # belonging to that wedge, not to the pie as a whole. The two right-side
    # wedges stack their labels in three lines, one block per wedge.
    SIDE_LABELS = [
        ("Drop during\nReorientation", (0.0, 1.03), "center", "bottom"),
        ("Pose\nTracking\nFailure", (1.07, 0.40), "left", "center"),
        ("Drop\nduring\nInsertion", (0.92, -0.78), "left", "center"),
        ("Timeout", (-1.06, -0.22), "right", "center"),
    ]
    for (text, (x, y), ha, va), (_, _, col) in zip(SIDE_LABELS, CATEGORIES):
        ax.text(x, y, text, ha=ha, va=va, fontsize=12, color=col,
                fontweight="bold", linespacing=1.15, clip_on=False)

    ax.set_aspect("equal")
    # Keep the x-span just wide enough for the side labels: any wider and the
    # equal-aspect axes become width-limited, letterboxing the pie vertically.
    # With no label below the pie any more, the bottom limit hugs the circle.
    # Wider limits than the pie needs -> the pie renders smaller, buying the
    # enlarged labels their air.
    ax.set_xlim(-1.70, 1.75)
    ax.set_ylim(-1.12, 1.38)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.995, bottom=0.005)
    # No tight bbox: the canvas IS the deliverable, at exactly 1.5:1.
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"failure_analysis.{ext}", dpi=600,
                    facecolor="white")
    print(f"wrote {HERE / 'failure_analysis.png'}")


if __name__ == "__main__":
    main()

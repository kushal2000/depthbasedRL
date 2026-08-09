"""Final rebuttal figure: occlusion example photos (2x2 grid).

Rebuilds Small_Object_Pose_Tracking_Failure_NEW4.pdf as a titled panel.
The four photos live in tile_{tl,tr,bl,br}.png (extracted at 300 dpi from
that PDF, red circles included). Rows are scenarios (robot hand / human
hand), columns are before/during occlusion.

Each square tile is cropped to 1.2:1 with a hand-picked window: the BR tile
caps the crop (red circle near its top, fixture box at its bottom), so the
grid cannot reach the canvas's 1.33 ratio and takes small side margins.

Writes: 03_occlusion_examples/occlusion_examples.{png,pdf}
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

# (tile, y0, y1) crop windows in the 870-px tiles, chosen per image so the
# red circles, the orange beam (top row) and the fixture box (bottom row)
# all survive. Width is never cropped.
TILES = {
    "tl": (60, 785),
    "tr": (90, 815),
    "bl": (120, 845),
    "br": (120, 845),
}
TILE_AR = 870 / 725  # width / cropped height = 1.2


def main() -> None:
    configure_rcparams()
    # Height matches panels 01/02; width is DERIVED from the photo grid so
    # there are no dead side margins. Lands near 1.06:1.
    FIG_H = 3.07
    top_in, bottom_in, vgap_in, hgap_in, side_in = 0.03, 0.03, 0.045, 0.045, 0.03
    tile_h = (FIG_H - top_in - bottom_in - vgap_in) / 2
    tile_w = tile_h * TILE_AR
    grid_w = 2 * tile_w + hgap_in
    FIG_W = grid_w + 2 * side_in
    x0_in = side_in

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    pos = {
        "tl": (x0_in, bottom_in + tile_h + vgap_in),
        "tr": (x0_in + tile_w + hgap_in, bottom_in + tile_h + vgap_in),
        "bl": (x0_in, bottom_in),
        "br": (x0_in + tile_w + hgap_in, bottom_in),
    }
    for k, (y0, y1) in TILES.items():
        img = Image.open(HERE / f"tile_{k}.png").crop((0, y0, 870, y1))
        x_in, y_in = pos[k]
        ax = fig.add_axes([x_in / FIG_W, y_in / FIG_H,
                           tile_w / FIG_W, tile_h / FIG_H])
        ax.imshow(img)
        ax.set_axis_off()

    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"occlusion_examples.{ext}", dpi=600,
                    facecolor="white")
    print(f"wrote {HERE / 'occlusion_examples.png'}")


if __name__ == "__main__":
    main()

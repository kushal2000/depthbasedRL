"""Montage rendered beam env tiles into a 3x5 parallel-env grid.

Default: 7 tiles from tiles_beam0 + 8 from tiles_beam2 = 15 (3 rows x 5 cols).
Pick specific tiles with --beam0-idx / --beam2-idx (else first-N in order).

Usage:
    python plot_figures/fig2/montage_beam_grid.py
    python plot_figures/fig2/montage_beam_grid.py --beam0-idx 0 1 3 4 5 7 8 \\
        --beam2-idx 0 2 3 4 5 6 8 9

Writes:
    plot_figures/fig2/outputs/thesis_presentation/beam_parallel_grid.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / "outputs" / "thesis_presentation"


def load_tiles(slug, idxs):
    d = OUT / f"tiles_{slug}"
    avail = sorted(d.glob("env_*.png"))
    if idxs is None:
        return [np.array(Image.open(p).convert("RGB")) for p in avail]
    return [np.array(Image.open(d / f"env_{i:02d}.png").convert("RGB")) for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--n-beam0", type=int, default=7)
    ap.add_argument("--n-beam2", type=int, default=8)
    ap.add_argument("--beam0-idx", type=int, nargs="*", default=None)
    ap.add_argument("--beam2-idx", type=int, nargs="*", default=None)
    ap.add_argument("--gap", type=int, default=6)
    ap.add_argument("--gap-color", type=int, nargs=3, default=(255, 255, 255))
    ap.add_argument("--out", default=str(OUT / "beam_parallel_grid.png"))
    args = ap.parse_args()

    b0 = load_tiles("beam0", args.beam0_idx)[: args.n_beam0]
    b2 = load_tiles("beam2", args.beam2_idx)[: args.n_beam2]
    tiles = b0 + b2
    need = args.rows * args.cols
    if len(tiles) < need:
        raise SystemExit(f"have {len(tiles)} tiles, need {need} for {args.rows}x{args.cols}")
    tiles = tiles[:need]

    H, W = tiles[0].shape[:2]
    g = args.gap
    canvas = np.full((args.rows * H + (args.rows - 1) * g,
                      args.cols * W + (args.cols - 1) * g, 3),
                     np.array(args.gap_color, np.uint8), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, args.cols)
        canvas[r * (H + g):r * (H + g) + H, c * (W + g):c * (W + g) + W] = t

    Image.fromarray(canvas).save(args.out)
    print(f"wrote {args.out}  ({canvas.shape[1]}x{canvas.shape[0]}; "
          f"{len(b0)} beam0 + {len(b2)} beam2)")


if __name__ == "__main__":
    main()

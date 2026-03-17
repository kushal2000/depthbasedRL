#!/usr/bin/env python3
"""Create side-by-side collision method comparison images.

Produces one PNG per part (1 row x 4 cols) with large text labels.

Usage:
    python fabrica/make_comparison.py
    python fabrica/make_comparison.py --frame frame_0200.png
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METHODS = [
    ("VHACD", "vhacd"),
    ("CoACD (default)", "coacd_old"),
    ("CoACD (tuned)", "coacd"),
    ("SDF", "sdf"),
]
PARTS = ["2", "6"]
DEBUG_OUTPUT = Path("debug_output")


def make_row(part, images, frame, output_path, font, small_font):
    """Create a single-part comparison image: 1 row x 4 cols with title."""
    cols = len(METHODS)

    # Get cell dimensions
    sample = next(iter(images.values()))
    cell_w, cell_h = sample.size

    title_h = 140
    label_h = 100
    padding = 10

    grid_w = cols * cell_w + (cols - 1) * padding
    grid_h = title_h + label_h + cell_h

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Title
    title = f"Beam Part {part} — Collision Method Comparison"
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((grid_w // 2 - tw // 2, 15), title, fill=(0, 0, 0), font=font)

    # Column headers + images
    for col_idx, (label, _) in enumerate(METHODS):
        x = col_idx * (cell_w + padding)

        # Header
        bbox = draw.textbbox((0, 0), label, font=small_font)
        lw = bbox[2] - bbox[0]
        draw.text((x + cell_w // 2 - lw // 2, title_h + 5), label, fill=(0, 0, 0), font=small_font)

        # Image
        y = title_h + label_h
        key = (part, label)
        if key in images:
            grid.paste(images[key], (x, y))
        else:
            draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(200, 200, 200))
            draw.text((x + 10, y + cell_h // 2), "Missing", fill=(100, 100, 100), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path))
    print(f"Saved {output_path} ({grid_w}x{grid_h})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=str, default="frame_0200.png",
                        help="Frame filename to use from video_frames/")
    parser.add_argument("--output-dir", type=str, default="debug_output/comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 96)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 72)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    # Load all images
    images = {}
    for part in PARTS:
        for label, method_dir in METHODS:
            img_path = DEBUG_OUTPUT / method_dir / f"part_{part}" / "video_frames" / args.frame
            if not img_path.exists():
                print(f"WARNING: Missing {img_path}")
                continue
            images[(part, label)] = Image.open(img_path)

    if not images:
        print("ERROR: No images found")
        return

    for part in PARTS:
        out = output_dir / f"collision_comparison_part_{part}.png"
        make_row(part, images, args.frame, out, font, small_font)


if __name__ == "__main__":
    main()

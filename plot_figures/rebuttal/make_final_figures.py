#!/usr/bin/env python3
"""Draft the five rebuttal figures at their final on-page sizes.

CoRL allows a one-page rebuttal, so every figure is generated at exactly the
size it will occupy (see outputs/final/README.md for the layout). Drafting at
final size is the point: a chart that reads fine at 5 in is often illegible at
1.65 in, and that only shows up if you draw it small.

Real vs placeholder, as of this draft:

  fig1  pie = REAL (14 logged real-world failures). Sweep = REAL but SINGLE
        TASK (Screw-Leg): only screw_leg_ratio_sweep.json survived; the peg and
        beam sweeps need re-running to add series.
  fig2  frames = REAL (pulled from inputs/slowmo_multi_part.mp4).
        table = REAL (16/20, 12/20).
  fig3  dimension panels = REAL (measured off the actual assets).
        rollout tiles = PLACEHOLDER; the teachers are still training.
  fig4  REAL.
  fig5  REAL, and matched -- the solo arms were re-run at the co-trained eval's
        placement range so nothing needs an asterisk.

    python plot_figures/rebuttal/make_final_figures.py [--textwidth 5.5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "plot_figures/rebuttal/outputs/final"
DATA = REPO / "plot_figures/rebuttal/experiments/outputs"
FRAMES = REPO / "plot_figures/rebuttal/inputs/frames"
INPUTS = REPO / "plot_figures/rebuttal/inputs"

# CVD-validated categorical order.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID = "#333333", "#777777", "#d8dae0"
PLACEHOLDER_FC, PLACEHOLDER_EC = "#eef0f4", "#b9bec9"

# Small type is the whole game at these widths.
plt.rcParams.update({
    "font.size": 6.0, "axes.labelsize": 6.0, "axes.titlesize": 6.5,
    "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "legend.fontsize": 5.6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "figure.dpi": 400, "savefig.dpi": 400,
})


def rate(stem):
    r = json.load(open(DATA / f"{stem}.json"))["results"][0]
    return 100.0 * r["early_drop_filtered"]["retract_rate"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white",
                    bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)
    print(f"  {name}")


def placeholder(ax, label):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           fc=PLACEHOLDER_FC, ec=PLACEHOLDER_EC,
                           lw=0.7, ls=(0, (3, 2))))
    ax.text(0.5, 0.5, label, transform=ax.transAxes, ha="center", va="center",
            fontsize=5.2, color=MUTED, linespacing=1.35)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def bare(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(INK); ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelcolor=INK, pad=1.5)


# --------------------------------------------------------------- fig 1
def fig1(W):
    """Left: success vs object-pose noise. Right: real-world failure modes.

    The left panel currently EMBEDS the previously rendered four-task figure
    (outputs/noise_sweep_all_tasks.png) rather than replotting, because only
    screw_leg_ratio_sweep.json is still on disk -- the other three sweeps' JSONs
    went to a scratch location that has been cleared. Sweeps 856567/8/9 are
    re-running them; once those land, set SWEEP_JSONS below and this switches to
    a native replot at the right point size.

    Digitising the PNG was tried and rejected: validated against the Screw-Leg
    series (whose JSON we do have) it was wrong by up to 17.9 points, because
    the marker discs never separate from the connecting lines.
    """
    SWEEP_JSONS = {"Tight Insertion": "peg_ratio_sweep",
                   "Assemble-beam Step 1": "beam1_ratio_sweep",
                   "Assemble-beam Step 2": "beam2_ratio_sweep",
                   "Screw-Leg": "screw_leg_ratio_sweep"}
    have_all = all((DATA / f"{v}.json").exists() for v in SWEEP_JSONS.values())

    fig, axes = plt.subplots(1, 2, figsize=(0.40 * W, 1.12),
                             gridspec_kw=dict(width_ratios=[1.42, 0.86],
                                              wspace=0.30))
    ax = axes[0]
    if have_all:
        for i, (label, stem) in enumerate(SWEEP_JSONS.items()):
            blob = json.load(open(DATA / f"{stem}.json"))
            xs = [float(r["name"].split("xyz")[1].split("m_")[0]) * 100
                  for r in blob["results"]]
            ys = [100.0 * r["early_drop_filtered"]["retract_rate"]
                  for r in blob["results"]]
            ax.plot(xs, ys, "-o", color=[C1, C2, C3, C4][i], lw=1.0, ms=1.8,
                    label=label)
        ax.axvline(1.0, color=MUTED, ls=":", lw=0.7)
        ax.set_xlabel("pose noise $\\sigma$ (cm)", labelpad=1.2)
        ax.set_ylabel("success (%)", labelpad=1.2)
        ax.set_ylim(-4, 104); ax.set_yticks([0, 50, 100])
        ax.legend(frameon=False, fontsize=3.9, handlelength=0.9,
                  handletextpad=0.4, borderpad=0.1, labelspacing=0.18)
        bare(ax)
    else:
        # stand-in: the existing render, cropped to drop its title
        src = REPO / "plot_figures/rebuttal/outputs/noise_sweep_all_tasks.png"
        im = plt.imread(src)
        h = im.shape[0]
        ax.imshow(im[int(0.055 * h):, :])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # REAL: 14 logged real-world failures.
    ax = axes[1]
    counts = [5, 4, 3, 2]
    labels = ["timeout", "drop in\ninsertion", "drop in\nreorient", "pose"]
    ax.pie(counts, colors=[C1, C2, C3, C4], startangle=90,
           radius=1.00, wedgeprops=dict(lw=0.5, edgecolor="white"),
           labels=labels, labeldistance=1.26,
           textprops=dict(fontsize=4.2, color=INK),
           autopct=lambda pc: f"{int(round(pc*14/100))}", pctdistance=0.60)
    for t in ax.texts:
        if t.get_text().isdigit():
            t.set_fontsize(4.9); t.set_color("white"); t.set_weight("bold")
    ax.set_aspect("equal")
    save(fig, "fig1_noise_and_failures")


# --------------------------------------------------------------- fig 2
def fig2(W):
    """Two frames SIDE BY SIDE over a compact two-column table.

    Side by side beats stacked here. Stacking forced a ~2.9:1 letterbox crop to
    fill 2.09 in of width within the height each frame could have; side by side
    gives each frame half the width, so a near-square crop keeps the hand, both
    beams and the fixture in shot. It also reads left-to-right as a sequence,
    which stacked did not.

    Table is two columns under a "success rate" header. Percentages were dropped
    as redundant -- 16/20 and 12/20 already carry the rate, and at 5.6 pt a
    second numeric column is the difference between legible and not.
    """
    # Height is DERIVED, not chosen. Each frame occupies half the panel width;
    # with a known crop aspect that fixes the image row's height exactly, so the
    # figure is only as tall as (images + table). Guessing it left a dead band
    # between the two, because imshow centres the image in a too-tall axes.
    panel_w = 0.38 * W
    crop_ar = (0.78 - 0.02) / (0.90 - 0.24)          # crop w/h, ~1.15
    img_h = (panel_w / 2) / crop_ar
    table_h = 0.34
    fig = plt.figure(figsize=(panel_w, img_h + table_h))
    gs = fig.add_gridspec(2, 2, height_ratios=[img_h, table_h],
                          hspace=0.04, wspace=0.04)
    frames = sorted(FRAMES.glob("f_*.png"), key=lambda p: float(p.stem[2:]))
    # Asymmetric crops: step 1's subject sits left of frame and step 2's right,
    # so trimming the far side of each squares them up and removes dead table.
    picks = [(frames[1] if len(frames) > 1 else None, "step 1", (0.02, 0.78)),
             (frames[2] if len(frames) > 2 else None, "step 2", (0.24, 1.00))]
    for i, (src, phase, (xa, xb)) in enumerate(picks):
        ax = fig.add_subplot(gs[0, i])
        if src is not None:
            im = plt.imread(src)
            h, w = im.shape[:2]
            ax.imshow(im[int(0.24 * h):int(0.90 * h), int(xa * w):int(xb * w)])
        else:
            placeholder(ax, "frame")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.04, 0.94, phase, transform=ax.transAxes, va="top", ha="left",
                fontsize=5.2, color="white",
                bbox=dict(fc="black", alpha=0.5, pad=1.0, lw=0))

    ax = fig.add_subplot(gs[1, :]); ax.axis("off")
    t = ax.table(cellText=[["Step 1", "16 / 20"],
                           ["Step 1 + 2", "12 / 20"]],
                 colLabels=["", "success rate"],
                 cellLoc="center", loc="center", colWidths=[0.50, 0.50])
    t.auto_set_font_size(False); t.set_fontsize(5.6); t.scale(1, 1.10)
    for (r, c), cell in t.get_celld().items():
        cell.set_linewidth(0.4); cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(color=MUTED)
        if r == 2 and c == 1:
            cell.set_text_props(weight="bold", color=C2)
    save(fig, "fig2_end_to_end")


def _silhouette(ax, mesh, axes2d, color, dx=0.0, dy=0.0):
    """Fill every projected triangle so their union reads as a silhouette.

    Simpler and more robust than an outline: no hull, no alpha shape, and
    concave features (the fork's tines, the gap between prongs) survive, which
    an convex outline would erase.
    """
    from matplotlib.collections import PolyCollection
    i, j = axes2d
    tris = mesh.vertices[mesh.faces][:, :, [i, j]] * 1000.0
    tris = tris + np.array([dx, dy])
    ax.add_collection(PolyCollection(tris, facecolors=color, edgecolors="none",
                                     linewidths=0, zorder=2))


def _dim(ax, p0, p1, text, off=0.0, vertical=False, fs=4.6):
    """A dimension line with arrows and a label."""
    x0, y0 = p0; x1, y1 = p1
    if vertical:
        x0 += off; x1 += off
    else:
        y0 += off; y1 += off
    ax.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.6,
                                shrinkA=0, shrinkB=0), zorder=4)
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, fontsize=fs, color=INK,
            ha="center", va="center", rotation=90 if vertical else 0,
            bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)


def _load_obj(which):
    import trimesh
    if which == "plug":
        sc = trimesh.load(REPO / "assets/urdf/plug_fork/plug/plug_visual.glb")
        return trimesh.util.concatenate(list(sc.geometry.values()))
    m = trimesh.load(REPO / "assets/urdf/plug_fork/fork/fork_visual_oriented.obj",
                     process=False)
    import trimesh as _t
    if isinstance(m, _t.Scene):
        m = _t.util.concatenate(list(m.geometry.values()))
    return m


def dimension_panel(ax):
    """Both objects in ONE panel, drawn TO THE SAME SCALE.

    Separate tiles were worse in two ways: the fork is 7:1 so its tile was
    mostly white, and separate panels cannot show RELATIVE size, which is the
    entire content of "naturally sized". At one scale a 42 mm plug against a
    198 mm fork makes the point without a caption.

    The plug is projected on (X, Z), NOT (Y, Z): its blades are separated along
    X, so a (Y, Z) view collapses them into one shape and hides that this is a
    two-feature insertion.

    Everything is placed in explicit mm bands so labels cannot collide -- the
    first version let matplotlib place them and "198 mm", "1.5 blades" and
    "Apple 20 W" all landed on top of each other.
    """
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    fork, plug = _load_obj("fork"), _load_obj("plug")
    fe, pe = fork.bounds * 1000, plug.bounds * 1000
    fl = fe[1][0] - fe[0][0]
    fw = fe[1][1] - fe[0][1]
    pw = pe[1][0] - pe[0][0]
    ph = pe[1][2] - pe[0][2]

    FORK_Y = 78.0                      # bands, in mm, chosen so nothing overlaps
    _silhouette(ax, fork, (0, 1), "#b9bec9", dx=-fe[0][0], dy=FORK_Y - fe[0][1])
    _silhouette(ax, plug, (0, 2), "#b9bec9", dx=-pe[0][0], dy=-pe[0][2])

    # fork: label above, dimension below
    ax.text(0, FORK_Y + fw + 6, f"YCB fork (unmodified)   {fl:.0f} mm",
            fontsize=4.4, color=MUTED, va="bottom")
    _dim(ax, (0, FORK_Y - 12), (fl, FORK_Y - 12), "")
    # plug: dimension below, name + callout to the right
    _dim(ax, (0, -12), (pw, -12), f"{pw:.1f}")
    ax.text(pw + 8, ph * 0.72, "Apple 20 W adapter", fontsize=4.4, color=MUTED,
            va="center")
    ax.text(pw + 8, ph * 0.16, "1.5 mm blades", fontsize=4.3, color=C2, va="center")
    ax.plot([pw * 0.55, pw + 6], [-1, ph * 0.16], color=C2, lw=0.45)

    ax.set_xlim(-10, fl + 14)
    ax.set_ylim(-26, FORK_Y + fw + 20)


# --------------------------------------------------------------- fig 3
def fig3(W):
    """One to-scale dimension panel + a 2x2 grid of rollouts.

    Dimensions sit on the LEFT so scale is established before the rollouts: the
    objection is that the parts are enlarged, not that the policy fails.
    """
    fig = plt.figure(figsize=(0.62 * W, 1.30))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.30, 1.0, 1.0],
                          hspace=0.06, wspace=0.04)
    ax = fig.add_subplot(gs[:, 0])
    dimension_panel(ax)

    spec = [("plug", INPUTS / "plug_socket_placeholder"),
            ("fork", INPUTS / "fork_rack_placeholder")]
    for r, (key, roll_dir) in enumerate(spec):
        rolls = sorted(roll_dir.glob("rollout_img*.png"))
        for c in (1, 2):
            ax = fig.add_subplot(gs[r, c])
            idx = c - 1
            if idx < len(rolls):
                rim = plt.imread(rolls[idx])
                rh, rw = rim.shape[:2]
                ax.imshow(rim[int(0.20 * rh):int(0.86 * rh),
                              int(0.06 * rw):int(0.94 * rw)])
            else:
                placeholder(ax, f"{key}\nrollout {c}")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0 and c == 1:
                ax.set_title("plug into socket", fontsize=4.9, color=MUTED, pad=1.4,
                             loc="left")
            if r == 1 and c == 1:
                ax.text(0.02, 0.04, "fork into holder", transform=ax.transAxes,
                        fontsize=4.6, color="white",
                        bbox=dict(fc="black", alpha=0.45, pad=0.8, lw=0))
    save(fig, "fig3_novel_geometries")


# --------------------------------------------------------------- fig 4 / 5
def grouped(W, name, frac, series, title_a, title_b, values):
    """2 groups x 2 bars, direct-labelled, no legend box (no room at 1.65in)."""
    fig, ax = plt.subplots(figsize=(frac * W, 1.12))
    x = np.arange(2); bw = 0.34
    a, b = values
    ax.bar(x - bw / 2 - 0.01, a, bw, color=C1, lw=0)
    ax.bar(x + bw / 2 + 0.01, b, bw, color=C2, lw=0)
    # One decimal: at these margins the difference that matters is fractional --
    # co-trained Step 1 is 97.7 vs 98.0, which rounds to "98 vs 98" and reads as
    # no result at all.
    for xi, v in zip(x - bw / 2 - 0.01, a):
        ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                    xytext=(0, 1.5), ha="center", fontsize=4.7, color=INK)
    for xi, v in zip(x + bw / 2 + 0.01, b):
        ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                    xytext=(0, 1.5), ha="center", fontsize=4.7, color=INK)
    # direct labels on the first group only -- a legend box does not fit
    # Both labels sit over the FIRST group. Putting the second on group 2 left
    # it floating at whatever height that bar happened to reach, where it read
    # as annotating the axis rather than naming the series.
    ax.annotate(title_a, (x[0] - bw / 2 - 0.01, a[0]), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=4.9, color=C1)
    ax.annotate(title_b, (x[0] + bw / 2 + 0.01, a[0]), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=4.9, color=C2)
    ax.set_xticks(x); ax.set_xticklabels(["Step 1", "Step 2"])
    ax.set_ylabel("success (%)", labelpad=1.5)
    ax.set_ylim(0, 122); ax.set_yticks([0, 50, 100])
    ax.set_xlim(-0.6, 1.6)
    bare(ax)
    save(fig, name)


def layout_preview(W):
    """Composite the five figures into the actual page layout.

    Drafting each figure at final size is necessary but not sufficient -- what
    matters is whether they balance against each other and how much of the page
    is left for prose, which only shows when they are placed together.
    """
    from PIL import Image
    dpi = 200
    # Plots first, images second. The plot row is shorter, so leading with it
    # puts the page's densest quantitative content directly under the opening
    # text, and the taller image row sits above the prose block rather than
    # between it and the numbers.
    rows = [[("fig1_noise_and_failures", 0.40), ("fig4_unbolted_fixture", 0.30),
             ("fig5_unified_policy", 0.30)],
            [("fig3_novel_geometries", 0.62), ("fig2_end_to_end", 0.38)]]
    page_w = int(W * dpi)
    canvas_rows, y = [], 0
    tiles = []
    for row in rows:
        placed, hmax = [], 0
        x = 0
        for name, frac in row:
            im = Image.open(OUT / f"{name}.png")
            tw = int(frac * page_w)
            th = int(im.height * tw / im.width)
            placed.append((im.resize((tw, th), Image.LANCZOS), x, tw, th))
            x += tw
            hmax = max(hmax, th)
        tiles.append((placed, hmax))
        y += hmax + int(0.16 * dpi)          # caption + gap allowance
    page_h = int(9.0 * dpi)
    canvas = Image.new("RGB", (page_w, page_h), "white")
    yy = 0
    for placed, hmax in tiles:
        for im, x, tw, th in placed:
            canvas.paste(im, (x, yy))          # top-align, not centre
        yy += hmax + int(0.16 * dpi)

    fig, ax = plt.subplots(figsize=(W, 9.0))
    ax.imshow(canvas, extent=[0, W, 0, 9.0])
    used = yy / dpi
    ax.axhline(9.0 - used, color="#eb6834", lw=1.0, ls="--")
    ax.text(W / 2, 9.0 - used - 0.16,
            f"figures use {used:.2f} in   |   {9.0-used:.2f} in left "
            f"= ~{int((9.0-used)/0.165)} lines of prose",
            ha="center", fontsize=8, color="#eb6834")
    ax.set_xlim(0, W); ax.set_ylim(0, 9.0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#b9bec9"); sp.set_linewidth(0.8)
    ax.set_title(f"one-page rebuttal, textwidth {W} in", fontsize=9, color=INK)
    fig.savefig(OUT / "layout_preview.png", dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("  layout_preview")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--textwidth", type=float, default=5.5)
    args = ap.parse_args()
    W = args.textwidth
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"drafting at textwidth {W} in ->")
    fig1(W)
    fig2(W)
    fig3(W)
    grouped(W, "fig4_unbolted_fixture", 0.30, None, "bolted", "free",
            ([rate("beam1_bolted"), rate("beam2mass_bolted_690g")],
             # Step 2's free-fixture bar is the BALLASTED fixture (1.5 kg, COM
             # lowered to 20 mm), not the as-authored 0.69 kg one, which scores
             # 49.8%. Step 2's receptacle is a tall plate (COM 59 mm) that tips
             # on contact; Step 1's is squat (19 mm) and barely moves. Ballasting
             # is the fix and the caption must say the fixture was modified.
             [rate("beam1_unbolted"), rate("beam2mass_unbolted_1502g_lowcom")]))
    grouped(W, "fig5_unified_policy", 0.30, None, "single-task", "co-trained",
            ([rate("solo_unclamped_step1_part2"), rate("solo_unclamped_step2_part0")],
             [rate("cotrained_beam_step1_part2"), rate("cotrained_beam_step2_part0")]))
    layout_preview(W)


if __name__ == "__main__":
    main()

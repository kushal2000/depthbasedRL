"""Bolted vs unbolted fixture success rate, grouped by task.

Usage:
    python plot_figures/rebuttal/plot_fixture_bars.py \
        --task "Peg:/path/bolted.json:/path/unbolted.json" \
        --task "Beam part 0:/path/b.json:/path/u.json"

Each JSON is an offline_eval_teacher_robustness.py output; the plotted number is
`early_drop_filtered.retract_rate` (the reportable success rate).
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

# Categorical slots 1-2 of the validated theme. Checked with the six-checks
# validator on all pairs (light): CVD worst protan/deutan dE 9.2 (target 8),
# normal-vision worst 24.0 (floor 15), both above the chroma floor.
C_BOLTED = "#2a78d6"
C_UNBOLTED = "#eb6834"
INK = "#333333"
MUTED = "#777777"


def _load(path: str | None):
    if not path:
        return None
    blob = json.load(open(path))
    return blob["results"][0], blob["num_envs"]


def _rates(bolted: str, unbolted: str | None):
    """Each condition uses its OWN post-early-drop denominator, so the plotted
    number equals what the eval logs report.

    This previously pinned both bars to the bolted run's denominator, on the
    reasoning that unstable initial placement belongs to the reset distribution
    and filtering the unbolted run by its own (assumed larger) drop count would
    credit it for failures the free fixture caused. Measurement went the other
    way: on beam Step 2 the bolted run dropped 82 envs at reset while the
    unbolted arms dropped 36-45, so the shared denominator inflated the unbolted
    bar by 4-6 points (5 kg read 94.4% against a true 88.1%) -- the opposite of
    the intent. Per-run denominators are the conservative reading and keep the
    figure consistent with every number in the logs.

    Read `early_drop_filtered.retract_rate` straight from the JSON rather than
    recomputing a denominator here -- an earlier attempt used
    `num_envs - n_dropped_early`, which is NOT that field's denominator: it is
    `num_envs - n_dropped_early - unfinished_envs`, so the plot silently
    disagreed with the logs by a few points.

    CAVEAT worth carrying into the caption: that denominator also excludes
    budget-censored envs (`unfinished_envs`), which is optimistic -- an env
    still running at the step cap is more likely a failure than a success. It
    matters unevenly across conditions: bolted censored 0, the unbolted arms
    6-21."""
    b, n_env = _load(bolted)
    br = 100.0 * b["early_drop_filtered"]["retract_rate"]
    b_n = (b["early_drop_filtered"]["n"], b["unfinished_envs"])
    ur = u_n = None
    if (u := _load(unbolted)) is not None:
        ur = 100.0 * u[0]["early_drop_filtered"]["retract_rate"]
        u_n = (u[0]["early_drop_filtered"]["n"], u[0]["unfinished_envs"])
    return br, ur, b_n, u_n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", required=True,
                    metavar="NAME:BOLTED_JSON[:UNBOLTED_JSON]")
    ap.add_argument("--name", default="fixture_bolted_vs_unbolted")
    ap.add_argument("--title",
                    default="Fixture bolted vs unbolted (no retraining)")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "outputs"))
    args = ap.parse_args()

    tasks = []
    for spec in args.task:
        parts = spec.split(":")
        # allow "\n" in a --task name to wrap the tick label
        name, bolted = parts[0].replace("\\n", "\n"), parts[1]
        unbolted = parts[2] if len(parts) > 2 and parts[2] else None
        b, u, b_n, u_n = _rates(bolted, unbolted)
        tasks.append({"name": name, "bolted": b, "unbolted": u,
                      "n": b_n, "nu": u_n})

    configure_rcparams()
    width = max(4.2, 1.9 * len(tasks) + 1.6)
    fig, ax = plt.subplots(figsize=(width, 3.8))

    bw = 0.34
    gap = 0.02  # 2px-ish surface gap between adjacent bars
    for i, t in enumerate(tasks):
        for val, off, color in ((t["bolted"], -(bw / 2 + gap / 2), C_BOLTED),
                                (t["unbolted"], +(bw / 2 + gap / 2), C_UNBOLTED)):
            if val is None:
                continue
            ax.bar(i + off, val, width=bw, color=color, linewidth=0, zorder=2)
            ax.annotate(f"{val:.1f}", (i + off, val), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8.5, color=INK)
        if t["unbolted"] is None:
            ax.annotate("unbolted\npending", (i + bw / 2 + gap / 2, 2),
                        ha="center", va="bottom", fontsize=8, color=MUTED,
                        multialignment="center")

    # Two series -> legend is mandatory. Use explicit Patch handles: an empty
    # bar() container does not carry its facecolor into the legend, which
    # silently renders both swatches the same colour.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C_BOLTED, label="Bolted (fixed fixture)"),
                       Patch(facecolor=C_UNBOLTED, label="Unbolted (free fixture)")],
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=False, fontsize=9, handlelength=1.1, handletextpad=0.5,
              columnspacing=1.4)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t["name"] for t in tasks])
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)])
    ax.set_xlim(-0.6, len(tasks) - 0.4)
    ax.set_title(args.title, loc="left", fontsize=10.5, pad=26)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelcolor=INK)

    fig.tight_layout()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{args.name}.{ext}", dpi=600, facecolor="white",
                    bbox_inches="tight", pad_inches=0.06)
    print(f"wrote {out / f'{args.name}.png'}")
    for t in tasks:
        def _f(rate, nn):
            if rate is None:
                return "pending"
            n, unf = nn
            return f"{rate:6.2f} (n={n}, censored={unf})"
        print(f"  {t['name']:22s} bolted={_f(t['bolted'], t['n'])}  "
              f"unbolted={_f(t['unbolted'], t['nu'])}")


if __name__ == "__main__":
    main()

"""Diff + plot the isaacgym vs isaacsim sine-hand rollouts.

Loads ``data/isaacgym_sine_hand.npz`` and ``data/isaacsim_sine_hand.npz``,
aligns by joint name (legacy is DFS canonical, Lab is BFS — same 29 joints,
different order), and writes:

  - ``plots/sine_hand_overlay.png`` — 5×6 grid of per-joint overlays
    (target / isaacgym q / isaacsim q vs time)
  - ``plots/sine_hand_summary.csv`` — per-joint mean & max |q_gym - q_sim|
  - prints the same summary to stdout, sorted by max absolute error.

    python debug_differences/plot_sine_hand_diff.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    npz = np.load(path, allow_pickle=True)
    return {
        "joint_names": [str(x) for x in npz["joint_names"]],
        "joint_pos": npz["joint_pos"],
        "joint_vel": npz["joint_vel"],
        "target": npz["target"],
        "time": npz["time"],
    }


def _reorder(data: dict, target_names: list[str]) -> dict:
    src_names = data["joint_names"]
    perm = np.array([src_names.index(n) for n in target_names], dtype=np.int64)
    return {
        "joint_names": list(target_names),
        "joint_pos": data["joint_pos"][:, perm],
        "joint_vel": data["joint_vel"][:, perm],
        "target": data["target"][:, perm],
        "time": data["time"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym_npz", default=str(REPO_ROOT / "debug_differences/data/isaacgym_sine_hand.npz"))
    parser.add_argument("--sim_npz", default=str(REPO_ROOT / "debug_differences/data/isaacsim_sine_hand.npz"))
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "debug_differences/plots"))
    args = parser.parse_args()

    gym_data = _load(Path(args.gym_npz))
    sim_data = _load(Path(args.sim_npz))

    canonical_names = gym_data["joint_names"]
    assert set(canonical_names) == set(sim_data["joint_names"]), (
        "joint name mismatch — different robots?"
    )
    sim_data = _reorder(sim_data, canonical_names)

    n_steps = min(gym_data["joint_pos"].shape[0], sim_data["joint_pos"].shape[0])
    t_axis = gym_data["time"][:n_steps]

    diff = gym_data["joint_pos"][:n_steps] - sim_data["joint_pos"][:n_steps]
    abs_err = np.abs(diff)
    mean_err = abs_err.mean(axis=0)
    max_err = abs_err.max(axis=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV summary ---
    csv_path = out_dir / "sine_hand_summary.csv"
    rows = sorted(
        zip(canonical_names, mean_err, max_err),
        key=lambda r: r[2],
        reverse=True,
    )
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["joint", "mean_abs_err_rad", "max_abs_err_rad"])
        for name, m, mx in rows:
            w.writerow([name, f"{m:.6f}", f"{mx:.6f}"])
    print(f"[diff] wrote {csv_path}")
    print(f"{'joint':<28s}  {'mean|err| (rad)':>18s}  {'max|err| (rad)':>18s}")
    for name, m, mx in rows:
        print(f"{name:<28s}  {m:>18.6f}  {mx:>18.6f}")

    # --- Per-joint overlay grid: all 29 DOFs (7 arm + 22 hand) ---
    rows_n, cols_n = 5, 6  # 30 cells for 29 joints
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n * 3.0, rows_n * 2.2),
                             sharex=True, constrained_layout=True)
    axes = axes.flatten()
    for i, name in enumerate(canonical_names):
        idx = canonical_names.index(name)
        ax = axes[i]
        ax.plot(t_axis, gym_data["target"][:n_steps, idx],
                "k:", lw=1, label="target")
        ax.plot(t_axis, gym_data["joint_pos"][:n_steps, idx],
                "C0", lw=1.2, label="isaacgym")
        ax.plot(t_axis, sim_data["joint_pos"][:n_steps, idx],
                "C1", lw=1.2, label="isaacsim")
        title_color = "tab:red" if name.startswith("iiwa14") else "black"
        ax.set_title(name, fontsize=8, color=title_color)
        ax.tick_params(labelsize=7)
    for j in range(len(canonical_names), len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.supxlabel("t (s)")
    fig.supylabel("joint position (rad)")
    png_path = out_dir / "sine_hand_overlay.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"[diff] wrote {png_path}")


if __name__ == "__main__":
    main()

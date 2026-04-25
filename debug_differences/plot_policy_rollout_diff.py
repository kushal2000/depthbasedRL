"""Diff + plot the isaacgym vs isaacsim pretrained-policy rollouts.

Loads the two npz traces produced by ``policy_rollout_isaacgym.py`` and
``policy_rollout_isaacsim.py``, aligns by joint name, normalizes
quaternion conventions, and writes:

  - ``plots/policy_joint_overlay.png`` — 5×6 grid: target / gym q / sim q
    vs time per DOF.
  - ``plots/policy_action_overlay.png`` — 5×6 grid: action_gym / action_sim
    per DOF (closed-loop divergence diagnostic).
  - ``plots/policy_object_traj.png`` — object & goal xyz trajectories +
    per-step ‖obj_pos_gym - obj_pos_sim‖ + quat-angle error.
  - ``plots/policy_reward_curve.png`` — reward_gym vs reward_sim vs t.
  - ``plots/policy_summary.csv`` — per-channel mean/max abs error.

Quaternion conventions: gym side uses xyzw, sim side uses wxyz. We
convert sim → xyzw at load time so all internal math is xyzw.

    python debug_differences/plot_policy_rollout_diff.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, sim_quat_wxyz: bool) -> dict:
    npz = np.load(path, allow_pickle=True)
    object_state = npz["object_state"].copy()
    goal_pose = npz["goal_pose"].copy()
    if sim_quat_wxyz:
        # Convert wxyz → xyzw so both sides share gym's convention.
        object_state[:, 3:7] = object_state[:, [4, 5, 6, 3]]
        goal_pose[:, 3:7] = goal_pose[:, [4, 5, 6, 3]]
    return {
        "joint_names": [str(x) for x in npz["joint_names"]],
        "obs": npz["obs"],
        "action": npz["action"],
        "joint_pos": npz["joint_pos"],
        "joint_vel": npz["joint_vel"],
        "joint_targets": npz["joint_targets"],
        "object_state": object_state,
        "goal_pose": goal_pose,
        "reward": npz["reward"],
    }


def _reorder(data: dict, target_names: list[str]) -> dict:
    src_names = data["joint_names"]
    perm = np.array([src_names.index(n) for n in target_names], dtype=np.int64)
    out = {k: v for k, v in data.items()}
    out["joint_names"] = list(target_names)
    for key in ("action", "joint_pos", "joint_vel", "joint_targets"):
        out[key] = data[key][:, perm]
    return out


def _quat_angle_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Smallest rotation angle (rad) between two quaternion sequences."""
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym_npz", default=str(REPO_ROOT / "debug_differences/data/isaacgym_policy_rollout.npz"))
    parser.add_argument("--sim_npz", default=str(REPO_ROOT / "debug_differences/data/isaacsim_policy_rollout.npz"))
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "debug_differences/plots"))
    parser.add_argument("--policy_dt", type=float, default=1.0 / 60.0)
    args = parser.parse_args()

    gym_data = _load(Path(args.gym_npz), sim_quat_wxyz=False)
    sim_data = _load(Path(args.sim_npz), sim_quat_wxyz=True)

    canonical_names = gym_data["joint_names"]
    assert set(canonical_names) == set(sim_data["joint_names"]), (
        "joint name mismatch — different robots?"
    )
    sim_data = _reorder(sim_data, canonical_names)

    n_steps = min(gym_data["joint_pos"].shape[0], sim_data["joint_pos"].shape[0])
    t_axis = np.arange(n_steps) * args.policy_dt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Per-joint q overlay grid ---
    rows_n, cols_n = 5, 6
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n * 3.0, rows_n * 2.2),
                             sharex=True, constrained_layout=True)
    axes = axes.flatten()
    for i, name in enumerate(canonical_names):
        ax = axes[i]
        ax.plot(t_axis, gym_data["joint_targets"][:n_steps, i],
                "k:", lw=1, label="target_gym")
        ax.plot(t_axis, gym_data["joint_pos"][:n_steps, i],
                "C0", lw=1.2, label="isaacgym")
        ax.plot(t_axis, sim_data["joint_pos"][:n_steps, i],
                "C1", lw=1.2, label="isaacsim")
        title_color = "tab:red" if name.startswith("iiwa14") else "black"
        ax.set_title(name, fontsize=8, color=title_color)
        ax.tick_params(labelsize=7)
    for j in range(len(canonical_names), len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.supxlabel("t (s)")
    fig.supylabel("joint position (rad)")
    png_path = out_dir / "policy_joint_overlay.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"[diff] wrote {png_path}")

    # --- 2. Per-joint action overlay grid ---
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n * 3.0, rows_n * 2.2),
                             sharex=True, constrained_layout=True)
    axes = axes.flatten()
    for i, name in enumerate(canonical_names):
        ax = axes[i]
        ax.plot(t_axis, gym_data["action"][:n_steps, i],
                "C0", lw=1.2, label="isaacgym")
        ax.plot(t_axis, sim_data["action"][:n_steps, i],
                "C1", lw=1.2, label="isaacsim")
        title_color = "tab:red" if name.startswith("iiwa14") else "black"
        ax.set_title(name, fontsize=8, color=title_color)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-1.1, 1.1)
    for j in range(len(canonical_names), len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.supxlabel("t (s)")
    fig.supylabel("normalized action [-1, 1]")
    png_path = out_dir / "policy_action_overlay.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"[diff] wrote {png_path}")

    # --- 3. Object + goal trajectory ---
    obj_pos_gym = gym_data["object_state"][:n_steps, :3]
    obj_pos_sim = sim_data["object_state"][:n_steps, :3]
    obj_quat_gym = gym_data["object_state"][:n_steps, 3:7]
    obj_quat_sim = sim_data["object_state"][:n_steps, 3:7]
    goal_pos_gym = gym_data["goal_pose"][:n_steps, :3]
    goal_pos_sim = sim_data["goal_pose"][:n_steps, :3]

    pos_l2 = np.linalg.norm(obj_pos_gym - obj_pos_sim, axis=-1)
    quat_err = _quat_angle_xyzw(obj_quat_gym, obj_quat_sim)
    goal_pos_l2 = np.linalg.norm(goal_pos_gym - goal_pos_sim, axis=-1)

    fig, axes = plt.subplots(3, 2, figsize=(11.0, 8.0), constrained_layout=True)
    for axis, label in enumerate("xyz"):
        axes[axis, 0].plot(t_axis, obj_pos_gym[:, axis], "C0", lw=1.2, label="gym")
        axes[axis, 0].plot(t_axis, obj_pos_sim[:, axis], "C1", lw=1.2, label="sim")
        axes[axis, 0].plot(t_axis, goal_pos_gym[:, axis], "C0--", lw=0.8, label="goal_gym")
        axes[axis, 0].plot(t_axis, goal_pos_sim[:, axis], "C1--", lw=0.8, label="goal_sim")
        axes[axis, 0].set_ylabel(f"{label} (m)")
        axes[axis, 0].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="best")
    axes[0, 0].set_title("Object & goal position vs t")
    axes[2, 0].set_xlabel("t (s)")

    axes[0, 1].plot(t_axis, pos_l2, "k", lw=1.2)
    axes[0, 1].set_ylabel("‖Δobj_pos‖ (m)")
    axes[0, 1].set_title("Per-step gym↔sim object divergence")
    axes[0, 1].grid(alpha=0.3)
    axes[1, 1].plot(t_axis, np.degrees(quat_err), "k", lw=1.2)
    axes[1, 1].set_ylabel("Δobj_rot (deg)")
    axes[1, 1].grid(alpha=0.3)
    axes[2, 1].plot(t_axis, goal_pos_l2, "k", lw=1.2)
    axes[2, 1].set_ylabel("‖Δgoal_pos‖ (m)")
    axes[2, 1].set_xlabel("t (s)")
    axes[2, 1].grid(alpha=0.3)

    png_path = out_dir / "policy_object_traj.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"[diff] wrote {png_path}")

    # --- 4. Reward curve ---
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.0), constrained_layout=True)
    ax.plot(t_axis, gym_data["reward"][:n_steps], "C0", lw=1.2, label="isaacgym")
    ax.plot(t_axis, sim_data["reward"][:n_steps], "C1", lw=1.2, label="isaacsim")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("reward")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    png_path = out_dir / "policy_reward_curve.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"[diff] wrote {png_path}")

    # --- 5. Summary CSV ---
    rows: list[tuple[str, float, float]] = []
    for i, name in enumerate(canonical_names):
        d = gym_data["joint_pos"][:n_steps, i] - sim_data["joint_pos"][:n_steps, i]
        rows.append((f"q[{name}]", float(np.abs(d).mean()), float(np.abs(d).max())))
    for i, name in enumerate(canonical_names):
        d = gym_data["action"][:n_steps, i] - sim_data["action"][:n_steps, i]
        rows.append((f"action[{name}]", float(np.abs(d).mean()), float(np.abs(d).max())))
    for axis, label in enumerate("xyz"):
        d = obj_pos_gym[:, axis] - obj_pos_sim[:, axis]
        rows.append((f"obj_pos[{label}]", float(np.abs(d).mean()), float(np.abs(d).max())))
    rows.append(("obj_pos_l2", float(pos_l2.mean()), float(pos_l2.max())))
    rows.append(("obj_rot_rad", float(quat_err.mean()), float(quat_err.max())))
    d_reward = gym_data["reward"][:n_steps] - sim_data["reward"][:n_steps]
    rows.append(("reward", float(np.abs(d_reward).mean()), float(np.abs(d_reward).max())))
    rows.sort(key=lambda r: r[2], reverse=True)

    csv_path = out_dir / "policy_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "mean_abs_err", "max_abs_err"])
        for name, m, mx in rows:
            w.writerow([name, f"{m:.6f}", f"{mx:.6f}"])
    print(f"[diff] wrote {csv_path}")
    print(f"{'channel':<32s}  {'mean|err|':>14s}  {'max|err|':>14s}")
    for name, m, mx in rows[:20]:
        print(f"{name:<32s}  {m:>14.6f}  {mx:>14.6f}")


if __name__ == "__main__":
    main()

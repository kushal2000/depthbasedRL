"""Aggregate-stats diff for the multi-env policy eval.

Loads the (T, N) traces from ``policy_eval_isaacgym.py`` and
``policy_eval_isaacsim.py`` and overlays distributional metrics:

  - ``reward.png``         — mean reward / step (mean ± std band).
  - ``success_rate.png``   — fraction of envs with ≥1 hit by step t,
                                         + per-step hit rate.
  - ``obj_to_goal.png``    — mean obj→goal dist / step.
  - ``lifted.png``         — fraction lifted / step.
  - ``episodes.png``       — first-reset-step CDF (proxy for episode
                                         length distribution under natural resets).
  - ``per_asset.png``      — per-asset-type bars (mean reward,
                                         lift rate, hit rate, mean obj-to-goal).
  - ``summary.csv``        — top-level aggregate metrics.

Both backends sample independent goal sequences (random per reset on each
side), so per-step trajectory overlay is meaningless — only aggregate
statistics over many envs / steps are interpretable.

    python debug_differences/plot_policy_eval_diff.py
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

# Both backends embed the handle-head type as token 1 of the filename:
# legacy ``{idx:03d}_{type}_handle_head_...``, sim ``{idx:03d}_{type}_handle_...``.
# Re-derive from asset_paths so an npz saved with an older driver (which
# tagged everything ``"unknown"`` because its regex required ``_handle_head_``)
# still reports a useful per-asset breakdown.
_TYPE_RE = re.compile(r"^\d{3}_([a-zA-Z]+)_handle")


def _asset_type_from_path(path: str) -> str:
    m = _TYPE_RE.match(Path(path).name)
    return m.group(1) if m else "unknown"


def _load(path: Path) -> dict:
    npz = np.load(path, allow_pickle=True)
    asset_paths = [str(p) for p in npz["asset_paths"]]
    # Always re-derive from paths — npz-saved tags may be stale.
    asset_types = [_asset_type_from_path(p) for p in asset_paths]
    return {
        "reward": npz["reward"],
        "is_success": npz["is_success"],
        "reset": npz["reset"],
        "progress": npz["progress"],
        "successes": npz["successes"],
        "lifted_object": npz["lifted_object"],
        "obj_pos_world": npz["obj_pos_world"],
        "goal_pos_world": npz["goal_pos_world"],
        "obj_to_goal_dist": npz["obj_to_goal_dist"],
        "asset_types": asset_types,
        "asset_paths": asset_paths,
    }


def _first_reset_step(reset_log: np.ndarray) -> np.ndarray:
    """For each env (column), return the step at which it first reset, or T
    if it never reset."""
    T, N = reset_log.shape
    out = np.full(N, T, dtype=np.int64)
    for i in range(N):
        idx = np.flatnonzero(reset_log[:, i])
        if idx.size > 0:
            out[i] = idx[0]
    return out


def _band(ax, t, x_per_env, color, label):
    """Mean line + ±std band across envs (axis=1)."""
    mean = x_per_env.mean(axis=1)
    std = x_per_env.std(axis=1)
    ax.plot(t, mean, color=color, lw=1.4, label=label)
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.18)


def _per_asset_bar(ax, gym_data, sim_data, key, agg, ylabel):
    """Bar chart: aggregate `key` per asset_type, gym vs sim side by side.

    `agg(arr_per_env_for_type)` reduces a (T, n_assets_of_this_type) slice
    to a scalar (typically mean over T then mean over envs).
    """
    types = sorted(set(gym_data["asset_types"]) | set(sim_data["asset_types"]))
    gym_vals = []
    sim_vals = []
    for tname in types:
        gym_idx = [i for i, t in enumerate(gym_data["asset_types"]) if t == tname]
        sim_idx = [i for i, t in enumerate(sim_data["asset_types"]) if t == tname]
        gym_vals.append(agg(gym_data[key][:, gym_idx]) if gym_idx else float("nan"))
        sim_vals.append(agg(sim_data[key][:, sim_idx]) if sim_idx else float("nan"))
    x = np.arange(len(types))
    w = 0.4
    ax.bar(x - w / 2, gym_vals, width=w, color="C0", label="isaacgym")
    ax.bar(x + w / 2, sim_vals, width=w, color="C1", label="isaacsim")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym_npz", default=str(REPO_ROOT / "debug_differences/data/policy_eval/isaacgym_policy_eval.npz"))
    parser.add_argument("--sim_npz", default=str(REPO_ROOT / "debug_differences/data/policy_eval/isaacsim_policy_eval.npz"))
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "debug_differences/plots/policy_eval"))
    parser.add_argument("--policy_dt", type=float, default=1.0 / 60.0)
    args = parser.parse_args()

    gym_data = _load(Path(args.gym_npz))
    sim_data = _load(Path(args.sim_npz))

    T_gym, N_gym = gym_data["reward"].shape
    T_sim, N_sim = sim_data["reward"].shape
    T = min(T_gym, T_sim)
    t_axis = np.arange(T) * args.policy_dt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Mean reward over time (mean ± std band across envs) ---
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.4), constrained_layout=True)
    _band(ax, t_axis, gym_data["reward"][:T], "C0", f"isaacgym (N={N_gym})")
    _band(ax, t_axis, sim_data["reward"][:T], "C1", f"isaacsim (N={N_sim})")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("reward (mean ± std over envs)")
    ax.set_title("Per-step reward — distributional eval")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out_dir / "reward.png", dpi=140)
    plt.close(fig)
    print(f"[diff] wrote {out_dir / 'reward.png'}")

    # --- 2. Success rate (fraction of envs that have hit ≥1 goal by step t)
    #         + per-step hit rate (fraction of envs hitting on this step) ---
    has_hit_gym = (np.cumsum(gym_data["is_success"][:T], axis=0) > 0).astype(np.float32)
    has_hit_sim = (np.cumsum(sim_data["is_success"][:T], axis=0) > 0).astype(np.float32)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True, constrained_layout=True)
    axes[0].plot(t_axis, has_hit_gym.mean(axis=1), "C0", lw=1.4, label="isaacgym")
    axes[0].plot(t_axis, has_hit_sim.mean(axis=1), "C1", lw=1.4, label="isaacsim")
    axes[0].set_ylabel("frac envs with ≥1 hit")
    axes[0].set_title("Cumulative success rate")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3); axes[0].set_ylim(-0.02, 1.02)
    axes[1].plot(t_axis, gym_data["is_success"][:T].mean(axis=1), "C0", lw=1.0, label="isaacgym")
    axes[1].plot(t_axis, sim_data["is_success"][:T].mean(axis=1), "C1", lw=1.0, label="isaacsim")
    axes[1].set_ylabel("frac envs hitting this step")
    axes[1].set_xlabel("t (s)")
    axes[1].set_title("Per-step hit rate")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.savefig(out_dir / "success_rate.png", dpi=140)
    plt.close(fig)
    print(f"[diff] wrote {out_dir / 'success_rate.png'}")

    # --- 3. Mean obj→goal distance over time (mean ± std) ---
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.4), constrained_layout=True)
    _band(ax, t_axis, gym_data["obj_to_goal_dist"][:T], "C0", "isaacgym")
    _band(ax, t_axis, sim_data["obj_to_goal_dist"][:T], "C1", "isaacsim")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("‖obj − goal‖ (m)")
    ax.set_title("Object→goal distance (mean ± std across envs)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.savefig(out_dir / "obj_to_goal.png", dpi=140)
    plt.close(fig)
    print(f"[diff] wrote {out_dir / 'obj_to_goal.png'}")

    # --- 4. Lift fraction over time ---
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.0), constrained_layout=True)
    ax.plot(t_axis, gym_data["lifted_object"][:T].mean(axis=1), "C0", lw=1.4, label="isaacgym")
    ax.plot(t_axis, sim_data["lifted_object"][:T].mean(axis=1), "C1", lw=1.4, label="isaacsim")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("frac envs with lifted_object=True")
    ax.set_title("Lift rate over time")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02)
    fig.savefig(out_dir / "lifted.png", dpi=140)
    plt.close(fig)
    print(f"[diff] wrote {out_dir / 'lifted.png'}")

    # --- 5. First-reset-step CDF (proxy for episode length under natural resets) ---
    first_reset_gym = _first_reset_step(gym_data["reset"][:T])
    first_reset_sim = _first_reset_step(sim_data["reset"][:T])
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.4), constrained_layout=True)
    sorted_gym = np.sort(first_reset_gym) * args.policy_dt
    sorted_sim = np.sort(first_reset_sim) * args.policy_dt
    cdf_gym = np.arange(1, len(sorted_gym) + 1) / len(sorted_gym)
    cdf_sim = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim)
    ax.step(sorted_gym, cdf_gym, "C0", lw=1.4, where="post", label="isaacgym")
    ax.step(sorted_sim, cdf_sim, "C1", lw=1.4, where="post", label="isaacsim")
    ax.set_xlabel("first-reset time (s); ≥T means env never reset in window")
    ax.set_ylabel("CDF over envs")
    ax.set_title("First-reset-step distribution")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02)
    fig.savefig(out_dir / "episodes.png", dpi=140)
    plt.close(fig)
    print(f"[diff] wrote {out_dir / 'episodes.png'}")

    # --- 6. Per-asset-type breakdown (4 bars per type) ---
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.0), constrained_layout=True)
    _per_asset_bar(
        axes[0, 0], gym_data, sim_data, "reward",
        agg=lambda x: float(x.mean()),
        ylabel="mean reward",
    )
    axes[0, 0].set_title("Per-asset mean reward")
    _per_asset_bar(
        axes[0, 1], gym_data, sim_data, "lifted_object",
        agg=lambda x: float(x.mean()),
        ylabel="lifted fraction",
    )
    axes[0, 1].set_title("Per-asset lift fraction (over T × N_type)")
    _per_asset_bar(
        axes[1, 0], gym_data, sim_data, "is_success",
        agg=lambda x: float(x.sum() / x.shape[1]) if x.size else float("nan"),
        ylabel="goal hits / env",
    )
    axes[1, 0].set_title("Per-asset total goal hits / env")
    _per_asset_bar(
        axes[1, 1], gym_data, sim_data, "obj_to_goal_dist",
        agg=lambda x: float(x.mean()),
        ylabel="mean ‖obj − goal‖ (m)",
    )
    axes[1, 1].set_title("Per-asset mean obj→goal distance")
    fig.savefig(out_dir / "per_asset.png", dpi=140)
    plt.close(fig)
    print(f"[diff] wrote {out_dir / 'per_asset.png'}")

    # --- 7. Per-env scatter (gym vs sim, one point per env) ---
    #     Asset shape is byte-identical per env across backends, but goal
    #     sequences differ — so a 45° line is not the expected outcome,
    #     just a useful reference. Big outliers from y=x flag asset shapes
    #     where the new physics meaningfully changes policy performance.
    #     Skipped when env counts differ (per-env pairing breaks down).
    if N_gym != N_sim:
        print(f"[diff] WARN: gym N={N_gym} ≠ sim N={N_sim} → "
              f"skipping per-env scatter + CSV (need matching env counts).")
    else:
        types = sorted(set(gym_data["asset_types"]) | set(sim_data["asset_types"]))
        type_to_color = {t: f"C{i}" for i, t in enumerate(types)}
        asset_types_per_env = gym_data["asset_types"]

        total_reward_gym = gym_data["reward"][:T].sum(axis=0)
        total_reward_sim = sim_data["reward"][:T].sum(axis=0)
        total_hits_gym = gym_data["is_success"][:T].sum(axis=0)
        total_hits_sim = sim_data["is_success"][:T].sum(axis=0)
        mean_lift_gym = gym_data["lifted_object"][:T].mean(axis=0)
        mean_lift_sim = sim_data["lifted_object"][:T].mean(axis=0)
        mean_otg_gym = gym_data["obj_to_goal_dist"][:T].mean(axis=0)
        mean_otg_sim = sim_data["obj_to_goal_dist"][:T].mean(axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0), constrained_layout=True)

        def _scatter(ax, x, y, title, xlabel, ylabel):
            for i in range(len(x)):
                t = asset_types_per_env[i] if i < len(asset_types_per_env) else "unknown"
                ax.scatter(x[i], y[i], c=type_to_color.get(t, "gray"),
                           s=22, alpha=0.85, edgecolor="k", linewidths=0.3)
            lo, hi = min(np.min(x), np.min(y)), max(np.max(x), np.max(y))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.6)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.grid(alpha=0.3)

        _scatter(axes[0, 0], total_reward_gym, total_reward_sim,
                 "Per-env total reward", "isaacgym", "isaacsim")
        _scatter(axes[0, 1], total_hits_gym, total_hits_sim,
                 "Per-env total goal hits", "isaacgym", "isaacsim")
        _scatter(axes[1, 0], mean_lift_gym, mean_lift_sim,
                 "Per-env mean lift fraction", "isaacgym", "isaacsim")
        _scatter(axes[1, 1], mean_otg_gym, mean_otg_sim,
                 "Per-env mean ‖obj − goal‖", "isaacgym", "isaacsim")
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=type_to_color[t],
                              label=t, markeredgecolor="k", markeredgewidth=0.3)
                   for t in types]
        axes[0, 0].legend(handles=handles, fontsize=8, loc="best")
        fig.savefig(out_dir / "per_env.png", dpi=140)
        plt.close(fig)
        print(f"[diff] wrote {out_dir / 'per_env.png'}")

        per_env_csv = out_dir / "per_env.csv"
        with per_env_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "env_id", "asset_type",
                "total_reward_gym", "total_reward_sim", "delta_reward",
                "total_hits_gym", "total_hits_sim", "delta_hits",
                "mean_lift_gym", "mean_lift_sim",
                "mean_otg_gym", "mean_otg_sim",
            ])
            for i in range(len(total_reward_gym)):
                t = asset_types_per_env[i] if i < len(asset_types_per_env) else "unknown"
                w.writerow([
                    i, t,
                    f"{total_reward_gym[i]:.3f}", f"{total_reward_sim[i]:.3f}",
                    f"{total_reward_sim[i] - total_reward_gym[i]:.3f}",
                    int(total_hits_gym[i]), int(total_hits_sim[i]),
                    int(total_hits_sim[i]) - int(total_hits_gym[i]),
                    f"{mean_lift_gym[i]:.4f}", f"{mean_lift_sim[i]:.4f}",
                    f"{mean_otg_gym[i]:.4f}", f"{mean_otg_sim[i]:.4f}",
                ])
        print(f"[diff] wrote {per_env_csv}")

    # --- 8. Summary CSV: top-level aggregates ---
    rows: list[tuple[str, float, float, float]] = []  # (metric, gym, sim, abs_diff)

    def add(name: str, g: float, s: float) -> None:
        rows.append((name, float(g), float(s), float(abs(g - s))))

    add("mean_reward",
        gym_data["reward"][:T].mean(), sim_data["reward"][:T].mean())
    add("total_goal_hits",
        gym_data["is_success"][:T].sum(), sim_data["is_success"][:T].sum())
    add("hits_per_env",
        gym_data["is_success"][:T].sum() / N_gym,
        sim_data["is_success"][:T].sum() / N_sim)
    add("frac_envs_with_hit",
        (gym_data["is_success"][:T].sum(axis=0) > 0).mean(),
        (sim_data["is_success"][:T].sum(axis=0) > 0).mean())
    add("frac_envs_ever_lifted",
        (gym_data["lifted_object"][:T].sum(axis=0) > 0).mean(),
        (sim_data["lifted_object"][:T].sum(axis=0) > 0).mean())
    add("mean_lift_rate",
        gym_data["lifted_object"][:T].mean(),
        sim_data["lifted_object"][:T].mean())
    add("total_resets",
        gym_data["reset"][:T].sum(), sim_data["reset"][:T].sum())
    add("frac_envs_ever_reset",
        (gym_data["reset"][:T].sum(axis=0) > 0).mean(),
        (sim_data["reset"][:T].sum(axis=0) > 0).mean())
    add("mean_obj_to_goal",
        gym_data["obj_to_goal_dist"][:T].mean(),
        sim_data["obj_to_goal_dist"][:T].mean())
    add("final_obj_to_goal",
        gym_data["obj_to_goal_dist"][T - 1].mean(),
        sim_data["obj_to_goal_dist"][T - 1].mean())

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "isaacgym", "isaacsim", "abs_diff"])
        for name, g, s, d in rows:
            w.writerow([name, f"{g:.6f}", f"{s:.6f}", f"{d:.6f}"])
    print(f"[diff] wrote {csv_path}")
    print(f"{'metric':<28s}  {'isaacgym':>14s}  {'isaacsim':>14s}  {'|diff|':>14s}")
    for name, g, s, d in rows:
        print(f"{name:<28s}  {g:>14.4f}  {s:>14.4f}  {d:>14.4f}")


if __name__ == "__main__":
    main()

"""Peg-in-hole-fixtured task.

Same RL interface as ``PegInHoleEnv``, but the scene includes a second
hole-fixture body (``peg_fixture``) on the table. Per-env start / goal poses
and the goal-trajectory waypoints are loaded from
``assets/urdf/peg_in_hole_fixtured/scenes/scenes.npz`` (ported from the
IsaacGym branch ``2026_04_28_peg_in_hole_fixtured``).

Supported goal modes (cfg.peg_in_hole.goal_mode):
  * ``preInsertAndFinal`` — last two scenes.npz waypoints per env. Matches the
    Play2Perfect IsaacGym ckpt's training.
  * ``dense`` — full trajectory from scenes.npz (up to ``traj_lengths[s, p]``
    waypoints). Matches the Scratch (dense reward) IsaacGym ckpt's training.
  * ``finalGoalOnly`` — last scenes.npz waypoint only.

The peg starts at ``start_poses[scene, peg]`` and the goal-hole at
``hole_positions[scene]``; the peg-holding fixture is placed at
``start_fixture_positions[scene, peg]``. Per-env scene/peg indices are
round-robin assigned at init.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from isaacsimenvs.tasks.peg_in_hole.peg_in_hole_env import (
    PegInHoleEnv,
    TABLE_HALF_HEIGHT,
)

from .peg_in_hole_fixtured_env_cfg import PegInHoleFixturedEnvCfg
from .scene_utils import setup_scene


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCENES_NPZ_PATH = _REPO_ROOT / "assets" / "urdf" / "peg_in_hole_fixtured" / "scenes" / "scenes.npz"


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Quaternion convention swap: scenes.npz stores xyzw, IsaacLab uses wxyz."""
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


class PegInHoleFixturedEnv(PegInHoleEnv):
    cfg: PegInHoleFixturedEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs) -> None:
        # Load scenes.npz first so _configure_problem (called by parent
        # __init__) can size _num_prelude_goals against the trajectory length.
        self._load_scenes_npz()
        super().__init__(cfg, render_mode, **kwargs)

        # Move scenes data to device now that self.device exists.
        d = self.device
        self._scenes_start_poses_t = self._scenes_start_poses_t.to(d)
        self._scenes_goals_t = self._scenes_goals_t.to(d)
        self._scenes_traj_lengths_t = self._scenes_traj_lengths_t.to(d)
        self._scenes_hole_positions_t = self._scenes_hole_positions_t.to(d)
        self._scenes_start_fixture_positions_t = (
            self._scenes_start_fixture_positions_t.to(d)
        )

        # Per-env (scene, peg) round-robin binding.
        N, M = self._scenes_start_poses_t.shape[:2]
        env_idx = torch.arange(self.num_envs, device=d)
        self._fix_scene_idx = (env_idx % N).long()
        self._fix_peg_idx = ((env_idx // N) % M).long()
        # Cap with available pegs per scene
        self._fix_peg_idx = self._fix_peg_idx.clamp(max=M - 1)

    def _load_scenes_npz(self) -> None:
        if not _SCENES_NPZ_PATH.exists():
            raise FileNotFoundError(
                f"PegInHoleFixturedEnv expects scenes.npz at {_SCENES_NPZ_PATH}; "
                f"pull it from branch 2026_04_28_peg_in_hole_fixtured "
                f"(assets/urdf/peg_in_hole_fixtured/scenes/scenes.npz)."
            )
        d = np.load(str(_SCENES_NPZ_PATH))
        start_poses = d["start_poses"].astype(np.float32)  # (N, M, 7) xyz+xyzw
        goals = d["goals"].astype(np.float32)              # (N, M, T, 7)
        traj_lengths = d["traj_lengths"].astype(np.int64)  # (N, M)
        hole_positions = d["hole_positions"].astype(np.float32)             # (N, 3)
        start_fixture_positions = d["start_fixture_positions"].astype(np.float32)  # (N, M, 3)

        # Convert quats xyzw -> wxyz (IsaacLab convention).
        start_poses_wxyz = np.concatenate(
            [start_poses[..., 0:3], _xyzw_to_wxyz(start_poses[..., 3:7])], axis=-1
        )
        goals_wxyz = np.concatenate(
            [goals[..., 0:3], _xyzw_to_wxyz(goals[..., 3:7])], axis=-1
        )

        self._scenes_start_poses_t = torch.from_numpy(start_poses_wxyz)
        self._scenes_goals_t = torch.from_numpy(goals_wxyz)
        self._scenes_traj_lengths_t = torch.from_numpy(traj_lengths)
        self._scenes_hole_positions_t = torch.from_numpy(hole_positions)
        self._scenes_start_fixture_positions_t = torch.from_numpy(start_fixture_positions)
        self._scenes_max_traj_len = int(traj_lengths.max())
        # Truncate / select waypoints per goal_mode. We do this lazily at
        # reset time so the cfg's goal_mode can change without reloading.

    def _configure_problem(self, cfg) -> None:
        super()._configure_problem(cfg)
        goal_mode = cfg.peg_in_hole.goal_mode
        # For fixtured modes, the trajectory comes from scenes.npz. We always
        # need enough prelude buffer slots to hold the full dense trajectory
        # (so we can switch goal_mode without re-allocating).
        if goal_mode in ("dense", "preInsertAndFinal", "finalGoalOnly"):
            self._num_prelude_goals = self._scenes_max_traj_len
            self._num_insertion_goals = 0
            self._num_total_insertion_goals = self._num_prelude_goals

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_peg_episode(self, env_ids: torch.Tensor) -> None:
        # Run parent reset for retract_phase / obs_noise / goal_pos_obs_noise etc.
        # We will overwrite hole, peg_fixture, peg, prelude_pose_world below.
        super()._reset_peg_episode(env_ids)

        n = env_ids.numel()
        d = self.device
        env_origins = self.scene.env_origins[env_ids]
        scene_ids = self._fix_scene_idx[env_ids]
        peg_ids = self._fix_peg_idx[env_ids]
        goal_mode = self.cfg.peg_in_hole.goal_mode

        # ── Goal-hole pose (the second fixture, the insertion target) ──
        hole_pos_local = self._scenes_hole_positions_t[scene_ids]  # (n, 3)
        hole_quat = torch.zeros(n, 4, device=d, dtype=torch.float32)
        hole_quat[:, 0] = 1.0  # identity wxyz
        self.hole_pos[env_ids] = hole_pos_local
        self.hole_quat_wxyz[env_ids] = hole_quat
        hole_pose = torch.cat([hole_pos_local + env_origins, hole_quat], dim=-1)
        self.hole.write_root_pose_to_sim(hole_pose, env_ids=env_ids)
        self.hole.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=d), env_ids=env_ids
        )

        # ── Peg-holding fixture pose ──
        pf_pos_local = self._scenes_start_fixture_positions_t[scene_ids, peg_ids]  # (n, 3)
        pf_pose = torch.cat([pf_pos_local + env_origins, hole_quat], dim=-1)
        self.peg_fixture.write_root_pose_to_sim(pf_pose, env_ids=env_ids)
        self.peg_fixture.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=d), env_ids=env_ids
        )

        # ── Peg pose ──
        peg_pose_local = self._scenes_start_poses_t[scene_ids, peg_ids]  # (n, 7) wxyz
        peg_pos = peg_pose_local[:, 0:3]
        peg_quat = peg_pose_local[:, 3:7]
        peg_pose = torch.cat([peg_pos + env_origins, peg_quat], dim=-1)
        self.object.write_root_pose_to_sim(peg_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=d), env_ids=env_ids
        )

        # ── Goal trajectory into prelude buffer ──
        # full_goals: (n, T_max, 7) but each env has its own traj_length.
        full_goals = self._scenes_goals_t[scene_ids, peg_ids]  # (n, T, 7) wxyz
        traj_lens = self._scenes_traj_lengths_t[scene_ids, peg_ids]  # (n,)

        # Mode-specific trimming. We write the relevant waypoints starting at
        # _prelude_pose_world[env_id, 0]; _write_goal_pose's prelude path
        # picks them by subgoal_idx, and env_max_goals caps episode length.
        T_max = full_goals.shape[1]
        # Make sure the prelude buffer is big enough (sized at init via
        # _num_prelude_goals = max_traj_len).
        assert T_max <= self._prelude_pose_world.shape[1], (
            f"prelude buffer too small: {self._prelude_pose_world.shape[1]} < {T_max}"
        )

        # Build per-env trimmed sequences. For preInsertAndFinal we want the
        # last two valid waypoints; for finalGoalOnly the last one; for dense
        # the full traj_lens[i].
        if goal_mode == "dense":
            # Use waypoints [0, traj_len) verbatim.
            for i in range(n):
                env_i = int(env_ids[i].item())
                L = int(traj_lens[i].item())
                self._prelude_pose_world[env_i, :L] = full_goals[i, :L]
            self.env_max_goals[env_ids] = traj_lens

        elif goal_mode == "preInsertAndFinal":
            # Last two valid waypoints (matches IsaacGym truncation).
            for i in range(n):
                env_i = int(env_ids[i].item())
                L = int(traj_lens[i].item())
                pre_idx = max(0, L - 2)
                final_idx = L - 1
                self._prelude_pose_world[env_i, 0] = full_goals[i, pre_idx]
                self._prelude_pose_world[env_i, 1] = full_goals[i, final_idx]
            self.env_max_goals[env_ids] = torch.full_like(traj_lens, 2)

        elif goal_mode == "finalGoalOnly":
            for i in range(n):
                env_i = int(env_ids[i].item())
                L = int(traj_lens[i].item())
                self._prelude_pose_world[env_i, 0] = full_goals[i, L - 1]
            self.env_max_goals[env_ids] = torch.full_like(traj_lens, 1)

        # Re-clear goal trackers + write first goal pose (so goal_viz shows
        # the new trajectory's first waypoint, not the old random goal).
        self._clear_goal_trackers(env_ids)
        self._write_goal_pose(env_ids, is_first_goal=True)


__all__ = ["PegInHoleFixturedEnv"]

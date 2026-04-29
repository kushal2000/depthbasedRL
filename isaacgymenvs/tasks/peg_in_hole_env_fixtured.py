"""PegInHoleEnvFixtured: peg starts upright in a tight starting fixture.

Variant of ``PegInHoleEnv`` for the reorientation-bottleneck hypothesis test
(see ``peg_in_hole_fixtured/`` and ``.claude/plans/in-this-folder-we-happy-falcon.md``).

Differences from ``PegInHoleEnv``:
  * Each scene URDF contains *two* fixed-base fixtures (start + goal) instead
    of one, baked at scene-generation time. The peg's starting fixture XY is
    therefore part of the URDF and varies per peg slot.
  * Per-env binding becomes a **triple-static** ``(scene, peg, tol)``
    round-robin over ``N × M × K`` URDFs (parent's binding is double-static
    on ``(scene, tol)`` only — peg_idx is dynamic at reset). With many envs
    each URDF is shared by ~num_envs / (N*M*K) envs.
  * On reset the peg spawns *at* the fixtured start pose (Z = in-fixture seat,
    not ``tableResetZ + tableObjectZOffset``), so it doesn't fall onto a
    table. ``object_init_state[:, 2]`` is overridden after the parent's
    ``reset_object_pose`` would have pinned it to the hover Z.

Goal trajectories already encode the fixtured start as their first waypoint;
``goalMode`` (``dense`` / ``preInsertAndFinal`` / ``finalGoalOnly``), reward,
retract, and DR plumbing are inherited unchanged.

Loads ``assets/urdf/peg_in_hole_fixtured/scenes/scenes.npz`` produced by
``peg_in_hole_fixtured/scene_generation/generate_scenes.py``.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from isaacgymenvs.tasks.peg_in_hole_env import PegInHoleEnv, VALID_GOAL_MODES
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
from isaacgymenvs.utils.torch_jit_utils import torch_rand_float


class PegInHoleEnvFixtured(PegInHoleEnv):
    # ────────────────────────────────────────────────────────────────
    # __init__
    # ────────────────────────────────────────────────────────────────

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(
            cfg, rl_device, sim_device, graphics_device_id,
            headless, virtual_screen_capture, force_render,
        )
        # Pin env_peg_idx to the static (scene, peg, tol) binding built in
        # _init_peg_in_hole_config. Parent zero-inits this and resamples each
        # reset; here it stays fixed because the start fixture XY is baked
        # into the URDF that env loaded.
        self.env_peg_idx[:] = torch.tensor(
            self._pih_env_peg_idx, dtype=torch.long, device=self.device,
        )
        # Recompute env_max_goals against the real peg_idx (parent seeded
        # with peg_idx=0 for every env's scene).
        self.env_max_goals[:] = self._pih_traj_lengths_t[
            self._pih_env_scene_idx_t, self.env_peg_idx
        ]
        self.prev_episode_env_max_goals[:] = self.env_max_goals.clone()

    # ────────────────────────────────────────────────────────────────
    # Scene loading + per-env binding (triple-static)
    # ────────────────────────────────────────────────────────────────

    def _init_peg_in_hole_config(self, cfg):
        """Load fixtured scenes.npz, truncate goals per goal_mode, build the
        ``(scene, peg, tol)`` triple-static per-env binding, and write
        ``urdf/peg_in_hole_fixtured/scenes/scene_{n}/peg_{m}/scene_tol{k}.urdf``
        paths.
        """
        assert self.goal_mode in VALID_GOAL_MODES, (
            f"goalMode must be one of {VALID_GOAL_MODES}, got {self.goal_mode!r}"
        )

        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
        )
        scenes_path = cfg["env"]["scenesPath"]
        if not os.path.isabs(scenes_path):
            scenes_path = os.path.join(repo_root, scenes_path)
        assert os.path.exists(scenes_path), (
            f"scenesPath {scenes_path} does not exist. Run "
            f"peg_in_hole_fixtured/scene_generation/generate_scenes.py first."
        )
        data = np.load(scenes_path)
        start_poses = data["start_poses"].astype(np.float32)              # (N, M, 7)
        goals = data["goals"].astype(np.float32)                          # (N, M, T, 7)
        traj_lengths = data["traj_lengths"].astype(np.int64)              # (N, M)
        hole_positions = data["hole_positions"].astype(np.float32)        # (N, 3)
        tol_pool_m = data["tolerance_pool_m"].astype(np.float32)
        scene_tol_indices = data["scene_tolerance_indices"].astype(np.int64)  # (N, K)
        # Fixtured-only fields (start fixture geometry baked into URDF; arrays
        # are kept for evaluation / visualization tooling).
        start_fixture_positions = data["start_fixture_positions"].astype(np.float32)  # (N, M, 3)
        start_tolerance_m = float(data["start_tolerance_m"])

        N, M, _ = start_poses.shape
        _, _, T, _ = goals.shape
        K = scene_tol_indices.shape[1]
        assert scene_tol_indices.shape[0] == N
        assert traj_lengths.shape == (N, M)

        # ── Goal-mode truncation (mirror PegInHoleEnv) ──────────────
        if self.goal_mode == "finalGoalOnly":
            n_idx = np.arange(N)[:, None].repeat(M, axis=1)
            m_idx = np.arange(M)[None, :].repeat(N, axis=0)
            final_idx = traj_lengths - 1
            final_goals = goals[n_idx, m_idx, final_idx]
            goals = final_goals[:, :, None, :]
            traj_lengths = np.ones_like(traj_lengths)
            T = 1
            print(
                f"[PegInHoleEnvFixtured] goalMode=finalGoalOnly: goals truncated to {goals.shape}"
            )
        elif self.goal_mode == "preInsertAndFinal":
            n_idx = np.arange(N)[:, None].repeat(M, axis=1)
            m_idx = np.arange(M)[None, :].repeat(N, axis=0)
            final_idx = traj_lengths - 1
            pre_idx = np.clip(traj_lengths - 2, a_min=0, a_max=None)
            pre_goals = goals[n_idx, m_idx, pre_idx]
            final_goals = goals[n_idx, m_idx, final_idx]
            goals = np.stack([pre_goals, final_goals], axis=2)
            traj_lengths = np.clip(traj_lengths, a_min=None, a_max=2)
            T = 2
            print(
                f"[PegInHoleEnvFixtured] goalMode=preInsertAndFinal: goals truncated to {goals.shape}"
            )

        self._pih_num_scenes = N
        self._pih_num_pegs = M
        self._pih_num_tol_slots = K
        self._pih_max_traj_len = T
        self._pih_start_poses = start_poses
        self._pih_goals = goals
        self._pih_traj_lengths = traj_lengths
        self._pih_hole_positions = hole_positions
        self._pih_tol_pool_m = tol_pool_m
        self._pih_scene_tol_indices = scene_tol_indices
        self._pih_start_fixture_positions = start_fixture_positions
        self._pih_start_tolerance_m = start_tolerance_m

        # ── Per-env (scene, peg, tol) triple-static round-robin ──
        # Optional override:
        #   forceSceneTolCombo = [scene_idx, tol_slot_idx]  (peg_idx → 0)
        #   forceSceneTolCombo = [scene_idx, peg_idx, tol_slot_idx]
        num_envs = cfg["env"]["numEnvs"]
        combo_count = N * M * K

        force_combo = cfg["env"].get("forceSceneTolCombo", None)
        if force_combo is not None:
            if len(force_combo) == 3:
                fs, fp, ft = (
                    int(force_combo[0]), int(force_combo[1]), int(force_combo[2])
                )
            elif len(force_combo) == 2:
                fs, ft = int(force_combo[0]), int(force_combo[1])
                fp = 0
            else:
                raise ValueError(
                    f"forceSceneTolCombo must be [scene, tol] or [scene, peg, tol]; got {force_combo}"
                )
            assert 0 <= fs < N and 0 <= fp < M and 0 <= ft < K, (
                f"forceSceneTolCombo=({fs},{fp},{ft}) out of range "
                f"(N={N}, M={M}, K={K})"
            )
            self._pih_env_scene_idx = np.full(num_envs, fs, dtype=np.int64)
            self._pih_env_peg_idx = np.full(num_envs, fp, dtype=np.int64)
            self._pih_env_tol_slot_idx = np.full(num_envs, ft, dtype=np.int64)
            print(
                f"[PegInHoleEnvFixtured] forceSceneTolCombo=({fs}, {fp}, {ft}) — "
                f"all {num_envs} envs pinned."
            )
        else:
            combo_ids = np.arange(num_envs) % combo_count
            self._pih_env_scene_idx = (combo_ids // (M * K)).astype(np.int64)
            self._pih_env_peg_idx = ((combo_ids // K) % M).astype(np.int64)
            self._pih_env_tol_slot_idx = (combo_ids % K).astype(np.int64)

        # Reuse parent's `forceTightestTolPerScene` / `tightestNTolSlotsPerScene`
        # ablations (they only touch tol_slot_idx; the peg axis is independent).
        if cfg["env"].get("forceTightestTolPerScene", False):
            per_scene_tightest_slot = np.argmin(
                tol_pool_m[scene_tol_indices], axis=1
            ).astype(np.int64)
            self._pih_env_tol_slot_idx = per_scene_tightest_slot[self._pih_env_scene_idx]
            print(
                f"[PegInHoleEnvFixtured] forceTightestTolPerScene=True — tol_slot "
                f"overridden to per-scene tightest."
            )
        tightest_n = int(cfg["env"].get("tightestNTolSlotsPerScene", -1))
        if (
            tightest_n > 0
            and tightest_n < K
            and not cfg["env"].get("forceTightestTolPerScene", False)
        ):
            actual_tols_per_scene = tol_pool_m[scene_tol_indices]
            tightest_slots_per_scene = np.argsort(actual_tols_per_scene, axis=1)[:, :tightest_n]
            env_pick_idx = np.arange(num_envs) % tightest_n
            self._pih_env_tol_slot_idx = tightest_slots_per_scene[
                self._pih_env_scene_idx, env_pick_idx
            ].astype(np.int64)
            seen_tols = actual_tols_per_scene[
                np.arange(N)[:, None], tightest_slots_per_scene
            ]
            print(
                f"[PegInHoleEnvFixtured] tightestNTolSlotsPerScene={tightest_n}/{K} — "
                f"seen tol range {seen_tols.min()*1000:.3f}–{seen_tols.max()*1000:.3f} mm."
            )

        # Per-env scene URDF paths under peg_in_hole_fixtured/.
        self._pih_scene_urdfs = []
        for env_i in range(num_envs):
            s = int(self._pih_env_scene_idx[env_i])
            p = int(self._pih_env_peg_idx[env_i])
            ts = int(self._pih_env_tol_slot_idx[env_i])
            self._pih_scene_urdfs.append(
                f"urdf/peg_in_hole_fixtured/scenes/"
                f"scene_{s:04d}/peg_{p:04d}/scene_tol{ts:02d}.urdf"
            )

        print(
            f"[PegInHoleEnvFixtured] Loaded {N} scenes × {M} pegs × {K} tol slots "
            f"(start tol {start_tolerance_m*1000:.2f}mm, "
            f"goal_mode={self.goal_mode}, max_traj_len={T})"
        )
        print(
            f"[PegInHoleEnvFixtured] {num_envs} envs → {combo_count} unique "
            f"(scene, peg, tol) combos ({num_envs / combo_count:.1f}x coverage)"
        )

    # ────────────────────────────────────────────────────────────────
    # Reset — peg_idx is static; Z is the in-fixture seat
    # ────────────────────────────────────────────────────────────────

    def reset_object_pose(self, env_ids, reset_buf_idxs=None, tensor_reset=True):
        """Place peg at its cached fixtured start pose. Skips the parent's
        peg-idx resampling (peg slot is baked into the URDF) and pins
        ``object_init_state[..., 2]`` to the in-fixture Z after the
        ``SimToolReal`` reset path would otherwise set it to ``tableResetZ +
        tableObjectZOffset`` (which would make the peg hover and fall).
        """
        if tensor_reset and len(env_ids) > 0 and reset_buf_idxs is None:
            self.prev_episode_env_max_goals[env_ids] = self.env_max_goals[env_ids]

            scene_ids = self._pih_env_scene_idx_t[env_ids]
            peg_ids = self.env_peg_idx[env_ids]
            self.env_max_goals[env_ids] = self._pih_traj_lengths_t[scene_ids, peg_ids]

            # X, Y, quat from cached fixtured start. Z is set after super.
            poses = self._pih_start_poses_t[scene_ids, peg_ids]            # (len, 7)
            self.object_init_state[env_ids, 0:2] = poses[:, 0:2]
            self.object_init_state[env_ids, 3:7] = poses[:, 3:7]

            self.goal_pos_obs_noise[env_ids, 0:2] = torch_rand_float(
                -self.cfg["env"]["goalXyObsNoise"],
                self.cfg["env"]["goalXyObsNoise"],
                (len(env_ids), 2), device=self.device,
            )

        # Skip PegInHoleEnv's peg-resampling reset path; call SimToolReal
        # directly. (Going through PegInHoleEnv.reset_object_pose would
        # randomize env_peg_idx, breaking the URDF↔peg_idx pairing.)
        SimToolReal.reset_object_pose(self, env_ids, reset_buf_idxs, tensor_reset)

        # After SimToolReal: override Z. SimToolReal pins
        # object_init_state[:, 2] = table_reset_z + tableObjectZOffset, and
        # then writes that into root_state_tensor. Replace both with the
        # cached in-fixture Z (offset by the table's reset-z delta so the peg
        # tracks the table's vertical jitter, mirroring how the goal-state
        # delta is applied in _reset_target).
        if tensor_reset and len(env_ids) > 0 and reset_buf_idxs is None:
            scene_ids = self._pih_env_scene_idx_t[env_ids]
            peg_ids = self.env_peg_idx[env_ids]
            poses = self._pih_start_poses_t[scene_ids, peg_ids]

            table_base_z = self.cfg["env"]["tableResetZ"]
            delta_z = self.table_init_state[env_ids, 2:3] - table_base_z
            new_z = poses[:, 2:3] + delta_z

            self.object_init_state[env_ids, 2:3] = new_z
            obj_indices = self.object_indices[env_ids]
            self.root_state_tensor[obj_indices, 2:3] = new_z

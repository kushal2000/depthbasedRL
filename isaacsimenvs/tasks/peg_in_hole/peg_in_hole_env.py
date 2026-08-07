"""Dynamic peg-in-hole task built as a SimToolRealEnv variant."""

from __future__ import annotations

import sys
from pathlib import Path

import math

import torch

from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
from isaacsimenvs.tasks.simtoolreal.utils.goal_sampling import (
    sample_absolute_goal_pose,
    sample_delta_goal_pose,
)
from isaacsimenvs.tasks.simtoolreal.utils.logging_utils import log_step_metrics
from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import (
    OBS_FIELD_SIZES,
    build_observations,
    compute_intermediate_values,
)
from isaacsimenvs.tasks.simtoolreal.utils.reward_utils import compute_rewards
from isaacsimenvs.tasks.simtoolreal.utils.termination_utils import (
    update_tolerance_curriculum,
)

from .peg_in_hole_env_cfg import PegInHoleEnvCfg, VALID_GOAL_MODES
from .scene_utils import REPO_ROOT, setup_scene


TABLE_HALF_HEIGHT = 0.15


def _xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[:, 3:4], quat[:, 0:3]], dim=-1)


def _obs_field_slice(fields: tuple[str, ...], field: str) -> slice | None:
    offset = 0
    for name in fields:
        size = OBS_FIELD_SIZES[name]
        if name == field:
            return slice(offset, offset + size)
        offset += size
    return None


def _resolve_asset_path(path: str | Path) -> Path:
    asset_path = Path(path)
    if asset_path.is_absolute():
        return asset_path

    candidates = (REPO_ROOT / asset_path, REPO_ROOT / "assets" / asset_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve asset path {path!r}; checked {checked}")


class PegInHoleEnv(SimToolRealEnv):
    cfg: PegInHoleEnvCfg

    def __init__(
        self, cfg: PegInHoleEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        self._configure_problem(cfg)

        super().__init__(cfg, render_mode, **kwargs)

        # When student_obs is enabled, the env exposes three obs groups via
        # `_get_observations`: "policy" = student obs (image+proprio, what
        # the depth-CNN reads), "critic" = state_list (privileged), and
        # "teacher_obs" = obs_list (proprio + object_state, noisy — what the
        # frozen state-MLP teacher reads). The default RlGamesVecEnvWrapper
        # drops keys it doesn't know; use DAggerRlGamesVecEnvWrapper from
        # `isaacsimenvs.utils.rlgames_utils` to pass "teacher_obs" through to
        # the agent as `self.obs["teacher"]`.
        student_cfg = getattr(cfg, "student_obs", None)
        if student_cfg is not None and student_cfg.enabled and student_cfg.image_enabled:
            from gymnasium import spaces
            import numpy as _np
            from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import _student_proprio_dict

            image_numel = (
                int(student_cfg.image_input_height)
                * int(student_cfg.image_input_width)
                * (1 if student_cfg.image_modality.lower() == "depth" else 3)
            )
            proprio_sample = _student_proprio_dict(self)
            proprio_dim = sum(
                int(proprio_sample[field].reshape(self.num_envs, -1).shape[-1])
                for field in student_cfg.proprio_list
            )
            student_dim = image_numel + proprio_dim
            actor_dim = int(cfg.observation_space)        # original obs_list dim (teacher actor input)
            critic_dim = int(cfg.state_space)             # state_list dim (asymmetric critic input)

            cfg.observation_space = student_dim
            box = lambda d: spaces.Box(low=-_np.inf, high=_np.inf, shape=(d,))
            self.observation_space = spaces.Dict({
                "policy": box(student_dim),
                "critic": box(critic_dim),
                "teacher_obs": box(actor_dim),
            })
            self.single_observation_space = spaces.Dict({
                "policy": box(student_dim),
                "critic": box(critic_dim),
                "teacher_obs": box(actor_dim),
            })
            self._teacher_actor_obs_dim = actor_dim

        # ---- Phase C: per-problem tables + per-env gathers ------------------
        # Problem-derived quantities are promoted from python scalars to (P,)
        # tables indexed by a per-env problem index. With P == 1 every (N,)
        # vector below is a constant fill, so all downstream arithmetic is
        # elementwise-identical to the pre-refactor scalar code.
        P = len(self._pih_insert_pose_seqs)
        s_max = max(len(s) for s in self._pih_insert_pose_seqs)
        # Pad with NaN, not identity: a padded slot must never look like a
        # reachable goal. If a tail index ever leaks past a short problem's
        # sequence the resulting pose is NaN and fails loudly, instead of
        # quietly aiming at the hole origin.
        pos_p = torch.full((P, s_max, 3), float("nan"),
                           dtype=torch.float32, device=self.device)
        quat_p = torch.full((P, s_max, 4), float("nan"),
                            dtype=torch.float32, device=self.device)
        for p, seq in enumerate(self._pih_insert_pose_seqs):
            poses = torch.as_tensor(seq, dtype=torch.float32, device=self.device)
            pos_p[p, : poses.shape[0]] = poses[:, 0:3]
            quat_p[p, : poses.shape[0]] = _xyzw_to_wxyz(poses[:, 3:7])
        self._insert_pos_rel_p = pos_p.contiguous()
        self._insert_quat_wxyz_p = quat_p.contiguous()

        def _p_tensor(values, dtype):
            return torch.as_tensor(values, dtype=dtype, device=self.device)

        self._num_prelude_goals_p = _p_tensor(self._pih_num_prelude_goals, torch.long)
        self._num_tail_goals_p = _p_tensor(self._pih_num_tail_goals, torch.long)
        self._num_total_goals_p = _p_tensor(self._pih_num_total_goals, torch.long)
        self._prelude_lift_off_p = _p_tensor(self._pih_prelude_lift_offs, torch.float32)
        self._hole_z_offset_p = _p_tensor(self._pih_hole_z_offsets, torch.float32)

        # setup_scene (Phase B) recovers this from the spawned prims when the
        # multi-asset path is used; single-problem runs are all-zeros.
        if getattr(self, "_problem_idx_per_env", None) is None:
            self._problem_idx_per_env = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
        pidx = self._problem_idx_per_env
        # Gather once here: an env's problem is fixed for its lifetime (it is
        # determined by the USD it was spawned with), so this never belongs in
        # the per-step path.
        self._num_prelude_goals_env = self._num_prelude_goals_p[pidx]
        self._num_total_goals_env = self._num_total_goals_p[pidx]
        self._prelude_lift_off_env = self._prelude_lift_off_p[pidx]
        self._hole_z_offset_env = self._hole_z_offset_p[pidx]

        if P > 1:
            self._check_problem_assignment(P, pidx)

        self.hole_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.hole_quat_wxyz = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )
        self.hole_quat_wxyz[:, 0] = 1.0  # identity default
        self.is_random_goal_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.prev_episode_is_random_goal = torch.zeros_like(self.is_random_goal_env)

        self.env_max_goals = self._num_total_goals_env.clone()
        pih_cfg = self.cfg.peg_in_hole
        random_goal_fraction = float(pih_cfg.random_goal_fraction)
        if random_goal_fraction > 0.0:
            self.is_random_goal_env[:] = (
                torch.rand(self.num_envs, device=self.device) < random_goal_fraction
            )
            random_goal_max = torch.full_like(
                self.env_max_goals, int(pih_cfg.random_goal_max_successes)
            )
            self.env_max_goals[:] = torch.where(
                self.is_random_goal_env, random_goal_max, self._num_total_goals_env
            )

        # Per-env world-frame prelude waypoints (lift_in_place, over_hole)
        # rebuilt at every reset for transportPreInsertFinal mode. Stored as
        # (pos_xyz, quat_wxyz) to match goal_viz convention.
        self._prelude_pose_world = torch.zeros(
            self.num_envs,
            max(int(self._num_prelude_goals_p.max().item()), 1),
            7,
            dtype=torch.float32,
            device=self.device,
        )

        self.prev_episode_env_max_goals = self.env_max_goals.clone()
        self.prev_episode_is_random_goal[:] = self.is_random_goal_env

        # Scalar curriculum threshold for transportPreInsertFinal, averaged over
        # the curriculum-eligible envs. update_tolerance_curriculum compares it
        # to successes.mean(), so it cannot be per-env. Mean of a constant when
        # P == 1, i.e. exactly float(_num_prelude_goals + 1).
        _elig = ~self.is_random_goal_env
        self._curriculum_threshold_dense = (
            float(self._num_prelude_goals_env[_elig].float().mean().item()) + 1.0
            if bool(_elig.any())
            else 1.0
        )

        self.insertion_success_tolerance = float(pih_cfg.insertion_success_tolerance)
        self.retract_success_bonus = float(
            int(pih_cfg.random_goal_max_successes) * self.cfg.reward.reach_goal_bonus
        )
        self.lift_bonus_active = True

        self.retract_phase = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.retract_succeeded = torch.zeros_like(self.retract_phase)
        self._just_entered_retract = torch.zeros_like(self.retract_phase)
        self._just_retracted = torch.zeros_like(self.retract_phase)

        self.goal_pos_obs_noise = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        # Per-env yaw noise (radians) applied to the observed goal orientation.
        self.goal_yaw_obs_noise = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._goal_kp_obs_slice = _obs_field_slice(
            tuple(cfg.obs.obs_list), "keypoints_rel_goal"
        )

        print(
            f"[PegInHoleEnv] problem={self._pih_problem_name} "
            f"object={cfg.assets.object_name} "
            f"goals={self._num_total_insertion_goals} "
            f"(prelude={self._num_prelude_goals}, tail={self._num_insertion_goals}) "
            f"goal_mode={pih_cfg.goal_mode} "
            f"random_goal_fraction={random_goal_fraction}",
            flush=True,
        )

    def _configure_problem(self, cfg: PegInHoleEnvCfg) -> None:
        pih_cfg = cfg.peg_in_hole
        if pih_cfg.goal_mode not in VALID_GOAL_MODES:
            raise ValueError(
                f"goal_mode must be one of {VALID_GOAL_MODES}, got {pih_cfg.goal_mode!r}"
            )

        random_goal_fraction = float(pih_cfg.random_goal_fraction)
        if not 0.0 <= random_goal_fraction <= 1.0:
            raise ValueError(
                "peg_in_hole.random_goal_fraction must be in [0, 1], "
                f"got {random_goal_fraction}"
            )

        repo_root_str = str(REPO_ROOT)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        from dextoolbench.objects import NAME_TO_OBJECT
        from peg_in_hole_dynamic import PROBLEM_REGISTRY

        # Multi-problem: `problems` overrides `problem` when non-empty. The
        # per-problem lists are built below; index 0 keeps feeding the legacy
        # scalars so the scene builder, pose_viewer and the fixtured subclass
        # are untouched at P == 1.
        problem_names = [str(p) for p in (pih_cfg.problems or ())]
        if problem_names:
            print(
                f"[PegInHoleEnv] multi-problem: peg_in_hole.problems has "
                f"{len(problem_names)} entries, IGNORING peg_in_hole.problem="
                f"{pih_cfg.problem!r}",
                flush=True,
            )
        else:
            problem_names = [str(pih_cfg.problem)]

        for name in problem_names:
            if name not in PROBLEM_REGISTRY:
                raise KeyError(
                    f"Unknown peg-in-hole problem {name!r}; "
                    f"known problems: {sorted(PROBLEM_REGISTRY)}"
                )

        mix = [int(v) for v in (pih_cfg.problem_mix or ())]
        if not mix:
            mix = [1] * len(problem_names)
        if len(mix) != len(problem_names):
            raise ValueError(
                f"peg_in_hole.problem_mix has {len(mix)} entries but "
                f"problems has {len(problem_names)}; they must match."
            )
        if any(c < 1 for c in mix):
            raise ValueError(f"peg_in_hole.problem_mix counts must be >= 1, got {mix}")
        n_slots = sum(mix)
        n_envs = int(cfg.scene.num_envs)
        if n_envs % n_slots != 0:
            raise ValueError(
                f"scene.num_envs ({n_envs}) must be divisible by the mix total "
                f"({n_slots} = sum(problem_mix)={mix}); otherwise the problem "
                "totals would not come out exact."
            )
        # One entry per env (not per mix-slot): a balanced multiset, shuffled.
        # Passing a length-num_envs list to MultiUsdFileCfg means the spawner's
        # round-robin hands each env its own entry, so the assignment is this
        # shuffle rather than a repeating period-L pattern. Balanced by
        # construction -- every problem gets exactly num_envs/n_slots*count envs.
        if bool(getattr(pih_cfg, "problem_assignment_shuffle", True)):
            reps = n_envs // n_slots
            slot_problem_idx = [i for i, c in enumerate(mix) for _ in range(c * reps)]
            if len(problem_names) > 1:
                import random as _random
                _random.Random(int(pih_cfg.problem_assignment_seed)).shuffle(
                    slot_problem_idx
                )
        else:
            # One entry per mix slot: the spawner cycles it, giving a repeating
            # period-L pattern instead of a shuffle. Same exact totals, much
            # shorter MultiUsdFileCfg list.
            slot_problem_idx = [i for i, c in enumerate(mix) for _ in range(c)]

        problem_name = problem_names[0]
        problem = PROBLEM_REGISTRY[problem_name]

        if problem.insertion_object_name not in NAME_TO_OBJECT:
            raise KeyError(
                f"Problem {problem_name!r} references object "
                f"{problem.insertion_object_name!r}, but it is not in NAME_TO_OBJECT."
            )
        object_spec = NAME_TO_OBJECT[problem.insertion_object_name]
        object_urdf = _resolve_asset_path(object_spec.urdf_path)
        receptive_urdf = _resolve_asset_path(problem.receptive_urdf)

        insert_pose_sequence = problem.insert_pose_rel_receptive
        if pih_cfg.goal_mode == "finalGoalOnly":
            insert_pose_sequence = (problem.final_insert_pose_rel_receptive,)
        if len(insert_pose_sequence) == 0:
            raise ValueError(f"Problem {problem_name!r} has no insertion subgoals.")

        # Resolve every problem in the mix into parallel length-P lists.
        # goal_mode is global, so the prelude count is derived per problem under
        # the same mode -- a problem lacking prelude_lift_offset under
        # transportPreInsertFinal is a hard error, same as the single case.
        p_names, p_objs, p_obj_urdfs, p_rec_urdfs = [], [], [], []
        p_scales, p_hole_z, p_seqs, p_tail = [], [], [], []
        p_prelude, p_lift = [], []
        for name in problem_names:
            prob = PROBLEM_REGISTRY[name]
            if prob.insertion_object_name not in NAME_TO_OBJECT:
                raise KeyError(
                    f"Problem {name!r} references object "
                    f"{prob.insertion_object_name!r}, not in NAME_TO_OBJECT."
                )
            spec = NAME_TO_OBJECT[prob.insertion_object_name]
            seq = prob.insert_pose_rel_receptive
            if pih_cfg.goal_mode == "finalGoalOnly":
                seq = (prob.final_insert_pose_rel_receptive,)
            if len(seq) == 0:
                raise ValueError(f"Problem {name!r} has no insertion subgoals.")
            if pih_cfg.goal_mode == "transportPreInsertFinal":
                if float(prob.prelude_lift_offset) <= 0.0:
                    raise ValueError(
                        f"goal_mode='transportPreInsertFinal' requires problem "
                        f"{name!r} to set prelude_lift_offset > 0; got "
                        f"{prob.prelude_lift_offset}."
                    )
                pre, lift = 2, float(prob.prelude_lift_offset)
            else:
                pre, lift = 0, 0.0
            p_names.append(name)
            p_objs.append(prob)
            p_obj_urdfs.append(str(_resolve_asset_path(spec.urdf_path)))
            p_rec_urdfs.append(str(_resolve_asset_path(prob.receptive_urdf)))
            p_scales.append(tuple(float(v) for v in spec.scale))
            p_hole_z.append(float(prob.hole_z_offset))
            p_seqs.append(tuple(seq))
            p_tail.append(len(seq))
            p_prelude.append(pre)
            p_lift.append(lift)

        self._pih_problem_name = problem_name
        self._pih_problem = problem
        self._pih_object_urdf_abs = str(object_urdf)
        self._pih_receptive_urdf_abs = str(receptive_urdf)
        self._pih_object_scale = tuple(float(v) for v in object_spec.scale)
        self._pih_hole_z_offset = float(problem.hole_z_offset)
        self._pih_insert_pose_sequence = tuple(insert_pose_sequence)
        self._num_insertion_goals = len(self._pih_insert_pose_sequence)

        # Phase A of multi-problem support: the same values as parallel lists of
        # length P. Today P == 1; the scalars above are kept as aliases so the
        # scene builder, the fixtured subclass and pose_viewer keep working
        # unchanged. Phase C (after super().__init__()) turns these into padded
        # (P, ...) tables and (N,) per-env gathers.
        # Objects must share link structure. RigidObject binds ONE PhysX view
        # over all spawned instances, and a view needs a consistent body layout;
        # mixing a 1-link object (lpeg) with 3-link ones (beam/furniture) makes
        # it bind only num_envs/P bodies and PhysX dies on the first step.
        # Measured: P=4@12288 -> 3072, P=4@3072 -> 768, P=3@12288 -> 4096, all
        # exactly num_envs/P. Checked here, before the multi-minute scene build.
        if len(p_names) > 1:
            import xml.etree.ElementTree as _ET
            layouts = {}
            for nm, urdf in zip(p_names, p_obj_urdfs):
                root = _ET.parse(urdf).getroot()
                layouts[nm] = tuple(l.get("name") for l in root.iter("link"))
            distinct = set(layouts.values())
            if len(distinct) > 1:
                detail = "; ".join(f"{n}: {list(v)}" for n, v in layouts.items())
                raise ValueError(
                    "multi-problem requires all insertion objects to share link "
                    "structure (same body count and names), but they differ -- "
                    f"{detail}. Group problems with matching object skeletons."
                )

        self._pih_problem_names = p_names
        self._pih_problems = p_objs
        self._pih_object_urdfs = p_obj_urdfs
        self._pih_receptive_urdfs = p_rec_urdfs
        self._pih_object_scales = p_scales
        self._pih_hole_z_offsets = p_hole_z
        self._pih_insert_pose_seqs = p_seqs
        self._pih_num_tail_goals = p_tail
        self._pih_slot_problem_idx = slot_problem_idx
        self._num_problems = len(p_names)

        # transportPreInsertFinal prepends two per-episode prelude waypoints
        # (lift-in-place, over-hole) onto the hole-frame insertion sequence.
        # The lift height is carried by the Problem (prelude_lift_offset).
        if pih_cfg.goal_mode == "transportPreInsertFinal":
            if float(problem.prelude_lift_offset) <= 0.0:
                raise ValueError(
                    f"goal_mode='transportPreInsertFinal' requires problem "
                    f"{problem_name!r} to set prelude_lift_offset > 0; got "
                    f"{problem.prelude_lift_offset}."
                )
            self._num_prelude_goals = 2
            self._prelude_lift_offset = float(problem.prelude_lift_offset)
        else:
            self._num_prelude_goals = 0
            self._prelude_lift_offset = 0.0
        self._num_total_insertion_goals = (
            self._num_insertion_goals + self._num_prelude_goals
        )
        self._pih_num_prelude_goals = p_prelude
        self._pih_prelude_lift_offs = p_lift
        self._pih_num_total_goals = [t + pr for t, pr in zip(p_tail, p_prelude)]
        if self._num_problems > 1:
            print(
                f"[PegInHoleEnv] problems={p_names} mix={mix} "
                f"slots={slot_problem_idx}",
                flush=True,
            )

        cfg.assets.object_name = problem.insertion_object_name
        cfg.assets.object_urdf = self._pih_object_urdf_abs
        cfg.assets.receptive_urdf = self._pih_receptive_urdf_abs
        cfg.assets.object_scale = self._pih_object_scale
        cfg.assets.table_urdf = "assets/urdf/table_narrow.urdf"

        random_max = int(pih_cfg.random_goal_max_successes)
        if random_goal_fraction > 0.0:
            cfg.termination.max_consecutive_successes = max(
                self._num_total_insertion_goals, random_max
            )
        else:
            cfg.termination.max_consecutive_successes = self._num_total_insertion_goals

    def _check_problem_assignment(self, num_problems: int, pidx: torch.Tensor) -> None:
        """Two guards on the env -> problem map, run once at construction.

        Everything upstream trusts that find_matching_prim_paths returns the
        spawner's ordering. If that ever broke, every env would get someone
        else's geometry AND waypoints, and training would look noisy-but-alive
        rather than failing. These checks do not share that assumption.
        """
        # (1) Order-independent physical cross-check: PhysX mass is read back
        # from the spawned bodies, so it cannot be fooled by a wrong ordering.
        # Within a problem group it must be constant.
        # A row-count mismatch here is NOT a limitation of the diagnostic -- it
        # means the RigidObject's physics view does not cover every env, i.e.
        # the multi-asset spawn is malformed. Observed at P=4 / 12288 envs:
        # default_mass came back with num_envs/P rows, and the run then died in
        # PhysX with "CUDA error: device-side assert triggered / Failed to
        # submit rigid body transforms" a few minutes later. Treating it as a
        # soft warning let that run burn 14.7 GPU-hours as a zombie, so it is a
        # hard error: fail at construction, loudly, in seconds.
        try:
            # default_mass lives on CPU in Isaac Lab while pidx is on the sim
            # device, so the mask has to be brought onto one of them.
            mass = self.object.data.default_mass[:, 0].to(pidx.device)
        except Exception as exc:  # pragma: no cover
            print(
                f"[PegInHoleEnv][warn] could not read default_mass ({exc}); "
                "skipping the mass cross-check.",
                flush=True,
            )
        else:
            if mass.shape[0] != pidx.shape[0]:
                raise RuntimeError(
                    f"Object physics view covers {mass.shape[0]} bodies but the "
                    f"scene has {pidx.shape[0]} envs "
                    f"(= num_envs/{num_problems}). RigidObject binds a single "
                    "PhysX view, which needs a consistent body layout across "
                    "the spawned assets; with mixed layouts it binds only one "
                    "asset's share and PhysX then fails on the first physics "
                    "step. This is NOT a scale or problem-count limit -- the "
                    "objects must share link structure. See the Phase A check."
                )
            means, spreads = [], []
            for p in range(num_problems):
                g = mass[pidx == p]
                if g.numel() == 0:
                    raise RuntimeError(f"problem {p} was assigned no envs")
                means.append(g.mean())
                spreads.append(g.std() if g.numel() > 1 else torch.zeros((), device=g.device))
            means_t, spreads_t = torch.stack(means), torch.stack(spreads)
            tol = 1e-6 * float(means_t.abs().max().clamp_min(1e-9))
            if float(spreads_t.max()) > max(tol, 1e-9):
                raise RuntimeError(
                    "env->problem map is wrong: object mass varies WITHIN a "
                    f"problem group (per-group std {spreads_t.tolist()}, "
                    f"means {means_t.tolist()})."
                )
            uniq = torch.unique(torch.round(means_t / max(tol, 1e-9)))
            if uniq.numel() < num_problems:
                print(
                    "[PegInHoleEnv][warn] two problems share an object mass; the "
                    "mass cross-check cannot distinguish them (inconclusive, not "
                    "a failure).", flush=True,
                )
            else:
                print(f"[PegInHoleEnv] mass cross-check OK: {means_t.tolist()}", flush=True)

        # (2) SAPG blocks are contiguous env ranges and rl_games reports stats
        # from the LAST block only, so a skewed block would make the headline
        # number describe a subset of the mix.
        block = int(getattr(self.cfg, "expl_coef_block_size", 0) or 0)
        if block > 0 and self.num_envs % block == 0 and self.num_envs > block:
            frac = torch.stack([
                (pidx.view(-1, block) == p).float().mean(dim=1) for p in range(num_problems)
            ])                                            # (P, n_blocks)
            target = (torch.bincount(pidx, minlength=num_problems).float()
                      / float(self.num_envs)).unsqueeze(1)
            worst = float((frac - target).abs().max())
            print(f"[PegInHoleEnv] SAPG block mix: worst deviation "
                  f"{100 * worst:.2f}% over {frac.shape[1]} blocks", flush=True)
            if worst > 0.10:
                raise RuntimeError(
                    f"problem mix is unbalanced across SAPG blocks (worst "
                    f"deviation {100 * worst:.1f}%); reported metrics would "
                    "describe a subset of the tasks."
                )

    def _override_goal_counts(self, prelude: int, tail: int) -> None:
        """Reshape the goal budget after _configure_problem.

        Single-problem subclasses (PegInHoleFixturedEnv) source their trajectory
        from scenes.npz rather than from the Problem, so they rewrite the goal
        counts. They must go through here rather than assigning the scalars
        directly: Phase C builds its (P,) tables from the *lists*, so a bare
        scalar write would be silently ignored and the env would get the wrong
        env_max_goals.
        """
        self._num_prelude_goals = int(prelude)
        self._num_insertion_goals = int(tail)
        self._num_total_insertion_goals = int(prelude) + int(tail)
        self._pih_num_prelude_goals = [int(prelude)]
        self._pih_num_tail_goals = [int(tail)]
        self._pih_num_total_goals = [int(prelude) + int(tail)]

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if hasattr(self, "prev_episode_env_max_goals"):
            self.prev_episode_env_max_goals[env_ids] = self.env_max_goals[env_ids]
            self.prev_episode_is_random_goal[env_ids] = self.is_random_goal_env[env_ids]
        super()._reset_idx(env_ids)
        self._reset_peg_episode(env_ids)

    def _reset_peg_episode(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        pih_cfg = self.cfg.peg_in_hole
        env_origins = self.scene.env_origins[env_ids]
        is_random_goal = self.is_random_goal_env[env_ids]

        insertion_max = self._num_total_goals_env[env_ids]
        random_max = torch.full(
            (n,),
            int(pih_cfg.random_goal_max_successes),
            dtype=torch.long,
            device=self.device,
        )
        self.env_max_goals[env_ids] = torch.where(
            is_random_goal, random_max, insertion_max
        )

        table_top_z = self._table_z_per_env[env_ids] + TABLE_HALF_HEIGHT

        # Object pose was already written by SimToolRealEnv._reset_object_pose
        # (called from super()._reset_idx via reset_env_state). It honors
        # cfg.reset.fixed_start_pose when set, otherwise samples
        # cfg.reset.reset_position_noise_x/y/z + random_orientation. We do not
        # override it here.

        hole_x_min, hole_x_max = (float(v) for v in pih_cfg.hole_x_range)
        hole_y_min, hole_y_max = (float(v) for v in pih_cfg.hole_y_range)
        self.hole_pos[env_ids, 0] = torch.empty(n, device=self.device).uniform_(
            hole_x_min, hole_x_max
        )
        self.hole_pos[env_ids, 1] = torch.empty(n, device=self.device).uniform_(
            hole_y_min, hole_y_max
        )
        self.hole_pos[env_ids, 2] = table_top_z + self._hole_z_offset_env[env_ids]
        if is_random_goal.any():
            rg_ids = env_ids[is_random_goal]
            self.hole_pos[rg_ids, 0:2] = 0.0
            self.hole_pos[rg_ids, 2] = -1.0

        yaw_range_deg = float(pih_cfg.hole_yaw_range_deg)
        if yaw_range_deg > 0.0:
            yaw = (
                torch.rand(n, device=self.device) * 2.0 - 1.0
            ) * yaw_range_deg * (math.pi / 180.0)
            z_axis = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=torch.float32
            ).expand(n, -1)
            hole_quat = quat_from_angle_axis(yaw, z_axis)
        else:
            hole_quat = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32
            ).unsqueeze(0).expand(n, -1).contiguous()
        self.hole_quat_wxyz[env_ids] = hole_quat
        hole_pose = torch.cat([self.hole_pos[env_ids] + env_origins, hole_quat], dim=-1)
        self.hole.write_root_pose_to_sim(hole_pose, env_ids=env_ids)
        self.hole.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids
        )

        self.goal_pos_obs_noise[env_ids] = 0.0
        self.goal_yaw_obs_noise[env_ids] = 0.0
        noise = float(pih_cfg.goal_xy_obs_noise)
        insertion_mask = ~is_random_goal
        if noise > 0.0 and insertion_mask.any():
            ins_ids = env_ids[insertion_mask]
            self.goal_pos_obs_noise[ins_ids, 0:2] = torch.empty(
                ins_ids.numel(), 2, device=self.device
            ).uniform_(-noise, noise)
        yaw_noise_deg = float(pih_cfg.goal_yaw_obs_noise_deg)
        if yaw_noise_deg > 0.0 and insertion_mask.any():
            ins_ids = env_ids[insertion_mask]
            yaw_max = yaw_noise_deg * (math.pi / 180.0)
            self.goal_yaw_obs_noise[ins_ids] = torch.empty(
                ins_ids.numel(), device=self.device
            ).uniform_(-yaw_max, yaw_max)

        self.retract_phase[env_ids] = False
        self.retract_succeeded[env_ids] = False
        self._just_entered_retract[env_ids] = False
        self._just_retracted[env_ids] = False

        # Build the per-env world-frame prelude (lift_in_place, over_hole) for
        # insertion-only envs when goal_mode=transportPreInsertFinal. Both
        # waypoints share the pre-insert orientation (so the policy reorients
        # during the lift) and sit at the same height = start_z + lift_offset.
        # Per-env prelude mask: a mixed batch can hold problems with and without
        # a prelude, so this is a mask rather than a global `if`.
        build_prelude = insertion_mask & (self._num_prelude_goals_env[env_ids] > 0)
        if build_prelude.any():
            ins_ids = env_ids[build_prelude]
            ins_prob = self._problem_idx_per_env[ins_ids]
            # post-reset peg world position (local to env_origin)
            start_pos = (
                self.object.data.root_pos_w[ins_ids]
                - self.scene.env_origins[ins_ids]
            )
            lift_z = start_pos[:, 2] + self._prelude_lift_off_env[ins_ids]
            # Pre-insert orientation is the orientation of the first hole-frame
            # tail waypoint (transport_above / pre_insert / final all share it).
            tail0_q = self._insert_quat_wxyz_p[ins_prob, 0]
            pre_insert_quat_world = quat_mul(self.hole_quat_wxyz[ins_ids], tail0_q)
            # waypoint 0: lift_in_place — directly above start XY
            self._prelude_pose_world[ins_ids, 0, 0] = start_pos[:, 0]
            self._prelude_pose_world[ins_ids, 0, 1] = start_pos[:, 1]
            self._prelude_pose_world[ins_ids, 0, 2] = lift_z
            self._prelude_pose_world[ins_ids, 0, 3:7] = pre_insert_quat_world
            # waypoint 1: over_hole — directly above hole XY, same height
            self._prelude_pose_world[ins_ids, 1, 0] = self.hole_pos[ins_ids, 0]
            self._prelude_pose_world[ins_ids, 1, 1] = self.hole_pos[ins_ids, 1]
            self._prelude_pose_world[ins_ids, 1, 2] = lift_z
            self._prelude_pose_world[ins_ids, 1, 3:7] = pre_insert_quat_world

        self._clear_goal_trackers(env_ids)
        self._write_goal_pose(env_ids, is_first_goal=True)

    def _write_goal_pose(self, env_ids: torch.Tensor, is_first_goal: bool = False,
                         subgoal_override: torch.Tensor | None = None) -> None:
        """subgoal_override: per-env subgoal index, bypassing the `_successes %
        env_max_goals` wrap. The wrap sends a completed env (successes ==
        max_goals) back to subgoal 0, which is correct for the random-goal
        cycle but wrong for a caller that re-writes the goal every step -- it
        would snap the marker back to pre-insert the moment insertion lands.
        See _refresh_free_fixture_goal."""
        n = env_ids.numel()
        env_origins = self.scene.env_origins[env_ids]
        is_random_goal = self.is_random_goal_env[env_ids]

        pos_local = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(n, -1).clone()

        insertion_mask = ~is_random_goal
        if insertion_mask.any():
            ins_ids = env_ids[insertion_mask]
            if subgoal_override is not None:
                subgoal_idx = subgoal_override[insertion_mask].long()
            else:
                subgoal_idx = (
                    self._successes[ins_ids] % self.env_max_goals[ins_ids]
                ).long()
            ins_pos_local = torch.zeros(
                ins_ids.numel(), 3, dtype=torch.float32, device=self.device
            )
            ins_quat = torch.zeros(
                ins_ids.numel(), 4, dtype=torch.float32, device=self.device
            )
            ins_quat[:, 0] = 1.0

            # Prelude path: world-frame poses prebuilt at reset.
            in_prelude = subgoal_idx < self._num_prelude_goals_env[ins_ids]
            if in_prelude.any():
                pre_ids = ins_ids[in_prelude]
                pre_idx = subgoal_idx[in_prelude]
                pre_pose = self._prelude_pose_world[pre_ids, pre_idx]
                ins_pos_local[in_prelude] = pre_pose[:, 0:3]
                ins_quat[in_prelude] = pre_pose[:, 3:7]

            # Tail path: hole-frame poses, transformed by current hole pose.
            in_tail = ~in_prelude
            if in_tail.any():
                tail_ids = ins_ids[in_tail]
                tail_prob = self._problem_idx_per_env[tail_ids]
                tail_idx = subgoal_idx[in_tail] - self._num_prelude_goals_env[tail_ids]
                # subgoal_idx < env_max_goals == total_goals, so this must hold.
                # Assert rather than clamp: a clamp would silently retarget a
                # different waypoint, whereas reading padding gives NaN.
                assert bool(
                    (tail_idx >= 0).all()
                    and (tail_idx < self._num_tail_goals_p[tail_prob]).all()
                ), "tail subgoal index out of range for its problem"
                hole_q = self.hole_quat_wxyz[tail_ids]
                insert_pos = self._insert_pos_rel_p[tail_prob, tail_idx]
                insert_q = self._insert_quat_wxyz_p[tail_prob, tail_idx]
                ins_pos_local[in_tail] = (
                    self.hole_pos[tail_ids] + quat_apply(hole_q, insert_pos)
                )
                ins_quat[in_tail] = quat_mul(hole_q, insert_q)

            pos_local[insertion_mask] = ins_pos_local
            quat[insertion_mask] = ins_quat

        if is_random_goal.any():
            rg_ids = env_ids[is_random_goal]
            reset_cfg = self.cfg.reset
            if reset_cfg.fixed_goal_pose is not None:
                fixed = torch.as_tensor(
                    reset_cfg.fixed_goal_pose,
                    device=self.device,
                    dtype=torch.float32,
                )
                pos_local[is_random_goal] = fixed[:3].unsqueeze(0).expand(
                    rg_ids.numel(), -1
                )
                quat[is_random_goal] = fixed[3:].unsqueeze(0).expand(
                    rg_ids.numel(), -1
                )
            elif is_first_goal or reset_cfg.goal_sampling_type == "absolute":
                rg_pos, rg_quat = sample_absolute_goal_pose(
                    mins=reset_cfg.target_volume_mins,
                    maxs=reset_cfg.target_volume_maxs,
                    scale=reset_cfg.target_volume_region_scale,
                    n_envs=rg_ids.numel(),
                    device=self.device,
                )
                pos_local[is_random_goal] = rg_pos
                quat[is_random_goal] = rg_quat
            elif reset_cfg.goal_sampling_type == "delta":
                rg_origins = self.scene.env_origins[rg_ids]
                prev_pos = self.goal_viz.data.root_pos_w[rg_ids] - rg_origins
                prev_quat = self.goal_viz.data.root_quat_w[rg_ids]
                rg_pos, rg_quat = sample_delta_goal_pose(
                    prev_pos=prev_pos,
                    prev_quat_wxyz=prev_quat,
                    delta_distance=reset_cfg.delta_goal_distance,
                    delta_rotation_degrees=reset_cfg.delta_rotation_degrees,
                    mins=reset_cfg.target_volume_mins,
                    maxs=reset_cfg.target_volume_maxs,
                    scale=reset_cfg.target_volume_region_scale,
                )
                pos_local[is_random_goal] = rg_pos
                quat[is_random_goal] = rg_quat
            else:
                raise ValueError(
                    "cfg.reset.goal_sampling_type must be 'delta' or 'absolute' "
                    f"for PegInHole random-goal envs, got {reset_cfg.goal_sampling_type!r}."
                )

        pose = torch.cat([pos_local + env_origins, quat], dim=-1)
        self.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.goal_viz.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids
        )

    def _clear_goal_trackers(self, env_ids: torch.Tensor) -> None:
        self._closest_keypoint_max_dist[env_ids] = -1.0
        self._closest_fingertip_dist[env_ids] = -1.0
        self._near_goal_steps[env_ids] = 0

    def _keypoint_success_tolerance_m(self) -> torch.Tensor:
        pih_cfg = self.cfg.peg_in_hole
        keypoint_scale = self.cfg.reward.keypoint_scale
        fixed_insertion_tol = pih_cfg.insertion_success_tolerance * keypoint_scale
        curriculum_tol = self._current_success_tolerance * keypoint_scale

        tol = torch.full(
            (self.num_envs,),
            float(fixed_insertion_tol),
            dtype=torch.float32,
            device=self.device,
        )
        if float(pih_cfg.random_goal_fraction) > 0.0:
            tol = torch.where(
                self.is_random_goal_env,
                torch.full_like(tol, float(curriculum_tol)),
                tol,
            )

        # In transportPreInsertFinal mode the coarse waypoints (prelude +
        # transport_above) use the curriculum tolerance (starts at
        # success_tolerance, shrinks toward target_success_tolerance), while
        # pre_insert and final keep the tight fixed insertion_success_tolerance.
        # The number of "coarse" stages is the prelude (2) plus the leading
        # transport_above pose (1) in the hole-frame tail = 3.
        has_prelude = self._num_prelude_goals_env > 0
        if pih_cfg.goal_mode == "transportPreInsertFinal" and bool(has_prelude.any()):
            num_coarse = self._num_prelude_goals_env + 1  # +1 = transport_above
            subgoal_idx = (self._successes % self.env_max_goals).long()
            # `& has_prelude` matters only in a mixed batch: a problem without a
            # prelude must keep the tight fixed tolerance.
            is_coarse = (
                (subgoal_idx < num_coarse) & ~self.is_random_goal_env & has_prelude
            )
            if is_coarse.any():
                tol = torch.where(
                    is_coarse,
                    torch.full_like(tol, float(curriculum_tol)),
                    tol,
                )
        return tol

    def _curriculum_eligible_mask(self) -> torch.Tensor | None:
        pih_cfg = self.cfg.peg_in_hole
        if pih_cfg.goal_mode == "transportPreInsertFinal":
            # Drive the curriculum off insertion-only envs in dense-traj mode.
            return ~self.is_random_goal_env
        if float(pih_cfg.random_goal_fraction) <= 0.0:
            return None
        return self.is_random_goal_env

    def _curriculum_success_threshold(self) -> float | None:
        pih_cfg = self.cfg.peg_in_hole
        if pih_cfg.goal_mode == "transportPreInsertFinal":
            # 5 subgoals; require averaging the 3 coarse waypoints before
            # we tighten — keeps the curriculum from shrinking on partial lifts.
            # Must stay scalar: update_tolerance_curriculum compares it against
            # successes.mean(). Averaged over the curriculum-eligible envs, which
            # equals (_num_prelude_goals + 1) exactly when P == 1.
            return float(self._curriculum_threshold_dense)
        if float(pih_cfg.random_goal_fraction) <= 0.0:
            return None
        return float(pih_cfg.random_goal_curriculum_success_threshold)

    def _wrench_dr_active_mask(self) -> torch.Tensor:
        """Disable wrench DR once the final insert is achieved (retract phase).

        The peg is unconstrained inside the hole during retract — random
        impulses there would knock it out before the fingers clear the
        keepout, defeating the retract reward.
        """
        return ~self.retract_phase

    def _refresh_free_fixture_goal(self) -> None:
        """Re-derive the insertion goal from the fixture's live pose.

        With ``fixture_bolted=False`` the hole is a dynamic body, so the peg can
        shove it. ``hole_pos`` / ``hole_quat_wxyz`` are otherwise only written at
        reset (they are pushed *to* sim, never read back), and the goal is only
        rewritten at reset and on subgoal transitions -- so both would go stale
        the moment the fixture moved, leaving the goal where the hole used to be.

        Reading the pose back each step and rewriting the goal keeps ``goal_viz``
        -- which is what the policy actually observes (``obs_utils`` reads
        ``env.goal_viz.data.root_pos_w``) -- glued to the fixture. The
        observation layout is unchanged, so no retraining is required.

        Only non-random-goal envs currently in the tail (post-prelude) phase are
        refreshed: random-goal envs have targets sampled at reset that must not
        be overwritten, and prelude goals are prebuilt world-frame poses.
        """
        if self.cfg.peg_in_hole.fixture_bolted:
            return

        insertion = ~self.is_random_goal_env
        if not bool(insertion.any()):
            return

        # Only insertion envs: random-goal envs park the hole at a z=-1 sentinel
        # at reset (see _reset_peg_episode), which must not be overwritten.
        env_origins = self.scene.env_origins
        live_pos = self.hole.data.root_pos_w - env_origins
        self.hole_pos[insertion] = live_pos[insertion]
        self.hole_quat_wxyz[insertion] = self.hole.data.root_quat_w[insertion]

        # Clamp instead of wrapping: an env that has hit every goal is in the
        # retract phase and its target must STAY at the final insertion pose
        # (still tracking the fixture, so retract is judged against where the
        # hole actually is now). The modulo used elsewhere would send it back to
        # subgoal 0 and yank the marker to pre-insert.
        subgoal_idx = torch.minimum(self._successes, self.env_max_goals - 1).long()
        eligible = insertion & (subgoal_idx >= self._num_prelude_goals_env)
        ids = eligible.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            self._write_goal_pose(ids, is_first_goal=False,
                                  subgoal_override=subgoal_idx[ids])

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        # Must precede compute_intermediate_values so rewards, terminations and
        # observations all see the same fixture-tracked goal this step.
        self._refresh_free_fixture_goal()
        compute_intermediate_values(self)

        pih_cfg = self.cfg.peg_in_hole
        is_success = self._is_success.clone()
        if pih_cfg.enable_retract:
            is_success &= ~self.retract_phase
        self._is_success = is_success

        self._successes += is_success.long()
        self._successes.copy_(torch.minimum(self._successes, self.env_max_goals))
        success_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
        if success_ids.numel() > 0:
            self.episode_length_buf[success_ids] = 0

        self._just_entered_retract[:] = False
        self._just_retracted[:] = False
        if pih_cfg.enable_retract:
            just_entered = (
                (self._successes >= self.env_max_goals)
                & ~self.retract_phase
                & ~self.is_random_goal_env
            )
            self._just_entered_retract.copy_(just_entered)
            self.retract_phase |= just_entered

            object_at_goal = (
                self._keypoints_max_dist
                <= pih_cfg.retract_success_tolerance * self.cfg.reward.keypoint_scale
            )
            mean_fingertip_dist = self._curr_fingertip_distances.mean(dim=-1)
            just_retracted = (
                (mean_fingertip_dist > pih_cfg.retract_distance_threshold)
                & self.retract_phase
                & ~self.retract_succeeded
                & object_at_goal
            )
            self._just_retracted.copy_(just_retracted)
            self.retract_succeeded |= just_retracted

        if success_ids.numel() > 0:
            next_goal = (
                is_success
                & (self._successes < self.env_max_goals)
                & ~self.retract_phase
            )
            next_goal_ids = next_goal.nonzero(as_tuple=False).squeeze(-1)
            if next_goal_ids.numel() > 0:
                self._clear_goal_trackers(next_goal_ids)
                self._write_goal_pose(next_goal_ids, is_first_goal=False)

        object_z_local = self.object.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        fall = object_z_local < 0.1
        if pih_cfg.enable_retract:
            random_goals_done = (
                (self._successes >= self.env_max_goals) & self.is_random_goal_env
            )
            max_successes = self.retract_succeeded | random_goals_done
            hand_far = (
                self._curr_fingertip_distances.max(dim=-1).values > 1.5
            ) & ~self.retract_phase
        else:
            max_successes = self._successes >= self.env_max_goals
            hand_far = self._curr_fingertip_distances.max(dim=-1).values > 1.5

        if pih_cfg.enable_dropped_on_table_term:
            table_top_local = self._table_z_per_env + TABLE_HALF_HEIGHT
            mean_ft_dist = self._curr_fingertip_distances.mean(dim=-1)
            dropped = (
                (object_z_local < table_top_local + pih_cfg.dropped_on_table_z_margin)
                & (mean_ft_dist > pih_cfg.dropped_on_table_ft_distance)
                & ~self.retract_phase
            )
        else:
            dropped = torch.zeros_like(fall)

        terminated = fall | max_successes | hand_far | dropped
        truncated = self.episode_length_buf >= self.max_episode_length
        self._termination_reasons = {
            "fall": fall,
            "max_successes": max_successes,
            "hand_far": hand_far,
            "dropped": dropped,
            "timeout": truncated,
        }
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        pih_cfg = self.cfg.peg_in_hole

        # The lift bonus is a single training-wide curriculum stage, so the latch
        # stays global. But the fade must be gated on the SLOWEST problem: a mean
        # pooled over all random-goal envs lets the easiest problem drag the
        # average past the threshold and switch off lift shaping (scale 20 /
        # 300) for problems still in the lift phase -- silently, since nothing
        # logs per-problem. Taking the min over per-problem means keeps one
        # curriculum for the whole run while removing that failure. Reduces
        # exactly to the pooled mean when P == 1.
        mean_rg_eps = 0.0
        fade_stat = 0.0
        rg = self.is_random_goal_env
        if rg.any():
            rg_succ = self._prev_episode_successes[rg].float()
            mean_rg_eps = rg_succ.mean().item()
            if getattr(self, "_num_problems", 1) > 1:
                pidx_rg = self._problem_idx_per_env[rg]
                per_problem = [
                    rg_succ[pidx_rg == p].mean().item()
                    for p in range(self._num_problems)
                    if bool((pidx_rg == p).any())
                ]
                # A problem with no random-goal envs cannot report progress, so
                # it is excluded rather than counted as 0 (which would freeze
                # the curriculum forever).
                fade_stat = min(per_problem) if per_problem else 0.0
                self.extras["lift_fade_stat_min_over_problems"] = fade_stat
            else:
                fade_stat = mean_rg_eps
        if (
            self.lift_bonus_active
            and fade_stat >= float(pih_cfg.lift_bonus_fade_threshold)
        ):
            self.lift_bonus_active = False

        if self.lift_bonus_active:
            if pih_cfg.force_lift_reward_active:
                lift_mask = torch.ones(
                    self.num_envs, dtype=torch.float32, device=self.device
                )
            else:
                lift_mask = self.is_random_goal_env.float()
        else:
            lift_mask = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        lift_rew = self._reward_terms["lifting_rew"]
        lift_bonus_rew = self._reward_terms["lift_bonus_rew"]
        masked_lift_rew = lift_rew * lift_mask
        masked_lift_bonus_rew = lift_bonus_rew * lift_mask
        reward = reward - lift_rew - lift_bonus_rew + masked_lift_rew + masked_lift_bonus_rew
        self._reward_terms["lifting_rew"] = masked_lift_rew
        self._reward_terms["lift_bonus_rew"] = masked_lift_bonus_rew
        self._reward_terms["total_reward"] = reward
        self.extras["lift_bonus_active"] = float(self.lift_bonus_active)
        self.extras["mean_rg_prev_ep_successes"] = mean_rg_eps

        if pih_cfg.enable_retract:
            object_at_goal = (
                self._keypoints_max_dist
                <= pih_cfg.retract_success_tolerance * self.cfg.reward.keypoint_scale
            ).float()
            mean_fingertip_dist = self._curr_fingertip_distances.mean(dim=-1)
            retract_rew = (
                mean_fingertip_dist * pih_cfg.retract_reward_scale * object_at_goal
                + self._just_retracted.float() * self.retract_success_bonus
            ) * self.retract_phase.float()

            already_in_retract = self.retract_phase & ~self._just_entered_retract
            action_penalty = (
                self._reward_terms["kuka_actions_penalty"]
                + self._reward_terms["hand_actions_penalty"]
            )
            reward = torch.where(already_in_retract, action_penalty + retract_rew, reward)
            self._reward_terms["retract_rew"] = retract_rew
            self._reward_terms["total_reward"] = reward

            self.extras["retract_phase_ratio"] = self.retract_phase.float().mean()
            self.extras["retract_success_ratio"] = self.retract_succeeded.float().mean()
            self.extras["retract_success_tolerance"] = float(
                pih_cfg.retract_success_tolerance
            )
            self.extras["retract_success_bonus"] = self.retract_success_bonus

        log_step_metrics(self)
        self._log_peg_metrics()
        return reward

    def _log_peg_metrics(self) -> None:
        success_ratio = self._successes.float() / self.env_max_goals.clamp_min(1).float()
        all_goals_hit = self._successes >= self.env_max_goals

        episode_final = self.extras.setdefault("episode_final", {})
        episode_final["success_ratio"] = success_ratio
        episode_final["all_goals_hit"] = all_goals_hit.float()
        if self.cfg.peg_in_hole.enable_retract:
            episode_final["retract_success"] = self.retract_succeeded.float()

        prev_ratio = (
            self._prev_episode_successes.float()
            / self.prev_episode_env_max_goals.clamp_min(1).float()
        )
        self.extras["success_ratio"] = prev_ratio.mean()
        self.extras["all_goals_hit_ratio"] = (
            self._prev_episode_successes >= self.prev_episode_env_max_goals
        ).float().mean()
        self.extras["insertion_success_tolerance"] = float(
            self.cfg.peg_in_hole.insertion_success_tolerance
        )

        # Per-problem breakdown. Without it a co-trained run reports one pooled
        # number and you cannot tell whether co-training helped or whether the
        # easiest task is carrying the mean -- which is the entire question the
        # experiment exists to answer.
        if getattr(self, "_num_problems", 1) > 1:
            hit = (
                self._prev_episode_successes >= self.prev_episode_env_max_goals
            ).float()
            ins_only = ~self.prev_episode_is_random_goal
            for p, name in enumerate(self._pih_problem_names):
                m = (self._problem_idx_per_env == p) & ins_only
                if not bool(m.any()):
                    continue
                self.extras[f"problem/{name}/all_goals_hit_ratio"] = hit[m].mean()
                self.extras[f"problem/{name}/success_ratio"] = prev_ratio[m].mean()
                if self.cfg.peg_in_hole.enable_retract:
                    self.extras[f"problem/{name}/retract_success"] = (
                        self.retract_succeeded[m].float().mean()
                    )

        if float(self.cfg.peg_in_hole.random_goal_fraction) > 0.0:
            prev_s = self._prev_episode_successes
            prev_mg = self.prev_episode_env_max_goals.clamp_min(1).float()
            ins_mask = ~self.prev_episode_is_random_goal
            rg_mask = self.prev_episode_is_random_goal
            self.extras["random_goal_frac"] = self.is_random_goal_env.float().mean()
            if ins_mask.any():
                self.extras["insertion_success_ratio"] = (
                    prev_s[ins_mask].float() / prev_mg[ins_mask]
                ).mean()
                self.extras["insertion_all_goals_hit_ratio"] = (
                    prev_s[ins_mask] >= self.prev_episode_env_max_goals[ins_mask]
                ).float().mean()
            else:
                self.extras["insertion_success_ratio"] = 0.0
                self.extras["insertion_all_goals_hit_ratio"] = 0.0
            if rg_mask.any():
                self.extras["random_goal_success_ratio"] = (
                    prev_s[rg_mask].float() / prev_mg[rg_mask]
                ).mean()
                self.extras["random_goal_all_goals_hit_ratio"] = (
                    prev_s[rg_mask] >= self.prev_episode_env_max_goals[rg_mask]
                ).float().mean()
            else:
                self.extras["random_goal_success_ratio"] = 0.0
                self.extras["random_goal_all_goals_hit_ratio"] = 0.0

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = build_observations(self)
        if self._goal_kp_obs_slice is not None:
            policy = obs["policy"]
            policy[:, self._goal_kp_obs_slice].view(self.num_envs, -1, 3).sub_(
                self.goal_pos_obs_noise.unsqueeze(1)
            )
            clip = self.cfg.obs.clamp_abs_observations
            obs["policy"] = policy.clamp(-clip, clip)

        # Depth-distillation contract: when student_obs is enabled, the env
        # exposes THREE obs groups (consumed by different parts of the dagger
        # pipeline):
        #   "policy"      = flattened student obs [image_flat, proprio] →
        #                   read by the depth-CNN student network
        #   "critic"      = state_list (privileged) → read by the asymmetric
        #                   central-value critic
        #   "teacher_obs" = obs_list (proprio + object state, noisy, what the
        #                   frozen state-MLP teacher trained on) → read by
        #                   DAggerA2CAgent for teacher labeling. The
        #                   DAggerRlGamesVecEnvWrapper passes this through as
        #                   `self.obs["teacher"]` in the agent.
        # SAPG appends 1 block-id column to "obs" and "states" only
        # (a2c_common.py:602-604) — "teacher" is left untouched, and our
        # `Teacher.get_action()` re-appends the block-id internally.
        student_cfg = getattr(self.cfg, "student_obs", None)
        if student_cfg is not None and student_cfg.enabled:
            student = self.get_student_obs()
            image_flat = student["image"].reshape(self.num_envs, -1)
            proprio = student["proprio"].reshape(self.num_envs, -1)
            student_flat = torch.cat([image_flat, proprio], dim=-1)
            obs = {
                "policy": student_flat,
                "critic": obs["critic"],
                "teacher_obs": obs["policy"],
            }
        return obs


__all__ = ["PegInHoleEnv", "PegInHoleEnvCfg"]

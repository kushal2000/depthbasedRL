#!/usr/bin/env python3
"""Direct smoke test for PegInHoleEnv (no rl_games).

Loads the env with a small number of envs, runs a reset + a few random
action steps, and asserts:
  - per-env (scene_idx, tol_slot_idx) follows the round-robin assignment,
  - env_max_goals matches traj_lengths[scene_idx, peg_idx] after reset,
  - retract state and reward tensors are initialized,
  - step() doesn't crash for a handful of zero-action steps.

Run:
    python peg_in_hole/smoke_test_env.py
    python peg_in_hole/smoke_test_env.py task.env.goalMode=finalGoalOnly
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

# isaacgym must be imported before torch — pulling it in transitively.
from isaacgymenvs.tasks import isaacgym_task_map  # noqa: E402,F401

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaacgymenvs.utils.reformat import omegaconf_to_dict  # noqa: E402


@hydra.main(
    config_name="config",
    config_path=str(REPO / "isaacgymenvs" / "cfg"),
    version_base="1.1",
)
def main(cfg):
    # Force small, headless settings for smoke.
    cfg.task.name = "PegInHoleEnv"
    cfg.task.env.numEnvs = 8
    cfg.task.env.viserViz = False
    cfg.task.env.enableCameraSensors = False
    cfg.task.env.capture_video = False
    cfg.headless = True
    cfg.capture_video = False

    cfg_dict = omegaconf_to_dict(cfg.task)
    env_cls = isaacgym_task_map[cfg.task.name]
    env = env_cls(
        cfg=cfg_dict,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )

    num_envs = env.num_envs
    num_acts = env.num_actions
    print(f"\n[smoke] Env created: num_envs={num_envs}, num_actions={num_acts}, "
          f"goal_mode={env.goal_mode}, max_consecutive_successes={env.max_consecutive_successes}")
    print(f"[smoke]   env_scene_idx    = {env._pih_env_scene_idx_t.cpu().tolist()}")
    print(f"[smoke]   env_tol_slot_idx = {env._pih_env_tol_slot_t.cpu().tolist()}")
    print(f"[smoke]   env_max_goals    = {env.env_max_goals.cpu().tolist()}")
    print(f"[smoke]   peg start xyz    =\n{env.object_init_state[:, :3].cpu().numpy()}")

    # ── Assertions on the static axis ──
    N = env._pih_num_scenes
    K = env._pih_num_tol_slots
    exp_combo = np.arange(num_envs) % (N * K)
    got_scene = env._pih_env_scene_idx_t.cpu().numpy()
    got_tol = env._pih_env_tol_slot_t.cpu().numpy()
    assert np.array_equal(got_scene, exp_combo // K), (got_scene, exp_combo // K)
    assert np.array_equal(got_tol, exp_combo % K), (got_tol, exp_combo % K)
    print("[smoke] PASS: round-robin (scene, tol) assignment")

    # ── Reset + peg_idx resampling ──
    # VecTask.reset() is a no-op (just returns obs); force a real reset by
    # triggering reset_idx on all envs (which calls reset_object_pose).
    all_envs = torch.arange(num_envs, device="cuda:0")
    env.reset_idx(all_envs, tensor_reset=True)
    print(f"\n[smoke] After forced reset: env_peg_idx = {env.env_peg_idx.cpu().tolist()}")

    scenes = env._pih_env_scene_idx_t
    pegs = env.env_peg_idx
    expected_max_goals = env._pih_traj_lengths_t[scenes, pegs]
    assert torch.equal(env.env_max_goals, expected_max_goals), (
        env.env_max_goals, expected_max_goals
    )
    print("[smoke] PASS: env_max_goals = traj_lengths[scene_idx, peg_idx]")

    # Only xy + quat are asserted: z is set by the parent (SimToolReal) to
    # table_reset_z + tableObjectZOffset, so the peg hovers 10 cm above the
    # table and settles via gravity. The cached z in scenes.npz is ignored
    # at runtime (matches fabrica multi_init_states behavior).
    expected_poses = env._pih_start_poses_t[scenes, pegs]
    diff_xy = (env.object_init_state[:, 0:2] - expected_poses[:, 0:2]).abs().max().item()
    diff_quat = (env.object_init_state[:, 3:7] - expected_poses[:, 3:7]).abs().max().item()
    print(f"[smoke]   object_init vs expected: max |dxy|={diff_xy:.6e}  max |dquat|={diff_quat:.6e}")
    assert diff_xy < 1e-5, (env.object_init_state[:, 0:2], expected_poses[:, 0:2])
    assert diff_quat < 1e-5
    print("[smoke] PASS: object_init_state[xy, quat] matches start_poses[scene_idx, peg_idx]")

    # ── Retract state initialized ──
    assert env.retract_phase.dtype == torch.bool and env.retract_phase.shape == (num_envs,)
    assert env.retract_succeeded.dtype == torch.bool
    assert "retract_rew" in env.rewards_episode
    print("[smoke] PASS: retract_phase / retract_succeeded / rewards_episode[retract_rew] initialized")

    # ── Step with zero actions ──
    for i in range(5):
        actions = torch.zeros(num_envs, num_acts, device="cuda:0")
        obs, rew, done, info = env.step(actions)
        print(f"[smoke] step {i}: rew_mean={rew.mean().item():+.3f}  "
              f"done={int(done.sum())}  retract={int(env.retract_phase.sum())}")

    print("\n[smoke] ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()

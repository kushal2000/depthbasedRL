"""SimToolReal task — Isaac Sim port of isaacgymenvs/tasks/simtoolreal/env.py.

**Status (overnight port, 2026-04-23):** inference-only wrapper around the
existing `isaacsim_conversion.IsaacSimEnv` setup. The DirectRLEnv skeleton +
configclass exist for registry parity with Cartpole, but the full DirectRLEnv
step / reward / reset path is **not** ported — the stock isaacgymenvs env is
6000+ lines, and our goal here is to validate that the pretrained policy runs
in Isaac Sim, not to retrain from scratch.

For a working rollout: see `play_simtoolreal.py` (single-env direct-sim loop;
no `gym.make` needed). For future training: port the reward/reset/obs pipeline
from `isaacgymenvs/tasks/simtoolreal/env.py` into `SimToolRealEnv._*` hooks.

Observation dim = 140, action dim = 29 (7-DOF IIWA + 22-DOF Sharpa left hand).
"""

from __future__ import annotations

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class SimToolRealEnvCfg(DirectRLEnvCfg):
    """Configclass mirroring the key fields from isaacgymenvs/cfg/task/SimToolReal.yaml.

    Only the subset needed for single-env pretrained-policy inference is modeled.
    Training-time fields (DR ranges, reward scales, curriculum) are intentionally
    omitted — they belong on a future full-port of the env.
    """

    # DirectRLEnvCfg required fields
    decimation = 1  # control runs at physics rate (60 Hz)
    episode_length_s = 10.0
    action_space = 29
    observation_space = 140
    state_space = 140

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=1.2, replicate_physics=True, clone_in_fabric=True
    )

    # Asset paths (relative to repo root) — populated by play_simtoolreal.py
    # from gym.register kwargs. Left empty here because the configclass is shared
    # across tasks that differ only by assembly/part and the play script overrides.
    robot_urdf: str = "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    table_urdf: str = ""
    object_urdf: str = ""
    trajectory_path: str = ""

    # Recording camera pose in env-local frame (mirrors Cartpole pattern).
    # Default matches isaacsim_conversion/rollout.py: (0, -1, 1.03) looking at (0, 0, 0.53).
    record_camera_eye: tuple[float, float, float] = (0.0, -1.0, 1.03)
    record_camera_target: tuple[float, float, float] = (0.0, 0.0, 0.53)

    # Moving-average action smoothing (hardcoded in the pretrained rollout loop;
    # documented here so a future training port can parametrize them).
    hand_moving_average: float = 0.1
    arm_moving_average: float = 0.1
    dof_speed_scale: float = 1.5

    # Keypoint reward / success-detection thresholds (from SimToolReal.yaml).
    keypoint_scale: float = 1.5
    target_success_tolerance: float = 0.01
    success_steps: int = 10


class SimToolRealEnv(DirectRLEnv):
    """Stub DirectRLEnv for registry parity. Not usable for training yet.

    The pretrained policy rollout is driven by ``play_simtoolreal.py``, which
    bypasses the gym API and drives the scene directly via
    ``isaacsim_conversion.IsaacSimEnv``. That path is the one validated to work
    against the checkpoint shipped in ``pretrained_policy/model.pth``.
    """

    cfg: SimToolRealEnvCfg

    def __init__(self, cfg: SimToolRealEnvCfg, render_mode: str | None = None, **kwargs):
        raise NotImplementedError(
            "SimToolRealEnv's training path is not yet ported. "
            "Use isaacsimenvs.tasks.simtoolreal.play_simtoolreal for pretrained-policy inference."
        )

    def _setup_scene(self):
        raise NotImplementedError

    def _pre_physics_step(self, actions):
        raise NotImplementedError

    def _apply_action(self):
        raise NotImplementedError

    def _get_observations(self):
        raise NotImplementedError

    def _get_rewards(self):
        raise NotImplementedError

    def _get_dones(self):
        raise NotImplementedError

    def _reset_idx(self, env_ids):
        raise NotImplementedError

"""Config for the peg-in-hole-fixtured task variant."""

from __future__ import annotations

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaacsimenvs.tasks.peg_in_hole.peg_in_hole_env_cfg import (
    PegInHoleAssetsCfg,
    PegInHoleCfg,
    PegInHoleEnvCfg,
    _default_peg_in_hole_sim_cfg,
)


def _quality_render_sim_cfg() -> SimulationCfg:
    """Cinematic-ish render settings for the fixtured env's video capture.

    Compared to the parent's 'performance' + AA-off (training throughput), this
    enables full RTX path: global illumination, reflections, shadows, ambient
    occlusion, DL denoiser, and higher samples-per-pixel. AA via DLAA.
    """
    sim_cfg = _default_peg_in_hole_sim_cfg()
    sim_cfg.render.rendering_mode = "quality"
    sim_cfg.render.antialiasing_mode = "DLAA"
    sim_cfg.render.enable_translucency = True
    sim_cfg.render.enable_reflections = True
    sim_cfg.render.enable_global_illumination = True
    sim_cfg.render.enable_direct_lighting = True
    sim_cfg.render.enable_shadows = True
    sim_cfg.render.enable_ambient_occlusion = True
    sim_cfg.render.enable_dl_denoiser = True
    sim_cfg.render.samples_per_pixel = 64
    return sim_cfg


@configclass
class PegInHoleFixturedCfg(PegInHoleCfg):
    # Matches the trained policies' problem.
    problem: str = "peg.tol0p5mm"
    # XY of the peg-holding fixture in env-local frame (relative to env origin).
    # The fixture spawns on the table at z = table_top + hole_z_offset with
    # identity orientation. Default sits to the left of the goal-hole's
    # randomized X range so the two fixtures don't overlap.
    peg_fixture_xy: tuple[float, float] = (-0.20, 0.0)
    # If True, _reset_peg_episode writes the peg pose so its cross-arm sits
    # in the peg_fixture opening (shaft pointing down). Default ON for the
    # fixtured task — that's the env's whole purpose.
    write_peg_in_fixture: bool = True
    # Height above the fixture top at which to spawn the peg origin (peg falls
    # in under gravity over the warmup steps).
    peg_drop_height_m: float = 0.10
    # X-offset from peg origin to its cross-arm in the URDF link frame.
    # The peg's link origin is at the handle center; the cross-arm is at
    # +0.115 along the handle in raw URDF coords (the box-size values in the
    # peg URDF are in real meters — object_scale is unrelated to real dims).
    peg_link_origin_to_crossarm_x_m: float = 0.115


@configclass
class PegInHoleFixturedEnvCfg(PegInHoleEnvCfg):
    sim: SimulationCfg = _quality_render_sim_cfg()
    assets: PegInHoleAssetsCfg = PegInHoleAssetsCfg()
    peg_in_hole: PegInHoleFixturedCfg = PegInHoleFixturedCfg()
    # Camera eye/target in env-local frame (added on env_origin), consumed by
    # ``isaacsimenvs/play_video.py``. Side view across the table — looking at
    # the table from the +X side so the two fixtures sit on a horizontal line
    # (peg_fixture left-of-frame, goal-hole right-of-frame, robot in the back).
    record_camera_eye: tuple[float, float, float] = (-1.4, 0.0, 0.85)
    record_camera_target: tuple[float, float, float] = (0.05, 0.0, 0.58)


__all__ = ["PegInHoleFixturedEnvCfg", "PegInHoleFixturedCfg"]

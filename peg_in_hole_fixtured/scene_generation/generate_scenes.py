#!/usr/bin/env python3
"""Generate fixtured peg-in-hole scenes (start fixture + goal fixture).

Mirrors ``peg_in_hole/scene_generation/generate_scenes.py`` but the peg starts
*upright* in a tight (0.5 mm) starting fixture instead of lying flat on the
table. The robot must extract → translate → insert; **no in-hand
reorientation**. We emit a separate URDF per (scene, peg, tol) because the
start fixture XY is baked into the URDF and varies across the M peg slots.

Per scene we:
  1. Sample one goal hole XY.
  2. Sample M start fixture XYs that don't overlap the goal hole or the table
     edge. (They aren't constrained against each other since they never
     coexist in a single URDF — each lives in its own URDF.)
  3. For each (scene, peg): build the upright start pose and a
     lift→transit→descent trajectory to the goal-fixture insertion pose.
  4. Subsample K goal-fixture tolerances per scene from a global log-uniform
     pool. Write M × K URDFs (`scenes/scene_{n}/peg_{m}/scene_tol{k}.urdf`),
     each containing `table + start fixture (0.5mm) + goal fixture (k-th tol)`.

Output schema (`scenes.npz`):
  - ``start_poses``            : (N, M, 7) float32 — peg xyz + xyzw quat (upright in fixture).
  - ``goals``                  : (N, M, max_traj_len, 7) float32 — zero-padded.
  - ``traj_lengths``           : (N, M) int32.
  - ``start_fixture_positions``: (N, M, 3) float32 — start fixture base corner per (scene, peg).
  - ``hole_positions``         : (N, 3) float32 — goal fixture base corner per scene.
  - ``tolerance_pool_m``       : (pool_size,) float32 — meters, sorted.
  - ``scene_tolerance_indices``: (N, K) int32 — into tolerance_pool_m (goal fixture only).
  - ``start_tolerance_m``      : scalar float32 — fixed 0.5 mm.

Usage:
    python peg_in_hole_fixtured/scene_generation/generate_scenes.py
    python peg_in_hole_fixtured/scene_generation/generate_scenes.py \\
        --num-scenes 5 --pegs-per-scene 3 --tolerance-pool-size 20 \\
        --tolerances-per-scene 3 --seed 0 --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from fabrica.scene_generation.trajectory_generation import generate_variable_trajectory

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "peg_in_hole_fixtured"
SCENES_DIR = ASSETS_DIR / "scenes"

# --- World-frame constants (mirror peg_in_hole/scene_generation/generate_scenes.py) ---
TABLE_Z = 0.38
TABLE_TOP_Z = 0.53
TABLE_SIZE = (0.475, 0.4, 0.3)
TABLE_HALF_X = TABLE_SIZE[0] / 2   # 0.2375
TABLE_HALF_Y = TABLE_SIZE[1] / 2   # 0.2
EDGE_INSET = 0.05

# Peg geometry
HANDLE_EXTENTS = (0.25, 0.03, 0.02)
HEAD_EXTENTS = (0.02, 0.10, 0.02)
HANDLE_CENTER = (0.0, 0.0, 0.0)
HEAD_CENTER = (0.115, 0.0, 0.0)
HANDLE_HALF = HANDLE_EXTENTS[0] / 2  # 0.125

# Hole / fixture geometry
HOLE_FOOTPRINT_X = 0.08
HOLE_FOOTPRINT_Y = 0.08
HOLE_SLOT_CORE_X = 0.02
HOLE_SLOT_CORE_Y = 0.03
HOLE_FLOOR_THICKNESS = 0.01
HOLE_DEPTH = 0.05
HOLE_HEIGHT = HOLE_FLOOR_THICKNESS + HOLE_DEPTH      # 0.06
HOLE_SCENE_Z = 0.15                                  # fixture base in scene-local z
HOLE_BASE_Z_WORLD = TABLE_Z + HOLE_SCENE_Z           # 0.53
HOLE_FLOOR_Z_WORLD = HOLE_BASE_Z_WORLD + HOLE_FLOOR_THICKNESS   # 0.54
HOLE_TOP_Z_WORLD = HOLE_BASE_Z_WORLD + HOLE_HEIGHT             # 0.59

HOLE_COLOR = (120, 120, 120)

# Sampling box — both start and goal fixtures live in this same XY range.
FIXTURE_CENTER_XY = (0.0, 0.0)
FIXTURE_HALF_WIDTHS = (0.15, 0.11)   # table interior - 0.05 edge - 0.04 footprint

# Min center-to-center distance between start and goal fixtures.
# Two 8 cm × 8 cm fixtures need ≥ 8 cm to not overlap; 12 cm leaves room for
# the peg to clear the start fixture during the lift-up phase.
FIXTURE_SEPARATION_M = 0.12

# Tolerance pool (goal fixture). Start fixture is fixed at 0.5 mm.
TOL_MIN_M = 0.1 * 1e-3
TOL_MAX_M = 10.0 * 1e-3
START_FIXTURE_TOL_M = 0.5 * 1e-3

# Collision margins / retries
PROXIMITY_MARGIN = 0.02              # m — min peg-to-goal-fixture-block clearance during lift+transit
MAX_PLACEMENT_RETRIES = 500
MAX_GOAL_RETRIES = 50                # re-sample goal hole XY if we can't fill M start fixtures

# Trajectory
TARGET_SPACING = 0.05
LIFT_INITIAL = TABLE_TOP_Z + 0.25    # 0.78 m — lift altitude
LIFT_INCREMENT = 0.05
MAX_LIFT_ITERATIONS = 5

# Insert pose (peg upright; body +X → world +Z). Same rotation used for start
# (peg in start fixture) and end (peg in goal fixture).
PEG_INSERT_QUAT_WXYZ = (0.70710678, 0.0, -0.70710678, 0.0)
PEG_FIXTURED_CENTER_Z = HOLE_FLOOR_Z_WORLD + 0.001 + HANDLE_HALF   # 0.666 — tip 1mm above slot floor


# --- Peg mesh / pose helpers ---

def peg_canonical_mesh() -> trimesh.Trimesh:
    handle = trimesh.creation.box(extents=np.asarray(HANDLE_EXTENTS, dtype=float))
    handle.apply_translation(np.asarray(HANDLE_CENTER, dtype=float))
    head = trimesh.creation.box(extents=np.asarray(HEAD_EXTENTS, dtype=float))
    head.apply_translation(np.asarray(HEAD_CENTER, dtype=float))
    return trimesh.util.concatenate([handle, head])


def peg_mesh_at(pos, quat_wxyz):
    mesh = peg_canonical_mesh()
    T = np.eye(4)
    xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    T[:3, :3] = R.from_quat(xyzw).as_matrix()
    T[:3, 3] = pos
    out = mesh.copy()
    out.apply_transform(T)
    return out


def hole_block_mesh(hole_x, hole_y):
    """Conservative bbox of a fixture BLOCK (outer footprint) at world pose."""
    box = trimesh.creation.box(
        extents=(HOLE_FOOTPRINT_X, HOLE_FOOTPRINT_Y, HOLE_HEIGHT)
    )
    box.apply_translation((hole_x, hole_y, HOLE_BASE_Z_WORLD + HOLE_HEIGHT / 2))
    return box


# --- Hole geometry ---

def hole_boxes(tol_m):
    slot_x = HOLE_SLOT_CORE_X + 2 * tol_m
    slot_y = HOLE_SLOT_CORE_Y + 2 * tol_m
    t = HOLE_FLOOR_THICKNESS
    d = HOLE_DEPTH
    ox, oy = HOLE_FOOTPRINT_X, HOLE_FOOTPRINT_Y

    assert slot_x < ox, (
        f"slot_x {slot_x} >= outer {ox}; increase footprint or reduce tol"
    )
    assert slot_y <= oy, (
        f"slot_y {slot_y} > outer {oy}; increase footprint or reduce tol"
    )

    boxes = [((0.0, 0.0, t / 2), (ox, oy, t))]
    zc = t + d / 2
    ew = (ox - slot_x) / 2
    if ew > 1e-6:
        boxes.append(((slot_x / 2 + ew / 2, 0.0, zc), (ew, oy, d)))
        boxes.append(((-(slot_x / 2 + ew / 2), 0.0, zc), (ew, oy, d)))
    nl = (oy - slot_y) / 2
    if nl > 1e-6:
        boxes.append(((0.0, slot_y / 2 + nl / 2, zc), (slot_x, nl, d)))
        boxes.append(((0.0, -(slot_y / 2 + nl / 2), zc), (slot_x, nl, d)))
    return boxes


# --- Sampling ---

def sample_log_uniform(rng, lo, hi, size):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))


def sample_fixture_xy(rng):
    x = FIXTURE_CENTER_XY[0] + float(rng.uniform(-FIXTURE_HALF_WIDTHS[0], FIXTURE_HALF_WIDTHS[0]))
    y = FIXTURE_CENTER_XY[1] + float(rng.uniform(-FIXTURE_HALF_WIDTHS[1], FIXTURE_HALF_WIDTHS[1]))
    return x, y


def sample_start_xy_for_goal(rng, goal_x, goal_y):
    """Reject-sample a start fixture XY ≥ FIXTURE_SEPARATION_M from the goal."""
    for _ in range(MAX_PLACEMENT_RETRIES):
        x, y = sample_fixture_xy(rng)
        d = np.hypot(x - goal_x, y - goal_y)
        if d >= FIXTURE_SEPARATION_M:
            return x, y
    return None


# --- Trajectory ---

def _lift_transit_clear(waypoints, lift_transit_end, hole_cm):
    """True iff peg mesh stays ≥ PROXIMITY_MARGIN from the goal fixture during
    the lift+transit phase. The descent phase is allowed to touch the goal."""
    active_cm = trimesh.collision.CollisionManager()
    active_cm.add_object("peg", peg_canonical_mesh())
    for i in range(lift_transit_end):
        pos = waypoints[i, :3]
        qx, qy, qz, qw = waypoints[i, 3:7]
        T = np.eye(4)
        T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = pos
        active_cm.set_transform("peg", T)
        if float(active_cm.min_distance_other(hole_cm)) < PROXIMITY_MARGIN:
            return False
    return True


def generate_fixtured_trajectory(start_x, start_y, goal_x, goal_y):
    """Lift→transit→descent trajectory from in-fixture upright start to in-goal
    upright end. Returns (waypoints, lift_transit_end) or (None, None)."""
    start_pos = np.array([start_x, start_y, PEG_FIXTURED_CENTER_Z])
    end_pos = np.array([goal_x, goal_y, PEG_FIXTURED_CENTER_Z])

    hole_block = hole_block_mesh(goal_x, goal_y)
    hole_cm = trimesh.collision.CollisionManager()
    hole_cm.add_object("hole", hole_block)

    for lift_iter in range(MAX_LIFT_ITERATIONS):
        clearance = LIFT_INITIAL + lift_iter * LIFT_INCREMENT
        waypoints, lift_transit_end = generate_variable_trajectory(
            start_pos,
            PEG_INSERT_QUAT_WXYZ,
            end_pos,
            PEG_INSERT_QUAT_WXYZ,
            clearance_z=clearance,
            insertion_dir=[0.0, 0.0, -1.0],
            target_spacing=TARGET_SPACING,
        )
        if _lift_transit_clear(waypoints, lift_transit_end, hole_cm):
            return waypoints, lift_transit_end

    return None, None


# --- URDF emission ---

def _box_xml(center, extents, material=None, indent="    "):
    cx, cy, cz = center
    ex, ey, ez = extents
    mat = f'\n{indent}  <material name="{material}"/>' if material else ""
    return (
        f'{indent}<visual>\n'
        f'{indent}  <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" rpy="0 0 0"/>\n'
        f'{indent}  <geometry><box size="{ex:.6f} {ey:.6f} {ez:.6f}"/></geometry>{mat}\n'
        f'{indent}</visual>\n'
        f'{indent}<collision>\n'
        f'{indent}  <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" rpy="0 0 0"/>\n'
        f'{indent}  <geometry><box size="{ex:.6f} {ey:.6f} {ez:.6f}"/></geometry>\n'
        f'{indent}</collision>'
    )


def write_scene_urdf(out_path: Path, start_xy, start_tol_m, goal_xy, goal_tol_m,
                     scene_idx: int, peg_idx: int, tol_slot_idx: int):
    """Scene URDF: table + start fixture (always 0.5mm) + goal fixture (k-th tol)."""
    sx, sy = start_xy
    gx, gy = goal_xy
    parts = [_box_xml((0.0, 0.0, 0.0), TABLE_SIZE, material="wood")]
    for (cx, cy, cz), ext in hole_boxes(start_tol_m):
        parts.append(_box_xml(
            (cx + sx, cy + sy, cz + HOLE_SCENE_Z), ext, material="start_fix",
        ))
    for (cx, cy, cz), ext in hole_boxes(goal_tol_m):
        parts.append(_box_xml(
            (cx + gx, cy + gy, cz + HOLE_SCENE_Z), ext, material="goal_fix",
        ))

    geom_xml = "\n".join(parts)
    r, g, b = HOLE_COLOR
    robot_name = (
        f"peg_in_hole_fixtured_scene_{scene_idx:04d}"
        f"_peg{peg_idx:04d}_tol{tol_slot_idx:02d}"
    )
    xml = f'''<?xml version="1.0"?>
<robot name="{robot_name}">
  <material name="wood"><color rgba="0.82 0.56 0.35 1.0"/></material>
  <material name="start_fix"><color rgba="0.30 0.50 0.85 1.0"/></material>
  <material name="goal_fix"><color rgba="{r/255:.4f} {g/255:.4f} {b/255:.4f} 1.0"/></material>
  <link name="box">
{geom_xml}
    <inertial>
      <mass value="500"/>
      <friction value="1.0"/>
      <inertia ixx="1000.0" ixy="0.0" ixz="0.0" iyy="1000.0" iyz="0.0" izz="1000.0"/>
    </inertial>
  </link>
</robot>
'''
    out_path.write_text(xml)


# --- Main pipeline ---

def generate_scenes(num_scenes, pegs_per_scene, tol_pool_size, tol_per_scene, seed):
    rng = np.random.default_rng(seed)

    tol_pool_m = np.sort(
        sample_log_uniform(rng, TOL_MIN_M, TOL_MAX_M, tol_pool_size)
    ).astype(np.float32)
    print(
        f"Tolerance pool: {tol_pool_size} log-uniform values in "
        f"[{tol_pool_m[0]*1000:.4f}, {tol_pool_m[-1]*1000:.4f}] mm. "
        f"Start fixture fixed at {START_FIXTURE_TOL_M*1000:.2f} mm."
    )

    start_poses = np.zeros((num_scenes, pegs_per_scene, 7), dtype=np.float32)
    start_fixture_positions = np.zeros((num_scenes, pegs_per_scene, 3), dtype=np.float32)
    hole_positions = np.zeros((num_scenes, 3), dtype=np.float32)
    scene_tolerance_indices = np.zeros((num_scenes, tol_per_scene), dtype=np.int32)
    traj_lengths = np.zeros((num_scenes, pegs_per_scene), dtype=np.int32)
    all_waypoints = [[None] * pegs_per_scene for _ in range(num_scenes)]

    print(
        f"\nGenerating {num_scenes} scenes × {pegs_per_scene} pegs × "
        f"{tol_per_scene} tolerances ({num_scenes*pegs_per_scene*tol_per_scene} total URDFs)..."
    )

    for scene_idx in range(num_scenes):
        # Outer loop: sample goal XY, then fill M start fixtures + trajectories.
        goal_x = goal_y = None
        scene_pegs = None  # list of (start_x, start_y, waypoints)
        for _ in range(MAX_GOAL_RETRIES):
            g_x, g_y = sample_fixture_xy(rng)
            pegs = []
            attempts_budget = pegs_per_scene * MAX_PLACEMENT_RETRIES
            while len(pegs) < pegs_per_scene and attempts_budget > 0:
                attempts_budget -= 1
                start_xy = sample_start_xy_for_goal(rng, g_x, g_y)
                if start_xy is None:
                    break
                s_x, s_y = start_xy
                wps, _ = generate_fixtured_trajectory(s_x, s_y, g_x, g_y)
                if wps is None:
                    continue  # no clearance worked; re-roll start XY
                pegs.append((s_x, s_y, wps))
            if len(pegs) == pegs_per_scene:
                goal_x, goal_y = g_x, g_y
                scene_pegs = pegs
                break

        if scene_pegs is None:
            raise RuntimeError(
                f"Scene {scene_idx}: couldn't fill {pegs_per_scene} "
                f"start fixtures + safe trajectories after {MAX_GOAL_RETRIES} goal re-rolls."
            )

        hole_positions[scene_idx] = [goal_x, goal_y, HOLE_BASE_Z_WORLD]

        for m, (s_x, s_y, waypoints) in enumerate(scene_pegs):
            # Upright peg pose at start fixture XY (PEG_INSERT_QUAT_WXYZ).
            start_poses[scene_idx, m, 0] = s_x
            start_poses[scene_idx, m, 1] = s_y
            start_poses[scene_idx, m, 2] = PEG_FIXTURED_CENTER_Z
            # wxyz → xyzw for storage
            qw, qx, qy, qz = PEG_INSERT_QUAT_WXYZ
            start_poses[scene_idx, m, 3] = qx
            start_poses[scene_idx, m, 4] = qy
            start_poses[scene_idx, m, 5] = qz
            start_poses[scene_idx, m, 6] = qw

            start_fixture_positions[scene_idx, m] = [s_x, s_y, HOLE_BASE_Z_WORLD]
            all_waypoints[scene_idx][m] = waypoints
            traj_lengths[scene_idx, m] = len(waypoints)

        # Per-scene goal-fixture tolerances.
        tol_indices = np.sort(
            rng.choice(tol_pool_size, size=tol_per_scene, replace=False)
        ).astype(np.int32)
        scene_tolerance_indices[scene_idx] = tol_indices

        # Write M × K URDFs for this scene.
        for m, (s_x, s_y, _) in enumerate(scene_pegs):
            peg_dir = SCENES_DIR / f"scene_{scene_idx:04d}" / f"peg_{m:04d}"
            peg_dir.mkdir(parents=True, exist_ok=True)
            for kk, tol_idx in enumerate(tol_indices):
                goal_tol_m = float(tol_pool_m[int(tol_idx)])
                urdf_path = peg_dir / f"scene_tol{kk:02d}.urdf"
                write_scene_urdf(
                    urdf_path,
                    start_xy=(s_x, s_y),
                    start_tol_m=START_FIXTURE_TOL_M,
                    goal_xy=(goal_x, goal_y),
                    goal_tol_m=goal_tol_m,
                    scene_idx=scene_idx,
                    peg_idx=m,
                    tol_slot_idx=kk,
                )

        if scene_idx % 5 == 0 or scene_idx == num_scenes - 1:
            lens = traj_lengths[scene_idx]
            print(
                f"  scene {scene_idx:4d}: goal=({goal_x:+.3f}, {goal_y:+.3f}) "
                f"traj_len={lens.min()}-{lens.max()}"
            )

    # Pad to uniform max trajectory length.
    max_traj = max(
        max(len(wp) for wp in scene_wps) for scene_wps in all_waypoints
    )
    goals = np.zeros((num_scenes, pegs_per_scene, max_traj, 7), dtype=np.float32)
    for s, scene_wps in enumerate(all_waypoints):
        for m, wp in enumerate(scene_wps):
            goals[s, m, : len(wp), :] = wp

    return (
        start_poses,
        goals,
        traj_lengths,
        start_fixture_positions,
        hole_positions,
        tol_pool_m,
        scene_tolerance_indices,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate fixtured peg-in-hole scenes."
    )
    parser.add_argument("--num-scenes", type=int, default=40)
    parser.add_argument("--pegs-per-scene", type=int, default=20)
    parser.add_argument("--tolerance-pool-size", type=int, default=100)
    parser.add_argument("--tolerances-per-scene", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing scenes.npz / URDFs.")
    args = parser.parse_args()

    out_npz = SCENES_DIR / "scenes.npz"
    if out_npz.exists() and not args.force:
        print(f"{out_npz} exists. Use --force to overwrite.")
        return

    SCENES_DIR.mkdir(parents=True, exist_ok=True)

    (
        starts,
        goals,
        lengths,
        start_fixtures,
        hole_positions,
        tol_pool,
        tol_indices,
    ) = generate_scenes(
        args.num_scenes,
        args.pegs_per_scene,
        args.tolerance_pool_size,
        args.tolerances_per_scene,
        args.seed,
    )

    np.savez(
        out_npz,
        start_poses=starts,
        goals=goals,
        traj_lengths=lengths,
        start_fixture_positions=start_fixtures,
        hole_positions=hole_positions,
        tolerance_pool_m=tol_pool,
        scene_tolerance_indices=tol_indices,
        start_tolerance_m=np.float32(START_FIXTURE_TOL_M),
    )

    (SCENES_DIR / "tolerance_pool.json").write_text(
        json.dumps(
            {
                "tolerance_pool_m": tol_pool.tolist(),
                "tolerance_pool_mm": (tol_pool * 1000).tolist(),
                "start_tolerance_m": float(START_FIXTURE_TOL_M),
                "start_tolerance_mm": float(START_FIXTURE_TOL_M * 1000),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nSaved {out_npz.relative_to(REPO_ROOT)}")
    print(f"  start_poses             : {starts.shape} {starts.dtype}")
    print(f"  goals                   : {goals.shape} {goals.dtype}")
    print(f"  traj_lengths            : {lengths.shape} {lengths.dtype}  "
          f"(min={lengths.min()}, max={lengths.max()})")
    print(f"  start_fixture_positions : {start_fixtures.shape} {start_fixtures.dtype}")
    print(f"  hole_positions          : {hole_positions.shape} {hole_positions.dtype}")
    print(f"  tolerance_pool_m        : {tol_pool.shape} {tol_pool.dtype}")
    print(f"  scene_tolerance_indices : {tol_indices.shape} {tol_indices.dtype}")
    total_urdfs = args.num_scenes * args.pegs_per_scene * args.tolerances_per_scene
    print(f"\nTotal URDFs written: {total_urdfs}")


if __name__ == "__main__":
    main()

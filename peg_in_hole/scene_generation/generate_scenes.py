#!/usr/bin/env python3
"""Generate multi-init, multi-goal, multi-tolerance peg-in-hole scenes.

A *scene* is defined by a single hole XY (the "goal") + K tolerance-variant
URDFs at that hole XY. Per scene we also cache M peg start poses and their
trajectories to the insertion pose. At training time the env randomizes
``(scene_idx, peg_idx, tol_slot_idx)`` — three independent axes — so the
policy sees every hole with many peg initial conditions and many clearances
without being told which clearance it faces.

URDFs depend only on (hole XY, tolerance) so we write N × K of them, not
N × M × K.

Pipeline (mirrors ``fabrica/scene_generation/generate_scenes.py``):
1. Load peg stable rests (``assets/urdf/peg_in_hole/stable_rests/peg.npz``),
   keeping only flat rests.
2. Sample a global log-uniform tolerance pool in [0.1, 10] mm.
3. For each of N scenes:
   - Sample hole XY in a ±(0.15, 0.11) box around (0, 0). Retry if we can't
     fill M peg starts below.
   - Sample M peg starts (stable rest + yaw + XY in a ±0.1 box around
     (0, 0)) rejection-filtered against this scene's hole:
       * peg-mesh-at-start bbox stays on the table (with edge inset),
       * peg-mesh-at-start does not overlap the hole block (min distance ≥
         PROXIMITY_MARGIN).
   - For each peg start, generate a crane-style trajectory to the insertion
     pose via ``fabrica.scene_generation.trajectory_generation.generate_variable_trajectory``.
   - Subsample K tolerances from the pool without replacement and write K
     scene URDFs (same hole XY, varying slot clearance).
4. Save ``assets/urdf/peg_in_hole/scenes/scenes.npz`` + tolerance_pool.json.

Output schema (``scenes.npz``):
  - ``start_poses``            : (N, M, 7) float32 — peg xyz + xyzw quat.
  - ``goals``                  : (N, M, max_traj_len, 7) float32 — zero-padded.
  - ``traj_lengths``           : (N, M) int32.
  - ``hole_positions``         : (N, 3) float32 — hole base corner world pos.
  - ``tolerance_pool_m``       : (pool_size,) float32 — meters, sorted.
  - ``scene_tolerance_indices``: (N, K) int32 — into tolerance_pool_m.

Usage:
    python peg_in_hole/scene_generation/generate_scenes.py
    python peg_in_hole/scene_generation/generate_scenes.py \\
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
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "peg_in_hole"
SCENES_DIR = ASSETS_DIR / "scenes"

# --- World-frame constants (see create_peg_and_holes.py for derivation) ---
TABLE_Z = 0.38              # scene-origin z in world (table box center)
TABLE_TOP_Z = 0.53          # world z of the table top surface
TABLE_SIZE = (0.475, 0.4, 0.3)
TABLE_HALF_X = TABLE_SIZE[0] / 2   # 0.2375
TABLE_HALF_Y = TABLE_SIZE[1] / 2   # 0.2
EDGE_INSET = 0.05

# Peg geometry (must match create_peg_and_holes.py)
HANDLE_EXTENTS = (0.25, 0.03, 0.02)
HEAD_EXTENTS = (0.02, 0.10, 0.02)
HANDLE_CENTER = (0.0, 0.0, 0.0)
HEAD_CENTER = (0.115, 0.0, 0.0)
HANDLE_HALF = HANDLE_EXTENTS[0] / 2  # 0.125

# Hole geometry
HOLE_FOOTPRINT_X = 0.08
HOLE_FOOTPRINT_Y = 0.08
HOLE_SLOT_CORE_X = 0.02
HOLE_SLOT_CORE_Y = 0.03
HOLE_FLOOR_THICKNESS = 0.01
HOLE_DEPTH = 0.05
HOLE_HEIGHT = HOLE_FLOOR_THICKNESS + HOLE_DEPTH  # 0.06
HOLE_SCENE_Z = 0.15         # hole base in scene-local z (matches HOLE_SCENE_OFFSET[2])
HOLE_BASE_Z_WORLD = TABLE_Z + HOLE_SCENE_Z                  # 0.53
HOLE_FLOOR_Z_WORLD = HOLE_BASE_Z_WORLD + HOLE_FLOOR_THICKNESS  # 0.54
HOLE_TOP_Z_WORLD = HOLE_BASE_Z_WORLD + HOLE_HEIGHT             # 0.59

HOLE_COLOR = (120, 120, 120)

# Sampling boxes (world frame; centers + half-widths on XY)
PEG_CENTER_XY = (0.0, 0.0)
PEG_HALF_WIDTHS = (0.1, 0.1)          # SimToolReal resetPositionNoiseX/Y
HOLE_CENTER_XY = (0.0, 0.0)
HOLE_HALF_WIDTHS = (0.15, 0.11)       # table interior - 0.05 edge - 0.04 footprint

# Tolerance pool
TOL_MIN_M = 0.1 * 1e-3   # 0.1 mm
TOL_MAX_M = 10.0 * 1e-3  # 10 mm

# Collision margins
PROXIMITY_MARGIN = 0.02  # m — min peg-to-hole-block distance at start
MAX_PLACEMENT_RETRIES = 500

# We only use flat rests (peg lying on the table). Standing rests (peg on
# handle end) are excluded — the policy should never start an episode with
# the peg upright. A rest is "flat" iff its z_lift ≤ this threshold.
REST_Z_LIFT_FLAT_MAX = 0.05

# Trajectory
TARGET_SPACING = 0.05
LIFT_INITIAL = TABLE_TOP_Z + 0.25   # 0.78 m — peg lifts to this z first
LIFT_INCREMENT = 0.05               # raise by this if lift-transit hits the hole
MAX_LIFT_ITERATIONS = 5

# Insertion end pose — rotate body +X to world +Z; tip 1 mm above hole floor.
PEG_INSERT_QUAT_WXYZ = (0.70710678, 0.0, -0.70710678, 0.0)
PEG_END_CENTER_Z = HOLE_FLOOR_Z_WORLD + 0.001 + HANDLE_HALF  # 0.666


# --- Peg mesh / pose helpers ---

def peg_canonical_mesh() -> trimesh.Trimesh:
    """Peg mesh built from the 2 URDF primitive boxes (handle + head)."""
    handle = trimesh.creation.box(extents=np.asarray(HANDLE_EXTENTS, dtype=float))
    handle.apply_translation(np.asarray(HANDLE_CENTER, dtype=float))
    head = trimesh.creation.box(extents=np.asarray(HEAD_EXTENTS, dtype=float))
    head.apply_translation(np.asarray(HEAD_CENTER, dtype=float))
    return trimesh.util.concatenate([handle, head])


def compose_peg_pose(rest_T, yaw_rad, x, y):
    """Compose URDF link pose for the peg at a stable rest + yaw + (x, y).

    The rest transform places the canonical mesh with its bottom at z=0.
    We then yaw about world z, translate to (x, y) on the table, and lift
    by TABLE_TOP_Z so the bottom sits on the table surface.

    Returns (pos [3], quat_wxyz [4]).
    """
    R_rest = rest_T[:3, :3]
    t_rest = rest_T[:3, 3]  # (0, 0, z_lift) in our compute_stable_rests output

    R_yaw = R.from_euler("z", yaw_rad).as_matrix()
    R_link = R_yaw @ R_rest
    p_link = R_yaw @ t_rest + np.array([x, y, TABLE_TOP_Z])

    xyzw = R.from_matrix(R_link).as_quat()
    quat_wxyz = [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]
    return p_link, quat_wxyz


def peg_mesh_at(pos, quat_wxyz):
    """Peg canonical mesh transformed to world pose."""
    mesh = peg_canonical_mesh()
    T = np.eye(4)
    xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    T[:3, :3] = R.from_quat(xyzw).as_matrix()
    T[:3, 3] = pos
    out = mesh.copy()
    out.apply_transform(T)
    return out


# --- Hole geometry ---

def hole_boxes(tol_m):
    """Return [((cx, cy, cz), (ex, ey, ez)), ...] in hole-local coords.

    Mirrors ``peg_in_hole/create_peg_and_holes.py::hole_boxes``: a floor box +
    up to 4 wall slabs defining the slot. Slot XY = core + 2*tol.
    """
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


def hole_block_mesh(hole_x, hole_y):
    """Conservative bbox of the hole BLOCK (outer footprint) at world pose.

    Used only for start-pose collision rejection — tolerance-independent
    because the footprint is constant across tolerances.
    """
    box = trimesh.creation.box(
        extents=(HOLE_FOOTPRINT_X, HOLE_FOOTPRINT_Y, HOLE_HEIGHT)
    )
    box.apply_translation((hole_x, hole_y, HOLE_BASE_Z_WORLD + HOLE_HEIGHT / 2))
    return box


# --- Sampling ---

def sample_log_uniform(rng, lo, hi, size):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))


def sample_peg_start(rng, stable_rests, rest_idx=None):
    """Sample peg start. If rest_idx is None, pick uniformly over rests."""
    if rest_idx is None:
        rest_idx = int(rng.integers(len(stable_rests)))
    rest_T = stable_rests[rest_idx]
    yaw = float(rng.uniform(-np.pi, np.pi))
    x = PEG_CENTER_XY[0] + float(rng.uniform(-PEG_HALF_WIDTHS[0], PEG_HALF_WIDTHS[0]))
    y = PEG_CENTER_XY[1] + float(rng.uniform(-PEG_HALF_WIDTHS[1], PEG_HALF_WIDTHS[1]))
    pos, quat_wxyz = compose_peg_pose(rest_T, yaw, x, y)
    return rest_idx, pos, quat_wxyz


def sample_hole_xy(rng):
    x = HOLE_CENTER_XY[0] + float(rng.uniform(-HOLE_HALF_WIDTHS[0], HOLE_HALF_WIDTHS[0]))
    y = HOLE_CENTER_XY[1] + float(rng.uniform(-HOLE_HALF_WIDTHS[1], HOLE_HALF_WIDTHS[1]))
    return x, y


def peg_bbox_inside_table(peg_mesh_world):
    lo, hi = peg_mesh_world.bounds
    return (
        lo[0] >= -(TABLE_HALF_X - EDGE_INSET)
        and hi[0] <= (TABLE_HALF_X - EDGE_INSET)
        and lo[1] >= -(TABLE_HALF_Y - EDGE_INSET)
        and hi[1] <= (TABLE_HALF_Y - EDGE_INSET)
    )


def sample_peg_for_hole(rng, stable_rests, hole_x, hole_y):
    """Reject-sample one peg start pose given a fixed hole XY.

    Constraints:
      1. peg_bbox_inside_table — peg mesh at start stays within table inset.
      2. peg-to-hole-block min distance ≥ PROXIMITY_MARGIN.
    (Hole is always inside the table by construction of HOLE_HALF_WIDTHS.)

    Returns (rest_idx, peg_pos, peg_quat_wxyz) on success, or None.
    """
    # Build hole collision manager once — it's fixed for all retries.
    hole_block = hole_block_mesh(hole_x, hole_y)
    cm_hole = trimesh.collision.CollisionManager()
    cm_hole.add_object("hole", hole_block)

    for _ in range(MAX_PLACEMENT_RETRIES):
        rest_idx, peg_pos, peg_quat_wxyz = sample_peg_start(rng, stable_rests)
        peg_world_mesh = peg_mesh_at(peg_pos, peg_quat_wxyz)
        if not peg_bbox_inside_table(peg_world_mesh):
            continue

        cm_peg = trimesh.collision.CollisionManager()
        cm_peg.add_object("peg", peg_world_mesh)
        if float(cm_peg.min_distance_other(cm_hole)) < PROXIMITY_MARGIN:
            continue

        return rest_idx, peg_pos, peg_quat_wxyz

    return None


# --- Trajectory + URDF emission ---

def _lift_transit_clear(waypoints, lift_transit_end, hole_cm):
    """Walk every lift+transit waypoint, return True iff the peg mesh at that
    pose stays ≥ PROXIMITY_MARGIN away from the hole block.

    Mirrors fabrica's ``generate_scenes.lift_transit_clear``. Uses a single
    active CollisionManager whose transform we update per waypoint.
    """
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


def generate_peg_trajectory(peg_pos, peg_quat_wxyz, hole_x, hole_y):
    """One vertical-insertion crane trajectory from peg start to inserted pose.

    Iteratively raises the lift altitude up to MAX_LIFT_ITERATIONS times if
    the peg mesh (at the rotating transit pose) would come within
    PROXIMITY_MARGIN of the hole block. Returns (waypoints, lift_transit_end)
    or (None, None) if no safe altitude in range works — caller should
    re-sample the peg start.
    """
    end_pos = np.array([hole_x, hole_y, PEG_END_CENTER_Z])
    # Hole block is fixed across lift iterations; build CM once.
    hole_block = hole_block_mesh(hole_x, hole_y)
    hole_cm = trimesh.collision.CollisionManager()
    hole_cm.add_object("hole", hole_block)

    for lift_iter in range(MAX_LIFT_ITERATIONS):
        clearance = LIFT_INITIAL + lift_iter * LIFT_INCREMENT
        waypoints, lift_transit_end = generate_variable_trajectory(
            peg_pos,
            peg_quat_wxyz,
            end_pos,
            PEG_INSERT_QUAT_WXYZ,
            clearance_z=clearance,
            insertion_dir=[0.0, 0.0, -1.0],
            target_spacing=TARGET_SPACING,
        )
        if _lift_transit_clear(waypoints, lift_transit_end, hole_cm):
            return waypoints, lift_transit_end

    return None, None


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


def write_scene_urdf(out_path: Path, tol_m: float, hole_x: float, hole_y: float,
                     scene_idx: int, tol_slot_idx: int):
    """Scene URDF: table (centered at scene origin) + hole at scene (hole_x,
    hole_y, HOLE_SCENE_Z).

    Mirrors ``peg_in_hole/create_peg_and_holes.py::write_scene_urdf`` but
    with sampled hole XY and an injected tolerance. Robot name encodes
    (scene, slot) so Isaac Gym doesn't get confused by duplicate names.
    """
    parts = [_box_xml((0.0, 0.0, 0.0), TABLE_SIZE, material="wood")]
    for (cx, cy, cz), ext in hole_boxes(tol_m):
        parts.append(_box_xml(
            (cx + hole_x, cy + hole_y, cz + HOLE_SCENE_Z),
            ext,
            material="hole_grey",
        ))

    geom_xml = "\n".join(parts)
    r, g, b = HOLE_COLOR
    robot_name = f"peg_in_hole_scene_{scene_idx:04d}_tol{tol_slot_idx:02d}"
    xml = f'''<?xml version="1.0"?>
<robot name="{robot_name}">
  <material name="wood"><color rgba="0.82 0.56 0.35 1.0"/></material>
  <material name="hole_grey"><color rgba="{r/255:.4f} {g/255:.4f} {b/255:.4f} 1.0"/></material>
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

MAX_HOLE_RETRIES = 50  # re-sample hole XY if we can't fill M pegs for one


def generate_scenes(num_scenes, pegs_per_scene, tol_pool_size, tol_per_scene, seed):
    rng = np.random.default_rng(seed)

    rests_path = ASSETS_DIR / "stable_rests" / "peg.npz"
    if not rests_path.exists():
        raise FileNotFoundError(
            f"Missing {rests_path}. Run compute_stable_rests_isaacgym.py first."
        )
    all_rests = np.load(rests_path)["transforms"]
    # Keep only flat rests (peg lying on table). Standing rests (z_lift > 0.05)
    # are excluded so the policy never starts an episode with the peg upright.
    flat_mask = all_rests[:, 2, 3] <= REST_Z_LIFT_FLAT_MAX
    stable_rests = all_rests[flat_mask]
    if len(stable_rests) == 0:
        raise RuntimeError(
            f"No flat rests (z_lift ≤ {REST_Z_LIFT_FLAT_MAX}) found in "
            f"{rests_path}. Check compute_stable_rests output."
        )
    print(
        f"Loaded {len(all_rests)} stable rests from {rests_path.name}; "
        f"using {len(stable_rests)} flat rests (excluded "
        f"{len(all_rests) - len(stable_rests)} standing rests)."
    )

    tol_pool_m = np.sort(
        sample_log_uniform(rng, TOL_MIN_M, TOL_MAX_M, tol_pool_size)
    ).astype(np.float32)
    print(
        f"Tolerance pool: {tol_pool_size} log-uniform values in "
        f"[{tol_pool_m[0]*1000:.4f}, {tol_pool_m[-1]*1000:.4f}] mm."
    )

    # Per-scene arrays factored as (N, M, ...) along (hole, peg) axes.
    start_poses = np.zeros((num_scenes, pegs_per_scene, 7), dtype=np.float32)
    hole_positions = np.zeros((num_scenes, 3), dtype=np.float32)
    scene_tolerance_indices = np.zeros((num_scenes, tol_per_scene), dtype=np.int32)
    traj_lengths = np.zeros((num_scenes, pegs_per_scene), dtype=np.int32)
    all_waypoints = [[None] * pegs_per_scene for _ in range(num_scenes)]

    print(
        f"\nGenerating {num_scenes} holes × {pegs_per_scene} pegs × "
        f"{tol_per_scene} tolerances..."
    )
    for scene_idx in range(num_scenes):
        # Outer loop: sample a hole XY and try to fill M peg starts against it.
        # A peg is "good" iff (a) its start pose clears the hole at PROXIMITY_MARGIN
        # AND (b) it admits a collision-safe lift-transit trajectory at some
        # clearance ≤ LIFT_INITIAL + MAX_LIFT_ITERATIONS*LIFT_INCREMENT.
        hole_x = hole_y = None
        pegs_for_scene = None  # list of (rest_idx, pos, quat_wxyz, waypoints)
        for _ in range(MAX_HOLE_RETRIES):
            h_x, h_y = sample_hole_xy(rng)
            pegs = []
            # Allow many peg attempts before giving up on this hole.
            peg_attempts_budget = pegs_per_scene * MAX_PLACEMENT_RETRIES
            while len(pegs) < pegs_per_scene and peg_attempts_budget > 0:
                peg_attempts_budget -= 1
                result = sample_peg_for_hole(rng, stable_rests, h_x, h_y)
                if result is None:
                    break
                rest_idx, peg_pos, peg_quat_wxyz = result
                wps, _ = generate_peg_trajectory(peg_pos, peg_quat_wxyz, h_x, h_y)
                if wps is None:
                    continue  # trajectory hits hole at every lift height; re-roll peg
                pegs.append((rest_idx, peg_pos, peg_quat_wxyz, wps))
            if len(pegs) == pegs_per_scene:
                hole_x, hole_y = h_x, h_y
                pegs_for_scene = pegs
                break

        if pegs_for_scene is None:
            raise RuntimeError(
                f"Scene {scene_idx}: couldn't fill {pegs_per_scene} "
                f"collision-safe pegs after {MAX_HOLE_RETRIES} hole re-rolls."
            )

        hole_positions[scene_idx] = [hole_x, hole_y, HOLE_BASE_Z_WORLD]

        # Store per-peg start pose + trajectory.
        rest_ids_this_scene = []
        for m, (rest_idx, peg_pos, peg_quat_wxyz, waypoints) in enumerate(pegs_for_scene):
            # xyz + xyzw quat (peg_quat_wxyz is wxyz → convert)
            start_poses[scene_idx, m, 0:3] = peg_pos
            start_poses[scene_idx, m, 3] = peg_quat_wxyz[1]
            start_poses[scene_idx, m, 4] = peg_quat_wxyz[2]
            start_poses[scene_idx, m, 5] = peg_quat_wxyz[3]
            start_poses[scene_idx, m, 6] = peg_quat_wxyz[0]
            all_waypoints[scene_idx][m] = waypoints
            traj_lengths[scene_idx, m] = len(waypoints)
            rest_ids_this_scene.append(rest_idx)

        # Tolerance URDFs: same for all M pegs in this scene (URDF depends on
        # hole XY + tolerance only).
        tol_indices = np.sort(
            rng.choice(tol_pool_size, size=tol_per_scene, replace=False)
        ).astype(np.int32)
        scene_tolerance_indices[scene_idx] = tol_indices

        scene_dir = SCENES_DIR / f"scene_{scene_idx:04d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for ii, tol_idx in enumerate(tol_indices):
            tol_m = float(tol_pool_m[int(tol_idx)])
            urdf_path = scene_dir / f"scene_tol{ii:02d}.urdf"
            write_scene_urdf(urdf_path, tol_m, hole_x, hole_y, scene_idx, ii)

        if scene_idx % 10 == 0 or scene_idx == num_scenes - 1:
            lens = traj_lengths[scene_idx]
            rest_counts = np.bincount(rest_ids_this_scene, minlength=len(stable_rests))
            print(
                f"  scene {scene_idx:4d}: hole=({hole_x:+.3f}, {hole_y:+.3f}) "
                f"peg_rests={rest_counts.tolist()} "
                f"traj_len={lens.min()}-{lens.max()}"
            )

    # Pad to uniform max traj length.
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
        hole_positions,
        tol_pool_m,
        scene_tolerance_indices,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate peg-in-hole multi-init, multi-goal, multi-tolerance scenes."
    )
    parser.add_argument("--num-scenes", type=int, default=100,
                        help="Number of unique hole positions.")
    parser.add_argument("--pegs-per-scene", type=int, default=10,
                        help="Peg starts + trajectories per hole position.")
    parser.add_argument("--tolerance-pool-size", type=int, default=100)
    parser.add_argument("--tolerances-per-scene", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing scenes.npz / URDFs.")
    args = parser.parse_args()

    out_npz = SCENES_DIR / "scenes.npz"
    if out_npz.exists() and not args.force:
        print(f"{out_npz} exists. Use --force to overwrite.")
        return

    SCENES_DIR.mkdir(parents=True, exist_ok=True)

    starts, goals, lengths, hole_positions, tol_pool, tol_indices = generate_scenes(
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
        hole_positions=hole_positions,
        tolerance_pool_m=tol_pool,
        scene_tolerance_indices=tol_indices,
    )

    (SCENES_DIR / "tolerance_pool.json").write_text(
        json.dumps(
            {
                "tolerance_pool_m": tol_pool.tolist(),
                "tolerance_pool_mm": (tol_pool * 1000).tolist(),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nSaved {out_npz.relative_to(REPO_ROOT)}")
    print(f"  start_poses            : {starts.shape} {starts.dtype}")
    print(f"  goals                  : {goals.shape} {goals.dtype}")
    print(f"  traj_lengths           : {lengths.shape} {lengths.dtype}  "
          f"(min={lengths.min()}, max={lengths.max()})")
    print(f"  hole_positions         : {hole_positions.shape} {hole_positions.dtype}")
    print(f"  tolerance_pool_m       : {tol_pool.shape} {tol_pool.dtype}")
    print(f"  scene_tolerance_indices: {tol_indices.shape} {tol_indices.dtype}")
    print(f"\nTotal URDFs written: {args.num_scenes * args.tolerances_per_scene}")
    print(
        f"Total (scene × peg × tol) triples available at train time: "
        f"{args.num_scenes * args.pegs_per_scene * args.tolerances_per_scene}"
    )


if __name__ == "__main__":
    main()

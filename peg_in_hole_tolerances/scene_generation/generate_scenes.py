#!/usr/bin/env python3
"""Generate train + val peg-in-hole scenes for the tolerance-generalization study.

Mirrors ``peg_in_hole/scene_generation/generate_scenes.py`` exactly except:
  - Output goes to ``assets/urdf/peg_in_hole_tolerances/{scenes,scenes_val}/``.
  - Tolerance pool range is mode-dependent (train: 0.5–1.0 mm; val: 0.1–0.5 mm).
  - Stable rests are reused from ``assets/urdf/peg_in_hole/stable_rests/peg.npz``
    (peg geometry is unchanged).

The val set is **never** used for training — only for held-out evaluation of
generalization to tighter tolerances than seen at training.

Usage:
    python peg_in_hole_tolerances/scene_generation/generate_scenes.py --mode train
    python peg_in_hole_tolerances/scene_generation/generate_scenes.py --mode val
    python peg_in_hole_tolerances/scene_generation/generate_scenes.py --mode train --force
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
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "peg_in_hole_tolerances"
# Stable rests live under peg_in_hole/ (peg geometry is unchanged across tasks).
STABLE_RESTS_PATH = REPO_ROOT / "assets" / "urdf" / "peg_in_hole" / "stable_rests" / "peg.npz"

# --- World-frame constants (mirror peg_in_hole/scene_generation/generate_scenes.py) ---
TABLE_Z = 0.38
TABLE_TOP_Z = 0.53
TABLE_SIZE = (0.475, 0.4, 0.3)
TABLE_HALF_X = TABLE_SIZE[0] / 2
TABLE_HALF_Y = TABLE_SIZE[1] / 2
EDGE_INSET = 0.05

# Peg geometry (must match peg_in_hole/create_peg_and_holes.py)
HANDLE_EXTENTS = (0.25, 0.03, 0.02)
HEAD_EXTENTS = (0.02, 0.10, 0.02)
HANDLE_CENTER = (0.0, 0.0, 0.0)
HEAD_CENTER = (0.115, 0.0, 0.0)
HANDLE_HALF = HANDLE_EXTENTS[0] / 2

# Hole geometry
HOLE_FOOTPRINT_X = 0.08
HOLE_FOOTPRINT_Y = 0.08
HOLE_SLOT_CORE_X = 0.02
HOLE_SLOT_CORE_Y = 0.03
HOLE_FLOOR_THICKNESS = 0.01
HOLE_DEPTH = 0.05
HOLE_HEIGHT = HOLE_FLOOR_THICKNESS + HOLE_DEPTH
HOLE_SCENE_Z = 0.15
HOLE_BASE_Z_WORLD = TABLE_Z + HOLE_SCENE_Z
HOLE_FLOOR_Z_WORLD = HOLE_BASE_Z_WORLD + HOLE_FLOOR_THICKNESS
HOLE_TOP_Z_WORLD = HOLE_BASE_Z_WORLD + HOLE_HEIGHT

HOLE_COLOR = (120, 120, 120)

# Sampling boxes
PEG_CENTER_XY = (0.0, 0.0)
PEG_HALF_WIDTHS = (0.1, 0.1)
HOLE_CENTER_XY = (0.0, 0.0)
HOLE_HALF_WIDTHS = (0.15, 0.11)

# Collision margins
PROXIMITY_MARGIN = 0.02
MAX_PLACEMENT_RETRIES = 500

REST_Z_LIFT_FLAT_MAX = 0.05

# Trajectory
TARGET_SPACING = 0.05
LIFT_INITIAL = TABLE_TOP_Z + 0.25
LIFT_INCREMENT = 0.05
MAX_LIFT_ITERATIONS = 5

PEG_INSERT_QUAT_WXYZ = (0.70710678, 0.0, -0.70710678, 0.0)
PEG_END_CENTER_Z = HOLE_FLOOR_Z_WORLD + 0.001 + HANDLE_HALF


# Mode-specific config: train (loose tolerances, used for training) vs val
# (tighter tolerances, held out for generalization eval).
MODES = {
    "train": {
        "num_scenes": 100,
        "pegs_per_scene": 10,
        "tolerances_per_scene": 10,
        "tolerance_pool_size": 100,
        "tol_min_mm": 0.5,
        "tol_max_mm": 1.0,
        "subdir": "scenes",
        "npz_name": "scenes.npz",
        "default_seed": 0,
    },
    "val": {
        "num_scenes": 10,
        "pegs_per_scene": 10,
        "tolerances_per_scene": 10,
        "tolerance_pool_size": 100,
        "tol_min_mm": 0.1,
        "tol_max_mm": 0.5,
        "subdir": "scenes_val",
        "npz_name": "scenes_val.npz",
        "default_seed": 1,  # different seed so val scene XYs don't collide with train indexing
    },
}


# --- Peg mesh / pose helpers ---

def peg_canonical_mesh() -> trimesh.Trimesh:
    handle = trimesh.creation.box(extents=np.asarray(HANDLE_EXTENTS, dtype=float))
    handle.apply_translation(np.asarray(HANDLE_CENTER, dtype=float))
    head = trimesh.creation.box(extents=np.asarray(HEAD_EXTENTS, dtype=float))
    head.apply_translation(np.asarray(HEAD_CENTER, dtype=float))
    return trimesh.util.concatenate([handle, head])


def compose_peg_pose(rest_T, yaw_rad, x, y):
    R_rest = rest_T[:3, :3]
    t_rest = rest_T[:3, 3]
    R_yaw = R.from_euler("z", yaw_rad).as_matrix()
    R_link = R_yaw @ R_rest
    p_link = R_yaw @ t_rest + np.array([x, y, TABLE_TOP_Z])
    xyzw = R.from_matrix(R_link).as_quat()
    quat_wxyz = [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]
    return p_link, quat_wxyz


def peg_mesh_at(pos, quat_wxyz):
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
    slot_x = HOLE_SLOT_CORE_X + 2 * tol_m
    slot_y = HOLE_SLOT_CORE_Y + 2 * tol_m
    t = HOLE_FLOOR_THICKNESS
    d = HOLE_DEPTH
    ox, oy = HOLE_FOOTPRINT_X, HOLE_FOOTPRINT_Y

    assert slot_x < ox, f"slot_x {slot_x} >= outer {ox}; reduce tol or grow footprint"
    assert slot_y <= oy, f"slot_y {slot_y} > outer {oy}; reduce tol or grow footprint"

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
    box = trimesh.creation.box(
        extents=(HOLE_FOOTPRINT_X, HOLE_FOOTPRINT_Y, HOLE_HEIGHT)
    )
    box.apply_translation((hole_x, hole_y, HOLE_BASE_Z_WORLD + HOLE_HEIGHT / 2))
    return box


# --- Sampling ---

def sample_log_uniform(rng, lo, hi, size):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))


def sample_peg_start(rng, stable_rests, rest_idx=None):
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
    end_pos = np.array([hole_x, hole_y, PEG_END_CENTER_Z])
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


def write_scene_urdf(out_path, tol_m, hole_x, hole_y, scene_idx, tol_slot_idx, mode_subdir):
    parts = [_box_xml((0.0, 0.0, 0.0), TABLE_SIZE, material="wood")]
    for (cx, cy, cz), ext in hole_boxes(tol_m):
        parts.append(_box_xml(
            (cx + hole_x, cy + hole_y, cz + HOLE_SCENE_Z),
            ext,
            material="hole_grey",
        ))
    geom_xml = "\n".join(parts)
    r, g, b = HOLE_COLOR
    robot_name = (
        f"peg_in_hole_tolerances_{mode_subdir}_scene_{scene_idx:04d}_tol{tol_slot_idx:02d}"
    )
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

MAX_HOLE_RETRIES = 50


def generate_scenes(
    num_scenes, pegs_per_scene, tol_pool_size, tol_per_scene,
    tol_min_m, tol_max_m, scenes_dir, npz_name, mode_subdir, seed,
):
    rng = np.random.default_rng(seed)

    if not STABLE_RESTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {STABLE_RESTS_PATH}. Run "
            f"peg_in_hole/scene_generation/compute_stable_rests_isaacgym.py first."
        )
    all_rests = np.load(STABLE_RESTS_PATH)["transforms"]
    flat_mask = all_rests[:, 2, 3] <= REST_Z_LIFT_FLAT_MAX
    stable_rests = all_rests[flat_mask]
    if len(stable_rests) == 0:
        raise RuntimeError(f"No flat rests in {STABLE_RESTS_PATH}.")
    print(
        f"Loaded {len(all_rests)} stable rests; using {len(stable_rests)} flat rests."
    )

    tol_pool_m = np.sort(
        sample_log_uniform(rng, tol_min_m, tol_max_m, tol_pool_size)
    ).astype(np.float32)
    print(
        f"Tolerance pool: {tol_pool_size} log-uniform values in "
        f"[{tol_pool_m[0]*1000:.4f}, {tol_pool_m[-1]*1000:.4f}] mm."
    )

    start_poses = np.zeros((num_scenes, pegs_per_scene, 7), dtype=np.float32)
    hole_positions = np.zeros((num_scenes, 3), dtype=np.float32)
    scene_tolerance_indices = np.zeros((num_scenes, tol_per_scene), dtype=np.int32)
    traj_lengths = np.zeros((num_scenes, pegs_per_scene), dtype=np.int32)
    all_waypoints = [[None] * pegs_per_scene for _ in range(num_scenes)]

    print(
        f"\nGenerating {num_scenes} scenes × {pegs_per_scene} pegs × "
        f"{tol_per_scene} tols ({num_scenes * tol_per_scene} URDFs total)..."
    )

    for scene_idx in range(num_scenes):
        hole_x = hole_y = None
        pegs_for_scene = None
        for _ in range(MAX_HOLE_RETRIES):
            h_x, h_y = sample_hole_xy(rng)
            pegs = []
            attempts_budget = pegs_per_scene * MAX_PLACEMENT_RETRIES
            while len(pegs) < pegs_per_scene and attempts_budget > 0:
                attempts_budget -= 1
                result = sample_peg_for_hole(rng, stable_rests, h_x, h_y)
                if result is None:
                    break
                rest_idx, peg_pos, peg_quat_wxyz = result
                wps, _ = generate_peg_trajectory(peg_pos, peg_quat_wxyz, h_x, h_y)
                if wps is None:
                    continue
                pegs.append((rest_idx, peg_pos, peg_quat_wxyz, wps))
            if len(pegs) == pegs_per_scene:
                hole_x, hole_y = h_x, h_y
                pegs_for_scene = pegs
                break

        if pegs_for_scene is None:
            raise RuntimeError(
                f"Scene {scene_idx}: couldn't fill {pegs_per_scene} "
                f"safe pegs after {MAX_HOLE_RETRIES} hole re-rolls."
            )

        hole_positions[scene_idx] = [hole_x, hole_y, HOLE_BASE_Z_WORLD]

        for m, (rest_idx, peg_pos, peg_quat_wxyz, waypoints) in enumerate(pegs_for_scene):
            start_poses[scene_idx, m, 0:3] = peg_pos
            start_poses[scene_idx, m, 3] = peg_quat_wxyz[1]
            start_poses[scene_idx, m, 4] = peg_quat_wxyz[2]
            start_poses[scene_idx, m, 5] = peg_quat_wxyz[3]
            start_poses[scene_idx, m, 6] = peg_quat_wxyz[0]
            all_waypoints[scene_idx][m] = waypoints
            traj_lengths[scene_idx, m] = len(waypoints)

        tol_indices = np.sort(
            rng.choice(tol_pool_size, size=tol_per_scene, replace=False)
        ).astype(np.int32)
        scene_tolerance_indices[scene_idx] = tol_indices

        scene_dir = scenes_dir / f"scene_{scene_idx:04d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for ii, tol_idx in enumerate(tol_indices):
            tol_m = float(tol_pool_m[int(tol_idx)])
            urdf_path = scene_dir / f"scene_tol{ii:02d}.urdf"
            write_scene_urdf(
                urdf_path, tol_m, hole_x, hole_y, scene_idx, ii, mode_subdir,
            )

        if scene_idx % 10 == 0 or scene_idx == num_scenes - 1:
            lens = traj_lengths[scene_idx]
            print(
                f"  scene {scene_idx:4d}: hole=({hole_x:+.3f}, {hole_y:+.3f}) "
                f"traj_len={lens.min()}-{lens.max()}"
            )

    max_traj = max(
        max(len(wp) for wp in scene_wps) for scene_wps in all_waypoints
    )
    goals = np.zeros((num_scenes, pegs_per_scene, max_traj, 7), dtype=np.float32)
    for s, scene_wps in enumerate(all_waypoints):
        for m, wp in enumerate(scene_wps):
            goals[s, m, : len(wp), :] = wp

    return (
        start_poses, goals, traj_lengths, hole_positions,
        tol_pool_m, scene_tolerance_indices,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val peg-in-hole scenes for tolerance generalization."
    )
    parser.add_argument("--mode", choices=list(MODES.keys()), required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the mode's default seed.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = MODES[args.mode]
    seed = args.seed if args.seed is not None else cfg["default_seed"]
    scenes_dir = ASSETS_DIR / cfg["subdir"]
    npz_path = scenes_dir / cfg["npz_name"]

    if npz_path.exists() and not args.force:
        print(f"{npz_path} exists. Use --force to overwrite.")
        return
    scenes_dir.mkdir(parents=True, exist_ok=True)

    starts, goals, lengths, hole_positions, tol_pool, tol_indices = generate_scenes(
        num_scenes=cfg["num_scenes"],
        pegs_per_scene=cfg["pegs_per_scene"],
        tol_pool_size=cfg["tolerance_pool_size"],
        tol_per_scene=cfg["tolerances_per_scene"],
        tol_min_m=cfg["tol_min_mm"] * 1e-3,
        tol_max_m=cfg["tol_max_mm"] * 1e-3,
        scenes_dir=scenes_dir,
        npz_name=cfg["npz_name"],
        mode_subdir=cfg["subdir"],
        seed=seed,
    )

    np.savez(
        npz_path,
        start_poses=starts,
        goals=goals,
        traj_lengths=lengths,
        hole_positions=hole_positions,
        tolerance_pool_m=tol_pool,
        scene_tolerance_indices=tol_indices,
    )

    (scenes_dir / "tolerance_pool.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "tol_min_mm": cfg["tol_min_mm"],
                "tol_max_mm": cfg["tol_max_mm"],
                "tolerance_pool_m": tol_pool.tolist(),
                "tolerance_pool_mm": (tol_pool * 1000).tolist(),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\n[mode={args.mode}] Saved {npz_path.relative_to(REPO_ROOT)}")
    print(f"  start_poses            : {starts.shape} {starts.dtype}")
    print(f"  goals                  : {goals.shape} {goals.dtype}")
    print(f"  traj_lengths           : {lengths.shape} {lengths.dtype}  "
          f"(min={lengths.min()}, max={lengths.max()})")
    print(f"  hole_positions         : {hole_positions.shape} {hole_positions.dtype}")
    print(f"  tolerance_pool_m       : {tol_pool.shape} {tol_pool.dtype}  "
          f"({tol_pool.min()*1000:.4f} – {tol_pool.max()*1000:.4f} mm)")
    print(f"  scene_tolerance_indices: {tol_indices.shape} {tol_indices.dtype}")
    print(f"\nTotal URDFs: {cfg['num_scenes'] * cfg['tolerances_per_scene']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fixture generation script ported from Fabrica's planning/run_fixture_gen.py.

Generates a fixture.obj and pickup.json for a given assembly.

Usage:
    python fabrica/generate_fixture.py \
        --assembly-dir /share/portal/kk837/Fabrica/assets/fabrica/beam \
        --log-dir /share/portal/kk837/Fabrica/logs/test/beam \
        --output-dir assets/urdf/fabrica/beam/fixture

Dependencies:
    - The Fabrica project must be accessible at FABRICA_ROOT (default: /share/portal/kk837/Fabrica)
      because grasps.pkl contains pickled Grasp objects from planning.robot.util_grasp.
    - Gripper collision meshes are loaded from FABRICA_ROOT/assets/.
    - manifold3d is required for reliable boolean union/intersection operations.
    - trimesh.boolean.difference works fine with the default engine.

Key insight: trimesh.boolean.union silently collapses meshes. All union and
intersection operations use manifold3d directly instead.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"

import sys
import json
import pickle
import numpy as np
import trimesh
import manifold3d as m3d
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
from scipy.spatial import Delaunay
from rectpack import newPacker
from time import time

# ---------------------------------------------------------------------------
# Fabrica sys.path setup -- required for unpickling Grasp objects
# ---------------------------------------------------------------------------
FABRICA_ROOT = os.environ.get(
    "FABRICA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Fabrica")),
)
if FABRICA_ROOT not in sys.path:
    sys.path.insert(0, FABRICA_ROOT)

# These imports are needed ONLY for unpickling grasps.pkl / precedence.pkl / tree_opt.pkl.
# The Grasp class lives in planning.robot.util_grasp and networkx DiGraph is standard.
# We also import SequenceOptimizer to extract sequences from the optimized tree.
from planning.run_seq_opt import SequenceOptimizer  # noqa: E402


# ===========================================================================
# Manifold3d helpers
# ===========================================================================

def _to_manifold(mesh: trimesh.Trimesh) -> m3d.Manifold:
    """Convert a trimesh.Trimesh to a manifold3d.Manifold."""
    return m3d.Manifold(
        mesh=m3d.Mesh(
            vert_properties=np.array(mesh.vertices, dtype=np.float32),
            tri_verts=np.array(mesh.faces, dtype=np.uint32),
        )
    )


def _to_trimesh(manifold: m3d.Manifold) -> trimesh.Trimesh:
    """Convert a manifold3d.Manifold back to a trimesh.Trimesh."""
    mesh_out = manifold.to_mesh()
    return trimesh.Trimesh(
        vertices=mesh_out.vert_properties[:, :3],
        faces=mesh_out.tri_verts,
    )


def manifold_union(meshes: list) -> trimesh.Trimesh:
    """
    Reliable boolean union of a list of trimesh.Trimesh objects using manifold3d.
    trimesh.boolean.union silently collapses meshes -- do NOT use it.
    """
    if len(meshes) == 0:
        return trimesh.Trimesh()
    if len(meshes) == 1:
        return meshes[0].copy()
    result = _to_manifold(meshes[0])
    for m in meshes[1:]:
        result = result + _to_manifold(m)
    return _to_trimesh(result)


def manifold_intersection(meshes: list) -> trimesh.Trimesh:
    """
    Reliable boolean intersection using manifold3d.
    trimesh.boolean.intersection may silently fail -- use this instead.
    """
    if len(meshes) == 0:
        return trimesh.Trimesh()
    if len(meshes) == 1:
        return meshes[0].copy()
    result = _to_manifold(meshes[0])
    for m in meshes[1:]:
        result = result ^ _to_manifold(m)
    return _to_trimesh(result)


def manifold_difference(meshes: list) -> trimesh.Trimesh:
    """
    Reliable boolean difference using manifold3d.
    meshes[0] - meshes[1] - meshes[2] - ...
    Use this instead of trimesh.boolean.difference when meshes may not be
    valid volumes (e.g., the countersunk pad hole).
    """
    if len(meshes) == 0:
        return trimesh.Trimesh()
    if len(meshes) == 1:
        return meshes[0].copy()
    result = _to_manifold(meshes[0])
    for m in meshes[1:]:
        result = result - _to_manifold(m)
    return _to_trimesh(result)


# ===========================================================================
# Transform utilities (inlined from Fabrica/assets/transform.py)
# ===========================================================================

def get_transform_matrix(state):
    """4x4 transform from [x, y, z] or [x, y, z, rx, ry, rz] (euler xyz)."""
    if len(state) == 3:
        T = np.eye(4)
        T[:3, 3] = state
        return T
    elif len(state) == 6:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", state[3:]).as_matrix()
        T[:3, 3] = state[:3]
        return T
    else:
        raise ValueError(f"Unexpected state length {len(state)}")


def get_transform_matrix_quat(pos, quat):
    """4x4 transform from position and quaternion (wxyz convention)."""
    pos = np.array(pos)
    quat = np.array(quat)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = pos
    return T


def mat_to_pos_quat(mat):
    """Extract position and quaternion (wxyz) from a 4x4 matrix."""
    pos = mat[:3, 3]
    quat = Rotation.from_matrix(mat[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return pos, quat


def get_pos_euler_from_transform_matrix(T):
    """Return [x, y, z, rx, ry, rz] from a 4x4 matrix."""
    pos = T[:3, 3]
    euler = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    return np.concatenate([pos, euler])


def get_pos_quat_from_pose(pos, quat, pose):
    """Apply a 4x4 pose transform to (pos, quat_wxyz)."""
    if pose is None:
        return pos, quat
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.array(quat)[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = pos
    T = pose @ T
    new_pos = T[:3, 3]
    new_quat = Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return new_pos, new_quat


def get_translate_matrix(t):
    T = np.eye(4)
    T[:3, 3] = t
    return T


def get_scale_matrix(s):
    T = np.eye(4)
    T[:3, :3] *= s
    return T


def get_revolute_matrix(axis, angle):
    axis_map = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.array(axis_map[axis]) * angle).as_matrix()
    return T


# ===========================================================================
# Geometry utilities (inlined from Fabrica/planning/robot/geometry.py)
# ===========================================================================

def add_buffer_to_mesh(mesh, buffer):
    """Expand/contract mesh along vertex normals."""
    new_verts = mesh.vertices + buffer * mesh.vertex_normals
    return trimesh.Trimesh(new_verts, mesh.faces, vertex_normals=mesh.vertex_normals)


def get_buffered_meshes(meshes, buffer):
    """Apply buffer to a single mesh, dict of meshes, or list of meshes."""
    if isinstance(meshes, trimesh.Trimesh):
        return add_buffer_to_mesh(meshes, buffer=buffer)
    elif isinstance(meshes, dict):
        return {k: add_buffer_to_mesh(v, buffer=buffer) for k, v in meshes.items()}
    elif isinstance(meshes, list):
        return [add_buffer_to_mesh(m, buffer=buffer) for m in meshes]
    else:
        raise TypeError(f"Unsupported type {type(meshes)}")


def get_combined_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        return trimesh.util.concatenate(
            [scene_or_mesh.geometry[n] for n in scene_or_mesh.geometry]
        )
    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    else:
        raise ValueError("Input is not Trimesh or Scene")


def get_combined_meshes(meshes):
    if isinstance(meshes, dict):
        return {k: get_combined_mesh(v) for k, v in meshes.items()}
    elif isinstance(meshes, list):
        return [get_combined_mesh(v) for v in meshes]
    raise ValueError("Input is not dict or list")


# ===========================================================================
# Part mesh loading (inlined from Fabrica/assets/load.py)
# ===========================================================================

def load_part_ids(obj_dir):
    """List part IDs (OBJ filenames without extension) sorted."""
    ids = []
    for name in os.listdir(obj_dir):
        if name.endswith(".obj"):
            ids.append(name.replace(".obj", ""))
    ids.sort()
    return ids


def load_config(obj_dir):
    cfg_path = os.path.join(obj_dir, "config.json")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, "r") as f:
        return json.load(f)


def load_part_meshes(assembly_dir, transform="final"):
    """Load part meshes with optional 'final' transform from config.json."""
    part_ids = load_part_ids(assembly_dir)
    config = load_config(assembly_dir)
    meshes = {}
    for pid in part_ids:
        mesh = trimesh.load_mesh(
            os.path.join(assembly_dir, f"{pid}.obj"), process=False, maintain_order=True
        )
        if config is not None and pid in config and transform == "final":
            mesh.apply_transform(get_transform_matrix(config[pid]["final_state"]))
        meshes[f"part{pid}"] = mesh
    return meshes


def load_pos_quat_dict(assembly_dir, transform="final"):
    """Load position/quaternion dicts. For 'final' transform, identity."""
    part_ids = load_part_ids(assembly_dir)
    pos_dict, quat_dict = {}, {}
    for pid in part_ids:
        if transform == "final":
            pos_dict[pid] = np.array([0.0, 0.0, 0.0])
            quat_dict[pid] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            pos_dict[pid], quat_dict[pid] = None, None
    return pos_dict, quat_dict


# ===========================================================================
# Workcell parameters (inlined from Fabrica/planning/robot/workcell.py)
# ===========================================================================

def get_board_dx():
    return 2.5


def get_assembly_center(arm_type):
    dx = get_board_dx()
    if arm_type in ("xarm7", "panda", "ur5e"):
        return np.array([0, -6 * dx, 0])
    raise NotImplementedError(f"Unknown arm type: {arm_type}")


def get_fixture_min_y(arm_type):
    dx = get_board_dx()
    table = {"xarm7": 6 * dx, "panda": 4 * dx, "ur5e": 6 * dx}
    if arm_type in table:
        return table[arm_type]
    raise NotImplementedError(f"Unknown arm type: {arm_type}")


# ===========================================================================
# Gripper mesh loading & transform
# (inlined from Fabrica/planning/robot/geometry.py)
# ===========================================================================

def load_panda_meshes(asset_folder):
    d = os.path.join(asset_folder, "panda", "collision")
    return {
        "panda_hand": trimesh.load(os.path.join(d, "hand.obj")),
        "panda_leftfinger": trimesh.load(os.path.join(d, "finger.obj")),
        "panda_rightfinger": trimesh.load(os.path.join(d, "finger.obj")),
    }


def load_robotiq_85_meshes(asset_folder):
    d = os.path.join(asset_folder, "robotiq_85", "collision")
    meshes = {"robotiq_base": trimesh.load(os.path.join(d, "robotiq_base_coarse.obj"))}
    for si in ("left", "right"):
        for sj in ("outer", "inner"):
            for link in ("knuckle", "finger"):
                meshes[f"robotiq_{si}_{sj}_{link}"] = trimesh.load(
                    os.path.join(d, f"{sj}_{link}_coarse.obj")
                )
    return meshes


def load_robotiq_140_meshes(asset_folder):
    d = os.path.join(asset_folder, "robotiq_140", "collision")
    meshes = {"robotiq_base": trimesh.load(os.path.join(d, "robotiq_base_coarse.obj"))}
    for side in ("left", "right"):
        for link in ("outer_knuckle", "outer_finger", "inner_finger"):
            meshes[f"robotiq_{side}_{link}"] = trimesh.load(
                os.path.join(d, f"{link}_coarse.obj")
            )
        meshes[f"robotiq_{side}_pad"] = trimesh.load(os.path.join(d, "pad_coarse.obj"))
        meshes[f"robotiq_{side}_inner_knuckle"] = trimesh.load(
            os.path.join(d, "inner_knuckle_coarse.obj")
        )
    return meshes


def load_gripper_meshes(gripper_type, asset_folder, combined=True):
    loaders = {
        "panda": load_panda_meshes,
        "robotiq-85": load_robotiq_85_meshes,
        "robotiq-140": load_robotiq_140_meshes,
    }
    if gripper_type not in loaders:
        raise NotImplementedError(f"Unknown gripper: {gripper_type}")
    meshes = loaders[gripper_type](asset_folder)
    if combined:
        meshes = get_combined_meshes(meshes)
    return meshes


# -- Gripper FK transforms --

def get_panda_meshes_transforms(meshes, open_ratio):
    transforms = {k: np.eye(4) for k in meshes}
    transforms["panda_leftfinger"] = get_translate_matrix([0, 4 * open_ratio, 5.84])
    transforms["panda_rightfinger"] = get_translate_matrix(
        [0, -4 * open_ratio, 5.84]
    ) @ get_scale_matrix([1, -1, 1])
    return transforms


def get_robotiq_85_meshes_transforms(meshes, open_ratio):
    transforms = {k: np.eye(4) for k in meshes}
    ce = 0.8757 * (1 - open_ratio)
    transforms["robotiq_left_outer_knuckle"] = (
        get_translate_matrix([3.06011444260539, 0.0, 6.27920162695395])
        @ get_revolute_matrix("Y", -ce)
    )
    transforms["robotiq_left_outer_finger"] = transforms[
        "robotiq_left_outer_knuckle"
    ] @ get_translate_matrix([3.16910442266543, 0.0, -0.193396375724605])
    transforms["robotiq_left_inner_knuckle"] = (
        get_translate_matrix([1.27000000001501, 0.0, 6.93074999999639])
        @ get_revolute_matrix("Y", -ce)
    )
    transforms["robotiq_left_inner_finger"] = (
        transforms["robotiq_left_inner_knuckle"]
        @ get_translate_matrix([3.4585310861294003, 0.0, 4.5497019381797505])
        @ get_revolute_matrix("Y", ce)
    )
    transforms["robotiq_right_outer_knuckle"] = (
        get_transform_matrix_quat(
            [-3.06011444260539, 0.0, 6.27920162695395], [0, 0, 0, 1]
        )
        @ get_revolute_matrix("Y", -ce)
    )
    transforms["robotiq_right_outer_finger"] = transforms[
        "robotiq_right_outer_knuckle"
    ] @ get_translate_matrix([3.16910442266543, 0.0, -0.193396375724605])
    transforms["robotiq_right_inner_knuckle"] = (
        get_transform_matrix_quat(
            [-1.27000000001501, 0.0, 6.93074999999639], [0, 0, 0, 1]
        )
        @ get_revolute_matrix("Y", -ce)
    )
    transforms["robotiq_right_inner_finger"] = (
        transforms["robotiq_right_inner_knuckle"]
        @ get_translate_matrix([3.4585310861294003, 0.0, 4.5497019381797505])
        @ get_revolute_matrix("Y", ce)
    )
    return transforms


def get_robotiq_140_meshes_transforms(meshes, open_ratio):
    transforms = {k: np.eye(4) for k in meshes}
    ce = 0.8757 * (1 - open_ratio)
    transforms["robotiq_left_outer_knuckle"] = (
        get_transform_matrix_quat(
            [0.0, -3.0601, 5.4905], [0.41040502, 0.91190335, 0.0, 0.0]
        )
        @ get_revolute_matrix("X", -ce)
    )
    transforms["robotiq_left_outer_finger"] = transforms[
        "robotiq_left_outer_knuckle"
    ] @ get_translate_matrix([0.0, 1.821998610742, 2.60018192872234])
    transforms["robotiq_left_inner_finger"] = (
        transforms["robotiq_left_outer_finger"]
        @ get_transform_matrix_quat(
            [0.0, 8.17554015893473, -2.82203446692936],
            [0.93501321, -0.35461287, 0.0, 0.0],
        )
        @ get_revolute_matrix("X", ce)
    )
    transforms["robotiq_left_pad"] = transforms[
        "robotiq_left_inner_finger"
    ] @ get_transform_matrix_quat(
        [0.0, 3.8, -2.3], [0, 0, 0.70710678, 0.70710678]
    )
    transforms["robotiq_left_inner_knuckle"] = (
        get_transform_matrix_quat(
            [0.0, -1.27, 6.142], [0.41040502, 0.91190335, 0.0, 0.0]
        )
        @ get_revolute_matrix("X", -ce)
    )
    transforms["robotiq_right_outer_knuckle"] = (
        get_transform_matrix_quat(
            [0.0, 3.0601, 5.4905], [0.0, 0.0, 0.91190335, 0.41040502]
        )
        @ get_revolute_matrix("X", -ce)
    )
    transforms["robotiq_right_outer_finger"] = transforms[
        "robotiq_right_outer_knuckle"
    ] @ get_translate_matrix([0.0, 1.821998610742, 2.60018192872234])
    transforms["robotiq_right_inner_finger"] = (
        transforms["robotiq_right_outer_finger"]
        @ get_transform_matrix_quat(
            [0.0, 8.17554015893473, -2.82203446692936],
            [0.93501321, -0.35461287, 0.0, 0.0],
        )
        @ get_revolute_matrix("X", ce)
    )
    transforms["robotiq_right_pad"] = transforms[
        "robotiq_right_inner_finger"
    ] @ get_transform_matrix_quat(
        [0.0, 3.8, -2.3], [0, 0, 0.70710678, 0.70710678]
    )
    transforms["robotiq_right_inner_knuckle"] = (
        get_transform_matrix_quat(
            [0.0, 1.27, 6.142], [0.0, 0.0, -0.91190335, -0.41040502]
        )
        @ get_revolute_matrix("X", -ce)
    )
    return transforms


def get_gripper_meshes_transforms(gripper_type, meshes, pos, quat, pose, open_ratio):
    fk_funcs = {
        "panda": get_panda_meshes_transforms,
        "robotiq-85": get_robotiq_85_meshes_transforms,
        "robotiq-140": get_robotiq_140_meshes_transforms,
    }
    transforms = fk_funcs[gripper_type](meshes, open_ratio)
    pos, quat = get_pos_quat_from_pose(pos, quat, pose)
    base_T = get_transform_matrix_quat(pos, quat)
    for name in transforms:
        transforms[name] = base_T @ transforms[name]
    return transforms


def transform_gripper_meshes(gripper_type, meshes, pos, quat, pose, open_ratio):
    transforms = get_gripper_meshes_transforms(
        gripper_type, meshes, pos, quat, pose, open_ratio
    )
    out = {k: v.copy() for k, v in meshes.items()}
    for name, mesh in out.items():
        mesh.apply_transform(transforms[name])
    return out


# ===========================================================================
# Countersunk pad (inlined from Fabrica/planning/utils/fixture_countersunk.py)
# ===========================================================================

COUNTERSUNK_DIAMETER = 1.05
HOLE_DIAMETER = 0.6
PAD_DIAMETER = 2.5
PAD_HEIGHT = 0.5


def create_solid_cone(radius, height, segments=32):
    cone = trimesh.creation.cone(radius=radius, height=height, sections=segments)
    theta = np.linspace(0, 2 * np.pi, segments)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(x, -height / 2)
    disk_verts = np.vstack((np.vstack((x, y, z)).T, [[0, 0, -height / 2]]))
    disk_faces = []
    for i in range(segments - 1):
        disk_faces.append([i, i + 1, len(disk_verts) - 1])
    disk_faces.append([segments - 1, 0, len(disk_verts) - 1])
    disk_faces = np.array(disk_faces)
    verts = np.vstack((cone.vertices, disk_verts))
    faces = np.vstack((cone.faces, disk_faces + len(cone.vertices)))
    solid = trimesh.Trimesh(vertices=verts, faces=faces)
    solid.merge_vertices()
    solid.fix_normals()
    return solid.convex_hull


def generate_countersunk_hole(countersunk_diameter, hole_diameter, pad_height, segments=32):
    cone_h = countersunk_diameter / 2 - hole_diameter / 2
    hole_h = pad_height - cone_h
    cone = create_solid_cone(countersunk_diameter / 2, countersunk_diameter / 2, segments=segments)
    cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    cone.apply_transform(trimesh.transformations.translation_matrix([0, 0, hole_h + cone_h]))
    cyl = trimesh.creation.cylinder(
        radius=hole_diameter / 2,
        height=hole_h,
        sections=segments,
        transform=trimesh.transformations.translation_matrix([0, 0, hole_h / 2]),
    )
    # Use manifold3d for union
    return manifold_union([cyl, cone])


def generate_countersunk_pad(
    countersunk_diameter=COUNTERSUNK_DIAMETER,
    hole_diameter=HOLE_DIAMETER,
    pad_diameter=PAD_DIAMETER,
    pad_height=PAD_HEIGHT,
):
    pad = trimesh.creation.box(
        [pad_diameter, pad_diameter, pad_height],
        transform=trimesh.transformations.translation_matrix([0, 0, pad_height / 2]),
    )
    hole = generate_countersunk_hole(countersunk_diameter, hole_diameter, pad_height)
    return manifold_difference([pad, hole])


# ===========================================================================
# Fixture board parameters
# ===========================================================================

DX = get_board_dx()
BOTTOM_THICKNESS = 0.5
EDGE_THICKNESS = 3.0
MIN_MOLD_DEPTH = 1.0
MOLD_EDGE_OFFSET_PART = [0.05, 0.05, 0.0]
MOLD_EDGE_OFFSET_GRIPPER = [0.8, 0.8, 0.4]
PART_BOUNDARY_OFFSET = 0.2
PART_GAP = 2.0
MAX_BIN_SIZE_SINGLE = [8 * DX, 10 * DX]
MAX_BIN_SIZE_DOUBLE = [8 * DX, 20 * DX]
DELTA_BIN_SIZE = 1 * DX
DELTA_BUFFER_SIZE = 2.0


# ===========================================================================
# Core fixture generation logic
# ===========================================================================

def generate_individual_pose_info(part_cfg_final, sequence, grasps_sequence):
    """Determine per-part pickup orientations from grasps."""
    part_meshes_final = part_cfg_final["mesh"]
    pose_info = {}
    sequence_forward = sequence[::-1]
    grasps_sequence_forward = grasps_sequence[::-1]

    for i, ((part_move, part_hold), (grasps_move, grasp_hold)) in enumerate(
        zip(sequence_forward, grasps_sequence_forward)
    ):
        grasp_move_final = grasps_move[0]

        if i == 0:
            # First step: both arms pick up
            gripper_l2r = Rotation.from_quat(
                grasp_hold.quat[[1, 2, 3, 0]]
            ).apply([0, -1, 0])
            gripper_b2f = Rotation.from_quat(
                grasp_hold.quat[[1, 2, 3, 0]]
            ).apply([0, 0, 1])
            pickup_rot = Rotation.align_vectors(
                [np.array([1, 0, 0]), np.array([0, 0, -1])],
                [gripper_l2r, gripper_b2f],
            )[0].as_matrix()

            hold_mesh = part_meshes_final[part_hold].copy()
            T = np.eye(4)
            T[:3, :3] = pickup_rot
            hold_mesh.apply_transform(T)

            pose_info[part_hold] = {
                "extent_x": hold_mesh.extents[0],
                "extent_y": hold_mesh.extents[1],
                "center_x": np.min(hold_mesh.vertices[:, 0]) + hold_mesh.extents[0] / 2,
                "center_y": np.min(hold_mesh.vertices[:, 1]) + hold_mesh.extents[1] / 2,
                "min_z": np.min(hold_mesh.vertices[:, 2]),
                "rot_mat": pickup_rot,
            }

        # Move arm
        gripper_l2r = Rotation.from_quat(
            grasp_move_final.quat[[1, 2, 3, 0]]
        ).apply([0, -1, 0])
        gripper_b2f = Rotation.from_quat(
            grasp_move_final.quat[[1, 2, 3, 0]]
        ).apply([0, 0, 1])
        pickup_rot = Rotation.align_vectors(
            [np.array([1, 0, 0]), np.array([0, 0, -1])],
            [gripper_l2r, gripper_b2f],
        )[0].as_matrix()

        move_mesh = part_meshes_final[part_move].copy()
        T = np.eye(4)
        T[:3, :3] = pickup_rot
        move_mesh.apply_transform(T)

        pose_info[part_move] = {
            "extent_x": move_mesh.extents[0],
            "extent_y": move_mesh.extents[1],
            "center_x": np.min(move_mesh.vertices[:, 0]) + move_mesh.extents[0] / 2,
            "center_y": np.min(move_mesh.vertices[:, 1]) + move_mesh.extents[1] / 2,
            "min_z": np.min(move_mesh.vertices[:, 2]),
            "rot_mat": pickup_rot,
        }

    return pose_info


def run_bin_packing(pose_info, bin_size):
    packer = newPacker(rotation=False)
    for pid, info in pose_info.items():
        packer.add_rect(info["extent_x"] + PART_GAP, info["extent_y"] + PART_GAP, pid)
    packer.add_bin(bin_size[0], bin_size[1])
    packer.pack()
    rects = packer.rect_list()
    if len(rects) == len(pose_info):
        return packer
    return None


def generate_pickup_pose(pose_info, min_fixture_y):
    """2D bin packing to find pickup positions for all parts."""
    packer = run_bin_packing(pose_info, MAX_BIN_SIZE_DOUBLE)
    if packer is None:
        return None, None

    packer = run_bin_packing(pose_info, MAX_BIN_SIZE_SINGLE)
    max_bin_size = MAX_BIN_SIZE_SINGLE if packer is not None else MAX_BIN_SIZE_DOUBLE

    best_packer = packer if packer is not None else run_bin_packing(pose_info, MAX_BIN_SIZE_DOUBLE)
    best_bin_size = max_bin_size
    best_bin_area = np.prod(max_bin_size)

    delta = DELTA_BIN_SIZE
    min_area = sum(
        (pose_info[pid]["extent_x"] + PART_GAP) * (pose_info[pid]["extent_y"] + PART_GAP)
        for pid in pose_info
    )
    min_bx = max(np.ceil(min_area / max_bin_size[1] / delta), 4) * delta
    min_by = max(np.ceil(min_area / max_bin_size[0] / delta), 4) * delta

    for bx in np.arange(min_bx, max_bin_size[0] + 0.5 * delta, delta):
        for by in np.arange(min_by, max_bin_size[1] + 0.5 * delta, delta):
            area = bx * by
            if area >= best_bin_area:
                continue
            p = run_bin_packing(pose_info, [bx, by])
            if p is not None:
                best_packer = p
                best_bin_size = [bx, by]
                best_bin_area = area

    packer, bin_size = best_packer, best_bin_size

    pickup_pose = {}
    for rect in packer[0]:
        pid = rect.rid
        T = np.eye(4)
        T[:3, :3] = pose_info[pid]["rot_mat"]
        T[:3, 3] = np.array([
            rect.x + rect.width / 2 - pose_info[pid]["center_x"] - bin_size[0] / 2,
            rect.y + rect.height / 2 - pose_info[pid]["center_y"] + min_fixture_y,
            BOTTOM_THICKNESS - pose_info[pid]["min_z"],
        ])
        pickup_pose[pid] = get_pos_euler_from_transform_matrix(T).tolist()

    return pickup_pose, bin_size


def get_swept_mesh(mesh_start, mesh_end):
    """Convex hull of two mesh vertex sets (swept volume approximation)."""
    pts = np.vstack([mesh_start.vertices, mesh_end.vertices])
    pts_unique = np.unique(pts.round(decimals=6), axis=0)
    return trimesh.convex.convex_hull(pts_unique, qhull_options="Qx Qs Qt")


def generate_pickup_meshes(
    part_cfg_final, sequence, grasps_sequence, gripper_type, pickup_pose
):
    """Compute pickup part meshes and gripper swept volumes."""
    part_meshes_final = part_cfg_final["mesh"]
    asset_folder = os.path.join(FABRICA_ROOT, "assets")
    gripper_meshes = load_gripper_meshes(gripper_type, asset_folder)

    # Transform part meshes to pickup poses
    part_meshes_pickup = {k: v.copy() for k, v in part_meshes_final.items()}
    for pid, pose in pickup_pose.items():
        part_meshes_pickup[pid].apply_transform(get_transform_matrix(pose))

    # Compute gripper swept volumes at pickup poses
    sequence_forward = sequence[::-1]
    grasps_forward = grasps_sequence[::-1]
    gripper_meshes_pickup = {}

    for i, ((part_move, part_hold), (grasps_move_t, grasp_hold)) in enumerate(
        zip(sequence_forward, grasps_forward)
    ):
        grasp_move_final = grasps_move_t[0]

        if i == 0:
            # Hold arm gripper
            gfm_hold = get_transform_matrix_quat(grasp_hold.pos, grasp_hold.quat)
            pp_hold = get_transform_matrix(pickup_pose[part_hold])
            gp_hold = pp_hold @ gfm_hold
            gp_pos, gp_quat = mat_to_pos_quat(gp_hold)
            tight = transform_gripper_meshes(
                gripper_type, gripper_meshes, gp_pos, gp_quat,
                np.eye(4), grasp_hold.open_ratio - 0.05,
            )
            loose = transform_gripper_meshes(
                gripper_type, gripper_meshes, gp_pos, gp_quat,
                np.eye(4), grasp_hold.open_ratio + 0.15,
            )
            swept_parts = [
                get_swept_mesh(tight[gp], loose[gp]) for gp in tight
            ]
            gripper_meshes_pickup[part_hold] = manifold_union(swept_parts)

        # Move arm gripper
        gfm_move = get_transform_matrix_quat(
            grasp_move_final.pos, grasp_move_final.quat
        )
        pp_move = get_transform_matrix(pickup_pose[part_move])
        gp_move = pp_move @ gfm_move
        gp_pos, gp_quat = mat_to_pos_quat(gp_move)
        tight = transform_gripper_meshes(
            gripper_type, gripper_meshes, gp_pos, gp_quat,
            np.eye(4), grasp_move_final.open_ratio - 0.05,
        )
        loose = transform_gripper_meshes(
            gripper_type, gripper_meshes, gp_pos, gp_quat,
            np.eye(4), grasp_move_final.open_ratio + 0.15,
        )
        swept_parts = [get_swept_mesh(tight[gp], loose[gp]) for gp in tight]
        gripper_meshes_pickup[part_move] = manifold_union(swept_parts)

    return part_meshes_pickup, gripper_meshes_pickup


def generate_fixture(part_meshes_pickup, gripper_meshes_pickup, bin_size, min_fixture_y):
    """Construct the fixture mesh by subtracting part/gripper molds from a board."""
    # Determine fixture height
    board_height_max = 0.0
    for pid, part_mesh in part_meshes_pickup.items():
        com = part_mesh.center_mass
        bh = BOTTOM_THICKNESS + MIN_MOLD_DEPTH
        while True:
            sliced = part_mesh.slice_plane([0, 0, bh], [0, 0, -1], cap=True)
            hull = Delaunay(sliced.vertices[:, :2])
            if hull.find_simplex(com[:2]) >= 0:
                break
            bh += 1.0
        board_height_max = max(board_height_max, bh)

    # Verify bin size
    concat = trimesh.util.concatenate(list(part_meshes_pickup.values()))
    in_fixture = concat.vertices[concat.vertices[:, 2] < board_height_max]
    vmin, vmax = in_fixture.min(axis=0), in_fixture.max(axis=0)
    part_extent = vmax - vmin
    edge_gap = (np.array(bin_size) - part_extent[:2]) / 2
    assert np.all(edge_gap >= 0), f"Bin size too small: gap={edge_gap}"

    # Compact fixture mesh
    box_units = np.floor(part_extent[:2] / DX) + 1
    box_units = np.ceil(box_units / 2) * 2
    box_extent = box_units * DX
    box_min = np.array([-box_extent[0] / 2, min_fixture_y, 0])
    box_max = np.array([box_extent[0] / 2, min_fixture_y + box_extent[1], board_height_max])
    board_mesh = trimesh.creation.box(bounds=[box_min, box_max])
    board_mesh_bottom = trimesh.creation.box(
        bounds=[box_min, [box_max[0], box_max[1], BOTTOM_THICKNESS]]
    )

    # Part translation to center
    box_center = (box_min + box_max) / 2.0
    part_center = (vmin + vmax) / 2.0
    part_translation = box_center - part_center
    part_translation[2] = 0.0

    # Create swept volumes for mold cavities
    part_meshes_swept = {}
    for pid, part_mesh in part_meshes_pickup.items():
        low = part_mesh.slice_plane(
            [0, 0, board_height_max + 0.01], [0, 0, -1], cap=True
        )
        high = low.copy()
        high.apply_translation([0, 0, board_height_max - BOTTOM_THICKNESS + 0.01])
        part_meshes_swept[pid] = get_swept_mesh(low, high)

    # Subtract parts and grippers from the board using manifold3d
    # Note: buffered meshes are non-manifold (inflated along normals), so we
    # use .convex_hull to make them manifold-safe before subtracting.
    board_m = _to_manifold(board_mesh)
    part_boxes = []
    for pid, swept in part_meshes_swept.items():
        buffered = get_buffered_meshes(swept, np.array(MOLD_EDGE_OFFSET_PART) / 2)
        buffered.apply_translation(part_translation)
        board_m = board_m - _to_manifold(buffered.convex_hull)

        pv = buffered.vertices
        pmin, pmax = pv.min(axis=0), pv.max(axis=0)
        pmin -= PART_BOUNDARY_OFFSET
        pmax += PART_BOUNDARY_OFFSET
        pmin[2] = -1e-2
        pmax[2] = part_meshes_pickup[pid].center_mass[2] + 0.5
        part_boxes.append(trimesh.creation.box(bounds=[pmin, pmax]))

        if gripper_meshes_pickup[pid].vertices.min(axis=0)[2] < board_height_max:
            gripper_hull = gripper_meshes_pickup[pid].slice_plane(
                [0, 0, board_height_max + 0.01], [0, 0, -1], cap=True
            ).convex_hull
            gripper_buffered = get_buffered_meshes(
                gripper_hull, np.array(MOLD_EDGE_OFFSET_GRIPPER) / 2
            )
            gripper_buffered.apply_translation(part_translation)
            board_m = board_m - _to_manifold(gripper_buffered.convex_hull)

    # Intersection with part boxes and union with bottom plate
    pbu = _to_manifold(part_boxes[0])
    for pb in part_boxes[1:]:
        pbu = pbu + _to_manifold(pb)
    board_m = board_m ^ pbu  # intersection
    board_m = board_m + _to_manifold(board_mesh_bottom)  # union with bottom
    board_mesh = _to_trimesh(board_m)

    return board_mesh, part_translation


def _aabb_overlap(mesh_a, mesh_b):
    """Quick AABB overlap test between two meshes."""
    a_min, a_max = mesh_a.bounds
    b_min, b_max = mesh_b.bounds
    return np.all(a_min <= b_max) and np.all(b_min <= a_max)


def _meshes_collide(mesh_a, mesh_b):
    """
    Check if two meshes collide using manifold3d intersection.
    If the intersection volume is non-trivial, they collide.
    Falls back to AABB check first for speed.
    """
    if not _aabb_overlap(mesh_a, mesh_b):
        return False
    try:
        ma = _to_manifold(mesh_a)
        mb = _to_manifold(mesh_b)
        intersection = ma ^ mb
        # If the intersection has negligible volume, no collision
        return intersection.num_tri() > 0
    except Exception:
        # If manifold conversion fails (non-watertight), use AABB as fallback
        return True


def check_part_gripper_collision(part_meshes_pickup, gripper_meshes_pickup, sequence):
    """
    Check which parts have gripper-other-part collisions in pickup layout.
    Uses manifold3d intersection instead of FCL CollisionManager.
    """
    disassembly_order = [pm for pm, _ in sequence] + [sequence[-1][1]]
    parts_to_buffer = []
    accumulated_parts = []  # list of (pid, mesh) already placed

    for pid in disassembly_order:
        gripper = gripper_meshes_pickup[pid]
        collides = False
        for other_pid, other_mesh in accumulated_parts:
            if _meshes_collide(gripper, other_mesh):
                collides = True
                break
        if collides:
            parts_to_buffer.append(pid)
        accumulated_parts.append((pid, part_meshes_pickup[pid]))

    return parts_to_buffer


def add_countersunk_pads_to_fixture(fixture_mesh, min_fixture_y):
    """Add screw-mount pads to the four corners of the fixture."""
    bsize = fixture_mesh.extents[:2]
    lo_x = -bsize[0] / 2 - DX / 2
    hi_x = bsize[0] / 2 + DX / 2
    lo_y = min_fixture_y + DX / 2
    hi_y = min(
        min_fixture_y + bsize[1] - DX / 2,
        min_fixture_y + MAX_BIN_SIZE_DOUBLE[1] // 2 + DX / 2,
    )
    centers = [(lo_x, lo_y), (hi_x, lo_y), (lo_x, hi_y), (hi_x, hi_y)]
    pads = []
    for cx, cy in centers:
        pad = generate_countersunk_pad()
        pad.apply_translation([cx, cy, 0.0])
        pads.append(pad)
    # Use manifold3d for union
    return manifold_union([fixture_mesh] + pads)


# ===========================================================================
# Main entry point
# ===========================================================================

def run_fixture_gen(assembly_dir, log_dir, output_dir, optimized=True, seed=0):
    """
    Generate fixture mesh and pickup poses.

    Args:
        assembly_dir: Path to directory with part OBJ files.
        log_dir:      Path to logs with grasps.pkl, precedence.pkl, tree_opt.pkl.
        output_dir:   Where to write fixture.obj and pickup.json.
        optimized:    Use tree_opt.pkl (True) or tree.pkl (False).
        seed:         Random seed for non-optimized sequence sampling.
    """
    # Load pickle data
    precedence_path = os.path.join(log_dir, "precedence.pkl")
    grasps_path = os.path.join(log_dir, "grasps.pkl")
    tree_path = os.path.join(
        log_dir, "tree_opt.pkl" if optimized else "tree.pkl"
    )

    for p in [precedence_path, grasps_path, tree_path]:
        if not os.path.exists(p):
            print(f"[generate_fixture] ERROR: {p} not found")
            return

    with open(precedence_path, "rb") as f:
        G_preced = pickle.load(f)
    with open(grasps_path, "rb") as f:
        grasps = pickle.load(f)
    with open(tree_path, "rb") as f:
        tree = pickle.load(f)

    arm_type = grasps["arm"]
    gripper_type = grasps["gripper"]

    # Extract sequence from tree
    if optimized:
        seq_optimizer = SequenceOptimizer(G_preced, grasps)
        sequence, grasps_sequence = seq_optimizer.get_sequence(tree)
    else:
        # SequencePlanner requires more Fabrica dependencies; raise clear error
        raise NotImplementedError(
            "Non-optimized mode requires SequencePlanner which has heavy Fabrica "
            "dependencies. Use --optimized (default) with tree_opt.pkl instead."
        )

    if sequence is None or grasps_sequence is None:
        print(f"[generate_fixture] No feasible sequence found in {tree_path}")
        return

    # Load part meshes in final pose
    part_meshes_final = load_part_meshes(assembly_dir, transform="final")
    part_meshes_final = {k.replace("part", ""): v for k, v in part_meshes_final.items()}
    assembly_center = get_assembly_center(arm_type)
    for mesh in part_meshes_final.values():
        mesh.apply_translation(assembly_center)

    pos_dict, quat_dict = load_pos_quat_dict(assembly_dir, transform="final")
    pos_dict = {pid: pos_dict[pid] + assembly_center for pid in part_meshes_final}
    part_cfg_final = {"mesh": part_meshes_final, "pos": pos_dict, "quat": quat_dict}

    min_y = get_fixture_min_y(arm_type)

    t_start = time()

    # Compute per-part pickup orientations
    pose_info = generate_individual_pose_info(part_cfg_final, sequence, grasps_sequence)

    # Iterative bin packing with gripper collision checking
    while True:
        pickup_pose, bin_size = generate_pickup_pose(pose_info, min_y)
        if bin_size is None:
            print("[generate_fixture] Bin size exceeds maximum")
            return

        part_meshes_pickup, gripper_meshes_pickup = generate_pickup_meshes(
            part_cfg_final, sequence, grasps_sequence, gripper_type, pickup_pose
        )

        parts_to_buffer = check_part_gripper_collision(
            part_meshes_pickup, gripper_meshes_pickup, sequence
        )
        if len(parts_to_buffer) == 0:
            break
        for pid in parts_to_buffer:
            pose_info[pid]["extent_x"] += DELTA_BUFFER_SIZE

    # Generate fixture
    fixture_mesh, part_translation = generate_fixture(
        part_meshes_pickup, gripper_meshes_pickup, bin_size, min_y
    )
    for mesh in part_meshes_pickup.values():
        mesh.apply_translation(part_translation)

    # Add countersunk mounting pads
    fixture_mesh = add_countersunk_pads_to_fixture(fixture_mesh, min_y)

    elapsed = round(time() - t_start, 2)
    print(f"[generate_fixture] Fixture generated in {elapsed}s")

    # Compute global pickup poses
    pickup_global = {}
    for pid, pose in pickup_pose.items():
        T = get_transform_matrix(pose) @ get_transform_matrix_quat(
            part_cfg_final["pos"][pid], part_cfg_final["quat"][pid]
        )
        T[:3, 3] += part_translation
        pickup_global[pid] = get_pos_euler_from_transform_matrix(T).tolist()

    # Write outputs
    os.makedirs(output_dir, exist_ok=True)
    fixture_path = os.path.join(output_dir, "fixture.obj")
    pickup_path = os.path.join(output_dir, "pickup.json")

    fixture_mesh.export(fixture_path)
    with open(pickup_path, "w") as f:
        json.dump(pickup_global, f)

    print(f"[generate_fixture] Wrote {fixture_path}")
    print(f"[generate_fixture] Wrote {pickup_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate fixture mesh and pickup poses.")
    parser.add_argument(
        "--assembly-dir",
        type=str,
        required=True,
        help="Path to directory containing part OBJ files",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Path to logs with grasps.pkl, precedence.pkl, tree_opt.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for fixture.obj and pickup.json",
    )
    parser.add_argument(
        "--no-optimized",
        action="store_true",
        default=False,
        help="Use tree.pkl instead of tree_opt.pkl (requires SequencePlanner)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (non-optimized only)")
    parser.add_argument(
        "--fabrica-root",
        type=str,
        default=None,
        help="Override FABRICA_ROOT path for loading gripper assets and pickle classes",
    )
    args = parser.parse_args()

    if args.fabrica_root is not None:
        FABRICA_ROOT = args.fabrica_root
        if FABRICA_ROOT not in sys.path:
            sys.path.insert(0, FABRICA_ROOT)

    run_fixture_gen(
        assembly_dir=args.assembly_dir,
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        optimized=not args.no_optimized,
        seed=args.seed,
    )

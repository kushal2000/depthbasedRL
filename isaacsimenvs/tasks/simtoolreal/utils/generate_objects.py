"""Procedural handle-head URDF generation for SimToolReal.

Clean port of the minimal subset of isaacgymenvs/tasks/simtoolreal/generate_objects.py
needed for the base goal-pose-reaching env: `generate_handle_urdf` (single cuboid
or cylinder) and `generate_handle_head_urdf` (handle + head composite with
variable densities + parallel-axis-adjusted inertia). No trimesh / vhacd code.

The top-level entry `generate_handle_head_urdfs(types, num_per_type, out_dir)`
mirrors legacy `env.py:_handle_head_primitives`: per-type sampling with
`np.random.seed(42)`, pool shuffle, and scales normalized by `object_base_size`.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS, Scale2, Scale3

Scale = Union[Scale2, Scale3]

# Matches legacy env.py:1732.
_SEED = 42
# Matches legacy env.py:1731; kept module-level so call sites can override for tests.
_NUM_OBJECTS_PER_TYPE_DEFAULT = 100
# Reward-space normalization constant (same value the env uses when rescaling).
_OBJECT_BASE_SIZE = 0.04

_FORCED_SIMPLE_CUBOID_MAX_CROSS_SECTION_M = 0.05
_FORCED_SIMPLE_CUBOID_MIN_CROSS_SECTION_M = 0.012
_FORCED_SIMPLE_CYLINDER_CROSS_SECTION_SCALE = 1.0 / 3.0


# ----------------------------------------------------------------------------
# Primitive URDF emitters
# ----------------------------------------------------------------------------


# All procedural object URDFs use this same link name so MultiUsdFileCfg-spawned
# prims have an identical internal structure across envs (env_*/Object/<LINK>).
# RigidObject's view regex is derived from env_0's structure and applied to all
# envs; if the link name varies, only matching envs are registered and
# set_transforms overflows the partial view.
_OBJECT_ROOT_LINK = "object_root"


def _cuboid_urdf_constant_density(path: Path, scale: Scale3, density: float) -> Path:
    lx, ly, lz = scale
    # Standard URDF requires <mass> + <inertia>. Legacy isaacgym's URDF
    # importer accepts <density> as a Gazebo extension and back-computes mass
    # from density × volume; Isaac Sim's importer follows strict URDF spec
    # and falls back to 1 kg when <mass> is missing — that breaks erasers and
    # any other handle-only types. Emit explicit <mass> + <inertia> so both
    # backends see byte-identical physics.
    m, ixx, iyy, izz = _compute_mass_and_inertia(scale, density)
    urdf = f"""<?xml version="1.0"?>
<robot name="cuboid">
  <link name="{_OBJECT_ROOT_LINK}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{lx} {ly} {lz}"/></geometry>
      <material name="brown"><color rgba="0.55 0.27 0.07 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{lx} {ly} {lz}"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{m}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""
    path.write_text(urdf)
    return path


def _cylinder_urdf_constant_density(
    path: Path, height: float, diameter: float, density: float
) -> Path:
    radius = diameter / 2
    # Same fix as the cuboid branch — emit explicit <mass> + <inertia> so
    # Isaac Sim's URDF importer doesn't fall back to a default 1 kg mass.
    # The geometry is rotated by rpy="0 -π/2 0" so the cylinder's length axis
    # aligns with link x. _compute_mass_and_inertia returns inertia in the
    # geometry-local frame (axis = z); we apply the same rpy on the inertial
    # origin so the values stay correct after rotation.
    m, ixx, iyy, izz = _compute_mass_and_inertia((height, diameter), density)
    urdf = f"""<?xml version="1.0"?>
<robot name="cylinder">
  <link name="{_OBJECT_ROOT_LINK}">
    <visual>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <geometry><cylinder length="{height}" radius="{radius}"/></geometry>
      <material name="brown"><color rgba="0.55 0.27 0.07 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <geometry><cylinder length="{height}" radius="{radius}"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <mass value="{m}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""
    path.write_text(urdf)
    return path


def generate_handle_urdf(
    path: Path, handle_scale: Scale, handle_density: float = 400.0
) -> Path:
    """Emit a URDF for a single cuboid (len(scale)==3) or cylinder (len(scale)==2) handle."""
    if len(handle_scale) == 3:
        return _cuboid_urdf_constant_density(path, handle_scale, handle_density)
    if len(handle_scale) == 2:
        return _cylinder_urdf_constant_density(
            path, handle_scale[0], handle_scale[1], handle_density
        )
    raise ValueError(f"Invalid handle_scale: {handle_scale}")


# ----------------------------------------------------------------------------
# Handle + head composite (variable density, single-link, with adjusted inertia)
# ----------------------------------------------------------------------------


def _compute_mass_and_inertia(scale: Scale, density: float):
    """Capsule-approximation for cylinders; exact for cuboids.

    Returns (m, ixx, iyy, izz) with scale-axis = z (caller flips if needed).
    """
    if len(scale) == 3:
        lx, ly, lz = scale
        v = lx * ly * lz
        m = v * density
        ixx = (1 / 12) * m * (ly * ly + lz * lz)
        iyy = (1 / 12) * m * (lx * lx + lz * lz)
        izz = (1 / 12) * m * (lx * lx + ly * ly)
        return m, ixx, iyy, izz
    if len(scale) == 2:
        h, d = scale[0], scale[1]
        r = d / 2
        # Capsule mass = cylinder + two hemispheres.
        m_c = density * math.pi * r * r * h
        m_h = density * (2 / 3) * math.pi * r ** 3
        m = m_c + 2 * m_h
        # Cylinder inertia about centroid (axis = z).
        i_c_axis = 0.5 * m_c * r * r
        i_c_perp = (1 / 12) * m_c * (3 * r * r + h * h)
        # Hemisphere inertia about its own centroid.
        i_h_axis = (2 / 5) * m_h * r * r
        i_h_perp = (83 / 320) * m_h * r * r
        d_com = (h / 2) + (3 * r / 8)
        izz = i_c_axis + 2 * i_h_axis
        ixx = iyy = i_c_perp + 2 * (i_h_perp + m_h * d_com * d_com)
        return m, ixx, iyy, izz
    raise ValueError(f"Invalid scale: {scale}")


def _handle_head_urdf_variable_density(
    path: Path,
    handle_scale: Scale,
    head_scale: Scale,
    handle_density: float,
    head_density: float,
) -> Path:
    # Handle geometry + inertia (axis along +x; for cylinders we rotate -pi/2 about y).
    if len(handle_scale) == 3:
        lx, ly, lz = handle_scale
        handle_geom = (
            f'<origin xyz="0 0 0" rpy="0 0 0"/>\n'
            f'        <geometry><box size="{lx} {ly} {lz}"/></geometry>'
        )
        handle_mass, handle_ixx, handle_iyy, handle_izz = _compute_mass_and_inertia(
            handle_scale, handle_density
        )
    else:
        h, d = handle_scale
        r = d / 2
        handle_geom = (
            f'<origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>\n'
            f'        <geometry><cylinder length="{h}" radius="{r}"/></geometry>'
        )
        # Rotated so handle axis is along +x; flip ixx ↔ izz.
        handle_mass, handle_izz, handle_iyy, handle_ixx = _compute_mass_and_inertia(
            handle_scale, handle_density
        )

    # Head geometry + inertia.
    if len(head_scale) == 3:
        hlx, hly, hlz = head_scale
        x_offset = handle_scale[0] / 2 + hlx / 2
        head_geom = (
            f'<origin xyz="{x_offset} 0 0" rpy="0 0 0"/>\n'
            f'        <geometry><box size="{hlx} {hly} {hlz}"/></geometry>'
        )
        head_mass, head_ixx, head_iyy, head_izz = _compute_mass_and_inertia(
            head_scale, head_density
        )
    else:
        hh, hd = head_scale
        hr = hd / 2
        x_offset = handle_scale[0] / 2 + hr
        head_geom = (
            f'<origin xyz="{x_offset} 0 0" rpy="-1.5707963267948966 0 0"/>\n'
            f'        <geometry><cylinder length="{hh}" radius="{hr}"/></geometry>'
        )
        # Rotated so head axis is along +y; flip iyy ↔ izz.
        head_mass, head_ixx, head_izz, head_iyy = _compute_mass_and_inertia(
            head_scale, head_density
        )

    # Parallel-axis shift to composite COM (handle at origin, head at +x_offset).
    total_mass = handle_mass + head_mass
    com_x = (handle_mass * 0.0 + head_mass * x_offset) / total_mass
    d_handle = -com_x
    d_head = x_offset - com_x

    ixx = handle_ixx + head_ixx
    iyy = (handle_iyy + handle_mass * d_handle * d_handle) + (
        head_iyy + head_mass * d_head * d_head
    )
    izz = (handle_izz + handle_mass * d_handle * d_handle) + (
        head_izz + head_mass * d_head * d_head
    )

    urdf = f"""<?xml version="1.0"?>
<robot name="handle_head">
  <link name="{_OBJECT_ROOT_LINK}">
    <visual>
      {handle_geom}
      <material name="brown"><color rgba="0.55 0.27 0.07 1.0"/></material>
    </visual>
    <collision>
      {handle_geom}
    </collision>
    <visual>
      {head_geom}
      <material name="gray"><color rgba="0.5 0.5 0.5 1.0"/></material>
    </visual>
    <collision>
      {head_geom}
    </collision>
    <inertial>
      <origin xyz="{com_x} 0 0" rpy="0 0 0"/>
      <mass value="{total_mass}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""
    path.write_text(urdf)
    return path


def generate_handle_head_urdf(
    path: Path,
    handle_scale: Scale,
    head_scale: Optional[Scale],
    handle_density: float = 400.0,
    head_density: Optional[float] = 800.0,
) -> Path:
    """Emit a URDF for a handle (no head) or a handle+head composite.

    The handle-only branch matches legacy `generate_handle_urdf`; the composite
    branch matches `generate_handle_head_urdf_variable_density` (single-link
    with parallel-axis-adjusted inertia — the 2-link variant was unstable per
    legacy env.py:1723-1724 and is not ported).
    """
    if head_scale is None and head_density is None:
        return generate_handle_urdf(path, handle_scale, handle_density)
    if head_scale is not None and head_density is not None:
        return _handle_head_urdf_variable_density(
            path, handle_scale, head_scale, handle_density, head_density
        )
    raise ValueError(
        f"head_scale and head_density must both be set or both None (got "
        f"{head_scale} and {head_density})"
    )


# ----------------------------------------------------------------------------
# Top-level: generate a pool of N×types URDFs for the env
# ----------------------------------------------------------------------------


def _scale_to_3d(scale: np.ndarray) -> tuple[float, float, float]:
    """Convert cylinder (h, d) → (h, d, d) so all returned scales are 3-tuples."""
    if len(scale) == 3:
        return (float(scale[0]), float(scale[1]), float(scale[2]))
    if len(scale) == 2:
        return (float(scale[0]), float(scale[1]), float(scale[1]))
    raise ValueError(f"Invalid scale shape: {scale.shape}")


def _shape_from_dims(dims: tuple[float, ...]) -> str:
    if len(dims) == 3:
        return "cuboid"
    if len(dims) == 2:
        return "cylinder"
    raise ValueError(f"Unsupported object dims: {dims}")


def _lerp_dims_band(
    min_lengths: tuple[float, ...],
    max_lengths: tuple[float, ...],
    rng: np.random.Generator,
    *,
    low: float = 0.25,
    high: float = 0.75,
) -> tuple[float, ...]:
    lo = np.asarray(min_lengths, dtype=float)
    hi = np.asarray(max_lengths, dtype=float)
    return tuple((lo + rng.uniform(low, high, size=lo.shape) * (hi - lo)).tolist())


def _forced_simple_distributions(shape: str, matching) -> list:
    out = []
    for dist in matching:
        if _shape_from_dims(tuple(dist.handle_min_lengths)) != shape:
            continue
        if shape == "cuboid":
            max_lengths = np.asarray(dist.handle_max_lengths, dtype=float)
            if np.max(max_lengths[1:]) > _FORCED_SIMPLE_CUBOID_MAX_CROSS_SECTION_M:
                continue
        out.append(dist)
    if not out:
        raise ValueError(f"No matching training distributions for forced simple shape {shape!r}.")
    return out


def _sample_forced_simple_scale(dist, rng: np.random.Generator, shape: str) -> tuple[float, ...]:
    handle_dims = _lerp_dims_band(dist.handle_min_lengths, dist.handle_max_lengths, rng)
    if shape == "cuboid":
        if len(handle_dims) != 3:
            raise ValueError("Forced simple cuboids must sample from cuboid-handle distributions.")
        length, width, height = (float(v) for v in handle_dims)
        width = max(width, _FORCED_SIMPLE_CUBOID_MIN_CROSS_SECTION_M)
        height = max(height, _FORCED_SIMPLE_CUBOID_MIN_CROSS_SECTION_M)
        return length, width, height
    if shape == "cylinder":
        if len(handle_dims) != 2:
            raise ValueError("Forced simple cylinders must sample from cylinder-handle distributions.")
        length, cross_section = (float(v) for v in handle_dims)
        return length, cross_section * _FORCED_SIMPLE_CYLINDER_CROSS_SECTION_SCALE
    raise ValueError(f"Unsupported simple shape: {shape}")


def _sample_training_object(dist, rng: np.random.Generator):
    handle_density = float(rng.uniform(dist.handle_min_density, dist.handle_max_density))
    head_density = (
        None
        if dist.head_min_density is None or dist.head_max_density is None
        else float(rng.uniform(dist.head_min_density, dist.head_max_density))
    )
    handle_min = np.asarray(dist.handle_min_lengths, dtype=float)
    handle_max = np.asarray(dist.handle_max_lengths, dtype=float)
    handle_scale = tuple((handle_min + rng.random(handle_min.shape) * (handle_max - handle_min)).tolist())
    if dist.head_min_lengths is None or dist.head_max_lengths is None:
        head_scale = None
    else:
        head_min = np.asarray(dist.head_min_lengths, dtype=float)
        head_max = np.asarray(dist.head_max_lengths, dtype=float)
        head_scale = tuple((head_min + rng.random(head_min.shape) * (head_max - head_min)).tolist())
    return handle_scale, head_scale, handle_density, head_density


def generate_handle_head_urdfs(
    handle_head_types: tuple[str, ...],
    num_per_type: int = _NUM_OBJECTS_PER_TYPE_DEFAULT,
    out_dir: Union[str, Path] = "/tmp/simtoolreal_assets",
    object_base_size: float = _OBJECT_BASE_SIZE,
    seed: int = _SEED,
    shuffle: bool = True,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Generate a pool of URDFs across the requested handle-head types.

    Mirrors legacy ``_handle_head_primitives`` at
    isaacgymenvs/tasks/simtoolreal/env.py:1706-1840:
      1. Seed NumPy with ``seed`` (deterministic pool per run).
      2. For each matching ``ObjectSizeDistribution``, sample ``num_per_type``
         (handle_scale, head_scale, handle_density, head_density) tuples.
      3. Emit one URDF per sample; return file paths paired with object scales
         normalized by ``object_base_size``.
      4. Shuffle (paths, scales) in lockstep so env ``i`` gets pool entry
         ``i % len(pool)`` with uniform coverage over types.

    Returns
    -------
    urdf_paths : list[str]
        Absolute paths to generated URDFs. Length = ``num_per_type × len(matching types)``.
    object_scales_normalized : list[tuple[float, float, float]]
        Per-pool-entry 3-tuple scales, normalized by ``object_base_size`` (reward
        math uses these directly; see env.py:1814-1821).
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.suffix == ".urdf":
                p.unlink()
    else:
        os.makedirs(out_dir)

    np.random.seed(seed)

    type_set = set(handle_head_types)
    matching = [d for d in OBJECT_SIZE_DISTRIBUTIONS if d.type in type_set]
    if not matching:
        raise ValueError(
            f"No matching ObjectSizeDistribution for handle_head_types={handle_head_types}. "
            f"Valid types: {sorted({d.type for d in OBJECT_SIZE_DISTRIBUTIONS})}"
        )

    paths: list[str] = []
    scales_raw: list[tuple[float, ...]] = []
    for dist in matching:
        # Sample order MUST match legacy env.py:1758-1772 (densities first,
        # scales second) — the two pools share seed=42 so first-asset URDFs
        # come out byte-identical across backends only when the np.random
        # draws line up step for step.
        handle_densities = dist.sample_handle_densities(num_per_type)
        head_densities = dist.sample_head_densities(num_per_type)
        handle_scales = dist.sample_handle_scales(num_per_type)
        head_scales = dist.sample_head_scales(num_per_type)

        for idx in range(num_per_type):
            h_scale = tuple(float(x) for x in handle_scales[idx])
            head = tuple(float(x) for x in head_scales[idx]) if head_scales is not None else None
            h_d = float(handle_densities[idx])
            head_d = float(head_densities[idx]) if head_densities is not None else None

            fname = (
                f"{idx:03d}_{dist.type}_handle_{h_scale}_head_{head}_d{h_d:.1f}_{head_d}"
                .replace(".", "-")
                + ".urdf"
            )
            urdf_path = out_dir / fname
            generate_handle_head_urdf(
                path=urdf_path,
                handle_scale=h_scale,
                head_scale=head,
                handle_density=h_d,
                head_density=head_d,
            )
            paths.append(str(urdf_path))
            scales_raw.append(h_scale)

    # Convert to 3-tuples and normalize by object_base_size (matches env.py:1814-1821).
    scales_3d = [_scale_to_3d(np.asarray(s)) for s in scales_raw]
    scales_norm = [
        (x / object_base_size, y / object_base_size, z / object_base_size)
        for (x, y, z) in scales_3d
    ]

    # Shuffle (paths, scales) in lockstep so the per-type ordering doesn't bias
    # env-i-gets-asset-i%N. Matches legacy env.py:1824-1830. Disabled for
    # debug-parity runs (debug_differences/policy_rollout_isaacsim.py) where
    # we want pool[0] = first matching ObjectSizeDistribution.
    if shuffle:
        indices = np.arange(len(paths))
        np.random.shuffle(indices)
        paths = [paths[i] for i in indices]
        scales_norm = [scales_norm[i] for i in indices]

    return paths, scales_norm


def generate_mixed_training_simple_urdfs(
    handle_head_types: tuple[str, ...],
    num_per_type: int = _NUM_OBJECTS_PER_TYPE_DEFAULT,
    out_dir: Union[str, Path] = "/tmp/simtoolreal_assets",
    object_base_size: float = _OBJECT_BASE_SIZE,
    seed: int = _SEED,
    shuffle: bool = True,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Generate 50% training objects, 25% simple boxes, and 25% simple cylinders.

    This mirrors the assembly-renderer "mixed_training_simple_25_25_50" figure
    distribution while keeping outputs compatible with SimToolReal's URDF->USD
    scene path. Forced simple shapes are sampled from the training handle
    ranges rather than arbitrary dimensions.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.suffix == ".urdf":
                p.unlink()
    else:
        os.makedirs(out_dir)

    rng = np.random.default_rng(seed)

    type_set = set(handle_head_types)
    matching = [d for d in OBJECT_SIZE_DISTRIBUTIONS if d.type in type_set]
    if not matching:
        raise ValueError(
            f"No matching ObjectSizeDistribution for handle_head_types={handle_head_types}. "
            f"Valid types: {sorted({d.type for d in OBJECT_SIZE_DISTRIBUTIONS})}"
        )
    forced_cuboids = _forced_simple_distributions("cuboid", matching)
    forced_cylinders = _forced_simple_distributions("cylinder", matching)

    total_count = max(1, int(num_per_type) * len(matching))
    paths: list[str] = []
    scales_raw: list[tuple[float, ...]] = []
    simple_box_idx = 0
    simple_cylinder_idx = 0

    for idx in range(total_count):
        slot = idx % 4
        if slot in {0, 2}:
            dist = matching[idx % len(matching)]
            h_scale, head, h_d, head_d = _sample_training_object(dist, rng)
            fname = (
                f"{idx:04d}_training_{dist.type}_handle_{h_scale}_head_{head}_d{h_d:.1f}_{head_d}"
                .replace(".", "-")
                + ".urdf"
            )
            urdf_path = out_dir / fname
            generate_handle_head_urdf(
                path=urdf_path,
                handle_scale=h_scale,
                head_scale=head,
                handle_density=h_d,
                head_density=head_d,
            )
            scales_raw.append(h_scale)
        elif slot == 1:
            dist = forced_cuboids[simple_box_idx % len(forced_cuboids)]
            simple_box_idx += 1
            h_scale = _sample_forced_simple_scale(dist, rng, "cuboid")
            h_d = float(rng.uniform(dist.handle_min_density, dist.handle_max_density))
            fname = f"{idx:04d}_simple_box_from_{dist.type}_handle_{h_scale}_d{h_d:.1f}".replace(".", "-") + ".urdf"
            urdf_path = out_dir / fname
            generate_handle_urdf(urdf_path, h_scale, h_d)
            scales_raw.append(h_scale)
        else:
            dist = forced_cylinders[simple_cylinder_idx % len(forced_cylinders)]
            simple_cylinder_idx += 1
            h_scale = _sample_forced_simple_scale(dist, rng, "cylinder")
            h_d = float(rng.uniform(dist.handle_min_density, dist.handle_max_density))
            fname = (
                f"{idx:04d}_simple_cylinder_from_{dist.type}_handle_{h_scale}_d{h_d:.1f}"
                .replace(".", "-")
                + ".urdf"
            )
            urdf_path = out_dir / fname
            generate_handle_urdf(urdf_path, h_scale, h_d)
            scales_raw.append(h_scale)
        paths.append(str(urdf_path))

    scales_3d = [_scale_to_3d(np.asarray(s)) for s in scales_raw]
    scales_norm = [
        (x / object_base_size, y / object_base_size, z / object_base_size)
        for (x, y, z) in scales_3d
    ]

    if shuffle:
        indices = np.arange(len(paths))
        rng.shuffle(indices)
        paths = [paths[i] for i in indices]
        scales_norm = [scales_norm[i] for i in indices]

    return paths, scales_norm


__all__ = [
    "generate_handle_urdf",
    "generate_handle_head_urdf",
    "generate_handle_head_urdfs",
    "generate_mixed_training_simple_urdfs",
]

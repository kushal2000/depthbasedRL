# Peg-in-hole scene generation

Generates training scenes for the peg-in-hole benchmark with three
independent axes of diversity: **hole position**, **peg initial pose**,
and **hole tolerance**.

A *scene* is keyed by one hole XY. For each scene we cache:
- `M` peg start poses + trajectories (same hole, many ways to reach it).
- `K` tolerance-variant URDFs of the table+hole at this XY.

URDFs depend only on `(hole XY, tolerance)`, so we write `N × K` of them and
share them across the `M` peg starts. At training time the env randomizes
`(scene_idx, peg_idx, tol_slot_idx)` — three independent axes — so the
policy is forced to generalize over all three without being told which
tolerance it faces.

## Pipeline

### 1. `compute_stable_rests_isaacgym.py`

Validates which of the 6 axis-aligned face-down orientations are dynamically
stable for the peg by dropping it above a ground plane in Isaac Gym and
checking the part-local down direction after settle. Outputs to
`assets/urdf/peg_in_hole/stable_rests/peg.npz`.

```bash
python peg_in_hole/scene_generation/compute_stable_rests_isaacgym.py
# Visual debug:
python peg_in_hole/scene_generation/compute_stable_rests_isaacgym.py --viewer
```

Typical result: 4 stable rests (±z up, ±x up). Only the **flat** rests
(±z up) are used downstream — standing rests (±x up) are filtered out so
the policy never starts with the peg upright.

### 2. `generate_scenes.py`

For each of `N` scenes:
- Samples hole XY in a `±(0.15, 0.11)` box around `(0, 0)` (table interior
  minus 0.05 edge inset and half the 0.08 m hole footprint).
- Rejection-samples `M` peg starts against this hole:
    - Stable rest + yaw ∼ `U[-π, π]` + XY in a `±(0.1, 0.1)` box around
      `(0, 0)` (SimToolReal `resetPositionNoiseX/Y`).
    - Peg mesh bbox must stay within table inset, and min distance to the
      hole block must be ≥ 0.02 m.
- If `M` peg starts can't be found for the sampled hole, re-rolls the hole.
- For each peg start, generates one crane-style trajectory (lift → transit
  → approach) via
  `fabrica.scene_generation.trajectory_generation.generate_variable_trajectory`
  with ~2.5 cm waypoint spacing and `clearance_z = 0.73 m`.
- Samples `K` tolerances from the global pool without replacement (sorted)
  and writes `K` scene URDFs for this hole XY.

```bash
# Smoke test
python peg_in_hole/scene_generation/generate_scenes.py \
    --num-scenes 5 --pegs-per-scene 3 \
    --tolerance-pool-size 20 --tolerances-per-scene 3 \
    --seed 0 --force

# Default (100 × 10 × 10)
python peg_in_hole/scene_generation/generate_scenes.py
# ↳ 100 holes × 10 pegs × 10 tols = 10 000 (scene, peg, tol) triples
# ↳ 1 000 URDFs on disk (100 × 10)
# ↳ Pool of 100 log-uniform tolerance values in [0.1, 10] mm
```

## Outputs

`assets/urdf/peg_in_hole/`:

```
stable_rests/
  peg.npz                       # (N_rests, 4, 4) transforms + uniform probs
scenes/
  scenes.npz                    # master metadata (schema below)
  tolerance_pool.json           # human-readable pool in m and mm
  scene_0000/
    scene_tol00.urdf            # K URDFs per scene (hole XY + tolerance)
    scene_tol01.urdf
    ...
    scene_tol09.urdf
  scene_0001/
    ...
```

### `scenes.npz` schema

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `start_poses` | `(N, M, 7)` | float32 | per-peg xyz + xyzw quat (world) |
| `goals` | `(N, M, max_traj_len, 7)` | float32 | zero-padded per-peg trajectories |
| `traj_lengths` | `(N, M)` | int32 | effective waypoint count per peg |
| `hole_positions` | `(N, 3)` | float32 | hole base corner world position |
| `tolerance_pool_m` | `(pool_size,)` | float32 | sorted log-uniform tolerances (m) |
| `scene_tolerance_indices` | `(N, K)` | int32 | indices into `tolerance_pool_m`; URDF `scene_tol{ii:02d}.urdf` → `tolerance_pool_m[scene_tolerance_indices[s, ii]]` |

## Sampling bounds

World-frame centers + half-widths on XY:

| | Center | Half-widths | Source |
|---|---|---|---|
| Peg start | `(0, 0)` | `(0.1, 0.1)` | SimToolReal `resetPositionNoiseX/Y` |
| Hole XY | `(0, 0)` | `(0.15, 0.11)` | Table interior − 0.05 edge − 0.04 footprint |

Peg z = `TABLE_TOP_Z + rest_z_lift = 0.54 m` for flat rests. Hole base z is
pinned at `TABLE_TOP_Z = 0.53 m`. The full table is `0.475 × 0.4 × 0.3` m,
top at world z=0.53.

## Tolerance continuum

- **Global pool**: `pool_size` log-uniform samples in `[0.1, 10]` mm (sorted,
  stored in meters in `tolerance_pool_m`).
- **Per scene**: `tolerances_per_scene` indices drawn without replacement
  from the pool. Same hole XY across the K URDFs; only the slot clearance
  varies.
- **At train time**: envs randomize over `(scene_idx, peg_idx, tol_slot_idx)`.
  The policy sees the peg + hole poses but not the tolerance → forced to be
  robust across the continuum.

## Downstream (not wired yet)

- Extend `FabricaEnv` to load `scenes.npz` and sample `(scene, peg, tol)`
  per env on reset.
- Multi-init viser eval analogous to `fabrica_multi_init_eval.py`.

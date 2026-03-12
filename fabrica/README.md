# Fabrica Assembly Evaluation

Interactive evaluation and debugging tools for assembly insertion tasks using IsaacGym.

## eval_assembly.py

Web-based GUI for running policy rollouts on assembly tasks. Select assembly/part from dropdowns, load the environment, and run episodes.

### Without SDF (V-HACD convex decomposition)

```bash
python fabrica/eval_assembly.py \
    --config-path pretrained_policy/config.yaml \
    --checkpoint-path pretrained_policy/model.pth \
    --port 8080
```

### With SDF (exact mesh collision)

```bash
python fabrica/eval_assembly.py \
    --config-path pretrained_policy/config.yaml \
    --checkpoint-path pretrained_policy/model.pth \
    --use-sdf \
    --final-goal-tolerance 0.001 \
    --override task.sim.physx.num_position_iterations 16 \
    --override task.sim.physx.max_depenetration_velocity 5.0 \
    --override task.sim.physx.contact_offset 0.005 \
    --override task.sim.physx.friction_offset_threshold 0.01 \
    --override task.sim.physx.friction_correlation_distance 0.00625 \
    --port 8080
```

### Key flags

| Flag | Description |
|------|-------------|
| `--use-sdf` | Use SDF collision meshes instead of V-HACD |
| `--final-goal-tolerance` | Tighter tolerance for the last subgoal (default: uses `successTolerance`) |
| `--override KEY VALUE` | Override any config value (repeatable) |
| `--no-headless` | Show the IsaacGym viewer window |

## debug_insertion.py

Teleports a part step-by-step along its insertion trajectory to diagnose collision issues. Reports desired vs actual pose at each step — large deltas indicate collision problems.

### Without SDF

```bash
python fabrica/debug_insertion.py \
    --config-path pretrained_policy/config.yaml \
    --checkpoint-path pretrained_policy/model.pth \
    --assembly beam --part 2 \
    --steps-per-waypoint 50 \
    --start-waypoint 8 \
    --port 8080
```

### With SDF

```bash
python fabrica/debug_insertion.py \
    --config-path pretrained_policy/config.yaml \
    --checkpoint-path pretrained_policy/model.pth \
    --assembly beam --part 2 \
    --object-name beam_2_sdf \
    --table-urdf "urdf/fabrica/environments/beam_2/scene_sdf.urdf" \
    --steps-per-waypoint 50 \
    --start-waypoint 8 \
    --port 8080
```

### Key flags

| Flag | Description |
|------|-------------|
| `--object-name` | Override object name (e.g. `beam_2_sdf` for SDF variant) |
| `--table-urdf` | Override table URDF path relative to assets root |
| `--start-waypoint` | First waypoint index to start from (default: 8) |
| `--end-waypoint` | Last waypoint index (default: last) |
| `--steps-per-waypoint` | Interpolation steps between waypoints (default: 50) |

## SDF vs V-HACD

**V-HACD** (default): Decomposes meshes into convex hulls. Fast but inflates collision geometry, preventing tight-tolerance insertions.

**SDF** (`--use-sdf`): Signed distance fields give exact mesh boundaries. Requires `thickness=0.0` on asset options and shape properties (handled automatically by `useSDF` config). Needs tuned physics params to avoid instability — see the Fabrica-matched overrides above.

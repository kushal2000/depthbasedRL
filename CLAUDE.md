# CLAUDE.md

## Project Overview
Depth-based RL for robotic assembly. The `fabrica/` module handles beam/part assembly tasks with visualization and trajectory generation.

## Key Directories
- `assets/urdf/fabrica/<assembly>/` — per-assembly mesh parts (OBJ) and config (assembly_order.json)
- `assets/urdf/fabrica/environments/` — pre-built environment URDFs
- `fabrica/` — Python scripts for visualization, trajectory generation, evaluation
- `assets/urdf/kuka_sharpa_description/` — robot URDF

## Assemblies
beam, car, cooling_manifold, duct, gamepad, plumbers_block, stool_circular

Each assembly has numbered part subdirs (e.g., `beam/0/0.obj`) and an `assembly_order.json` with:
```json
{"steps": ["6","3","2","1","0"], "start_rotations": {"2": [1,0,0,90]}}
```

## Scene Constants (eval.py)
- TABLE_Z = 0.38, table surface = 0.53
- Robot at (0, 0.8, 0)
- Fixture at (0.12, -0.152, 0.15) relative to table

## Visualization
- Uses **viser** for web-based 3D viewers
- `viser_utils.py` has shared loaders: `load_all_assemblies`, `load_assembly_order`, `SceneManager`
- Viser `add_line_segments` does NOT reliably render — use mesh geometry for visual markers
- Default viser port: 8082

## Conventions
- Assembly task specs (order, rotations) go in JSON config files, not hardcoded in Python
- OBJ meshes are in assembled reference frame — no rotation needed at final position
- Part IDs are strings ("0", "1", ...) not ints

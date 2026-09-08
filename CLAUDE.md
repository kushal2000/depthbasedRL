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

## Workflow
- **Never kill ports blindly** — other processes (training jobs, user sessions) may be using them. Always use a free port instead of killing existing ones.
- Use `lsof -i:<port>` to check if a port is occupied before binding to it
- Only run one eval server at a time, always on port 8080

## Conventions
- Assembly task specs (order, rotations) go in JSON config files, not hardcoded in Python
- OBJ meshes are in assembled reference frame — no rotation needed at final position
- Part IDs are strings ("0", "1", ...) not ints
- Branch names start with the creation date in `YYYY_MM_DD` format (e.g., `2025_03_18_my_feature`)

## Python packaging for venv_isaacsim
- No `PYTHONPATH=...` hacks, even if the alternative costs ~30s of startup. Prefer proper installation.
- Repo-local packages (`deployment`, `isaacgymenvs`, `isaacsimenvs`, `fabrica`, …): register via `uv pip install -e . --no-deps` into the target venv. `--no-deps` is required in `.venv_isaacsim` because `pyproject.toml` pins conflict with Python 3.11 (numpy==1.23.0, warp-lang==0.10.1, isaacgym-stubs).
- New top-level package dirs must be added to `[tool.setuptools.packages.find]` in `pyproject.toml`, then `uv pip install -e . --no-deps` re-run.
- Isaac Lab sub-namespaces (`isaaclab.sim`, `isaaclab.envs`, …) resolve only after `AppLauncher(args)` runs. Any script that imports `isaaclab.*` must instantiate `AppLauncher` first. Do not export the bundled `site-packages/isaaclab/source/isaaclab` dir onto `PYTHONPATH`.
- Scripts that don't need Isaac Lab (e.g., `isaacsim_conversion/test_inference.py`) must not call `AppLauncher` — skip the Kit startup cost.

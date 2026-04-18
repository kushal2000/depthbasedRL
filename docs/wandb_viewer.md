# WandB Interactive 3D Viewer

During training, SimToolReal can log an interactive 3D pose-based viewer to WandB alongside the standard video capture. The viewer runs entirely from robot/object pose state — no camera sensors or GPU rendering required — so it works in headless training.

## How it works

At a configurable cadence the environment captures one episode's worth of robot joint positions, robot base pose, object pose, goal pose, and table pose. These are passed to `viewer_api.create_html()` which embeds a self-contained Three.js scene into a single HTML file, logged as `wandb.Html`. The standard `wandb.Video` is logged at the same time so both can be compared side-by-side in the WandB UI.

The viewer is implemented in `isaacgymenvs/tasks/simtoolreal/interactive_viewer/`:

| File | Purpose |
|------|---------|
| `viewer_api.py` | `create_html()`, `make_url_robot()`, `make_embedded_robot()` |
| `viewer_common.py` | `render_template()` and Jinja helpers |
| `index.template.html` | Browser runtime (Three.js + urdf-loader) |
| `__init__.py` | Re-exports the public API |

## Config keys (`isaacgymenvs/cfg/task/SimToolReal.yaml`)

```yaml
env:
  capture_viewer: True          # enable the interactive viewer
  capture_viewer_freq: 6_000    # log every N env steps (same cadence as video)
  capture_viewer_len: 600       # number of frames to capture per episode

  # Base URL for GitHub raw URDF fetching.
  # "" (default) = /main/ branch (stable; always has all released URDF + mesh files).
  # To target your exact pushed commit instead, set _VIEWER_USE_CURRENT_COMMIT = True
  # in env.py, or override this key with an explicit URL:
  #   task.env.capture_viewer_github_raw_base=https://raw.githubusercontent.com/tylerlum/simtoolreal/abc123def456/
  capture_viewer_github_raw_base: ""

  # What to do when a URDF URL is unreachable (HTTP HEAD request, with timing):
  #   "warn"  (default) — print a loud warning box but continue
  #   "error"           — raise ValueError and abort
  #   "skip"            — no check (use when no internet access)
  capture_viewer_url_check: "warn"
```

### Adding a new URDF / mesh

When adding a new robot or object URDF with mesh files:

1. Push the new files to GitHub.
2. The default `/main/` base will serve the files as long as they are merged to main.
   While iterating on an unpushed branch, override with your commit/branch URL or set
   `_VIEWER_USE_CURRENT_COMMIT = True` in `env.py` to auto-derive from `git rev-parse HEAD`.
3. If the URL is unreachable you will see a URL check failure. Set
   `capture_viewer_url_check=skip` to bypass the check when offline.

## Quaternion convention

Isaac Gym root state tensor layout: `[x, y, z, qx, qy, qz, qw]` (xyzw).
`viewer_api.py` expects the same convention.
Three.js `quaternion.set(x, y, z, w)` also takes xyzw — all consistent, no reordering needed.

## Supported robot

The viewer is hardcoded to `iiwa14_left_sharpa_adjusted_restricted` (the only robot currently
used in SimToolReal). A `ValueError` is raised at env init if a different robot asset is
configured. The object URDF is read at env creation time and embedded inline (pure
`<box>`/`<cylinder>` primitives only — no external mesh references).

## Smoke test

```bash
cd /home/tylerlum/github_repos/simtoolreal_private
source .venv/bin/activate

# freq=50 so viewer arms immediately; captures 30 frames
python isaacgymenvs/train.py task=SimToolReal headless=True num_envs=64 \
    task.env.capture_viewer=True \
    task.env.capture_viewer_freq=50 \
    task.env.capture_viewer_len=30 \
    wandb_activate=True wandb_project=simtoolreal_test \
    train.params.config.minibatch_size=512
```

After the first capture fires you will see a `viewer/` section in the WandB run with an
interactive HTML panel showing the robot arm + hand + object + goal in 3D.

## WandB run ID change

`isaacgymenvs/utils/wandb_utils.py` was updated to append a datetime timestamp to every
WandB run ID and display name (`{experiment}_{YYYYMMDD_HHMMSS}`). This ensures each
training run creates its own WandB entry instead of resuming a previous one with the same
experiment name.

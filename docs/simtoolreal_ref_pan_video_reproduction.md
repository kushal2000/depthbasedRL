# SimToolReal Reference Pan Video Reproduction

This records the exact setup used to create the current favorite SimToolReal
pretraining/diversity video.

## Output

Video:

`local_logs/2026-06-04_22-22-23_2026-06-04_22-22-23_simtoolreal_ref_pan_best_golden_gradient_20s/rollout.mp4`

Review sheet:

`local_logs/2026-06-04_22-22-23_2026-06-04_22-22-23_simtoolreal_ref_pan_best_golden_gradient_20s/contact_sheet_start_mid_end.png`

## Code

Branch when generated:

`2026-06-03_Tyler_SimVideos`

Commit when generated:

`770b1b8f0cb2cea2d48f55daf30928318d2e45b8`

The same commit is the parent of branch:

`2026-06-03_Tyler_SimVideosFinetune`

Main scripts:

- `bash_scripts/95_render_simtoolreal_ref_pan_cinematic.sh`
- `isaacsimenvs/render_simtoolreal_pretrained.py`

## Exact Command

```bash
OUT_DIR="local_logs/$(date +%F_%H-%M-%S)_simtoolreal_ref_pan_best_golden_gradient_20s" \
HEADLESS=1 \
NUM_ENVS=100 \
GRID_COLS=10 \
ENV_SPACING_X=0.8 \
ENV_SPACING_Y=2.45 \
STEPS=1200 \
CAPTURE_PNG_STEPS=0,300,600,900,1200 \
MAKE_VIDEO=1 \
WIDTH=1920 \
HEIGHT=1080 \
QUALITY=high \
RENDER_MODE=rt \
RENDER_SPP=64 \
SKY_STYLE=blue_dome \
BACKDROP_STYLE=gradient_sky \
LIGHTING_STYLE=single_sun \
bash_scripts/95_render_simtoolreal_ref_pan_cinematic.sh
```

## Important Defaults From The Wrapper

- Checkpoint:
  `/juno/u/kedia/depthbasedRL/train_dir/TrainingObjective/Play2Win/model.pth`
- Robot URDF:
  `/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted_pretty.urdf`
- Object distribution:
  `mixed_training_simple_25_25_50`
- `num_assets_per_type=8`
- Goals hidden.
- Camera motion:
  `sapg_ref_pan`
- Camera start:
  target `(0.0, 0.0, 0.63)`, eye offset `(-0.75, -0.85, 0.55)`
- Camera end:
  target grid scale `(-0.5, -0.5, 0.0)`, eye grid scale `(-0.8, -1.2, 0.5)`
- Floor:
  `nvidia_precast_concrete_dark_gray`
- Table:
  matte display color `(0.50, 0.32, 0.18)`
- Sky:
  `blue_dome`, color `(0.50, 0.66, 0.86)`, dome intensity `1200`
- Backdrop:
  `gradient_sky`, top color `(0.25, 0.48, 0.76)`, horizon color `(0.68, 0.78, 0.88)`
- Lighting:
  `single_sun`, default light intensity `360`, sun exposure `9.35`, sun angle `0.12`,
  color temperature `5250K`, yaw offset `70 deg`
- Image postprocess:
  object saturation `1.6`, object value scale `0.82`

## Notes For Generating Variants

For a batch of candidate videos, vary only:

- `SEED`
- `OUT_DIR`
- optionally `CHECKPOINT`

Keep the camera, render palette, and object distribution fixed unless the goal is
an explicit visual ablation. This makes it easy to compare candidate rollouts
and choose the best-looking one without changing the visual language.

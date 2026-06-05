# SimToolReal Pretraining Video, 2026-06-05

Branch: `2026-06-05_Tyler_SimVideos`

Goal: regenerate the pretrained Play2Win / 1000-object cinematic video with the
same camera/rendering setup as the previous pretraining video, but make the
rollout cleaner for presentation.

## Entry Point

```bash
MAKE_VIDEO=1 \
STEPS=1200 \
CAPTURE_PNG_STEPS=0,600,1200 \
bash_scripts/95_render_simtoolreal_ref_pan_cinematic.sh
```

The script writes to a timestamped folder under `local_logs/` unless `OUT_DIR`
is set explicitly.

## Video-Only Rollout Settings

The cinematic script now defaults to:

- `OBJECT_DISTRIBUTION_MODE=mixed_training_easy_video_25_25_50`
- `RESET_POSITION_NOISE_M="0 0 0"`
- `RESET_ORIENTATION_MODE=identity`
- `TABLE_RESET_Z_RANGE_M=0`
- `RESET_DOF_POS_NOISE_ARM=0`
- `RESET_DOF_POS_NOISE_FINGERS=0`
- `RESET_DOF_VEL_NOISE=0`
- `FORCE_SCALE=0`
- `TORQUE_SCALE=0`

This means the object starts at the default table-relative pose, the robot
starts at its default joint state with zero joint velocity, and no random
external wrench perturbations are applied.

## Easy Video Object Distribution

`mixed_training_easy_video_25_25_50` is only intended for presentation videos.
It keeps the same qualitative split as the previous mixed distribution:

- 50% training-distribution handle-head objects
- 25% handle-only boxes sampled from training handle size ranges
- 25% handle-only cylinders sampled from training handle size ranges

Changes for cleaner videos:

- Excludes `screwdriver` and `spatula`, which are the thinnest original
  categories and are likely to produce the worst rollouts.
- Keeps hammer/marker/eraser/brush geometry, but clamps all head densities to a
  low handle-like range, so eccentric heavy heads do not dominate failures.

Expected easier cases: eraser, marker, simple boxes, simple cylinders, and
brushes with low-density heads.

Expected harder original cases: screwdriver/spatula because of thin geometry,
and high-density hammer heads because of eccentric mass and inertia.

## Rendering Defaults

The script preserves the previous cinematic composition:

- `NUM_ENVS=100`
- `GRID_COLS=10`
- `ENV_SPACING_X=0.8`
- `ENV_SPACING_Y=2.45`
- `camera_motion=sapg_ref_pan`
- Pretty robot URDF from
  `/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted_pretty.urdf`
- `render_quality_preset=beauty`
- `render_mode=rt`
- `render_samples_per_pixel=64`
- Synthetic `gradient_sky` backdrop, which was more reliable in the camera
  sensor than Isaac Sim HDRI/dynamic sky backgrounds in this setup.

The run manifest records the reset and wrench settings so the generated video
can be audited later.

# FurnitureBench 200 mm Finetuned Video Reproduction

This records the first working fixed-camera FurnitureBench screwing video setup.

## Checkpoint And Task

- Checkpoint:
  `/juno/u/kedia/depthbasedRL/train_dir/May26/screwing_newer/model.pth`
- Task:
  `Isaacsimenvs-PegInHole-Direct-v0`
- Problem:
  `furniture_bench.one_leg_leg4_200mm_matchedmass_sdf_hybrid_super_dense`
- Goal mode:
  `preInsertAndFinal`
- Video setting:
  `random_goal_fraction=0.0` so the single-env rollout is always the screwing
  task, not a random-goal episode.

## Code

Branch:

`2026-06-03_Tyler_SimVideosFinetune`

Main scripts:

- `bash_scripts/96_render_furniturebench_200mm_finetuned.sh`
- `isaacsimenvs/render_peg_in_hole_finetuned.py`

## Quantitative Smoke Test

This verifies the checkpoint/env wiring before rendering:

```bash
NO_RENDER=1 \
NUM_ENVS=1 \
STEPS=1200 \
SEED=0 \
RANDOM_GOAL_FRACTION=0.0 \
TRAIN_DR=1 \
OUT_DIR=local_logs/furniturebench_200mm_quant_smoke_seed0 \
bash_scripts/96_render_furniturebench_200mm_finetuned.sh
```

Observed result on 2026-06-04:

- First episode completed all `10/10` goals at step `460`.
- Second episode reset at step `1060` with `0/10`, so use a shorter video
  horizon if the goal is a clean success clip.

## First Working 1080p Success Clip

```bash
OUT_DIR=local_logs/furniturebench_200mm_highqual_success_seed0 \
NUM_ENVS=1 \
STEPS=430 \
SEED=0 \
RANDOM_GOAL_FRACTION=0.0 \
TRAIN_DR=1 \
CAPTURE_PNG_STEPS=0,240,420,430 \
WIDTH=1920 \
HEIGHT=1080 \
MAKE_VIDEO=1 \
VIDEO_FPS=30 \
bash_scripts/96_render_furniturebench_200mm_finetuned.sh
```

Output:

`local_logs/2026-06-04_22-48-53_furniturebench_200mm_highqual_success_seed0/rollout.mp4`

Review sheet:

`local_logs/2026-06-04_22-48-53_furniturebench_200mm_highqual_success_seed0/contact_sheet_steps.png`

Manifest summary:

- Ended before reset.
- `current_successes=10/10`.

## Current Visual Defaults

The current wrapper defaults intentionally match the SimToolReal Play pretraining
render settings where possible, while using a fixed close camera for the
FurnitureBench task.

- Camera xyz:
  `(0.0, -0.4115866854641337, 0.7392877590177354)`
- Camera forward world:
  `(0.0, 0.9644869986422913, -0.26413032663816677)`
- Camera target distance:
  `1.4 m`
- Camera focal length:
  `10 cm`
- Robot URDF:
  `/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted_pretty.urdf`
- Default light intensity:
  `360`
- Sky dome intensity:
  `1200`
- Sun exposure:
  `9.35`
- Sun angle:
  `0.12`
- Image exposure:
  `0.0`
- Image contrast:
  `1.0`
- Image saturation:
  `1.0`
- Backdrop style:
  `gradient_sky`
- Backdrop color:
  `(0.25, 0.48, 0.76)`
- Backdrop horizon color:
  `(0.68, 0.78, 0.88)`
- Table color:
  `(0.42, 0.27, 0.15)`
- Floor color:
  `(0.44, 0.45, 0.43)`

The earlier close-up version used `BACKDROP_STYLE=blue_walls`, which looked
flatter and more blue than the Play pretraining video. The current default uses
the same gradient-sky colors as the Play pretraining script. Because the
FurnitureBench camera is much closer and lower, it can still expose the
backdrop wall geometry at the left edge; use `BACKDROP_STYLE=blue_walls` only if
the clean flat background is preferred over matching the Play sky palette.

## Notes For Generating Variants

For multiple candidate clips, vary `SEED`, `OUT_DIR`, and possibly `STEPS`.
The seed-0 successful episode hits `10/10` by step `420` and resets at step
`460`, so a `430` step video is intentionally cleaner than a longer `600` step
clip.

## 20 Second Viser-Pose Pretty-Robot Clip

This is the requested longer fixed-camera clip using Tyler's Viser camera pose
and the visually nicer robot URDF.

```bash
OUT_DIR=local_logs/furniturebench_200mm_viser_pose_pretty_20s_seed0 \
SEED=0 \
RANDOM_GOAL_FRACTION=0.0 \
TRAIN_DR=1 \
bash_scripts/96_render_furniturebench_200mm_finetuned.sh
```

Historical wrapper defaults for this command:

- `STEPS=1200`, which is 20 seconds at the 60 Hz policy/env step.
- `MAKE_VIDEO=1`
- `WIDTH=1920`, `HEIGHT=1080`
- `RENDER_QUALITY_PRESET=beauty`
- `RENDER_MODE=rt`
- `RENDER_SPP=64`
- Camera xyz:
  `(0.017118738815094965, -0.4115866854641337, 0.7392877590177354)`
- Camera wxyz:
  `(0.6051540840361335, -0.7945251569598455, -0.014533776794422243, 0.04803206079704674)`
- Viser forward-axis interpretation:
  camera-local `+Z`, which maps to roughly world `+Y` for this pose.
- Camera focal length:
  `10 cm`, widened from the default `24 cm` so the close Viser pose frames the
  task instead of clipping into it.
- Robot URDF:
  `/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted_pretty.urdf`

Output:

`local_logs/2026-06-04_23-03-58_furniturebench_200mm_viser_pose_pretty_20s_seed0/rollout.mp4`

Review sheet:

`local_logs/2026-06-04_23-03-58_furniturebench_200mm_viser_pose_pretty_20s_seed0/contact_sheet_steps.png`

Observed result:

- Episode 1 completed all `10/10` goals at step `460`.
- Episode 2 reset at step `1060` with `0/10`.

For polished success-only clips, prefer either the 430-step command above or
run a seed sweep and select a 20-second rollout with cleaner later episodes.

## 20 Second Axis-Aligned Fixed-Fixture Clip

This is the current preferred FurnitureBench fixed-camera clip. The camera is
centered at `x=0`, looks straight along world `+Y`, keeps the same downward pitch
as the Viser pose, fixes the fixture pose, and uses only tiny object position
noise. This avoids the visually bad initial object/fixture collisions while
keeping the task recognizable.

```bash
OUT_DIR=local_logs/furniturebench_200mm_axis_camera_fixed_reset_play_sky_20s_seed0 \
STEPS=1200 \
CAPTURE_PNG_STEPS=0,300,600,900,1200 \
MAKE_VIDEO=1 \
SEED=0 \
RANDOM_GOAL_FRACTION=0.0 \
TRAIN_DR=1 \
bash_scripts/96_render_furniturebench_200mm_finetuned.sh
```

Output:

`local_logs/2026-06-04_23-29-59_furniturebench_200mm_axis_camera_fixed_reset_play_sky_20s_seed0/rollout.mp4`

Observed result:

- Episode 1 completed all `10/10` goals at step `608`.
- Episode 2 reached `5/10` by step `1200`.

Key reset overrides:

- Hole/fixture pose:
  `x=0`, `y=-0.08`, yaw `0 deg`
- Object reset center:
  `(x=0, y=0.07)`
- Object reset position noise:
  `(0.015, 0.015, 0.005) m`
- Object reset orientation:
  `identity`

## 20 Second Left-Fixture Right-Leg Clip

This is the updated preferred clip for presentation composition. The fixture is
biased to camera-left/world `-X`, and the leg starts camera-right/world `+X`.
Both still have small randomized offsets, and the leg has yaw-only orientation
noise so it does not look like a single hard-coded pose.

```bash
OUT_DIR=local_logs/furniturebench_200mm_left_fixture_right_leg_yaw_20s_seed0 \
STEPS=1200 \
CAPTURE_PNG_STEPS=0,300,600,900,1200 \
MAKE_VIDEO=1 \
SEED=0 \
RANDOM_GOAL_FRACTION=0.0 \
TRAIN_DR=1 \
bash_scripts/96_render_furniturebench_200mm_finetuned.sh
```

Output:

`local_logs/2026-06-04_23-39-38_furniturebench_200mm_left_fixture_right_leg_yaw_20s_seed0/rollout.mp4`

Observed result:

- Episode 1 completed all `10/10` goals at step `593`.
- Episode 2 completed all `10/10` goals at step `1157`.

Key reset overrides:

- Hole/fixture `x` range:
  `[-0.085, -0.055] m`
- Hole/fixture `y` range:
  `[-0.075, -0.055] m`
- Hole/fixture yaw range:
  `±3 deg`
- Object reset center:
  `(x=0.075, y=0.07)`
- Object reset position noise:
  `(0.025, 0.015, 0.005) m`
- Object reset orientation:
  yaw-only, `±20 deg`

## 2026-06-09 Lighting-Refine Clip

This branch ports the FurnitureBench 200 mm finetuned video renderer onto the
latest SimToolReal lighting-refine visual stack selected from:

`local_logs/2026-06-06_03-01-41_simtoolreal_ref_pan_refined_crisper_objectpop_20s/rollout.mp4`

Branch:

`2026-06-09_Tyler_SimVideos_LightingRefine_Finetune`

Command:

```bash
OUT_DIR=local_logs/furniturebench_200mm_lighting_refine_latest_20s_seed0 \
SEED=0 \
STEPS=1200 \
CAPTURE_PNG_STEPS=0,300,600,900,1200 \
MAKE_VIDEO=1 \
RANDOM_GOAL_FRACTION=0.0 \
TRAIN_DR=1 \
bash_scripts/96_render_furniturebench_200mm_finetuned.sh
```

Output:

`local_logs/2026-06-09_13-15-24_furniturebench_200mm_lighting_refine_latest_20s_seed0/rollout.mp4`

Review strip:

`local_logs/2026-06-09_13-15-24_furniturebench_200mm_lighting_refine_latest_20s_seed0/review_strip.png`

Observed result:

- Episode 1 completed all `10/10` goals at step `556`.
- Episode 2 reached `3/10` by step `1200`.

Visual stack:

- Floor:
  `soft_concrete_pbr_tiles`
- Table color:
  `(0.40, 0.30, 0.22)`
- Backdrop:
  32-band `gradient_sky`
- Sky:
  `blue_dome`, intensity `1260`
- Sun:
  exposure `9.62`, angle `0.24`, color temperature `5150 K`,
  color `(1.0, 0.965, 0.87)`, elevation `48 deg`, yaw offset `105 deg`
- Image grading:
  exposure `-0.20`, contrast `1.08`, saturation `1.06`, gamma `0.98`

## 2026-06-09 Safer Left/Right Spawn Revision

The first lighting-refine clip still allowed the 200 mm leg and fixture to
start too close in the worst case. The default wrapper now keeps the fixture
clearly camera-left/world `-X` and the leg camera-right/world `+X`:

- Hole/fixture `x` range:
  `[-0.130, -0.110] m`
- Hole/fixture `y` range:
  `[-0.075, -0.055] m`
- Hole/fixture yaw range:
  `±3 deg`
- Object reset center:
  `(x=0.160, y=0.070)`
- Object reset position noise:
  `(0.010, 0.015, 0.005) m`
- Object reset orientation:
  yaw-only, `±12 deg`

This preserves visible reset variation while avoiding the ugly initial
fixture/leg collisions.

Command:

```bash
OUT_DIR=local_logs/furniturebench_200mm_lighting_refine_more_separated_spawn_20s_seed0 \
SEED=0 \
STEPS=1200 \
CAPTURE_PNG_STEPS=0,300,600,900,1200 \
MAKE_VIDEO=1 \
RANDOM_GOAL_FRACTION=0.0 \
TRAIN_DR=1 \
bash_scripts/96_render_furniturebench_200mm_finetuned.sh
```

Output:

`local_logs/2026-06-09_13-26-23_furniturebench_200mm_lighting_refine_more_separated_spawn_20s_seed0/rollout.mp4`

Review strip:

`local_logs/2026-06-09_13-26-23_furniturebench_200mm_lighting_refine_more_separated_spawn_20s_seed0/review_strip.png`

Observed result:

- Episode 1 completed all `10/10` goals at step `961`.
- Episode 2 was at `0/10` by step `1200`.

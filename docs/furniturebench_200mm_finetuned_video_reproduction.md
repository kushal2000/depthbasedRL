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

## 2026-06-09 Reset-Center Fix And Preferred Lighting-Refine Clip

The earlier wrapper exposed `RESET_POSITION_CENTER_X/Y`, but
`_reset_object_pose()` was not actually adding these center offsets. As of this
revision, `ResetCfg` has explicit `reset_position_center_x/y` fields and reset
sampling uses:

`object_xy = reset_position_center_xy + uniform_noise * reset_position_noise_xy`

The default values for normal training remain `0.0`, so this is backwards
compatible for existing training configs. For the FurnitureBench video, this
makes the wrapper's intended left-fixture/right-leg composition real.

The current preferred setup keeps the old good fixture pose range from
`2026-06-03_Tyler_SimVideosFinetune`, places the leg camera-right/world `+X`,
and uses a fixed camera-facing gradient backdrop so the left edge no longer
shows the diagonal wall/cutoff artifact.

- Hole/fixture `x` range:
  `[-0.085, -0.055] m`
- Hole/fixture `y` range:
  `[-0.075, -0.055] m`
- Hole/fixture yaw range:
  `±3 deg`
- Object reset center:
  `(x=0.180, y=0.070)`
- Object reset position noise:
  `(0.010, 0.015, 0.005) m`
- Object reset orientation:
  yaw-only, `±8 deg`
- Backdrop:
  `fixed_gradient_sky`, a 32-band +Y-facing wall behind the table

This preserves visible reset variation while avoiding the initial fixture/leg
collision and the gray left-side backdrop artifact.

The near fixed wall fixed the gray wedge but made the lighting look too muted:
the wall sat close to the table (`BACKDROP_Y=2.6`) and visually dominated the
horizon. The v2 setup keeps the same fixed camera-facing gradient, but pushes it
far behind the task (`BACKDROP_Y=50`) so the floor/sky/light read much closer to
the SimToolReal pretraining reference while still avoiding the diagonal
`gradient_sky` wedge.

Current v2 command:

```bash
OUT_DIR=local_logs/furniturebench_v2_lighting_refine_far_fixed_backdrop_20s_seed0 \
SEED=0 \
STEPS=1200 \
CAPTURE_PNG_STEPS=0,300,600,900,1200 \
MAKE_VIDEO=1 \
RANDOM_GOAL_FRACTION=0.0 \
TRAIN_DR=1 \
bash_scripts/96_render_furniturebench_200mm_finetuned.sh
```

Output:

`local_logs/2026-06-09_15-05-08_furniturebench_v2_lighting_refine_far_fixed_backdrop_20s_seed0/rollout.mp4`

Review strip:

`local_logs/2026-06-09_15-05-08_furniturebench_v2_lighting_refine_far_fixed_backdrop_20s_seed0/review_strip.png`

Comparison against the best pretraining reference:

- `local_logs/2026-06-09_furniturebench_v2_render_compare/v2_vs_pretrain_t5.png`
- `local_logs/2026-06-09_furniturebench_v2_render_compare/v2_vs_pretrain_t10.png`

Observed result:

- Episode 1 completed all `10/10` goals and reset at step `663`.
- Episode 2 reached `9/10` by step `1200`.

Current wrapper defaults:

- `BACKDROP_STYLE=fixed_gradient_sky`
- `BACKDROP_WIDTH=180`
- `BACKDROP_Y=50`
- `BACKDROP_COLOR=(0.24, 0.46, 0.75)`
- `BACKDROP_HORIZON_COLOR=(0.60, 0.72, 0.84)`
- `DEFAULT_LIGHT_INTENSITY=390`
- `SKY_DOME_INTENSITY=1260`
- `SINGLE_SUN_EXPOSURE=9.62`
- `SINGLE_SUN_ANGLE=0.24`
- `SINGLE_SUN_COLOR_TEMPERATURE=5150`
- `SINGLE_SUN_ELEVATION_DEG=48`
- `SINGLE_SUN_YAW_OFFSET_DEG=105`
- `IMAGE_EXPOSURE=-0.20`
- `IMAGE_CONTRAST=1.08`
- `IMAGE_SATURATION=1.06`
- `IMAGE_GAMMA=0.98`

The exact SimToolReal panning backdrop (`gradient_sky`) is still available via
`BACKDROP_STYLE=gradient_sky`, but for the close straight-on FurnitureBench
camera it can reveal a gray diagonal wall/cutoff at the image edge. The fixed
camera-facing gradient is the current safer default for this script. In v2 it is
placed far from the task to avoid the muted near-wall look.

If the fixed backdrop ever needs to be tested with multiple envs, use
`CAMERA_ENV_ID` with `ENV_SPACING_X/Y` and `GRID_COLS`. The camera and fixed
backdrop both use the selected env origin, so the camera pose remains identical
relative to that env's robot/table.

## 2026-06-10 Nine-Seed V2 Review Batch

This batch keeps the v2 visual stack fixed and varies only `SEED=0..8`. With
the current reset config, the seed changes the small randomized fixture and leg
initial poses while preserving the same camera, lighting, policy, task, and
20-second horizon.

Branch:

`2026-06-09_Tyler_SimVideos_LightingRefine_Finetune_v2`

Base command:

```bash
RUN_ROOT="local_logs/$(date +%Y-%m-%d_%H-%M-%S)_furniturebench_v2_9_seed_rollouts"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" | tee local_logs/latest_furniturebench_v2_9_seed_rollouts.txt

for seed in $(seq 0 8); do
  seed_dir="$RUN_ROOT/seed_$(printf '%02d' "$seed")"
  echo "[furniturebench_v2_batch] seed=${seed} out=${seed_dir}" | tee -a "$RUN_ROOT/batch.log"
  OUT_DIR="$seed_dir" \
  EXTRA_ARGS="--no_timestamp_out_dir" \
  SEED="$seed" \
  STEPS=1200 \
  CAPTURE_PNG_STEPS=0,300,600,900,1200 \
  MAKE_VIDEO=1 \
  RANDOM_GOAL_FRACTION=0.0 \
  TRAIN_DR=1 \
  bash_scripts/96_render_furniturebench_200mm_finetuned.sh 2>&1 | tee "$seed_dir.render.log"
done
```

Actual output root:

`local_logs/2026-06-10_02-36-59_furniturebench_v2_9_seed_rollouts`

Per-seed videos:

- `seed_00/rollout.mp4`
- `seed_01/rollout.mp4`
- `seed_02/rollout.mp4`
- `seed_03/rollout.mp4`
- `seed_04/rollout.mp4`
- `seed_05/rollout.mp4`
- `seed_06/rollout.mp4`
- `seed_07/rollout.mp4`
- `seed_08/rollout.mp4`

Review outputs:

- `local_logs/2026-06-10_02-36-59_furniturebench_v2_9_seed_rollouts/furniturebench_v2_9_seed_grid_3x3.mp4`
- `local_logs/2026-06-10_02-36-59_furniturebench_v2_9_seed_rollouts/furniturebench_v2_9_seed_grid_3x3_t0.png`
- `local_logs/2026-06-10_02-36-59_furniturebench_v2_9_seed_rollouts/furniturebench_v2_9_seed_grid_3x3_t8.png`

Observed result:

- All 9 seeds completed one full `10/10` episode at step `663`.
- All 9 seeds reached `9/10` in the second episode by step `1200`.
- First-frame hashes differ across seeds, confirming the seed changes the
  reset initialization while leaving the fixed render setup intact.

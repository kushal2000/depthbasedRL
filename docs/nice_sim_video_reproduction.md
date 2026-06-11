# Nice Sim Video Reproduction

This is the short reproduction index for the current nice SimToolReal
pretraining video and FurnitureBench 200 mm finetuned video.

Branch used when this was documented:

`2026-06-09_Tyler_SimVideos_LightingRefine_Finetune_v2`

The pretty KUKA/Sharpa robot is vendored in this repo:

`assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted_pretty.urdf`

The mesh dependencies are also vendored:

- `assets/urdf/kuka_allegro_description/`
- `assets/urdf/kuka_allegro_sharpa_merged/`

## Pretraining Object-Diversity Video

This is the dense 100-env panning video for the Play2Win / 1000-object
pretrained policy. It uses the final refined lighting/floor/backdrop settings.

Quick command:

```bash
bash_scripts/105_render_simtoolreal_ref_pan_refined_crisper_objectpop.sh
```

Equivalent explicit command:

```bash
OUT_DIR=local_logs/$(date +%F_%H-%M-%S)_simtoolreal_ref_pan_refined_crisper_objectpop_20s \
  MAKE_VIDEO=1 STEPS=1200 CAPTURE_PNG_STEPS=0,300,600,900,1200 \
  BACKDROP_GRADIENT_BANDS=32 \
  FLOOR_STYLE=soft_concrete_pbr_tiles \
  FLOOR_COLOR_R=0.56 FLOOR_COLOR_G=0.56 FLOOR_COLOR_B=0.53 \
  FLOOR_TILE_COUNT=1 FLOOR_TILE_SIZE=120 FLOOR_TILE_GAP=0 \
  FLOOR_TEXTURE_SCALE=8.0 FLOOR_ROUGHNESS=0.92 FLOOR_NORMAL_STRENGTH=0.05 FLOOR_SPECULAR_LEVEL=0.04 \
  TABLE_COLOR_R=0.40 TABLE_COLOR_G=0.30 TABLE_COLOR_B=0.22 \
  BACKDROP_COLOR_R=0.24 BACKDROP_COLOR_G=0.46 BACKDROP_COLOR_B=0.75 \
  BACKDROP_HORIZON_COLOR_R=0.60 BACKDROP_HORIZON_COLOR_G=0.72 BACKDROP_HORIZON_COLOR_B=0.84 \
  OBJECT_COLOR_SATURATION=1.75 \
  OBJECT_COLOR_VALUE_SCALE=0.88 \
  SINGLE_SUN_ELEVATION_DEG=48 \
  SINGLE_SUN_EXPOSURE=9.62 \
  SINGLE_SUN_ANGLE=0.24 \
  SINGLE_SUN_COLOR_TEMPERATURE=5150 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.965 SINGLE_SUN_COLOR_B=0.87 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=390 \
  SKY_DOME_INTENSITY=1260 \
  IMAGE_EXPOSURE=-0.20 \
  IMAGE_CONTRAST=1.08 \
  IMAGE_SATURATION=1.06 \
  IMAGE_GAMMA=0.98 \
  bash_scripts/98_render_simtoolreal_ref_pan_cinematic_greige_floor.sh
```

Reference output from the selected candidate:

`local_logs/2026-06-06_03-01-41_simtoolreal_ref_pan_refined_crisper_objectpop_20s/rollout.mp4`

Detailed notes:

`docs/simtoolreal_cinematic_rendering.md`

## FurnitureBench 200 mm Finetuned Video

This is the fixed-camera video for the May 26 screwing checkpoint:

`/juno/u/kedia/depthbasedRL/train_dir/May26/screwing_newer/model.pth`

The current wrapper defaults reproduce the v2 setup: repo-local pretty robot,
fixed far gradient backdrop, soft concrete floor, refined sun/sky lighting, and
left-fixture/right-leg reset composition.

Single seed command:

```bash
OUT_DIR=local_logs/$(date +%F_%H-%M-%S)_furniturebench_v2_lighting_refine_20s_seed0 \
  SEED=0 \
  STEPS=1200 \
  CAPTURE_PNG_STEPS=0,300,600,900,1200 \
  MAKE_VIDEO=1 \
  RANDOM_GOAL_FRACTION=0.0 \
  TRAIN_DR=1 \
  bash_scripts/96_render_furniturebench_200mm_finetuned.sh
```

Nine-seed review batch:

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

Reference v2 seed-0 output:

`local_logs/2026-06-09_15-05-08_furniturebench_v2_lighting_refine_far_fixed_backdrop_20s_seed0/rollout.mp4`

Reference nine-seed review output:

`local_logs/2026-06-10_02-36-59_furniturebench_v2_9_seed_rollouts/furniturebench_v2_9_seed_grid_3x3.mp4`

Observed nine-seed behavior:

- All 9 seeds completed one full `10/10` episode at step `663`.
- All 9 seeds reached `9/10` in the second episode by step `1200`.

Detailed notes:

`docs/furniturebench_200mm_finetuned_video_reproduction.md`

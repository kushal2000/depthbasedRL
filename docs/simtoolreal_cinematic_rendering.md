# SimToolReal Cinematic Rendering Notes

Branch: `2026-06-03_Tyler_SimVideos`

## Current Locked Choices

- Use the reference SAPG pan trajectory from `local_logs/2026-06-04_17-21-59_simtoolreal_cinematic_warm_gray_high_quality_less_light_20s`.
- Keep `num_envs=100`, `grid_cols=10`, `env_spacing_xy=0.8 2.45`, and `camera_motion=sapg_ref_pan`.
- Use the pretty visual URDF:
  `/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted_pretty.urdf`
- Keep goals hidden and use the mixed object distribution: 50% normal training objects, 25% handle-only box-like objects, 25% handle-only cylinder-like objects.

## What Worked

- The reference camera trajectory works well. It starts with a close foreground robot/object interaction and zooms out to show dense object diversity.
- The pretty URDF is a clear improvement over the orange default robot for this video.
- RT mode with `render_quality_preset=beauty` and `render_samples_per_pixel=64` is fast enough for 1080p 20s videos and looks clean.
- The no-extra-light version avoids the heavy orange/sunset look and keeps the robot visually clean.

## What Did Not Work

- White floor plus white background has weak contrast. It makes the scene look clean but washed out.
- A hard single-sun setup with the default world light dimmed to near zero produced stronger shadows but made the horizon/background black and too dramatic.
- Neutral/white tables blended into the floor and reduced object/robot readability.
- Blue wall/backdrop variants looked artificial when the wall was too saturated or too visible.

## Current Art Direction

The target should be closer to a bright outdoor architectural render, not a saturated sunset:

- Floor: medium-light warm gray concrete or stone, not pure white. This should contrast with the white/gray robot while still feeling bright.
- Background: pale blue-gray sky gradient, not white. The horizon should be very light and the top slightly cooler/bluer.
- Table: warm beige/brown. It should be darker than the floor but not glossy or saturated.
- Lighting: keep ambient fill, then add a warm daylight/golden sun for readable shadows. The best current direction is warmer than neutral daylight, but not full orange sunset.
- Shadows: visible and directional, but not so dark that they dominate the foreground.

## Current Palette Decision

The white floor and white background are the main reason the current render lacks contrast. The robot has a white/gray body, so a pure white floor makes the robot, object colors, and shadows read as low-contrast. A dark floor has the opposite problem: it looks like a debug viewport and makes the scene feel less polished. The best compromise is a medium-light gray concrete/stone floor.

Current script defaults in `bash_scripts/95_render_simtoolreal_ref_pan_cinematic.sh` after the latest contrast pass:

- Floor: `nvidia_precast_concrete_dark_gray`. Visually this reads as medium gray in the camera render, not charcoal.
- Background: `gradient_sky` with a blue-gray horizon `(0.68, 0.78, 0.88)` and blue top color `(0.25, 0.48, 0.76)`.
- Backdrop extent margin: `80.0`. The wide SAPG pan otherwise sees the edge of a smaller diagonal sky panel.
- Dome sky/fill: muted blue dome `(0.50, 0.66, 0.86)`.
- Table: matte warm brown `(0.50, 0.32, 0.18)`.
- Light: `single_sun` with default world/dome fill kept at `360`, sun exposure `9.35`, sun angle `0.12`, color temperature `5250K`, and yaw offset `70 deg`.

This is intentionally more golden/sunlit than the flat no-extra-light render. The sun is low enough to make shadows visible, but there is still enough ambient fill that the background does not collapse to black. The single-sun color in `render_simtoolreal_pretrained.py` is neutral-warm `(1.0, 0.97, 0.90)` instead of orange `(1.0, 0.92, 0.78)`. The backdrop horizon is configurable because a near-white horizon made the rendered camera image look like white floor against white background.

Latest reviewed frame:

`local_logs/2026-06-04_21-44-25_2026-06-04_21-44-25_refpan_palette_bright_golden_sun_shadow_probe/step_0600.png`

Assessment:

- This is the best sunlight/shadow setting so far. It is warmer and more directional than the previous high-contrast default.
- It is still a synthetic gradient backdrop, not a true HDRI/cloud sky. That is acceptable for now because IsaacLab camera captures did not reliably show the dome/HDRI sky like the viewport.
- The floor now has enough contrast with the white robot and colored objects.
- Directional shadows are visible without making the whole render too dark.

Comparison sheet for the current lighting/sky probes:

`local_logs/2026-06-04_render_sky_sun_probe_review/sky_sun_probe_comparison_with_bright_step0600.png`

## Proper Sky / HDRI Probe Results

The "proper sky" paths were re-tested with the synthetic backdrop disabled and extra script lighting disabled:

- `SKY_STYLE=dynamic_clear_sky BACKDROP_STYLE=none LIGHTING_STYLE=none`: dynamic clear sky imports, but the camera image is badly overexposed and reads as a white background.
- `SKY_STYLE=dynamic_clear_sky BACKDROP_STYLE=none LIGHTING_STYLE=none DEFAULT_LIGHT_INTENSITY=0`: removing the default light avoids the overexposure, but the saved camera frame background becomes black.
- `SKY_STYLE=hdri SKY_HDRI_PRESET=cloudy_vondelpark BACKDROP_STYLE=none LIGHTING_STYLE=none`: HDRI at normal dome intensity is also overexposed/white.
- `SKY_STYLE=hdri SKY_HDRI_PRESET=cloudy_vondelpark SKY_DOME_INTENSITY=250 BACKDROP_STYLE=none LIGHTING_STYLE=none DEFAULT_LIGHT_INTENSITY=0`: lowering the HDRI intensity avoids overexposure, but the camera frame background becomes black.
- `SKY_STYLE=hdri SKY_HDRI_PATH=/home/tylerlum/github_repos/RoboLab/assets/backgrounds/default/empty_warehouse.hdr DOME_LIGHT_UPPER_LOWER_STRATEGY=0 BACKDROP_STYLE=none LIGHTING_STYLE=none DEFAULT_LIGHT_INTENSITY=0 CAPTURE_SOURCE=camera_sensor`: real `.hdr` texture plus full IBL strategy still gives black background in the IsaacLab camera sensor output.
- The viewport/path-tracing capture path was also tried with the same `.hdr` setup, but the Kit capture extension hung in this headless script. Do not rely on `CAPTURE_SOURCE=viewport` until that path is debugged separately.

Conclusion: for this IsaacLab camera sensor capture path, a synthetic gradient backdrop is currently more reliable than the dynamic/HDRI sky. The dome/HDRI can illuminate the scene, but it is not reliably visible as the camera background in saved RGB sensor frames. The proper sky may still work in an interactive viewport/path-traced capture, but it is not reliable in the saved camera frames used by this script.

Relevant render controls now exposed by `bash_scripts/95_render_simtoolreal_ref_pan_cinematic.sh`:

- `CAPTURE_SOURCE=camera_sensor|viewport`
- `DOME_LIGHT_UPPER_LOWER_STRATEGY=0|3|4`
- `SKY_HDRI_PATH=/path/to/file.hdr`
- `SKY_DOME_INTENSITY=...`

## Current Best Output

Latest high-contrast daylight candidate:

`local_logs/2026-06-04_21-19-59_2026-06-04_21-19-59_simtoolreal_ref_pan_high_contrast_daylight_20s/rollout.mp4`

Quick review sheet:

`local_logs/2026-06-04_21-19-59_2026-06-04_21-19-59_simtoolreal_ref_pan_high_contrast_daylight_20s/contact_sheet_start_mid_end.png`

Main remaining issues:

- The sky is still a synthetic gradient backdrop, so it has a visible horizon/band rather than a natural HDRI/cloud sky.
- The table is a matte warm-brown material, not a true wood texture.
- The floor is much better for contrast, but less like the bright white-stone reference than the earlier white variants.
- The current default lighting has only been rendered as a frame probe so far. A full MP4 should be regenerated once the palette is accepted.

## Next Test Matrix

Keep the camera and policy fixed. Only vary visual stack:

- Candidate A: current brighter golden default, full 20s MP4.
- Candidate B: same palette, but slightly cooler/less golden if the robot/table look too yellow in motion.
- Candidate C: revisit proper HDRI only if using an interactive viewport/path-traced capture path instead of IsaacLab camera frames.

Compare frames at `0`, `600`, and `1200` before committing to a full MP4.

## 2026-06-05 Lighting/Backdrop Probe Update

Branch: `2026-06-05_Tyler_SimVideos_LightingAdjust`

New controls added:

- `--dynamic_sky_preset remote_clear|local_simple|local_sunstudy`
- `--dynamic_sky_path /path/or/url/to/sky.usd`
- `--backdrop_gradient_bands N`
- Bash pass-through for `FLOOR_COLOR_R/G/B`, `FLOOR_TEXTURE_SCALE`, `DYNAMIC_SKY_PRESET`, `DYNAMIC_SKY_PATH`, and `BACKDROP_GRADIENT_BANDS`.

Probe workflow:

```bash
ROOT_DIR=local_logs/$(date +%F_%H-%M-%S)_simtoolreal_visual_stack_probe \
  NUM_ENVS=16 GRID_COLS=4 STEPS=600 CAPTURE_PNG_STEPS=0,600 \
  WIDTH=960 HEIGHT=540 RENDER_SPP=16 \
  bash_scripts/96_probe_simtoolreal_visual_stack.sh
```

Quick probe generated:

`local_logs/2026-06-05_01-34-06_simtoolreal_visual_stack_probe_quick/`

Review sheets:

- `contact_sheet_step_0000.png`
- `contact_sheet_step_0600.png`

Findings:

- `01_smooth_gradient_current_concrete` is the safest improvement. It preserves the accepted concrete/table palette but uses 16 gradient bands, smaller tile gaps, and less repetitive floor texture scaling.
- `02_slate_blue_gray_floor_smooth_gradient` and `04_dark_desaturated_blue_floor` add contrast, but the blue-gray floor reads more artificial than the concrete floor.
- `05_dynamic_local_simple_no_backdrop` and `06_dynamic_local_sunstudy_no_backdrop` render visible local dynamic sky assets in camera-sensor mode, but they behave like a wrapped environment/floor and do not create a clean outdoor horizon.
- `07_hdri_carlight_no_backdrop_low_intensity` is too dark with default light disabled; earlier high-intensity HDRI variants overexposed to white. This reinforces that the robust final path should use the geometry gradient backdrop for now.

Selected candidate:

```bash
OUT_DIR=local_logs/$(date +%F_%H-%M-%S)_simtoolreal_ref_pan_smooth_gradient_concrete_candidate_20s \
  MAKE_VIDEO=1 STEPS=1200 CAPTURE_PNG_STEPS=0,600,1200 \
  NUM_ENVS=100 GRID_COLS=10 WIDTH=1920 HEIGHT=1080 RENDER_SPP=64 \
  BACKDROP_GRADIENT_BANDS=16 FLOOR_TILE_GAP=0.002 FLOOR_TEXTURE_SCALE=2.5 \
  bash_scripts/95_render_simtoolreal_ref_pan_cinematic.sh
```

Output:

`local_logs/2026-06-05_01-44-51_simtoolreal_ref_pan_smooth_gradient_concrete_candidate_20s/rollout.mp4`

## 2026-06-05 Floor Material Probe Update

Branch: `2026-06-05_Tyler_SimVideos_LightingAdjust`

Goal: improve the floor while keeping the accepted SAPG reference camera path, pretty URDF, gradient-sky backdrop, and single-sun lighting. The old/default floor path remains available through `bash_scripts/95_render_simtoolreal_ref_pan_cinematic.sh`; the new candidate has its own wrapper.

What was added:

- Procedural floor texture assets in `assets/textures/`.
- New floor styles in `isaacsimenvs/render_simtoolreal_pretrained.py`:
  `soft_concrete_pbr_tiles`, `matte_warm_gray_pbr_tiles`, `matte_slate_pbr_tiles`, and `matte_greige_pbr_tiles`.
- New floor controls exposed through script `95`:
  `FLOOR_NORMAL_PATH`, `FLOOR_ROUGHNESS`, `FLOOR_NORMAL_STRENGTH`, and `FLOOR_SPECULAR_LEVEL`.
- The PBR tile underlay now uses `FLOOR_COLOR_R/G/B` instead of hardcoded light gray, so dark/matte floors do not get bright white seams.
- New wrapper:
  `bash_scripts/98_render_simtoolreal_ref_pan_cinematic_greige_floor.sh`

Floor probe command:

```bash
STEPS=300 CAPTURE_PNG_STEPS=0,300 \
  bash_scripts/97_probe_simtoolreal_floor_materials.sh
```

Probe output:

`local_logs/2026-06-05_02-27-13_simtoolreal_floor_material_probe/contact_sheet_step_0300.png`

Findings:

- White/soft concrete PBR floors look clean in small probes but wash out in the full 100-env shot.
- Dark slate and cool concrete give contrast but read too artificial/debug-like.
- Warm limestone/fieldstone are too patterned or too brown for the dense robot-grid composition.
- Small PBR tiles are distracting because visible seams compete with the robots and object colors.
- The best direction is a continuous, matte, mid-tone greige concrete surface: enough contrast against the gray/white robot, no black debug-floor feeling, no visible tile grid.

Current best floor candidate:

```bash
MAKE_VIDEO=1 STEPS=300 CAPTURE_PNG_STEPS=0,150,300 \
  BACKDROP_GRADIENT_BANDS=16 \
  bash_scripts/98_render_simtoolreal_ref_pan_cinematic_greige_floor.sh
```

Equivalent explicit settings:

```bash
FLOOR_STYLE=matte_greige_pbr_tiles \
FLOOR_COLOR_R=0.50 FLOOR_COLOR_G=0.48 FLOOR_COLOR_B=0.43 \
FLOOR_TILE_COUNT=1 FLOOR_TILE_SIZE=120 FLOOR_TILE_GAP=0 \
FLOOR_TEXTURE_SCALE=12.0 \
FLOOR_ROUGHNESS=0.92 \
FLOOR_NORMAL_STRENGTH=0.05 \
FLOOR_SPECULAR_LEVEL=0.06
```

Candidate output:

`local_logs/2026-06-05_02-44-32_simtoolreal_ref_pan_matte_greige_continuous_floor_candidate_5s/rollout.mp4`

Representative stills:

- `local_logs/2026-06-05_02-42-48_simtoolreal_ref_pan_matte_greige_continuous_floor_candidate_stills/step_0150.png`
- `local_logs/2026-06-05_02-42-48_simtoolreal_ref_pan_matte_greige_continuous_floor_candidate_stills/step_0300.png`

Old floor fallback:

```bash
bash_scripts/95_render_simtoolreal_ref_pan_cinematic.sh
```

or explicitly:

```bash
FLOOR_STYLE=nvidia_precast_concrete_dark_gray \
  bash_scripts/95_render_simtoolreal_ref_pan_cinematic.sh
```

## 2026-06-05 Daylight / Golden Sun Probe Update

Branch: `2026-06-05_Tyler_SimVideos_LightingAdjust`

Reference target:

`beautiful_rendering_lighting_white_terrain.png`

Interpretation of the reference lighting:

- Primary light source: one low, warm directional "sun" from camera side/front-side, high enough to light the robots but low enough to cast long shadows.
- Secondary light source: broad cool sky/ambient fill. This prevents robot backs and undersides from going black.
- Exposure: controlled below clipping. The reference is bright, but the floor and robots keep visible gradients rather than becoming flat white.
- Shadow style: visible, directional, moderately soft. Shadows add depth but do not obscure the object colors.

New controls added:

- `--single_sun_color R G B`
- Bash pass-through: `SINGLE_SUN_COLOR_R/G/B`
- Bash pass-through: `IMAGE_EXPOSURE`

Broad lighting probe:

```bash
STEPS=300 CAPTURE_PNG_STEPS=0,150,300 \
  bash_scripts/99_probe_simtoolreal_daylight_lighting.sh
```

Output:

`local_logs/2026-06-05_02-54-12_simtoolreal_daylight_lighting_probe/`

Review sheets:

- `contact_sheet_step_0000.png`
- `contact_sheet_step_0150.png`
- `contact_sheet_step_0300.png`

Broad probe findings:

- `04_side_sun_less_front_shadow` was closest to the reference in visible sun/shadow direction, but its shadows were too graphic for the final dense-grid video.
- `01_golden_key_cool_fill` and `07_cooler_clear_daylight` were safe/readable but only modestly different from the current accepted lighting.
- `06_beauty_stack_control` remained too studio-like and flat relative to the reference.

Focused lighting probe:

```bash
STEPS=300 CAPTURE_PNG_STEPS=0,150,300 \
  bash_scripts/100_probe_simtoolreal_daylight_lighting_focused.sh
```

Output:

`local_logs/2026-06-05_03-00-53_simtoolreal_daylight_lighting_focused_probe/`

Review sheets:

- `contact_sheet_step_0000.png`
- `contact_sheet_step_0150.png`
- `contact_sheet_step_0300.png`

Focused probe findings:

- `02_golden_contrast_balanced` is the most reference-like candidate: stronger daylight contrast, visible side shadows, and clear sun direction.
- `01_side_sun_softened` is the better conservative default: similar sunlight direction, but less tan/gold color cast and less aggressive shadows.
- `05_crisp_sun_controlled_fill` has the strongest shadow definition among the focused variants, but is likely too harsh for the dense 100-env final.

Current recommended daylight candidate:

```bash
OUT_DIR=local_logs/$(date +%F_%H-%M-%S)_simtoolreal_ref_pan_side_sun_softened_20s \
  MAKE_VIDEO=1 STEPS=1200 CAPTURE_PNG_STEPS=0,600,1200 \
  SINGLE_SUN_EXPOSURE=9.65 \
  SINGLE_SUN_ANGLE=0.28 \
  SINGLE_SUN_COLOR_TEMPERATURE=5050 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.96 SINGLE_SUN_COLOR_B=0.84 \
  SINGLE_SUN_YAW_OFFSET_DEG=110 \
  DEFAULT_LIGHT_INTENSITY=520 \
  SKY_DOME_INTENSITY=1700 \
  IMAGE_EXPOSURE=-0.10 \
  bash_scripts/98_render_simtoolreal_ref_pan_cinematic_greige_floor.sh
```

Short 100-env test output:

`local_logs/2026-06-05_03-06-53_simtoolreal_ref_pan_side_sun_softened_5s/rollout.mp4`

More dramatic/reference-like alternative:

```bash
OUT_DIR=local_logs/$(date +%F_%H-%M-%S)_simtoolreal_ref_pan_golden_contrast_balanced_20s \
  MAKE_VIDEO=1 STEPS=1200 CAPTURE_PNG_STEPS=0,600,1200 \
  SINGLE_SUN_EXPOSURE=9.90 \
  SINGLE_SUN_ANGLE=0.22 \
  SINGLE_SUN_COLOR_TEMPERATURE=5000 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.95 SINGLE_SUN_COLOR_B=0.82 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=470 \
  SKY_DOME_INTENSITY=1600 \
  IMAGE_EXPOSURE=-0.16 \
  bash_scripts/98_render_simtoolreal_ref_pan_cinematic_greige_floor.sh
```

Short 100-env test output:

`local_logs/2026-06-05_03-05-19_simtoolreal_ref_pan_golden_contrast_balanced_5s/rollout.mp4`

Current lighting takeaway:

- Keep the greige continuous floor and synthetic gradient sky; those are still the reliable base.
- Use a side-biased low sun to create the reference-like daylight/shadow cue.
- Do not drive sun exposure alone; pair stronger sun with more sky/default fill and a small negative output exposure, otherwise the table/floor overexposes and the robot colors flatten.
- For final videos, prefer `01_side_sun_softened` unless the video needs a more dramatic hero-shot feel, in which case use `02_golden_contrast_balanced`.

## Shorter Shadows And Grey/White Floor

The low side-sun variants above had the right sunlit feel, but the light was too close to the horizon and created very long shadows. The script now supports an explicit sun elevation override:

- `SINGLE_SUN_ELEVATION_DEG=...`
- This controls sun height above the horizon.
- `SINGLE_SUN_ANGLE` remains the angular size/softness of the sun, not the sun height.

The greige floor wrapper also now defaults to `BACKDROP_GRADIENT_BANDS=16` so final videos do not accidentally use the coarse 4-band synthetic sky. For tighter lighting/floor comparisons, use:

```bash
bash_scripts/102_probe_simtoolreal_overhead_sun_gray_floors.sh
```

Useful probe outputs:

- `local_logs/2026-06-06_02-14-03_simtoolreal_shorter_shadow_floor_probe/contact_sheet_step_0300.png`
- `local_logs/2026-06-06_02-21-31_simtoolreal_overhead_sun_gray_floor_probe/contact_sheet_step_0300.png`

Result:

- Brown/sand-like floors blend too much with the table and make the scene read flatter.
- The best non-brown direction so far is `04_soft_concrete_elev55`: light grey/white concrete floor, explicit 55 degree sun elevation, shorter shadows, and enough shadow contrast to keep depth cues.
- The visible horizon is now mainly the synthetic floor/backdrop boundary, not the old coarse sky banding.

Current short-shadow 20s candidate:

`local_logs/2026-06-06_02-25-40_simtoolreal_ref_pan_soft_concrete_elev55_short_shadows_20s/rollout.mp4`

Comparison reference:

`local_logs/2026-06-05_video_comparisons/pretraining_10_rollouts/01_seed43.mp4`

Matched-frame contact sheet:

`local_logs/2026-06-06_video_comparisons/soft_concrete_elev55_vs_seed43_contact.png`

Comparison notes:

- The new `soft_concrete_elev55` candidate has shorter, less side-biased shadows than `01_seed43`.
- The new floor is closer to the desired white/grey concrete direction and avoids the old debug-grid look.
- `01_seed43` has stronger floor/sky contrast and the grid gives useful depth cues, but it looks more synthetic and less like a clean cinematic floor.
- The new candidate is cleaner overall, but still has two weaknesses: the table is still close to the floor/table color family, and the synthetic horizon remains visible.
- If iterating further, the highest-value changes are to make the table slightly darker/desaturated brown for stronger floor separation, tune the backdrop horizon color lower/darker, and try one more sun elevation around 45-50 degrees with the same soft-concrete floor.

Exact command:

```bash
OUT_DIR=local_logs/$(date +%F_%H-%M-%S)_simtoolreal_ref_pan_soft_concrete_elev55_short_shadows_20s \
  MAKE_VIDEO=1 STEPS=1200 CAPTURE_PNG_STEPS=0,300,600,900,1200 \
  BACKDROP_GRADIENT_BANDS=32 \
  FLOOR_STYLE=soft_concrete_pbr_tiles \
  FLOOR_COLOR_R=0.58 FLOOR_COLOR_G=0.58 FLOOR_COLOR_B=0.55 \
  FLOOR_TILE_COUNT=1 FLOOR_TILE_SIZE=120 FLOOR_TILE_GAP=0 \
  FLOOR_TEXTURE_SCALE=8.0 FLOOR_ROUGHNESS=0.92 FLOOR_NORMAL_STRENGTH=0.05 FLOOR_SPECULAR_LEVEL=0.04 \
  SINGLE_SUN_ELEVATION_DEG=55 \
  SINGLE_SUN_EXPOSURE=9.35 \
  SINGLE_SUN_ANGLE=0.34 \
  SINGLE_SUN_COLOR_TEMPERATURE=5250 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.97 SINGLE_SUN_COLOR_B=0.90 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=440 \
  SKY_DOME_INTENSITY=1400 \
  IMAGE_EXPOSURE=-0.18 \
  bash_scripts/98_render_simtoolreal_ref_pan_cinematic_greige_floor.sh
```

## 2026-06-06 Lighting Polish Probe

Branch: `2026-06-06_Tyler_SimVideos_LightingPolish`

Goal: preserve the accepted soft-concrete/short-shadow look, then make small improvements to table/floor separation, sky contrast, and shadow depth without returning to the overly long side-sun shadows.

Probe command:

```bash
STEPS=300 CAPTURE_PNG_STEPS=0,150,300 \
  bash_scripts/103_probe_simtoolreal_lighting_polish.sh
```

Probe output:

`local_logs/2026-06-06_02-43-30_simtoolreal_lighting_polish_probe/`

Review sheets:

- `contact_sheet_step_0000.png`
- `contact_sheet_step_0150.png`
- `contact_sheet_step_0300.png`

Finding:

- `04_elev50_desaturated_table` is the best polish candidate. It keeps the light grey/white soft-concrete floor, lowers the sun from 55 to 50 degrees for slightly clearer depth cues, desaturates the table, and makes the gradient backdrop a little more blue/grey.
- The change is intentionally subtle. It is more readable and less tan/yellow than the accepted `soft_concrete_elev55` candidate, but it does not solve the remaining synthetic horizon/backdrop limitation.

Current polished 20s candidate:

`local_logs/2026-06-06_02-47-46_simtoolreal_ref_pan_polished_desat_table_elev50_20s/rollout.mp4`

Comparison sheet against the previously accepted short-shadow candidate:

`local_logs/2026-06-06_video_comparisons/polished_elev50_vs_accepted_elev55_contact.png`

Exact command:

```bash
OUT_DIR=local_logs/$(date +%F_%H-%M-%S)_simtoolreal_ref_pan_polished_desat_table_elev50_20s \
  MAKE_VIDEO=1 STEPS=1200 CAPTURE_PNG_STEPS=0,300,600,900,1200 \
  BACKDROP_GRADIENT_BANDS=32 \
  FLOOR_STYLE=soft_concrete_pbr_tiles \
  FLOOR_COLOR_R=0.57 FLOOR_COLOR_G=0.57 FLOOR_COLOR_B=0.54 \
  FLOOR_TILE_COUNT=1 FLOOR_TILE_SIZE=120 FLOOR_TILE_GAP=0 \
  FLOOR_TEXTURE_SCALE=8.0 FLOOR_ROUGHNESS=0.92 FLOOR_NORMAL_STRENGTH=0.05 FLOOR_SPECULAR_LEVEL=0.04 \
  TABLE_COLOR_R=0.45 TABLE_COLOR_G=0.34 TABLE_COLOR_B=0.24 \
  BACKDROP_COLOR_R=0.24 BACKDROP_COLOR_G=0.46 BACKDROP_COLOR_B=0.75 \
  BACKDROP_HORIZON_COLOR_R=0.62 BACKDROP_HORIZON_COLOR_G=0.73 BACKDROP_HORIZON_COLOR_B=0.84 \
  SINGLE_SUN_ELEVATION_DEG=50 \
  SINGLE_SUN_EXPOSURE=9.45 \
  SINGLE_SUN_ANGLE=0.32 \
  SINGLE_SUN_COLOR_TEMPERATURE=5200 \
  SINGLE_SUN_COLOR_R=1.00 SINGLE_SUN_COLOR_G=0.97 SINGLE_SUN_COLOR_B=0.89 \
  SINGLE_SUN_YAW_OFFSET_DEG=105 \
  DEFAULT_LIGHT_INTENSITY=430 \
  SKY_DOME_INTENSITY=1350 \
  IMAGE_EXPOSURE=-0.15 \
  bash_scripts/98_render_simtoolreal_ref_pan_cinematic_greige_floor.sh
```

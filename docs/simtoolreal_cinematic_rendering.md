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

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
- Lighting: keep ambient fill, then add a warm daylight sun for readable shadows. Use roughly 5600-6200K daylight, not orange 5200K sunset tones.
- Shadows: visible and directional, but not so dark that they dominate the foreground.

## Current Palette Decision

The white floor and white background are the main reason the current render lacks contrast. The robot has a white/gray body, so a pure white floor makes the robot, object colors, and shadows read as low-contrast. A dark floor has the opposite problem: it looks like a debug viewport and makes the scene feel less polished. The best compromise is a medium-light gray concrete/stone floor.

Current script defaults in `bash_scripts/95_render_simtoolreal_ref_pan_cinematic.sh` after the latest contrast pass:

- Floor: `nvidia_precast_concrete_dark_gray`. Visually this reads as medium gray in the camera render, not charcoal.
- Background: `gradient_sky` with a blue-gray horizon `(0.68, 0.78, 0.88)` and blue top color `(0.25, 0.48, 0.76)`.
- Backdrop extent margin: `80.0`. The wide SAPG pan otherwise sees the edge of a smaller diagonal sky panel.
- Dome sky/fill: muted blue dome `(0.50, 0.66, 0.86)`.
- Table: matte warm brown `(0.50, 0.32, 0.18)`.
- Light: `single_sun` with default world/dome fill kept at `450`, sun exposure `8.0`, sun angle `0.28`, color temperature `5900K`, and yaw offset `80 deg`.

This is intentionally more in the direction of sunset/sunlight than the flat no-extra-light render, but avoids the orange cast from the previous hard-sun test. The single-sun color in `render_simtoolreal_pretrained.py` is now neutral-warm `(1.0, 0.97, 0.90)` instead of orange `(1.0, 0.92, 0.78)`. The backdrop horizon is configurable because a near-white horizon made the rendered camera image look like white floor against white background.

Latest reviewed frame:

`local_logs/2026-06-04_21-16-53_2026-06-04_21-16-53_refpan_palette_dark_floor_blue_sky_big_backdrop_frame600/step_0600.png`

Assessment:

- This is the best contrast so far. The scene no longer reads as white floor on white sky.
- It is still a synthetic gradient backdrop, not a true HDRI/cloud sky. That is acceptable for now because IsaacLab camera captures did not reliably show the dome/HDRI sky like the viewport.
- The floor now has enough contrast with the white robot and colored objects.
- Directional shadows are visible but not extremely dramatic.

## Current Best Output

Latest high-contrast daylight candidate:

`local_logs/2026-06-04_21-19-59_2026-06-04_21-19-59_simtoolreal_ref_pan_high_contrast_daylight_20s/rollout.mp4`

Quick review sheet:

`local_logs/2026-06-04_21-19-59_2026-06-04_21-19-59_simtoolreal_ref_pan_high_contrast_daylight_20s/contact_sheet_start_mid_end.png`

Main remaining issues:

- The sky is still a synthetic gradient backdrop, so it has a visible horizon/band rather than a natural HDRI/cloud sky.
- The table is a matte warm-brown material, not a true wood texture.
- The floor is much better for contrast, but less like the bright white-stone reference than the earlier white variants.

## Next Test Matrix

Keep the camera and policy fixed. Only vary visual stack:

- Candidate A: medium gray concrete floor, brown table, blue-gray gradient backdrop, no extra sun.
- Candidate B: same as A, but with a warm daylight sun and strong ambient fill left on.
- Candidate C: slightly darker warm gray floor, brown table, blue-gray gradient backdrop, softer daylight sun.

Compare frames at `0`, `600`, and `1200` before committing to a full MP4.

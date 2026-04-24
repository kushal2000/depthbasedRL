# 2026-04-24 Rendering Findings

This note records the camera-rendering checks run on branch `2026-04-24_debugstudentrollout`.

## Main conclusions

- `80x45` RGB is possible, but not on default-ish `balanced` settings.
- `160x90` RGB rendered cleanly in all verified tests.
- For `80x45`, the render settings matter a lot:
  - `rendering_mode="quality"`
  - `antialiasing_mode="DLAA"`
  - `enable_dl_denoiser=True`
  - `samples_per_pixel=4`
- Wrist-mounted tiled cameras can work, but only if they are spawned under the wrist link and not driven later with `set_world_poses()`.

## Camera-path findings

### Wrist tiled bug

The original wrist path was not really using tiled rendering correctly.

Required fixes:

- Use `TiledCameraCfg` for wrist when `camera_backend == "tiled"` and `ISAACSIM_FORCE_TILED_WRIST=1`.
- Spawn the wrist camera prim under the link:
  - `"{ENV_REGEX_NS}/Robot/<link_name>/DistillCamera"`
- For tiled cameras, cache world poses but do not call `camera.set_world_poses(...)`.

Without those changes, wrist tiled rendering produced invalid black/white bar outputs.

### Third-person path

I did not find a separate third-person camera bug after rechecking the path.

Third-person and wrist both improved substantially at `80x45` once the render settings were changed to `quality + DLAA + denoiser`.

## Verified image-quality matrix

All of the following were run with `num_envs=4`.

### 80x45

Third-person, tiled, `balanced`:

- noisy
- artifact: [third80_balanced_debug](/home/tylerlum/github_repos/depthbasedRL/distillation_runs_rendercheck/third80_balanced_debug/camera_debug/rgb_grid.png)

Third-person, tiled, `quality + DLAA + denoiser + spp=4`:

- much cleaner / usable
- artifact: [third80_quality_dlaa_debug_serial](/home/tylerlum/github_repos/depthbasedRL/distillation_runs_rendercheck/third80_quality_dlaa_debug_serial/camera_debug/rgb_grid.png)

Wrist, tiled, `balanced`:

- noisy
- artifact: [wrist80_balanced_debug_serial](/home/tylerlum/github_repos/depthbasedRL/distillation_runs_rendercheck/wrist80_balanced_debug_serial/camera_debug/rgb_grid.png)

Wrist, tiled, `quality + DLAA + denoiser + spp=4`:

- much cleaner / usable
- artifact: [wrist80_quality_dlaa_debug](/home/tylerlum/github_repos/depthbasedRL/distillation_runs_rendercheck/wrist80_quality_dlaa_debug/camera_debug/rgb_grid.png)

### 160x90

Third-person, tiled, `balanced`:

- clean
- artifact: [third160_balanced_debug_serial](/home/tylerlum/github_repos/depthbasedRL/distillation_runs_rendercheck/third160_balanced_debug_serial/camera_debug/rgb_grid.png)

Third-person, tiled, `quality + DLAA + denoiser + spp=4`:

- clean
- artifact: [third160_quality_dlaa_debug_serial](/home/tylerlum/github_repos/depthbasedRL/distillation_runs_rendercheck/third160_quality_dlaa_debug_serial/camera_debug/rgb_grid.png)

Wrist, tiled, `balanced`:

- clean
- artifact: [wrist160_balanced_debug_serial](/home/tylerlum/github_repos/depthbasedRL/distillation_runs_rendercheck/wrist160_balanced_debug_serial/camera_debug/rgb_grid.png)

Wrist, tiled, `quality + DLAA + denoiser + spp=4`:

- clean
- artifact: [wrist160_quality_dlaa_debug_serial](/home/tylerlum/github_repos/depthbasedRL/distillation_runs_rendercheck/wrist160_quality_dlaa_debug_serial/camera_debug/rgb_grid.png)

## Recommendation

If keeping the student at `80x45`, use:

- `ISAACSIM_RENDER_MODE=quality`
- `ISAACSIM_RENDER_AA=DLAA`
- `ISAACSIM_RENDER_DENOISER=1`
- `ISAACSIM_RENDER_SPP=4`

If changing camera resolution is acceptable, `160x90` is the safer path and looked good even at `balanced`.

## Notes

- The low-res issue was visible in both wrist and third-person RGB.
- Depth was not the focus of these render-quality checks.
- The old-base branch still has a shared `/tmp/isaaclab_usd_cache_distill` race when multiple Isaac Sim jobs import in parallel.

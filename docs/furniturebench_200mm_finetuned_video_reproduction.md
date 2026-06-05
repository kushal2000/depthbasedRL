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

## Visual Defaults

The close-up view uses darker defaults than the wide SimToolReal diversity video
because the original beauty lighting overexposed the single-table scene.

- Camera eye:
  `(0.12, -1.55, 0.98)`
- Camera target:
  `(0.02, 0.02, 0.58)`
- Default light intensity:
  `120`
- Sky dome intensity:
  `650`
- Sun exposure:
  `6.8`
- Sun angle:
  `0.45`
- Image exposure:
  `0.72`
- Image contrast:
  `1.08`
- Image saturation:
  `1.03`
- Table color:
  `(0.42, 0.27, 0.15)`
- Floor color:
  `(0.44, 0.45, 0.43)`

## Notes For Generating Variants

For multiple candidate clips, vary `SEED`, `OUT_DIR`, and possibly `STEPS`.
The seed-0 successful episode hits `10/10` by step `420` and resets at step
`460`, so a `430` step video is intentionally cleaner than a longer `600` step
clip.

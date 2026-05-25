# L-Peg Seed-Aggregated Curves

The original Fig. 2 scripts read committed source JSONs under
`outputs/fig2_panel_bcd/`.  No generator for those JSONs is tracked in this
repo.  The old panel-A JSONs use feasible-normalized success:

```text
episode_final/all_goals_hit / (1 - episode_final/done_fall)
```

These scripts generate separate Tyler L-peg seed-aggregation JSONs from W&B
history by default, with local TensorBoard event files as a fallback, and then
render mean/std shaded curves.

The plotter also renders a hybrid ObjectDiversity comparison where the
`1000_obj` curve is replaced by the TrainingObjective `Play2Win` run data,
displayed as `Play2Perfect (1000 objects)`. This is useful because those two
checkpoints are intended to be comparable.

## Full Collection

W&B is the default because the local TensorBoard event files are large,
especially while jobs are still running.

```bash
.venv-isaacsim-py311/bin/python plot_figures/fig2/collect_lpeg_seed_curves.py
.venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_seed_plots.py --show-seeds
```

The plotter uses relative env frames by default (`global_step - first_logged_step`)
because some older W&B seed-0 runs have large absolute `global_step` offsets from
prior/resumed runs. Mean/std curves require all available seeds by default; use
`--min-seeds 2` if you intentionally want partial tails after shorter seeds end.

Outputs:

```text
outputs/fig2_tyler_seed_curves/lpeg_objectdiversity_seed_curves.json
outputs/fig2_tyler_seed_curves/lpeg_trainingobjective_seed_curves.json
plot_figures/fig2/outputs/lpeg_objectdiversity_episode_final_feasible_normalized_all_goals_hit_relative_frames_allseeds_mean_std_with_seed_overlay.png
plot_figures/fig2/outputs/lpeg_objectdiversity_episode_final_feasible_normalized_all_goals_hit_relative_frames_allseeds_mean_std_clean.png
plot_figures/fig2/outputs/lpeg_trainingobjective_episode_final_feasible_normalized_all_goals_hit_relative_frames_allseeds_mean_std_with_seed_overlay.png
plot_figures/fig2/outputs/lpeg_trainingobjective_episode_final_feasible_normalized_all_goals_hit_relative_frames_allseeds_mean_std_clean.png
plot_figures/fig2/outputs/lpeg_objectdiversity_play2win_episode_final_feasible_normalized_all_goals_hit_relative_frames_allseeds_mean_std_with_seed_overlay.png
plot_figures/fig2/outputs/lpeg_objectdiversity_play2win_episode_final_feasible_normalized_all_goals_hit_relative_frames_allseeds_mean_std_clean.png
plot_figures/fig2/outputs/lpeg_seed_ablation_summary_grid_3x3_relative_frames_allseeds.png
```

## Smoke Test

```bash
.venv-isaacsim-py311/bin/python plot_figures/fig2/collect_lpeg_seed_curves.py \
  --out-dir /tmp/fig2_lpeg_seed_smoke \
  --checkpoint-tags Play2Win 1_obj \
  --seeds 1 \
  --wandb-samples 2000

.venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_seed_plots.py \
  --data-dir /tmp/fig2_lpeg_seed_smoke \
  --out-dir /tmp/fig2_lpeg_seed_smoke_plots \
  --show-seeds
```

## Individual Seed Curves

Color indicates checkpoint; line style indicates seed.

```bash
.venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_seed_plots.py \
  --individual
```

For capped 4B summary plots with and without faint seed overlays:

```bash
.venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_seed_plots.py \
  --show-seeds --x-max-billions 4

.venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_seed_plots.py \
  --x-max-billions 4

.venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_seed_summary_grid.py
```

For the three newer ObjectDiversity seeds only:

```bash
.venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_seed_plots.py \
  --individual --seeds 1 2 3
```

## TensorBoard Fallback

This reads local event files directly. It can be much slower than W&B on active
runs because event files can exceed 100 MB.

```bash
.venv-isaacsim-py311/bin/python plot_figures/fig2/collect_lpeg_seed_curves.py \
  --source tensorboard
```

## Alternate Raw-Success Plot

To plot raw full task success instead of old-Fig2-compatible feasible-normalized
success:

```bash
.venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_seed_plots.py \
  --metric episode_final/all_goals_hit \
  --show-seeds
```

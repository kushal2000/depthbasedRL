# L-Peg Selected "Ours" Curves

For the final L-peg ablation panels, use the following three curves as the
shared mean/std definition of "Ours" in every panel:

| Source | Checkpoint | Seed | Wrench |
|---|---|---:|---|
| ObjectDiversity | `1000_obj` | 0 | yes |
| ObjectDiversity | `1000_obj` | 1 | yes |
| ObjectDiversity | `1000_obj` | 2 | yes |

This intentionally uses three seeds of the same policy/checkpoint. The earlier
candidate ranking had a slightly faster mixed set, but that mixed set reused
`1000_obj` seed 0 and `Play2Win` seed 0, making the seed comparison less clean.
We therefore use ranks 1, 3, and 4 from the candidate table:

| Rank | Policy | Seed | Wrench? | Final Success | t70 | t80 | Env Steps |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | 1000 Objects | 2 | yes | 97.8% | 0.59B | 0.63B | 6.37B |
| 3 | 1000 Objects | 0 | yes | 96.8% | 0.71B | 0.78B | 8.03B |
| 4 | 1000 Objects | 1 | yes | 96.3% | 0.97B | 1.16B | 4.43B |

Use the shared selected-Ours aggregate with different legend names depending on
the panel:

- Object Diversity: `1000 Objects (Ours)`
- Training Objective: `Full Pose (Ours)`
- Trajectory Diversity: `Random Trajectories (Ours)`
- Play Precision: `1 cm (Ours)`

The dedicated plotting entrypoint is:

```bash
.venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_selected_ours_four_panels.py
```

## Current Panel Definitions

All panels plot feasible-normalized success:

```text
episode_final/feasible_normalized_all_goals_hit
  = episode_final/all_goals_hit / (1 - episode_final/done_fall)
```

The x-axis is relative finetuning environment steps:

```text
relative_env_steps = global_step - first_logged_global_step_for_that_run
```

Each curve is the mean over selected seeds on the common support of those
seeds. The shaded region is one standard deviation over those seeds. Common
support means the aggregate only extends to the shortest selected seed for that
curve.

The current four panels are:

| Panel | Curves |
|---|---|
| Object Diversity | `1000 Objects (Ours)`, `100 Objects`, `10 Objects` |
| Training Objective | `Full Pose (Ours)`, `Rotation-Only`, `Translation-Only` |
| Trajectory Diversity | `Random Trajectories (Ours)`, `100 Trajectories (Wrench)`, `100 Trajectories (No Wrench)` |
| Play Precision | `1 cm (Ours)`, `5 cm (Wrench)`, `5 cm (No Wrench)` |

The blue Ours curve is exactly the same selected three runs in every panel; only
the legend label changes to match the ablation being shown.

## Data Sources

The plotting script reads these cached W&B exports:

| Cache | Contents |
|---|---|
| `outputs/fig2_tyler_3curve_seed_curves/lpeg_wrench_seed_curves_latest.json` | Wrench L-peg object-diversity and training-objective runs |
| `outputs/fig2_tyler_seed_curves/lpeg_trainingobjective_seed_curves.json` | No-wrench training-objective runs |
| `outputs/fig2_tyler_traj_precision_seed_curves/lpeg_traj_precision_wrench_compare_latest.json` | Trajectory-count and precision-threshold runs, with and without wrench |

Refresh and regenerate with:

```bash
.venv-isaacsim-py311/bin/python plot_figures/fig2/make_lpeg_selected_ours_four_panels.py \
  --refresh-wrench \
  --refresh-traj-precision \
  --x-max-billions 4 \
  --slug selected_ours_4B
```

## Current Outputs

The current generated files are:

| Panel | PNG |
|---|---|
| Object Diversity | `plot_figures/fig2/outputs/lpeg_final_object_diversity_selected_ours_4B.png` |
| Training Objective | `plot_figures/fig2/outputs/lpeg_final_training_objective_selected_ours_4B.png` |
| Trajectory Diversity | `plot_figures/fig2/outputs/lpeg_final_trajectory_diversity_selected_ours_4B.png` |
| Play Precision | `plot_figures/fig2/outputs/lpeg_final_play_precision_selected_ours_4B.png` |
| Review grid | `plot_figures/fig2/outputs/lpeg_final_four_panel_review_selected_ours_4B.png` |

PDFs with the same basenames are also written for the four standalone panels.

## Current Play Precision Caveat

The `5 cm (No Wrench)` Play Precision curve currently ends around 1.9B-2.0B
environment steps because two of its three selected runs failed early:

| Seed | W&B State | Env Steps | Final Norm Success | Cause |
|---:|---|---:|---:|---|
| 0 | failed | 1.91B | 62.9% | CUDA OOM |
| 1 | failed | 2.36B | 90.4% | CUDA OOM |
| 2 | running | 3.99B at last refresh | 100.0% | still running |

Both failed runs used full L-peg scale on 24GB A5000-class GPUs:

```text
NUM_ENVS=12288
MINIBATCH_SIZE=98304
EXPL_COEF_BLOCK_SIZE=2048
```

The failure was in `rl_games` central-value training during backward:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB.
```

Cluster run dirs checked:

```text
/move/u/tylerlum/github_repos/depthbasedRL/train_dir/fig2/panel_a_teachers_tyler_traj_precision_wrench_compare/lpeg_tol0p5mm_finetune_rgf0_dr_no_wrench_seed0_Precision_5cm_2026-05-27_00-53-20
/move/u/tylerlum/github_repos/depthbasedRL/train_dir/fig2/panel_a_teachers_tyler_traj_precision_wrench_compare/lpeg_tol0p5mm_finetune_rgf0_dr_no_wrench_seed1_Precision_5cm_2026-05-27_00-53-27
```

If this no-wrench 5 cm curve is needed for a final comparison, rerun seeds 0
and 1 either on larger GPUs or with reduced L-peg scale, for example half
`NUM_ENVS`, half `MINIBATCH_SIZE`, and corresponding smaller
`EXPL_COEF_BLOCK_SIZE`.

# TrainingObjective Sweep, 2026-05-24

Submitted from branch `2026-05-21_TylerJobs` at commit `e1e5a0626b9c5871b6d6f4fc9d6852138286afbb`.

All jobs use `CHECKPOINT_FAMILY=TrainingObjective`, `--checkpoint_load_mode weights`, 24 hour walltime, and the task hyperparameters from `panel_a_tyler_finetune.sub`.

## Placement

| Job ID | Task | Checkpoint | Node | Partition | Memory |
| --- | --- | --- | --- | --- | --- |
| 15543785 | `furniture_bench_one_leg_finetune_rgf10_dr` | `Play2Win` | `move5` | `move` | 90 GB |
| 15543786 | `furniture_bench_one_leg_finetune_rgf10_dr` | `RotationOnly` | `move5` | `move` | 90 GB |
| 15543787 | `furniture_bench_one_leg_finetune_rgf10_dr` | `SingleGoal` | `move5` | `move` | 90 GB |
| 15543788 | `furniture_bench_one_leg_finetune_rgf10_dr` | `TranslationOnly` | `move5` | `move` | 90 GB |
| 15543789 | `beam_3x_part_2_finetune_rgf0_dr` | `Play2Win` | `move5` | `move` | 90 GB |
| 15543790 | `beam_3x_part_2_finetune_rgf0_dr` | `RotationOnly` | `move5` | `move` | 90 GB |
| 15543791 | `beam_3x_part_2_finetune_rgf0_dr` | `SingleGoal` | `move5` | `move` | 90 GB |
| 15543792 | `beam_3x_part_2_finetune_rgf0_dr` | `TranslationOnly` | `move5` | `move` | 90 GB |
| 15543793 | `beam_3x_part_0_finetune_rgf0_dr` | `Play2Win` | `move4` | `move` | 80 GB |
| 15543794 | `beam_3x_part_0_finetune_rgf0_dr` | `RotationOnly` | `move4` | `move` | 80 GB |
| 15543795 | `beam_3x_part_0_finetune_rgf0_dr` | `SingleGoal` | `move4` | `move` | 80 GB |
| 15543796 | `beam_3x_part_0_finetune_rgf0_dr` | `TranslationOnly` | `move4` | `move` | 80 GB |
| 15543797 | `lpeg_tol0p5mm_finetune_rgf0_dr` | `Play2Win` | `juno2` | `juno` | 100 GB |
| 15543798 | `lpeg_tol0p5mm_finetune_rgf0_dr` | `RotationOnly` | `juno2` | `juno` | 100 GB |
| 15543799 | `lpeg_tol0p5mm_finetune_rgf0_dr` | `SingleGoal` | `juno2` | `juno-lo` | 100 GB |
| 15543800 | `lpeg_tol0p5mm_finetune_rgf0_dr` | `TranslationOnly` | `juno2` | `juno-lo` | 100 GB |

Run directories are under:

```text
/move/u/tylerlum/github_repos/depthbasedRL/train_dir/fig2/panel_a_teachers_tyler_training_objective/
```

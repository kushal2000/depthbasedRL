# L-Peg Seed-1/Seed-3 Sweep, 2026-05-25

Submitted from branch `2026-05-21_TylerJobs` at commit
`2a9e9958c2b6d9d3975fa565a51f7d51cffd598c`.

This sweep runs only the L-peg task and covers two new random seeds for all
eight checkpoints:

- `TrainingObjective`: `Play2Win`, `RotationOnly`, `SingleGoal`, `TranslationOnly`
- `ObjectDiversity`: `1_obj`, `10_obj`, `100_obj`, `1000_obj`

All jobs use `--checkpoint_load_mode weights`, 24 hour walltime, full L-peg
scale (`NUM_ENVS=12288`, `MINIBATCH_SIZE=98304`,
`EXPL_COEF_BLOCK_SIZE=2048`), and the task hyperparameters from
`panel_a_tyler_finetune.sub`.

Seed is passed as `agent.params.seed` through the launcher `SEED` variable and
is also embedded in the run name via `EXPERIMENT_TAG`:

```text
lpeg_tol0p5mm_finetune_rgf0_dr_seed${SEED}
```

## Existing L-Peg Seed Coverage

- ObjectDiversity full L-peg runs already covered seed `0` and seed `2`.
- TrainingObjective full L-peg runs already covered seed `0`.
- This sweep adds seed `1` and seed `3`.

## Placement

| Job ID | Family | Checkpoint | Seed | Node | Partition | Memory |
| --- | --- | --- | --- | --- | --- | --- |
| 15551054 | `TrainingObjective` | `Play2Win` | `1` | `move5` | `move` | 90 GB |
| 15551055 | `TrainingObjective` | `RotationOnly` | `1` | `move5` | `move` | 90 GB |
| 15551056 | `TrainingObjective` | `SingleGoal` | `1` | `move5` | `move` | 90 GB |
| 15551057 | `TrainingObjective` | `TranslationOnly` | `1` | `move5` | `move` | 90 GB |
| 15551058 | `TrainingObjective` | `Play2Win` | `3` | `move5` | `move` | 90 GB |
| 15551059 | `TrainingObjective` | `RotationOnly` | `3` | `move5` | `move` | 90 GB |
| 15551060 | `TrainingObjective` | `SingleGoal` | `3` | `move5` | `move` | 90 GB |
| 15551061 | `TrainingObjective` | `TranslationOnly` | `3` | `move5` | `move` | 90 GB |
| 15551062 | `ObjectDiversity` | `1_obj` | `1` | `move4` | `move` | 80 GB |
| 15551063 | `ObjectDiversity` | `10_obj` | `1` | `move4` | `move` | 80 GB |
| 15551064 | `ObjectDiversity` | `100_obj` | `1` | `move4` | `move` | 80 GB |
| 15551065 | `ObjectDiversity` | `1000_obj` | `1` | `move4` | `move` | 80 GB |
| 15551066 | `ObjectDiversity` | `1_obj` | `3` | `juno2` | `juno` | 100 GB |
| 15551067 | `ObjectDiversity` | `10_obj` | `3` | `juno2` | `juno` | 100 GB |
| 15551068 | `ObjectDiversity` | `100_obj` | `3` | `juno2` | `juno-lo` | 100 GB |
| 15551069 | `ObjectDiversity` | `1000_obj` | `3` | `juno2` | `juno-lo` | 100 GB |

Run directories are under:

```text
/move/u/tylerlum/github_repos/depthbasedRL/train_dir/fig2/
```

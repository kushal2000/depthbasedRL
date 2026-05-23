# Seed-2 Replicate Jobs, 2026-05-23

Submitted from branch `2026-05-21_TylerJobs` at commit `c24f3cabd4d9d5de33057b90b45030ab2b4fca88`.

All jobs use `CHECKPOINT_FAMILY=ObjectDiversity`, `SEED=2`, `--checkpoint_load_mode weights`, 24 hour walltime, and the same task hyperparameters as `panel_a_tyler_finetune.sub`.

## Placement

| Job ID | Task | Checkpoint | Node | Partition | Memory |
| --- | --- | --- | --- | --- | --- |
| 15535760 | `lpeg_tol0p5mm_finetune_rgf0_dr_seed2` | `1_obj` | `juno2` | `juno` | 100 GB |
| 15535774 | `lpeg_tol0p5mm_finetune_rgf0_dr_seed2` | `10_obj` | `juno2` | `juno-lo` | 100 GB |
| 15535775 | `lpeg_tol0p5mm_finetune_rgf0_dr_seed2` | `100_obj` | `juno2` | `juno-lo` | 100 GB |
| 15535776 | `lpeg_tol0p5mm_finetune_rgf0_dr_seed2` | `1000_obj` | `juno2` | `juno-lo` | 100 GB |
| 15535764 | `beam_3x_part_0_finetune_rgf0_dr_seed2` | `1_obj` | `move3` | `move` | 90 GB |
| 15535765 | `beam_3x_part_0_finetune_rgf0_dr_seed2` | `10_obj` | `move3` | `move` | 90 GB |
| 15535766 | `beam_3x_part_0_finetune_rgf0_dr_seed2` | `100_obj` | `move3` | `move` | 90 GB |
| 15535767 | `beam_3x_part_0_finetune_rgf0_dr_seed2` | `1000_obj` | `move3` | `move` | 90 GB |
| 15535768 | `furniture_bench_one_leg_finetune_rgf10_dr_seed2` | `1_obj` | `move5` | `move` | 90 GB |
| 15535769 | `furniture_bench_one_leg_finetune_rgf10_dr_seed2` | `10_obj` | `move5` | `move` | 90 GB |
| 15535770 | `furniture_bench_one_leg_finetune_rgf10_dr_seed2` | `100_obj` | `move5` | `move` | 90 GB |
| 15535771 | `furniture_bench_one_leg_finetune_rgf10_dr_seed2` | `1000_obj` | `move5` | `move` | 90 GB |

The first L-peg `1_obj` job used normal `juno`; the remaining three L-peg jobs were resubmitted to `juno-lo` after `juno` hit `QOSMaxGRESPerUser`.

## Follow-Up Fix

The initial full-scale move3 A5000 beam part-0 jobs `15535764`-`15535767` failed with CUDA OOM after environment setup. They were replaced by half-scale A5000 jobs with `NUM_ENVS=6144`, `MINIBATCH_SIZE=49152`, and auto `EXPL_COEF_BLOCK_SIZE=1024`:

| Job ID | Task | Checkpoint | Node | Partition | Memory |
| --- | --- | --- | --- | --- | --- |
| 15535834 | `beam_3x_part_0_finetune_rgf0_dr_seed2_a5000_halfenv` | `1_obj` | `move3` | `move` | 90 GB |
| 15535835 | `beam_3x_part_0_finetune_rgf0_dr_seed2_a5000_halfenv` | `10_obj` | `move3` | `move` | 90 GB |
| 15535836 | `beam_3x_part_0_finetune_rgf0_dr_seed2_a5000_halfenv` | `100_obj` | `move3` | `move` | 90 GB |
| 15535837 | `beam_3x_part_0_finetune_rgf0_dr_seed2_a5000_halfenv` | `1000_obj` | `move3` | `move` | 90 GB |

Run directories are under:

```text
/move/u/tylerlum/github_repos/depthbasedRL/train_dir/fig2/panel_a_teachers_tyler_object_diversity/*seed2*
```

# Fig 2 Panel A Tyler Jobs

Tyler-local launchers for the Fig 2 panel-a teacher finetuning jobs. The
original collaborator scripts in `plot_figures/fig2/experiments/` are left
unchanged.

## Runner

Use:

```bash
plot_figures/fig2/experiments_tyler/panel_a_tyler_finetune.sub
```

The runner defaults to `move5` RTX PRO 6000 with:

```text
REPO_ROOT=/move/u/tylerlum/github_repos/depthbasedRL_rtx6000
ISAACSIM_ENV_DIR=/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-cu128-py311
WANDB_ENTITY=tylerlum
WANDB_PROJECT=fig2
WANDB_GROUP=panel_a_teachers_tyler_object_diversity
```

It requests `--mem=90000`, which is approximately `move5` node RAM divided by
8 GPUs with margin. For `move4`, `juno2`, or `humanoid1`, set memory to roughly
node RAM divided by GPU count.

## Task Tags

```text
lpeg_tol0p5mm
beam_3x_part_0
beam_3x_part_2
furniture_bench_one_leg
```

The FurnitureBench preset keeps the original different settings:

```text
RANDOM_GOAL_FRACTION=0.1
INSERTION_SUCCESS_TOLERANCE=0.005
TARGET_SUCCESS_TOLERANCE=0.002
```

## Checkpoint Tags

Primary ObjectDiversity checkpoints:

```text
CHECKPOINT_FAMILY=ObjectDiversity CHECKPOINT_TAG=1_obj
CHECKPOINT_FAMILY=ObjectDiversity CHECKPOINT_TAG=10_obj
CHECKPOINT_FAMILY=ObjectDiversity CHECKPOINT_TAG=100_obj
CHECKPOINT_FAMILY=ObjectDiversity CHECKPOINT_TAG=1000_obj
```

Prepared but not launched initially:

```text
CHECKPOINT_FAMILY=TrainingObjective CHECKPOINT_TAG=Play2Win
CHECKPOINT_FAMILY=TrainingObjective CHECKPOINT_TAG=RotationOnly
CHECKPOINT_FAMILY=TrainingObjective CHECKPOINT_TAG=SingleGoal
CHECKPOINT_FAMILY=TrainingObjective CHECKPOINT_TAG=TranslationOnly
```

## Gated Launch

Run exactly one tiny sanity job first:

```bash
NUM_ENVS=256 MINIBATCH_SIZE=2048 MAX_ITERATIONS=2 \
TASK_TAG=lpeg_tol0p5mm CHECKPOINT_FAMILY=ObjectDiversity CHECKPOINT_TAG=1_obj \
sbatch --time=1:00:00 plot_figures/fig2/experiments_tyler/panel_a_tyler_finetune.sub
```

Monitor:

```bash
squeue -u tylerlum -o "%.18i %.9P %.30j %.8u %.2t %.10M %.10l %.12R"
tail -f /move/u/tylerlum/github_repos/depthbasedRL_rtx6000/train_dir/fig2/panel_a_teachers_tyler_object_diversity/*/slurm.log
```

Only after the first sanity job reaches training iterations, run the other
three task sanity checks:

```bash
for task in beam_3x_part_0 beam_3x_part_2 furniture_bench_one_leg; do
  NUM_ENVS=256 MINIBATCH_SIZE=2048 MAX_ITERATIONS=2 \
  TASK_TAG="$task" CHECKPOINT_FAMILY=ObjectDiversity CHECKPOINT_TAG=1_obj \
  sbatch --time=1:00:00 plot_figures/fig2/experiments_tyler/panel_a_tyler_finetune.sub
done
```

## First Full Comparison

After all task sanity checks pass, launch only `1_obj` vs `1000_obj` across all
four tasks. Full jobs use 48 h walltime and high `MAX_ITERATIONS` so they should
stop by walltime, not by iteration count.

```bash
for task in lpeg_tol0p5mm beam_3x_part_0 beam_3x_part_2 furniture_bench_one_leg; do
  for ckpt in 1_obj 1000_obj; do
    TASK_TAG="$task" CHECKPOINT_FAMILY=ObjectDiversity CHECKPOINT_TAG="$ckpt" \
    sbatch plot_figures/fig2/experiments_tyler/panel_a_tyler_finetune.sub
  done
done
```

If those jobs are stable, launch `10_obj` and `100_obj` later:

```bash
for task in lpeg_tol0p5mm beam_3x_part_0 beam_3x_part_2 furniture_bench_one_leg; do
  for ckpt in 10_obj 100_obj; do
    TASK_TAG="$task" CHECKPOINT_FAMILY=ObjectDiversity CHECKPOINT_TAG="$ckpt" \
    sbatch plot_figures/fig2/experiments_tyler/panel_a_tyler_finetune.sub
  done
done
```

Do not launch the TrainingObjective sweep until the ObjectDiversity jobs are
healthy and GPUs are available.

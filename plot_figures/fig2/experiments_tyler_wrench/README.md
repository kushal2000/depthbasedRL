# Tyler L-peg Wrench Finetunes

This folder prepares the 2026-05-25 rerun of the Fig. 2 L-peg finetunes on top
of `origin/2026_05_25_fig4_improve_real_world`.

Compared with `plot_figures/fig2/experiments_tyler`, these jobs keep the same
L-peg task, checkpoints, seed sweep, training scale, and W&B project, but enable
object wrench perturbations:

```text
env.domain_randomization.force_scale=20.0
env.domain_randomization.torque_scale=2.0
env.domain_randomization.force_only_when_lifted=False
env.domain_randomization.torque_only_when_lifted=False
env.domain_randomization.force_prob_range=[0.001,0.1]
env.domain_randomization.torque_prob_range=[0.001,0.1]
```

The merged env code also gates wrench perturbations off during retract phase via
`PegInHoleEnv._wrench_dr_active_mask()`.

The launcher is dry-run by default:

```bash
bash plot_figures/fig2/experiments_tyler_wrench/submit_lpeg_wrench_24.sh
```

Submit only the first 16-job wave when GPUs are free. Wave 1 is all
ObjectDiversity seeds `0,1,2` plus TrainingObjective seed `0`. Wave 2 is the
remaining TrainingObjective seeds `1,2`.

```bash
DRY_RUN=0 WAVE=sanity bash plot_figures/fig2/experiments_tyler_wrench/submit_lpeg_wrench_24.sh

DRY_RUN=0 WAVE=1 bash plot_figures/fig2/experiments_tyler_wrench/submit_lpeg_wrench_24.sh
```

Submit the remaining 8 jobs later:

```bash
DRY_RUN=0 WAVE=2 bash plot_figures/fig2/experiments_tyler_wrench/submit_lpeg_wrench_24.sh
```

Prepared GPU layout for wave 1:

```text
8 x move5 RTX PRO 6000  partition=move    account=move
4 x move4 L40S          partition=move    account=move
2 x juno2 A5000         partition=juno    account=juno
2 x juno2 A5000         partition=juno-lo account=juno
```

Wave 2 reuses the first 8 slots from the same layout when those remaining
TrainingObjective seeds are ready.

W&B:

```text
entity:  tylerlum
project: fig2
group:   panel_a_teachers_tyler_wrench_lpeg
```

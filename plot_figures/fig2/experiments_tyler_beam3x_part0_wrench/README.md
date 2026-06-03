# Beam 3x Part 0 Wrench Sweep

Created for the 2026-06-03 Fabrica beam_3x part-0 finetuning sweep.

The sweep runs `fabrica.beam_3x.part_0_matchedmass_sdf_hybrid` with the same
Fig. 2 `_dr` task settings as `panel_a_beam_3x_part_0_finetune_rgf0_dr.sub`,
plus Fig. 4 wrench perturbations:

```text
force_scale=20.0
torque_scale=2.0
force_only_when_lifted=False
torque_only_when_lifted=False
```

Run matrix:

```text
TrainingObjective/Play2Win        seeds 0,1
ObjectDiversity/100_obj           seeds 0,1
ObjectDiversity/10_obj            seeds 0,1
TrainingObjective/RotationOnly    seeds 0,1
TrainingObjective/TranslationOnly seeds 0,1
Trajectory_Count/100              seeds 0,1
Trajectory_Count/10               seeds 0,1
Precision/10cm                    seeds 0,1
Precision/5cm                     seeds 0,1
```

Placement is encoded in `submit_beam3x_part0_wrench_18.sh`. The preferred
layout was 7 move4 L40S + 1 humanoid1 L40S, but at launch time move4 had only
6 L40S free. The second Trajectory_Count/100 seed is therefore assigned to the
free move5 RTX PRO 6000 with the RTX-specific IsaacSim env:

```text
6 x move4 L40S
1 x humanoid1 L40S
1 x move5 RTX PRO 6000
5 x juno2 A5000 (2 juno, 3 juno-lo)
5 x move3 A5000
```

Jobs default to full scale:

```text
NUM_ENVS=12288
MINIBATCH_SIZE=98304
EXPL_COEF_BLOCK_SIZE=2048
TIME_LIMIT=1-12:00:00
```

Previous full-scale `beam_3x_part_0` move3 A5000 jobs OOMed immediately after
environment setup. If that repeats, resubmit failed A5000 jobs with:

```text
NUM_ENVS=6144
MINIBATCH_SIZE=49152
EXPL_COEF_BLOCK_SIZE=1024
```

# 2026-04-24 Peg Distillation Plan

## Baseline jobs

Use the peg-in-hole teacher on:
- scene `82`
- peg `5`
- tolerance slot `5`
- goal mode `preInsertAndFinal`

Planned runs:

1. `1_peg_teacher_obs_fixed`
- teacher-observation distillation
- fixed peg initialization
- target machine: `L40S`

2. `2_peg_teacher_obs_randomized`
- teacher-observation distillation
- randomized peg initialization
- target machine: `L40S`

3. `3_peg_depth_image_160x90_fixed`
- depth-image distillation
- `160x90` camera input
- fixed peg initialization
- target machine: `RTX6000`

4. `4_peg_depth_image_160x90_randomized`
- depth-image distillation
- `160x90` camera input
- randomized peg initialization
- target machine: `RTX6000`

5. `5_peg_rgb_image_160x90_fixed_local`
- local third-person RGB image distillation
- `160x90` camera input
- fixed peg initialization

Suggested W&B group:
- `2026-04-24_PegInHole`

## Later robustness plan

After the first baseline runs are working, add robustness in stages:

1. Camera pose randomization
- randomize camera position and rotation around the nominal real-camera pose
- keep ranges small first
- verify the peg and hole remain visible

2. Hole position randomization
- randomize the hole / scene target position
- update goal poses consistently with the same transform

3. Hole orientation randomization
- randomize the hole orientation
- rotate goal poses consistently with the same transform
- preserve the intended pre-insert and final-goal semantics

4. Depth image noise models
- port the depth augmentation path from `2026-04-23_DextrahImageRobustness`
- likely include correlated noise, normal-direction noise, dropout, artifact blobs, and stick artifacts

## Important rule

If hole position or orientation is randomized, the supervision must be transformed too:
- teacher goals
- success/keypoint targets tied to the hole pose

Do not visually move or rotate the hole without moving the goals with it.

## Recommended order

1. Baseline fixed/randomized teacher-observation and depth-image runs
2. Camera pose randomization
3. Hole position randomization with goal-pose updates
4. Hole orientation randomization with goal-pose updates
5. Depth noise augmentation

# Depth Student Noise And Deployment Notes

Date: 2026-05-02

## Recommended Training Noise

DEXTRAH's camera randomization and depth-noise constants are a useful reference point:

- Main camera extrinsic randomization: uniform `+-0.03 m` xyz and `+-3 deg` roll/pitch/yaw, resampled on reset.
- Depth valid window: `0.5-1.3 m`.
- Pixel dropout: `p_dropout = 0.0125 / 4 = 0.003125`.
- Random uniform artifact pixels: `p_randu = 0.003125`.
- Stick artifacts: `p_stick = 0.001 / 4 = 0.00025`, max length `18 px`, max width `3 px`.
- Handa-style depth noise: correlated noise `sigma_s = 0.5`, `sigma_d = 1/6`, normal noise `sigma_theta = 0.01`.

Our current `depth_noise_profile=medium` is intentionally close to the DEXTRAH artifact levels:

- Dropout/randu: `0.003`, essentially DEXTRAH-level.
- Stick probability: `0.00025`, matching DEXTRAH.
- Stick geometry: `18 px x 3 px`, matching DEXTRAH.
- Additional metric Gaussian/correlated noise is simpler than DEXTRAH's Handa implementation, but the visual/debug outputs look realistic and not trivially clean.

For camera pose randomization:

- `20mm/2deg` is conservative and likely realistic for a hand-tuned extrinsic calibration.
- `30mm/3deg` is the closest DEXTRAH-matched level.
- `50mm/5deg` is stronger than DEXTRAH and should be treated as a stress-test robustness target.

Recommended serious candidate setting:

```bash
--depth_noise_profile medium \
--camera_pose_randomization_profile custom \
--camera_pose_randomization_mode startup \
--camera_pos_noise_m 0.03 0.03 0.03 \
--camera_rot_noise_deg 3 3 3
```

The current `50mm/5deg` runs are useful, but they are deliberately harder than DEXTRAH. If they work, they give margin; if they lose insertion performance, `30mm/3deg` is the better target.

## Deployment Image Pipeline

The current peg-in-hole depth student was trained with the `PegInHoleDepthStudent.yaml` student observation settings:

- Source/render size: `160 x 90`.
- Policy crop: `x=[90,160)`, `y=[0,70)`.
- Final policy image shape: `1 x 70 x 70`.
- Depth preprocessing: `window_normalize`.
- Depth window: `[0.70, 1.10] m`.

The deployment node should therefore process the ZED depth image as:

```text
ZED SDK depth frame
  -> convert units to meters
  -> set invalid depths and depths < 1 mm to 0, matching FoundationPose ZED handling
  -> resize to 160 x 90 with nearest-neighbor interpolation when resizing is needed
  -> window normalize: clip((depth_m - 0.70) / (1.10 - 0.70), 0, 1)
  -> crop [y=0:70, x=90:160]
  -> tensor shape (1, 1, 70, 70), float32
```

For real deployment, use direct ZED SDK capture in the policy node rather than a
ROS depth-image topic. Depth images are high bandwidth and can add avoidable
network/serialization load. The node defaults to:

```bash
--depth_source zed_sdk
```

Use `--depth_source ros_topic --depth_topic <topic>` only for debugging or if the
ZED SDK is not available in the deployment shell.

The node saves debug artifacts that make each stage visible:

- raw ZED metric depth in meters
- resized metric depth in meters
- normalized full `160 x 90` policy-window depth
- final cropped `70 x 70` policy image
- a review grid PNG, and optionally an mp4

The node also emits warnings when the depth distribution is clearly wrong, for example if the median depth suggests millimeters were not converted to meters or if almost none of the crop lies in the expected `[0.70, 1.10] m` window.

The ZED SDK path requests millimeter units from the camera and the deployment
node defaults to `--depth_units auto`, which treats integer/`16U` depth, encodings
containing `mm`, or raw median depth above `10` as millimeters and divides by
`1000`.

The ZED SDK settings are intentionally aligned with
`/juno/u/kedia/FoundationPose/live_tracking_with_ros_reset.py`: `HD1080`,
`NEURAL` depth, millimeter coordinate units, `svo_real_time_mode=True`, manual
exposure `25`, manual gain `40`, optional upside-down image/depth flipping, and
nearest-neighbor depth resizing. The student node additionally defaults to
retrieving depth directly at `160 x 90` for policy-loop speed. The non-blocking
ZED reader is rate-capped at `30 Hz`, matching the camera FPS; this should be a
no-op when `grab()` blocks normally, but it prevents accidental over-polling if
the SDK returns quickly. The ZED reader also caches the depth frame at
`160 x 90` before the policy loop sees it, so if a particular ZED Python binding
falls back to full-frame retrieval, the resize is kept in the camera reader path
instead of repeatedly hitting the control loop.
Set `--zed_retrieve_width 0 --zed_retrieve_height 0` to retrieve
full-resolution depth and resize in Python for a closer FoundationPose-style
path, at higher latency.

The policy proprio input matches the current Isaac Lab distillation setup:

- normalized full-URDF-limit joint positions, shape `29`
- raw joint velocities, shape `29`
- previous joint position targets, shape `29`

Total proprio shape is `87`. Deployment action post-processing clips commands to the same full URDF joint limits.

Deployment safety/default behavior:

- Joint command publishing is off by default.
- The node waits for the first depth, iiwa joint state, and Sharpa joint state messages before starting.
- The policy runs `--warmup_steps 30` forward passes by default to warm GPU kernels, then resets the recurrent hidden state before the real run.
- After warmup, the node runs a cached-frame startup policy benchmark. This prints median/p95 pure policy latency against the requested control-rate budget, so a slow ZED path is not confused with a slow model/GPU path.
- Warmup does not publish policy outputs. If needed, `--warmup_publish_current_targets` publishes current sensed joint positions as hold targets during warmup only.
- If joint publishing is disabled or a duration window has elapsed, `prev_action_targets` defaults to the current sensed joints. Use `--prev_targets_when_not_publishing computed` only if you explicitly want a dry-run rollout of the policy's hypothetical command history.
- Joint targets are clipped to full URDF joint limits. The extra arm-delta publish guard is off by default; pass `--max_arm_target_delta_deg <degrees>` if you want to block unexpectedly large arm target jumps during a cautious test.
- Joint-position proprioception is normalized with the full URDF joint limits, matching Isaac Sim training.
- Action smoothing uses the fixed training values: hand moving average `0.1`, arm moving average `0.1`, and arm velocity-delta scale `1.5`.
- In non-blocking ZED mode at 60 Hz control and 30 Hz camera FPS, repeated depth frames are expected. Status logs include `depth_frame_id` and `depth_reused=True/False`; proprio is still updated every control step. Reused depth frames reuse the cached preprocessed policy tensor, so the control loop does not repeat depth resizing/windowing/debug saving when the camera frame has not changed. Timing logs break out ZED grab time/period, depth read, depth preprocessing, debug saving, policy submit time, action GPU sync time, target computation, and pose publishing.

## Deployment Command

Example dry-run command with no joint command publishing:

```bash
PYTHONPATH=/home/tylerlum/github_repos/depthbasedRL:$PYTHONPATH \
python deployment/student_depth_policy_node_nonblocking.py \
  --checkpoint_path distillation_runs/09ctd_rot6d_medium_noise_camrand50mm5deg/checkpoints/student_latest.pt \
  --debug_depth_dir /tmp/depth_student_debug \
  --debug_depth_every_n 30 \
  --debug_depth_video_path /tmp/depth_student_debug/depth_debug.mp4 \
  --no-publish_joint_commands
```

To publish joint targets for only the first second:

```bash
PYTHONPATH=/home/tylerlum/github_repos/depthbasedRL:$PYTHONPATH \
python deployment/student_depth_policy_node_nonblocking.py \
  --checkpoint_path distillation_runs/09ctd_rot6d_medium_noise_camrand50mm5deg/checkpoints/student_latest.pt \
  --publish_joint_commands \
  --publish_joint_commands_duration_s 1.0
```

For a hardware hold-style warmup similar to `deployment/rl_policy_node.py`, add:

```bash
--warmup_publish_current_targets
```

This still never publishes policy actions during warmup; it only republishes the current sensed joint positions as hold targets.

For a DEXTRAH-matched policy candidate, prefer a checkpoint trained with `medium` depth noise and about `30mm/3deg` camera pose randomization. The existing `20mm/2deg` checkpoints are conservative; the `50mm/5deg` checkpoints are stress-test robust.

## ZED Non-Blocking Diagnostic

Use the standalone diagnostic before blaming the policy node:

```bash
.venv-isaacsim-py311/bin/python deployment/test_zed_nonblocking.py \
  --duration_s 30 \
  --consumer_hz 60 \
  --producer_preprocess none \
  --consumer_preprocess none
```

Useful variants:

- `--consumer_hz 120` checks whether the consumer loop can reuse cached frames at a higher rate.
- `--producer_preprocess policy` checks the intended deployment design: the ZED thread does policy-style resize/window/crop once per camera frame.
- `--consumer_preprocess policy` simulates the bad design where the consumer repeats preprocessing every control tick.
- `--zed_grab_hz 0` removes the producer rate cap so you can see whether over-polling the ZED SDK hurts timing.
- `--zed_depth_mode PERFORMANCE` or `--zed_depth_mode NEURAL_LIGHT` compares cheaper depth modes against the default `NEURAL`.
- `--save_dir /tmp/zed_nonblocking_debug` saves raw depth arrays plus window/crop PNGs for visual inspection.

## Relationship To `deployment/rl_policy_node.py`

The old real-world `rl_policy_node.py` uses a reliable deployment sequence that the student node now mirrors:

- wait for all required ROS observations
- initialize `prev_targets` from current joints
- warm up policy computation before the real loop
- reset recurrent state after warmup
- print loop health and block unsafe large arm target jumps

The student node differs intentionally:

- it has no privileged object-pose input; the depth image and proprio drive the student directly
- joint command publishing defaults to off for safe policy/pose visualization
- debug depth saving is first-class and saves metric arrays plus raw/resized/windowed/cropped visualizations
- predicted object pose is published to `/robot_frame/current_object_pose`, with full pose when the checkpoint has `object_rot6d`

## Aux Pose Output

The deployment node supports both aux-head formats:

- `object_pos` only: publishes predicted object position with identity orientation, or the orientation passed through `--position_only_quat_xyzw`.
- `object_pos` plus `object_rot6d`: converts the predicted 6D rotation to a valid rotation matrix/quaternion and publishes the full predicted object pose.

The pose is published in `robot_frame` by default to `/robot_frame/current_object_pose`.

## Future Item

Image observation delay is not currently modeled in deployment beyond using the latest received ROS image. We should add a training run with camera/observation delay once the noise and extrinsic randomization level is settled.

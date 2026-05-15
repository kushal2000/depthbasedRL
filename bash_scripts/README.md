# ZED And Student Policy Debug Scripts

Run these from anywhere; each script changes to the repository root first.

ZED defaults for the student policy node are `HD1080 @ 30 Hz` with `NEURAL`
depth and SDK retrieval directly at `160x90`. This is the deployment default
because it has looked more stable in real hardware testing. The standalone ZED
timing/debug scripts are still configurable and may default to faster diagnostic
modes. The scripts print the SDK-confirmed opened resolution/FPS and left-camera
intrinsics.

1. `01_zed_baseline_60hz_no_preprocess.sh`
   Baseline producer/consumer test at 60 Hz with no depth preprocessing.
   Check `tick_period_med/p95/max`: for a healthy 60 Hz consumer this should be near `16.7 ms`.

2. `02_zed_consumer_120hz_no_preprocess.sh`
   Checks whether the consumer loop can reuse cached ZED frames at 120 Hz.

3. `03_zed_producer_policy_preprocess_60hz.sh`
   Intended deployment design: policy-style depth preprocessing happens once per ZED frame in the producer process.

4. `04_zed_consumer_policy_preprocess_60hz.sh`
   Bad-path comparison: policy-style depth preprocessing happens every consumer tick.

5. `05_zed_no_grab_rate_cap_60hz.sh`
   Removes the producer rate cap to see whether over-polling the ZED SDK hurts timing.

6. `06_zed_save_depth_debug_images.sh`
   Saves raw metric depth arrays plus window/crop PNGs to `./zed_nonblocking_debug`.

7. `07_student_depth_policy_dry_run_nonblocking.sh [checkpoint]`
   Runs the full nonblocking student policy node for 30 seconds without publishing joint commands. This still publishes the predicted object pose to `/robot_frame/current_object_pose`.

8. `08_student_depth_policy_publish_1s_nonblocking.sh [checkpoint]`
   Runs the full nonblocking student policy node until Ctrl-C by default and publishes joint targets for `PUBLISH_DURATION_S` seconds. The default is one second.

9. `09_zed_multiprocess_60hz_no_preprocess.sh`
   Same baseline as script 1, but ZED capture runs in a separate process and shares the latest image through shared memory.

10. `10_zed_multiprocess_120hz_no_preprocess.sh`
    Same as script 9 with a 120 Hz consumer loop.

11. `11_zed_multiprocess_producer_policy_preprocess_60hz.sh`
    Multiprocess version of the intended deployment design: policy-style depth preprocessing happens once per ZED frame in the producer process.

12. `12_student_depth_policy_save_debug_nonblocking.sh [checkpoint]`
    Runs the full student node without joint commands and saves raw/resized/windowed/cropped depth debug files to `./depth_student_debug`.

13. `13_student_depth_policy_timing_60hz_nonblocking.sh [checkpoint]`
    Longer 60-second full-node timing run without debug image writes or joint commands.

14. `14_eval_depth_policy_32c_L_defaultcam.sh [checkpoint]`
    Isaac Sim depth-policy eval for the 32c L-peg checkpoint. It writes
    `metrics.csv`, `episodes.csv`, `summary.json`, pose-viewer HTML, and depth
    rollout/debug media. Set `SERVE_VISER=1` to add live viser visualization
    with camera frustums and point cloud.

20. `20_student_depth_32c_dry_run_nonblocking.sh [checkpoint]`
    Full 32c student policy dry run. Publishes object pose, but does not publish joint commands.

21. `21_student_depth_32c_save_debug_nonblocking.sh [checkpoint]`
    Same 32c dry run, plus raw/resized/windowed/cropped depth PNG/NPZ/mp4 debug output.

22. `22_student_depth_32c_timing_60hz_nonblocking.sh [checkpoint]`
    Longer 32c timing run without depth-video writes or joint commands.

23. `23_student_depth_32c_warmup_hold_publish_nonblocking.sh [checkpoint]`
    Publishes current sensed joint positions as hold targets during warmup only, then runs with policy joint commands disabled.

24. `24_student_depth_32c_publish_0p25s_nonblocking.sh [checkpoint]`
    Publishes 32c policy joint targets for 0.25 seconds, then keeps running without publishing.

25. `25_student_depth_32c_publish_1s_nonblocking.sh [checkpoint]`
    Publishes 32c policy joint targets for 1 second, then keeps running without publishing.

26. `26_student_depth_32c_publish_continuous_nonblocking.sh [checkpoint]`
    Publishes 32c policy joint targets continuously until Ctrl-C.

30. `30_depth_debug_roscore_local.sh`
    Starts a localhost-only `roscore` with `ROS_MASTER_URI=http://127.0.0.1:11311`.

31. `31_depth_debug_fake_robot.sh`
    Publishes fake `/iiwa/joint_states` and `/sharpa/joint_states`, subscribes to the policy command topics, and interpolates toward commands.

32. `32_depth_debug_fake_depth_from_file.sh /path/to/depth_file`
    Publishes a fixed or replayed metric depth image on `/zed/zed_node/depth/depth_registered` plus optional CameraInfo. Supports `.npz`, `.npy`, image files, and mp4/avi/mov.

33. `33_depth_debug_student_ros_topic_dry_run.sh [checkpoint]`
    Runs the student policy against the fake ROS depth topic and fake robot, publishes predicted object pose, does not publish joint commands, and records a rollout NPZ on shutdown.

34. `34_depth_debug_student_ros_topic_publish_3s.sh [checkpoint]`
    Same as 33, but publishes joint commands for `PUBLISH_DURATION_S=3` seconds by default.

35. `35_depth_debug_visualization_with_depth.sh`
    Runs the listener-only Viser visualization with optional live depth image display. Set `LOAD_POINT_CLOUD=1` to also show a point cloud when CameraInfo is available.

Recommended student-policy test order:

1. `01_zed_baseline_60hz_no_preprocess.sh`
2. `02_zed_consumer_120hz_no_preprocess.sh`
3. `07_student_depth_policy_dry_run_nonblocking.sh`
4. `12_student_depth_policy_save_debug_nonblocking.sh`
5. `13_student_depth_policy_timing_60hz_nonblocking.sh`
6. `08_student_depth_policy_publish_1s_nonblocking.sh`

Recommended 32c real-world rollout order, progressively allowing more output:

1. `20_student_depth_32c_dry_run_nonblocking.sh`
2. `21_student_depth_32c_save_debug_nonblocking.sh`
3. `22_student_depth_32c_timing_60hz_nonblocking.sh`
4. `23_student_depth_32c_warmup_hold_publish_nonblocking.sh`
5. `24_student_depth_32c_publish_0p25s_nonblocking.sh`
6. `25_student_depth_32c_publish_1s_nonblocking.sh`
7. `26_student_depth_32c_publish_continuous_nonblocking.sh`

The `20+` scripts default to:

`/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/32c_juno_a5000_L_defaultcam_q1_medium_noise_camrand20mm2deg/checkpoints/student_best.pt`

The policy-publishing scripts default to `MAX_ARM_TARGET_DELTA_DEG=0`, matching
the node default and disabling the extra arm-delta publish guard. Set
`MAX_ARM_TARGET_DELTA_DEG=...` only if you explicitly want to block unexpectedly
large arm target jumps during a cautious test.

Depth-policy eval examples:

- `bash_scripts/14_eval_depth_policy_32c_L_defaultcam.sh`
- `SERVE_VISER=1 bash_scripts/14_eval_depth_policy_32c_L_defaultcam.sh`
- `WANDB=1 NUM_ENVS=64 NUM_COMPLETED_EPISODES=256 bash_scripts/14_eval_depth_policy_32c_L_defaultcam.sh`

The student-policy scripts default to
`/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/10_juno_rot6d_medium_noise_camrand20mm2deg_256env_48h/checkpoints/student_latest.pt`
when it exists, otherwise they fall back to the local `10_local` checkpoint. You
can pass a checkpoint as the first argument or set `DEFAULT_STUDENT_CHECKPOINT`.

Useful env overrides:

- `RUN_DURATION_S=120 bash_scripts/07_student_depth_policy_dry_run_nonblocking.sh`
- `DEBUG_DIR=./my_depth_debug bash_scripts/12_student_depth_policy_save_debug_nonblocking.sh`
- `PUBLISH_DURATION_S=3 bash_scripts/08_student_depth_policy_publish_1s_nonblocking.sh`
- `PUBLISH_DURATION_S=-1 bash_scripts/08_student_depth_policy_publish_1s_nonblocking.sh`
- `PUBLISH_DURATION_S=0.5 bash_scripts/24_student_depth_32c_publish_0p25s_nonblocking.sh`
- `MAX_ARM_TARGET_DELTA_DEG=10 bash_scripts/25_student_depth_32c_publish_1s_nonblocking.sh`

Local fake-ROS safety/debug sequence:

1. Terminal A: `bash_scripts/30_depth_debug_roscore_local.sh`
2. Terminal B: `bash_scripts/31_depth_debug_fake_robot.sh`
3. Terminal C: `bash_scripts/32_depth_debug_fake_depth_from_file.sh /path/to/depth.npz`
4. Terminal D: `bash_scripts/35_depth_debug_visualization_with_depth.sh`
5. Terminal E: `RUN_DURATION_S=30 bash_scripts/33_depth_debug_student_ros_topic_dry_run.sh`

All `30+` scripts source `bash_scripts/depth_deploy_debug_env.sh`, activate
`${DEPTH_DEPLOY_CONDA_ENV:-simtoolreal_ros_env}`, and force localhost ROS
networking. They should not talk to `bohg-ws-2`, `bohg-ws-19`, or the real
robot unless you explicitly override the environment after sourcing.

Rollout recordings from scripts 33/34 are saved once on shutdown under
`RECORD_DIR`. Inspect one with:

```bash
python deployment/visualize_student_depth_rollout.py \
  --recording ./student_depth_ros_topic_recording/<recording>.npz \
  --object-name peg_L
```

HD1080 comparison example:

```bash
python deployment/test_zed_multiprocess.py \
  --zed_resolution HD1080 \
  --zed_camera_fps 30 \
  --zed_grab_hz 30 \
  --duration_s 30 \
  --consumer_hz 60 \
  --producer_preprocess policy
```

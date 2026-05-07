# ZED And Student Policy Debug Scripts

Run these from anywhere; each script changes to the repository root first.

1. `01_zed_baseline_60hz_no_preprocess.sh`
   Baseline producer/consumer test at 60 Hz with no depth preprocessing.
   Check `tick_period_med/p95/max`: for a healthy 60 Hz consumer this should be near `16.7 ms`.

2. `02_zed_consumer_120hz_no_preprocess.sh`
   Checks whether the consumer loop can reuse cached ZED frames at 120 Hz.

3. `03_zed_producer_policy_preprocess_60hz.sh`
   Intended deployment design: policy-style depth preprocessing happens once per ZED frame in the producer thread.

4. `04_zed_consumer_policy_preprocess_60hz.sh`
   Bad-path comparison: policy-style depth preprocessing happens every consumer tick.

5. `05_zed_no_grab_rate_cap_60hz.sh`
   Removes the producer rate cap to see whether over-polling the ZED SDK hurts timing.

6. `06_zed_save_depth_debug_images.sh`
   Saves raw metric depth arrays plus window/crop PNGs to `/tmp/zed_nonblocking_debug`.

7. `07_student_depth_policy_dry_run_nonblocking.sh [checkpoint]`
   Runs the full nonblocking student policy node without publishing joint commands.

8. `08_student_depth_policy_publish_1s_nonblocking.sh [checkpoint]`
   Runs the full nonblocking student policy node and publishes joint targets for one second.

9. `09_zed_multiprocess_60hz_no_preprocess.sh`
   Same baseline as script 1, but ZED capture runs in a separate process and shares the latest image through shared memory.

10. `10_zed_multiprocess_120hz_no_preprocess.sh`
    Same as script 9 with a 120 Hz consumer loop.

11. `11_zed_multiprocess_producer_policy_preprocess_60hz.sh`
    Multiprocess version of the intended deployment design: policy-style depth preprocessing happens once per ZED frame in the producer process.

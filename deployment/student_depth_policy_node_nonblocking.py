#!/usr/bin/env python
"""Run the depth-student ROS node with non-blocking ZED capture enabled.

This entrypoint is intentionally tiny so the deployment logic stays in
student_depth_policy_node.py. It only defaults the node to --zed_nonblocking,
which captures ZED depth in a background thread and lets the policy loop reuse
the latest cached depth frame instead of blocking on the camera FPS.
"""

from __future__ import annotations

import sys

from student_depth_policy_node import main


if __name__ == "__main__":
    if "--zed_nonblocking" not in sys.argv and "--no-zed_nonblocking" not in sys.argv:
        sys.argv.append("--zed_nonblocking")
    main()

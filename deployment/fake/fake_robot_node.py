#!/usr/bin/env python
"""Simple fake robot ROS node for local policy deployment tests.

It publishes joint states and optionally interpolates them toward commands sent
by the policy node. This is not a physics simulator; it is a low-overhead way to
test proprioception, command publishing, and policy timing without hardware.
"""

from __future__ import annotations

import argparse
import time
from typing import Literal

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from termcolor import colored


NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 22

DEFAULT_ARM_Q = np.array([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 1.308], dtype=np.float64)
DEFAULT_HAND_Q = np.zeros(NUM_HAND_JOINTS, dtype=np.float64)

IIWA_JOINT_NAMES = [f"iiwa_joint_{i}" for i in range(1, NUM_ARM_JOINTS + 1)]
SHARPA_JOINT_NAMES = [f"joint_{i}.0" for i in range(NUM_HAND_JOINTS)]


def warn(message: str) -> None:
    print(colored(message, "yellow"), flush=True)


def info(message: str) -> None:
    print(colored(message, "green"), flush=True)


def warn_every(message: str, n_seconds: float, key=None) -> None:
    if not hasattr(warn_every, "_last_times"):
        warn_every._last_times = {}
    key = key or message
    last_times = warn_every._last_times
    last_time = last_times.get(key, 0.0)
    if time.time() - last_time > n_seconds:
        warn(message)
        last_times[key] = time.time()


def _clip_delta_by_norm(delta: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0.0:
        return delta
    norm = float(np.linalg.norm(delta))
    if norm > max_norm:
        return max_norm * delta / norm
    return delta


class FakeRobotNode:
    def __init__(self, args: argparse.Namespace):
        rospy.init_node("fake_robot_node")
        self.args = args

        self.iiwa_joint_cmd = None
        self.sharpa_joint_cmd = None

        self.iiwa_pub = rospy.Publisher(args.iiwa_joint_state_topic, JointState, queue_size=1)
        self.sharpa_pub = rospy.Publisher(args.sharpa_joint_state_topic, JointState, queue_size=1)
        self.iiwa_cmd_sub = rospy.Subscriber(
            args.iiwa_joint_cmd_topic, JointState, self.iiwa_joint_cmd_callback, queue_size=1
        )
        self.sharpa_cmd_sub = rospy.Subscriber(
            args.sharpa_joint_cmd_topic, JointState, self.sharpa_joint_cmd_callback, queue_size=1
        )

        self.iiwa_joint_q = self._initial_arm_q(args)
        self.sharpa_joint_q = self._initial_hand_q(args)
        self.initial_iiwa_joint_q = self.iiwa_joint_q.copy()
        self.initial_sharpa_joint_q = self.sharpa_joint_q.copy()
        self.iiwa_joint_qd = np.zeros(NUM_ARM_JOINTS, dtype=np.float64)
        self.sharpa_joint_qd = np.zeros(NUM_HAND_JOINTS, dtype=np.float64)

        if not args.wait_for_commands:
            self.iiwa_joint_cmd = self.iiwa_joint_q.copy()
            self.sharpa_joint_cmd = self.sharpa_joint_q.copy()
            warn("Not waiting for joint commands; initialized commands to current fake joint state.")

        self.dt = 1.0 / float(args.rate_hz)
        self.rate = rospy.Rate(args.rate_hz)
        self.start_time = time.time()
        self.last_status_time = 0.0

    @staticmethod
    def _initial_arm_q(args: argparse.Namespace) -> np.ndarray:
        if args.initial_arm_q is not None:
            q = np.asarray(args.initial_arm_q, dtype=np.float64)
        elif args.initial_pose == "zeros":
            q = np.zeros(NUM_ARM_JOINTS, dtype=np.float64)
        elif args.initial_pose == "student_default":
            q = DEFAULT_ARM_Q.copy()
        else:
            raise ValueError(f"Unsupported initial_pose={args.initial_pose!r}")
        if q.shape != (NUM_ARM_JOINTS,):
            raise ValueError(f"Expected initial arm q shape {(NUM_ARM_JOINTS,)}, got {q.shape}")
        return q

    @staticmethod
    def _initial_hand_q(args: argparse.Namespace) -> np.ndarray:
        if args.initial_hand_q is not None:
            q = np.asarray(args.initial_hand_q, dtype=np.float64)
        else:
            q = DEFAULT_HAND_Q.copy()
        if q.shape != (NUM_HAND_JOINTS,):
            raise ValueError(f"Expected initial hand q shape {(NUM_HAND_JOINTS,)}, got {q.shape}")
        return q

    def iiwa_joint_cmd_callback(self, msg: JointState) -> None:
        cmd = np.asarray(msg.position, dtype=np.float64)
        if cmd.shape != (NUM_ARM_JOINTS,):
            warn_every(f"Ignoring iiwa cmd with shape {cmd.shape}; expected {(NUM_ARM_JOINTS,)}", 1.0)
            return
        self.iiwa_joint_cmd = cmd

    def sharpa_joint_cmd_callback(self, msg: JointState) -> None:
        cmd = np.asarray(msg.position, dtype=np.float64)
        if cmd.shape != (NUM_HAND_JOINTS,):
            warn_every(f"Ignoring sharpa cmd with shape {cmd.shape}; expected {(NUM_HAND_JOINTS,)}", 1.0)
            return
        self.sharpa_joint_cmd = cmd

    def update_joint_states(self) -> None:
        if self.iiwa_joint_cmd is None or self.sharpa_joint_cmd is None:
            warn_every(
                f"Waiting for commands: iiwa={self.iiwa_joint_cmd is not None}, sharpa={self.sharpa_joint_cmd is not None}",
                n_seconds=1.0,
            )
            self.iiwa_joint_qd[:] = 0.0
            self.sharpa_joint_qd[:] = 0.0
            return

        mode: Literal["interpolate", "pd_control"] = self.args.mode
        if mode == "interpolate":
            delta_iiwa = _clip_delta_by_norm(
                self.iiwa_joint_cmd - self.iiwa_joint_q,
                self.args.max_delta_iiwa,
            )
            delta_sharpa = _clip_delta_by_norm(
                self.sharpa_joint_cmd - self.sharpa_joint_q,
                self.args.max_delta_sharpa,
            )
            self.iiwa_joint_q += delta_iiwa
            self.sharpa_joint_q += delta_sharpa
            self.iiwa_joint_qd = delta_iiwa / self.dt
            self.sharpa_joint_qd = delta_sharpa / self.dt
        elif mode == "pd_control":
            p_gain = self.args.p_gain
            d_gain = self.args.d_gain
            iiwa_qdd = p_gain * (self.iiwa_joint_cmd - self.iiwa_joint_q) - d_gain * self.iiwa_joint_qd
            sharpa_qdd = p_gain * (self.sharpa_joint_cmd - self.sharpa_joint_q) - d_gain * self.sharpa_joint_qd
            self.iiwa_joint_qd += iiwa_qdd * self.dt
            self.sharpa_joint_qd += sharpa_qdd * self.dt
            self.iiwa_joint_q += self.iiwa_joint_qd * self.dt
            self.sharpa_joint_q += self.sharpa_joint_qd * self.dt
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def publish_joint_states(self) -> None:
        stamp = rospy.Time.now()

        iiwa_msg = JointState()
        iiwa_msg.header.stamp = stamp
        iiwa_msg.name = IIWA_JOINT_NAMES
        iiwa_msg.position = self.iiwa_joint_q.tolist()
        iiwa_msg.velocity = self.iiwa_joint_qd.tolist()
        self.iiwa_pub.publish(iiwa_msg)

        sharpa_msg = JointState()
        sharpa_msg.header.stamp = stamp
        sharpa_msg.name = SHARPA_JOINT_NAMES
        sharpa_msg.position = self.sharpa_joint_q.tolist()
        sharpa_msg.velocity = self.sharpa_joint_qd.tolist()
        self.sharpa_pub.publish(sharpa_msg)

    def _print_status(self) -> None:
        now = time.time()
        if now - self.last_status_time < self.args.status_interval_s:
            return
        self.last_status_time = now
        info(
            "[fake_robot_node] "
            f"iiwa_cmd={self.iiwa_joint_cmd is not None} sharpa_cmd={self.sharpa_joint_cmd is not None} "
            f"iiwa_delta_norm={np.linalg.norm((self.iiwa_joint_cmd if self.iiwa_joint_cmd is not None else self.iiwa_joint_q) - self.iiwa_joint_q):.4f} "
            f"sharpa_delta_norm={np.linalg.norm((self.sharpa_joint_cmd if self.sharpa_joint_cmd is not None else self.sharpa_joint_q) - self.sharpa_joint_q):.4f} "
            f"iiwa_from_start={np.linalg.norm(self.iiwa_joint_q - self.initial_iiwa_joint_q):.4f} "
            f"sharpa_from_start={np.linalg.norm(self.sharpa_joint_q - self.initial_sharpa_joint_q):.4f}"
        )

    def run(self) -> None:
        loop_no_sleep_dts, loop_dts = [], []
        while not rospy.is_shutdown():
            if self.args.run_duration_s >= 0.0 and time.time() - self.start_time >= self.args.run_duration_s:
                info(f"Reached run_duration_s={self.args.run_duration_s:.2f}; shutting down fake robot.")
                break

            start_time = rospy.Time.now()
            self.update_joint_states()
            self.publish_joint_states()
            self._print_status()

            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()

            loop_no_sleep_dt = (before_sleep_time - start_time).to_sec()
            loop_dt = (after_sleep_time - start_time).to_sec()
            loop_no_sleep_dts.append(loop_no_sleep_dt)
            loop_dts.append(loop_dt)

            print_steps = max(1, int(self.args.fps_print_interval_s / self.dt))
            if len(loop_dts) == print_steps:
                loop_dt_array = np.array(loop_dts)
                loop_no_sleep_dt_array = np.array(loop_no_sleep_dts)
                fps_array = 1.0 / np.clip(loop_dt_array, 1e-9, None)
                fps_no_sleep_array = 1.0 / np.clip(loop_no_sleep_dt_array, 1e-9, None)
                print(
                    "[fake_robot_node] FPS "
                    f"sleep mean/med={np.mean(fps_array):.1f}/{np.median(fps_array):.1f}; "
                    f"no_sleep mean/med={np.mean(fps_no_sleep_array):.1f}/{np.median(fps_no_sleep_array):.1f}",
                    flush=True,
                )
                loop_no_sleep_dts, loop_dts = [], []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate_hz", type=float, default=60.0)
    parser.add_argument("--run_duration_s", type=float, default=-1.0)
    parser.add_argument("--status_interval_s", type=float, default=1.0)
    parser.add_argument("--fps_print_interval_s", type=float, default=5.0)
    parser.add_argument("--mode", choices=("interpolate", "pd_control"), default="interpolate")
    parser.add_argument("--max_delta_iiwa", type=float, default=0.1)
    parser.add_argument("--max_delta_sharpa", type=float, default=0.1)
    parser.add_argument("--p_gain", type=float, default=10.0)
    parser.add_argument("--d_gain", type=float, default=0.0)
    parser.add_argument("--initial_pose", choices=("student_default", "zeros"), default="student_default")
    parser.add_argument("--initial_arm_q", type=float, nargs=NUM_ARM_JOINTS, default=None)
    parser.add_argument("--initial_hand_q", type=float, nargs=NUM_HAND_JOINTS, default=None)
    parser.add_argument("--iiwa_joint_state_topic", default="/iiwa/joint_states")
    parser.add_argument("--sharpa_joint_state_topic", default="/sharpa/joint_states")
    parser.add_argument("--iiwa_joint_cmd_topic", default="/iiwa/joint_cmd")
    parser.add_argument("--sharpa_joint_cmd_topic", default="/sharpa/joint_cmd")
    parser.set_defaults(wait_for_commands=False)
    wait_group = parser.add_mutually_exclusive_group()
    wait_group.add_argument("--wait_for_commands", dest="wait_for_commands", action="store_true")
    wait_group.add_argument("--no-wait_for_commands", dest="wait_for_commands", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        FakeRobotNode(parse_args()).run()
    except rospy.ROSInterruptException:
        pass

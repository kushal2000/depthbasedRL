#!/usr/bin/env python

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import rospy
import tyro
import viser
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, JointState
from termcolor import colored
from viser.extras import ViserUrdf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dextoolbench.metadata import ALL_OBJECT_NAMES
from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.utils.utils import get_repo_root_dir
import peg_in_hole.objects  # noqa: F401 - registers peg/peg_L/L_peg into NAME_TO_OBJECT

VISUALIZATION_OBJECT_NAMES = sorted(set(ALL_OBJECT_NAMES) | set(NAME_TO_OBJECT))

T_W_R = np.eye(4)
T_W_R[:3, 3] = np.array([0.0, 0.8, 0.0])


def warn(message: str):
    print(colored(message, "yellow"))


def warn_every(message: str, n_seconds: float, key=None):
    """
    Print a warning message at most once every n_seconds per unique key.
    Stores state inside the function itself (no globals).
    """
    if not hasattr(warn_every, "_last_times"):
        warn_every._last_times = {}  # create on first call

    key = key or message
    last_times = warn_every._last_times
    last_time = last_times.get(key, 0)

    if time.time() - last_time > n_seconds:
        warn(message)
        last_times[key] = time.time()


def info(message: str):
    print(colored(message, "green"))


NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 22

BLUE_RGB = (0, 0, 255)
GREEN_RGBA = (0, 255, 0, 0.5)
LIGHT_BLUE_RGBA = (80, 200, 255, 0.55)
BLACK_RGBA = (0, 0, 0, 1.0)

AXES_LENGTH = 0.1
AXES_RADIUS = 0.001
DEFAULT_DEPTH_NEAR_M = 0.70
DEFAULT_DEPTH_FAR_M = 1.10
DEFAULT_CAMERA_FRAME = "/student_depth_camera"
DEFAULT_CAMERA_IMAGE_TOPIC = "/zed/zed_node/depth/depth_registered"
DEFAULT_CAMERA_INFO_TOPIC = "/zed/zed_node/rgb/camera_info"
DEFAULT_OBJECT_POSE_TOPIC = "/robot_frame/current_object_pose"
DEFAULT_PREDICTED_OBJECT_POSE_TOPIC = "/robot_frame/predicted_object_pose"
# Sim default student-camera pose in the world frame. This is only used for
# optional depth debugging in Viser; object/robot visualization is unchanged.
DEFAULT_CAMERA_POS_WORLD = (-0.5002050422666431, -0.6385715691360607, 1.0201893282998005)
DEFAULT_CAMERA_QUAT_WXYZ = (-0.5314110448277682, 0.833810802683381, -0.14035163049226862, 0.051606846267884886)

# Viser Server global variable
SERVER = viser.ViserServer()


@SERVER.on_client_connect
def _(client: viser.ClientHandle) -> None:
    """For each client that connects, set the camera pose."""
    with client.atomic():
        client.camera.position = (0.0, -1.0, 1.03)
        client.camera.look_at = (0.0, 0.0, 0.53)
        # client.camera.wxyz = (w, x, y, z)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4), T.shape
    n_pts = points.shape[0]
    assert points.shape == (n_pts, 3), points.shape

    return (T[:3, :3] @ points.T + T[:3, 3][:, None]).T


def _decode_depth_image(msg: Image, depth_units: str) -> np.ndarray:
    """Decode a ROS depth image into metric meters without requiring cv_bridge."""

    encoding = msg.encoding.lower()
    if encoding in ("32fc1", "type_32fc1"):
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        inferred_units = "m"
    elif encoding in ("16uc1", "mono16"):
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32)
        inferred_units = "mm"
    else:
        raise ValueError(f"Unsupported depth image encoding={msg.encoding!r}; expected 32FC1 or 16UC1.")

    units = depth_units
    finite = np.isfinite(depth)
    if units == "auto":
        units = inferred_units
        if finite.any() and float(np.nanmedian(depth[finite])) > 10.0:
            units = "mm"

    if units == "m":
        depth_m = depth.astype(np.float32, copy=True)
    elif units == "mm":
        depth_m = depth.astype(np.float32, copy=False) / 1000.0
    else:
        raise ValueError(f"depth_units must be auto, m, or mm; got {depth_units!r}")

    depth_m[~np.isfinite(depth_m)] = 0.0
    return depth_m


def _depth_to_rgb(depth_m: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.clip((depth - near_m) / max(far_m - near_m, 1e-6), 0.0, 1.0)
    gray = (255.0 * (1.0 - normalized)).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    rgb[~valid] = np.array([30, 30, 30], dtype=np.uint8)
    return rgb


def _viser_frustum_image(rgb: np.ndarray, *, flip_y: bool) -> np.ndarray:
    image = np.asarray(rgb)
    return np.flipud(image) if flip_y else image


def _points_from_depth(
    depth_m: np.ndarray,
    K: np.ndarray,
    stride: int,
    max_points: int,
    near_m: float,
    far_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[yy, xx]
    valid = np.isfinite(z) & (z > 0.0)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    x = (xx.astype(np.float32) - float(K[0, 2])) / float(K[0, 0]) * z
    y = (yy.astype(np.float32) - float(K[1, 2])) / float(K[1, 1]) * z
    points = np.stack([x, y, z], axis=-1)[valid].reshape(-1, 3)
    colors = _depth_to_rgb(z, near_m, far_m)[valid].reshape(-1, 3)
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[idx]
        colors = colors[idx]
    return points.astype(np.float32, copy=False), colors.astype(np.uint8, copy=False)


@dataclass
class RosSnapshot:
    iiwa_joint_cmd: Optional[np.ndarray]
    sharpa_joint_cmd: Optional[np.ndarray]
    iiwa_joint_state: Optional[np.ndarray]
    sharpa_joint_state: Optional[np.ndarray]
    object_pose: Optional[np.ndarray]
    predicted_object_pose: Optional[np.ndarray]
    goal_object_pose: Optional[np.ndarray]
    depth_image_m: Optional[np.ndarray]
    depth_stamp: Optional[rospy.Time]
    camera_K: Optional[np.ndarray]

    @classmethod
    def make_with_nones(cls) -> RosSnapshot:
        return cls(
            iiwa_joint_cmd=None,
            sharpa_joint_cmd=None,
            iiwa_joint_state=None,
            sharpa_joint_state=None,
            object_pose=None,
            predicted_object_pose=None,
            goal_object_pose=None,
            depth_image_m=None,
            depth_stamp=None,
            camera_K=None,
        )

    def make_copy_with_defaults(self) -> RosSnapshot:
        if self.iiwa_joint_cmd is None:
            warn_every("iiwa_joint_cmd is None", n_seconds=1.0)
            iiwa_joint_cmd = np.zeros(NUM_ARM_JOINTS)
        else:
            iiwa_joint_cmd = self.iiwa_joint_cmd

        if self.sharpa_joint_cmd is None:
            warn_every("sharpa_joint_cmd is None", n_seconds=1.0)
            sharpa_joint_cmd = (
                np.zeros(NUM_HAND_JOINTS) + 0.5
            )  # Not 0 so it is obvious when not there
        else:
            sharpa_joint_cmd = self.sharpa_joint_cmd

        if self.iiwa_joint_state is None:
            warn_every("iiwa_joint_state is None", n_seconds=1.0)
            iiwa_joint_state = np.zeros(NUM_ARM_JOINTS)
        else:
            iiwa_joint_state = self.iiwa_joint_state

        if self.sharpa_joint_state is None:
            warn_every("sharpa_joint_state is None", n_seconds=1.0)
            sharpa_joint_state = np.zeros(NUM_HAND_JOINTS)
        else:
            sharpa_joint_state = self.sharpa_joint_state

        if self.object_pose is None:
            warn_every("object_pose is None", n_seconds=1.0)
            object_pose = np.eye(4)
            object_pose[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            object_pose = self.object_pose

        if self.predicted_object_pose is None:
            warn_every("predicted_object_pose is None", n_seconds=1.0)
            predicted_object_pose = np.eye(4)
            predicted_object_pose[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            predicted_object_pose = self.predicted_object_pose

        if self.goal_object_pose is None:
            warn_every("goal_object_pose is None", n_seconds=1.0)
            goal_object_pose = np.eye(4)
            goal_object_pose[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            goal_object_pose = self.goal_object_pose

        return RosSnapshot(
            iiwa_joint_cmd=iiwa_joint_cmd,
            sharpa_joint_cmd=sharpa_joint_cmd,
            iiwa_joint_state=iiwa_joint_state,
            sharpa_joint_state=sharpa_joint_state,
            object_pose=object_pose,
            predicted_object_pose=predicted_object_pose,
            goal_object_pose=goal_object_pose,
            depth_image_m=self.depth_image_m,
            depth_stamp=self.depth_stamp,
            camera_K=self.camera_K,
        )


class VisualizationNode:
    def __init__(self, args: VisualizationNodeArgs):
        self.args = args
        self.depth_frustum = None
        self.depth_point_cloud = None
        self._last_depth_stamp = None

        # ROS setup
        rospy.init_node("visualization_node")

        # Store snapshot
        self.ros_snapshot = RosSnapshot.make_with_nones()

        # Subscribers
        self.initialize_ros_subscribers()

        # Initialize Viser
        self.initialize_viser(object_name=args.object_name)

        # Set update rate to 10Hz
        self.rate_hz = 10
        self.dt = 1 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

    def initialize_ros_subscribers(self):
        self.iiwa_sub = rospy.Subscriber(
            "/iiwa/joint_states",
            JointState,
            self.iiwa_joint_state_callback,
            queue_size=1,
        )
        self.sharpa_sub = rospy.Subscriber(
            "/sharpa/joint_states",
            JointState,
            self.sharpa_joint_state_callback,
            queue_size=1,
        )
        self.iiwa_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self.iiwa_joint_cmd_callback, queue_size=1
        )
        self.sharpa_cmd_sub = rospy.Subscriber(
            "/sharpa/joint_cmd",
            JointState,
            self.sharpa_joint_cmd_callback,
            queue_size=1,
        )
        self.object_pose_sub = rospy.Subscriber(
            self.args.object_pose_topic,
            PoseStamped,
            self.object_pose_callback,
            queue_size=1,
        )
        self.predicted_object_pose_sub = rospy.Subscriber(
            self.args.predicted_object_pose_topic,
            PoseStamped,
            self.predicted_object_pose_callback,
            queue_size=1,
        )
        self.goal_object_pose_sub = rospy.Subscriber(
            "/robot_frame/goal_object_pose",
            Pose,
            self.goal_object_pose_callback,
            queue_size=1,
        )
        if self.args.load_depth_image or self.args.load_point_cloud:
            self.depth_image_sub = rospy.Subscriber(
                self.args.depth_topic,
                Image,
                self.depth_image_callback,
                queue_size=1,
            )
            self.camera_info_sub = rospy.Subscriber(
                self.args.camera_info_topic,
                CameraInfo,
                self.camera_info_callback,
                queue_size=1,
            )
            info(
                f"Subscribing to depth_topic={self.args.depth_topic} "
                f"camera_info_topic={self.args.camera_info_topic}"
            )

    def initialize_viser(self, object_name: str):
        SERVER.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        # Create a real robot (simulating real robot) and a command robot (visualizing commands)
        # Load robot URDF with a fixed base
        robot_urdf_path = (
            get_repo_root_dir()
            / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        )
        assert robot_urdf_path.exists(), f"robot_urdf_path not found: {robot_urdf_path}"

        SERVER.scene.add_frame(
            "/robot/state",
            position=(0, 0.8, 0),
            wxyz=(1, 0, 0, 0),
            show_axes=False,
        )
        SERVER.scene.add_frame(
            "/robot/cmd",
            position=(0, 0.8, 0),
            wxyz=(1, 0, 0, 0),
            show_axes=False,
        )
        self.robot_viser = ViserUrdf(
            SERVER, robot_urdf_path, root_node_name="/robot/state"
        )
        self.robot_cmd_viser = ViserUrdf(
            SERVER,
            robot_urdf_path,
            root_node_name="/robot/cmd",
            mesh_color_override=BLUE_RGB,
        )

        # Set the cmd robot to be translucent
        # NOTE: To change opacity, you must create ViserUrdf with mesh_color_override
        for robot_cmd_mesh in self.robot_cmd_viser._meshes:
            assert isinstance(robot_cmd_mesh, viser.MeshHandle), (
                f"robot_cmd_mesh is not a MeshHandle, you must create ViserUrdf with mesh_color_override: {type(robot_cmd_mesh)}"
            )
            robot_cmd_mesh.opacity = 0.5

        LOAD_TABLE = True
        if LOAD_TABLE:
            # table_urdf_path = get_repo_root_dir() / "assets/urdf/table_narrow.urdf"
            table_urdf_path = (
                get_repo_root_dir() / "assets/urdf/table_narrow_whiteboard.urdf"
            )
            assert table_urdf_path.exists(), (
                f"table_urdf_path not found: {table_urdf_path}"
            )

            SERVER.scene.add_frame(
                "/table",
                position=(0.0, 0.0, 0.38),
                wxyz=(1, 0, 0, 0),
                show_axes=False,
            )
            table_viser = ViserUrdf(
                SERVER,
                table_urdf_path,
                root_node_name="/table",
                mesh_color_override=BLACK_RGBA,
            )

            TRANSPARENT_TABLE = True
            if TRANSPARENT_TABLE:
                # NOTE: To change opacity, you must create ViserUrdf with mesh_color_override
                # Make the table transparent
                # Change the color of each link (including the base)
                for table_mesh in table_viser._meshes:
                    assert isinstance(table_mesh, viser.MeshHandle), (
                        f"table_mesh is not a MeshHandle, you must create ViserUrdf with mesh_color_override: {type(table_mesh)}"
                    )
                    # table_mesh.color = (0, 0, 0)
                    table_mesh.opacity = 0.5

        # Load the object mesh
        FAR_AWAY_OBJECT_POSITION = np.ones(3)

        if object_name not in NAME_TO_OBJECT:
            raise ValueError(
                f"Unknown object_name={object_name!r}. Options: {', '.join(VISUALIZATION_OBJECT_NAMES)}"
            )
        object_urdf = NAME_TO_OBJECT[object_name].urdf_path
        goal_object_urdf = object_urdf
        assert object_urdf.exists(), f"object_urdf does not exist: {object_urdf}"

        self.object_viser = SERVER.scene.add_frame(
            "/object",
            position=FAR_AWAY_OBJECT_POSITION,
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )
        self.object_urdf_viser = ViserUrdf(
            SERVER, object_urdf, root_node_name="/object"
        )
        self.goal_object_viser = SERVER.scene.add_frame(
            "/goal_object",
            position=FAR_AWAY_OBJECT_POSITION + np.array([0.2, 0.2, 0.2]),
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )
        self.goal_object_urdf_viser = ViserUrdf(
            SERVER,
            goal_object_urdf,
            root_node_name="/goal_object",
            mesh_color_override=GREEN_RGBA,
        )
        self.predicted_object_viser = SERVER.scene.add_frame(
            "/predicted_object",
            position=FAR_AWAY_OBJECT_POSITION + np.array([0.4, 0.4, 0.4]),
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )
        self.predicted_object_urdf_viser = ViserUrdf(
            SERVER,
            object_urdf,
            root_node_name="/predicted_object",
            mesh_color_override=LIGHT_BLUE_RGBA,
        )

        # Set the robot to a default pose
        DEFAULT_ARM_Q = np.zeros(NUM_ARM_JOINTS)
        DEFAULT_HAND_Q = np.zeros(NUM_HAND_JOINTS)
        assert DEFAULT_ARM_Q.shape == (NUM_ARM_JOINTS,)
        assert DEFAULT_HAND_Q.shape == (NUM_HAND_JOINTS,)
        DEFAULT_Q = np.concatenate([DEFAULT_ARM_Q, DEFAULT_HAND_Q])
        self.robot_viser.update_cfg(DEFAULT_Q)
        self.robot_cmd_viser.update_cfg(DEFAULT_Q)

        if self.args.load_depth_image or self.args.load_point_cloud:
            self.camera_frame = SERVER.scene.add_frame(
                DEFAULT_CAMERA_FRAME,
                position=tuple(self.args.camera_pos_world),
                wxyz=tuple(self.args.camera_quat_wxyz),
                show_axes=True,
                axes_length=0.08,
                axes_radius=0.002,
            )
            self.depth_frustum = SERVER.scene.add_camera_frustum(
                f"{DEFAULT_CAMERA_FRAME}/depth_image",
                fov=0.7,
                aspect=16.0 / 9.0,
                scale=0.25,
                image=np.zeros((90, 160, 3), dtype=np.uint8),
            )
            if self.args.load_point_cloud:
                self.depth_point_cloud = SERVER.scene.add_point_cloud(
                    f"{DEFAULT_CAMERA_FRAME}/point_cloud",
                    points=np.empty((0, 3), dtype=np.float32),
                    colors=np.empty((0, 3), dtype=np.uint8),
                    point_size=self.args.point_size,
                )

    def iiwa_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.ros_snapshot.iiwa_joint_cmd = np.array(msg.position)

    def sharpa_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.ros_snapshot.sharpa_joint_cmd = np.array(msg.position)

    def iiwa_joint_state_callback(self, msg: JointState):
        """Callback to update the current joint positions."""
        self.ros_snapshot.iiwa_joint_state = np.array(msg.position)

    def sharpa_joint_state_callback(self, msg: JointState):
        """Callback to update the current joint positions."""
        self.ros_snapshot.sharpa_joint_state = np.array(msg.position)

    def object_pose_callback(self, msg: PoseStamped):
        """ "Callback to update the current object pose."""
        self.ros_snapshot.object_pose = self._pose_stamped_to_matrix(msg)

    def predicted_object_pose_callback(self, msg: PoseStamped):
        """Callback to update the student-predicted object pose."""
        self.ros_snapshot.predicted_object_pose = self._pose_stamped_to_matrix(msg)

    @staticmethod
    def _pose_stamped_to_matrix(msg: PoseStamped) -> np.ndarray:
        msg = msg.pose
        xyz = np.array([msg.position.x, msg.position.y, msg.position.z])
        quat_xyzw = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]
        )
        latest_pose = np.eye(4)
        latest_pose[:3, 3] = xyz
        latest_pose[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        return latest_pose

    def goal_object_pose_callback(self, msg: Pose):
        """ "Callback to update the goal object pose."""
        xyz = np.array([msg.position.x, msg.position.y, msg.position.z])
        quat_xyzw = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]
        )
        latest_pose = np.eye(4)
        latest_pose[:3, 3] = xyz
        latest_pose[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        self.ros_snapshot.goal_object_pose = latest_pose

    def depth_image_callback(self, msg: Image):
        """Callback to update the latest optional depth visualization image."""
        try:
            self.ros_snapshot.depth_image_m = _decode_depth_image(msg, self.args.depth_units)
            self.ros_snapshot.depth_stamp = msg.header.stamp
        except Exception as exc:
            warn_every(f"Failed to decode depth image: {exc}", n_seconds=2.0, key="depth_decode")

    def camera_info_callback(self, msg: CameraInfo):
        """Callback to update camera intrinsics for optional point-cloud visualization."""
        self.ros_snapshot.camera_K = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)

    def update_depth_viser(self):
        if not (self.args.load_depth_image or self.args.load_point_cloud):
            return

        depth = self.ros_snapshot.depth_image_m
        if depth is None:
            warn_every("depth_image is None", n_seconds=2.0)
            return
        if self.ros_snapshot.depth_stamp is not None and self.ros_snapshot.depth_stamp == self._last_depth_stamp:
            return
        self._last_depth_stamp = self.ros_snapshot.depth_stamp

        rgb = _depth_to_rgb(depth, self.args.depth_near_m, self.args.depth_far_m)
        if self.depth_frustum is not None:
            self.depth_frustum.image = _viser_frustum_image(rgb, flip_y=self.args.flip_depth_image_y)
            K = self.ros_snapshot.camera_K
            h, w = depth.shape
            self.depth_frustum.aspect = float(w) / float(h)
            if K is not None and float(K[0, 0]) > 0.0:
                self.depth_frustum.fov = 2.0 * np.arctan2(float(h), 2.0 * float(K[1, 1]))

        if self.args.load_point_cloud and self.depth_point_cloud is not None:
            K = self.ros_snapshot.camera_K
            if K is None:
                warn_every(
                    "camera_K is None; point cloud disabled until CameraInfo arrives",
                    n_seconds=2.0,
                    key="camera_K_none",
                )
                return
            points, colors = _points_from_depth(
                depth,
                K,
                stride=self.args.point_stride,
                max_points=self.args.max_points,
                near_m=self.args.depth_near_m,
                far_m=self.args.depth_far_m,
            )
            self.depth_point_cloud.points = points
            self.depth_point_cloud.colors = colors

    def update_viser(self):
        """Update the viser simulation with the commanded joint positions."""
        ros_snapshot = self.ros_snapshot.make_copy_with_defaults()
        iiwa_joint_cmd = ros_snapshot.iiwa_joint_cmd
        sharpa_joint_cmd = ros_snapshot.sharpa_joint_cmd
        iiwa_joint_state = ros_snapshot.iiwa_joint_state
        sharpa_joint_state = ros_snapshot.sharpa_joint_state
        object_pose = ros_snapshot.object_pose
        predicted_object_pose = ros_snapshot.predicted_object_pose
        goal_object_pose = ros_snapshot.goal_object_pose

        assert iiwa_joint_cmd is not None
        assert sharpa_joint_cmd is not None
        assert iiwa_joint_state is not None
        assert sharpa_joint_state is not None
        assert object_pose is not None
        assert predicted_object_pose is not None
        assert goal_object_pose is not None

        # Command Robot: Set the commanded joint positions
        q_cmd = np.concatenate([iiwa_joint_cmd, sharpa_joint_cmd])
        q_state = np.concatenate([iiwa_joint_state, sharpa_joint_state])
        self.robot_viser.update_cfg(q_state)
        self.robot_cmd_viser.update_cfg(q_cmd)

        # Update the object pose
        # Object pose is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        T_R_O = object_pose
        T_W_O = T_W_R @ T_R_O
        object_pos = T_W_O[:3, 3]
        object_quat_xyzw = R.from_matrix(T_W_O[:3, :3]).as_quat()
        self.object_viser.position = object_pos
        self.object_viser.wxyz = object_quat_xyzw[[3, 0, 1, 2]]

        # Update the student-predicted object pose.
        T_R_P = predicted_object_pose
        T_W_P = T_W_R @ T_R_P
        predicted_object_pos = T_W_P[:3, 3]
        predicted_object_quat_xyzw = R.from_matrix(T_W_P[:3, :3]).as_quat()
        self.predicted_object_viser.position = predicted_object_pos
        self.predicted_object_viser.wxyz = predicted_object_quat_xyzw[[3, 0, 1, 2]]

        # Update the goal object pose
        # Goal object pose is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        T_R_G = goal_object_pose
        T_W_G = T_W_R @ T_R_G
        goal_object_pos = T_W_G[:3, 3]
        goal_object_quat_xyzw = R.from_matrix(T_W_G[:3, :3]).as_quat()
        self.goal_object_viser.position = goal_object_pos
        self.goal_object_viser.wxyz = goal_object_quat_xyzw[[3, 0, 1, 2]]
        self.update_depth_viser()

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""
        loop_no_sleep_dts, loop_dts = [], []
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update the viser simulation with the current joint commands
            self.update_viser()

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()

            loop_no_sleep_dt = (before_sleep_time - start_time).to_sec()
            loop_no_sleep_dts.append(loop_no_sleep_dt)
            loop_dt = (after_sleep_time - start_time).to_sec()
            loop_dts.append(loop_dt)

            PRINT_FPS_EVERY_N_SECONDS = 5.0
            PRINT_FPS_EVERY_N_STEPS = int(PRINT_FPS_EVERY_N_SECONDS / self.dt)
            if len(loop_dts) == PRINT_FPS_EVERY_N_STEPS:
                loop_dt_array = np.array(loop_dts)
                loop_no_sleep_dt_array = np.array(loop_no_sleep_dts)
                fps_array = 1.0 / loop_dt_array
                fps_no_sleep_array = 1.0 / loop_no_sleep_dt_array
                print("FPS with sleep:")
                print(f"  Mean: {np.mean(fps_array):.1f}")
                print(f"  Median: {np.median(fps_array):.1f}")
                print(f"  Max: {np.max(fps_array):.1f}")
                print(f"  Min: {np.min(fps_array):.1f}")
                print(f"  Std: {np.std(fps_array):.1f}")
                print("FPS without sleep:")
                print(f"  Mean: {np.mean(fps_no_sleep_array):.1f}")
                print(f"  Median: {np.median(fps_no_sleep_array):.1f}")
                print(f"  Max: {np.max(fps_no_sleep_array):.1f}")
                print(f"  Min: {np.min(fps_no_sleep_array):.1f}")
                print(f"  Std: {np.std(fps_no_sleep_array):.1f}")
                print()
                loop_no_sleep_dts, loop_dts = [], []


@dataclass
class VisualizationNodeArgs:
    object_name: str = "claw_hammer"
    f"""The name of the object to visualize. Options: {", ".join(VISUALIZATION_OBJECT_NAMES)}"""
    object_pose_topic: str = DEFAULT_OBJECT_POSE_TOPIC
    """Ground-truth/current object pose topic, usually from IsaacSim or real perception."""
    predicted_object_pose_topic: str = DEFAULT_PREDICTED_OBJECT_POSE_TOPIC
    """Student auxiliary predicted object pose topic."""
    load_depth_image: bool = False
    """If true, subscribe to the depth image topic and show it as a Viser camera frustum."""
    load_point_cloud: bool = False
    """If true, also build a point cloud from the depth topic and CameraInfo intrinsics."""
    depth_topic: str = DEFAULT_CAMERA_IMAGE_TOPIC
    """ROS depth image topic. Supports 32FC1 meters and 16UC1 millimeters."""
    camera_info_topic: str = DEFAULT_CAMERA_INFO_TOPIC
    """ROS CameraInfo topic used for frustum FOV and optional point cloud projection."""
    depth_units: str = "auto"
    """Depth units: auto, m, or mm."""
    depth_near_m: float = DEFAULT_DEPTH_NEAR_M
    """Near depth value for grayscale visualization."""
    depth_far_m: float = DEFAULT_DEPTH_FAR_M
    """Far depth value for grayscale visualization."""
    camera_pos_world: tuple[float, float, float] = DEFAULT_CAMERA_POS_WORLD
    """Viser camera-frame position in the visualization world frame."""
    camera_quat_wxyz: tuple[float, float, float, float] = DEFAULT_CAMERA_QUAT_WXYZ
    """Viser camera-frame orientation as wxyz."""
    point_stride: int = 4
    """Subsample stride for optional point cloud generation."""
    max_points: int = 20000
    """Maximum number of point-cloud points shown."""
    point_size: float = 0.006
    """Viser point-cloud point size."""
    flip_depth_image_y: bool = False
    """Flip depth image vertically before displaying in the Viser frustum."""


def main():
    args: VisualizationNodeArgs = tyro.cli(VisualizationNodeArgs)
    try:
        # Create and run the VisualizationNode
        node = VisualizationNode(args=args)
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

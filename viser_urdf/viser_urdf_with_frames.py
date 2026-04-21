"""URDF robot visualizer

Visualize robot models from URDF files with interactive joint controls.

Requires yourdfpy and robot_descriptions. Any URDF supported by yourdfpy should work.

- https://github.com/robot-descriptions/robot_descriptions.py
- https://github.com/clemense/yourdfpy

**Features:**

* :class:`viser.extras.ViserUrdf` for URDF file parsing and visualization
* Interactive joint sliders for robot articulation
* Real-time robot pose updates
* Support for local URDF files and robot_descriptions library
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import tyro
import viser
from scipy.spatial.transform import Rotation as R
from viser._scene_handles import FrameHandle
from viser.extras import ViserUrdf



def create_robot_control_sliders(
    server: viser.ViserServer,
    viser_urdf: ViserUrdf,
    link_name_to_frame: Dict[str, FrameHandle],
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """Create slider for each joint of the robot. We also update robot model
    when slider moves."""
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []

    print()
    print("Actuated joint limits:")
    for joint_name, (lower, upper) in viser_urdf.get_actuated_joint_limits().items():
        print(f"joint_name: {joint_name}, lower: {lower}, upper: {upper}")
    print()

    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )

        def slider_on_update(_):
            viser_urdf.update_cfg(np.array([slider.value for slider in slider_handles]))
            update_frames(viser_urdf, link_name_to_frame)

        slider.on_update(  # When sliders move, we update the URDF configuration.
            slider_on_update
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def update_frames(viser_urdf: ViserUrdf, link_name_to_frame: Dict[str, FrameHandle]):
    for link_name, frame in link_name_to_frame.items():
        link_pose = viser_urdf._urdf.get_transform(frame_to=link_name).copy()
        assert link_pose.shape == (4, 4), f"link_pose.shape: {link_pose.shape}"

        xyz = link_pose[:3, 3]
        xyzw = R.from_matrix(link_pose[:3, :3]).as_quat()
        wxyz = xyzw[[3, 0, 1, 2]]

        frame.position = xyz
        frame.wxyz = wxyz


def main(
    urdf_path: Path = Path("assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"),
    load_meshes: bool = True,
    load_collision_meshes: bool = True,
) -> None:
    # Start viser server.
    server = viser.ViserServer()

    # Load URDF.
    #
    # This takes either a yourdfpy.URDF object or a path to a .urdf file.
    assert urdf_path.exists(), f"URDF path {urdf_path} does not exist"
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
        root_node_name="/robot",
    )

    # Create frames
    link_name_to_frame = {}
    FAR_AWAY_POS = (100, 0, 0)
    AXES_LENGTH = 0.05
    AXES_RADIUS = 0.001
    INCLUDE_ALL_FRAMES = True
    if INCLUDE_ALL_FRAMES:
        for link_name in viser_urdf._urdf.link_map.keys():
            link_name_to_frame[link_name] = server.scene.add_frame(
                f"/robot/{link_name}",
                position=FAR_AWAY_POS,
                wxyz=(1, 0, 0, 0),
                show_axes=True,
                axes_length=AXES_LENGTH,
                axes_radius=AXES_RADIUS,
            )

    # Create sliders in GUI that help us move the robot joints.
    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf, link_name_to_frame
        )

    # Add visibility checkboxes.
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    # Hide checkboxes if meshes are not loaded.
    show_meshes_cb.visible = load_meshes
    show_collision_meshes_cb.visible = load_collision_meshes

    # Set initial robot configuration.
    initial_config[:7] = np.array(
    [
        -1.571,
        1.571 - np.deg2rad(10),
        -0.000,
        1.376 + np.deg2rad(10),
        -0.000,
        1.485,
        1.308,
    ]
)
    viser_urdf.update_cfg(np.array(initial_config))
    update_frames(viser_urdf, link_name_to_frame)

    # Create grid.
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            # Get the minimum z value of the trimesh scene.
            trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
        ),
    )

    # Create joint reset button.
    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q

    # Sleep forever.
    while True:
        breakpoint()
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)

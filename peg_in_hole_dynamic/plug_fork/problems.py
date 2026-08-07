"""Register plug_fork Problem entries.

Real-world-scale insertion tasks, added because reviewers noted the existing
suite uses enlarged geometry (beam_3x is literally 3x, the screw leg 3x longer)
and that its objects "are limited to thin and long pieces, essentially the same
family". They asked specifically about "complex or unseen object shapes such as
electric plugs, keys, or forks".

  fork.tol{N}mm   YCB 030_fork, a real scan at true scale
                  (197.6 x 27.1 x 15.9 mm) -> flatware holder, HANDLE first
  plug.tol{N}mm   Apple 20 W USB-C adapter A2940, built to published dimensions
                  (41.5 x 42.5 x 27 mm) -> power board socket, blades first

The plug answers the shape-family point that the fork does not: it is compact
rather than elongated, and it is the only TWO-feature insertion in the suite --
both 1.5 mm blades enter together, so contact itself constrains yaw.

Every pose below is in the receptive frame, whose origin is the receptacle's
centroid, and every number is derived from the meshes rather than authored.
Objects are authored long-axis-X with the entry end at -X (repo convention:
reward.fixed_size and Object.scale both lay their keypoints along X, so a part
built along another axis gets a keypoint box rotated against its own mesh), so
the insertion quaternion is the peg's -90 deg about Y.
"""

from peg_in_hole_dynamic import PROBLEM_REGISTRY, Problem

# Same quaternion the peg uses: stands the body's +X axis up so the -X end --
# the fork's handle, the plug's blades -- points down along the insertion axis.
_INSERT_QUAT = (0.0, -0.70710678, 0.0, 0.70710678)

# --- fork -> flatware holder ----------------------------------------------
#   hole_z_offset 0.0425 = half the 85 mm block, so it sits ON the table; the
#     receptive origin is its centroid, so a zero offset sinks half of it.
#   slot size = the fork's SWEPT envelope over its leading half
#     (15.43 x 18.69 mm) plus clearance, NOT any single cross-section. The head
#     is dished, so a slot cut to the widest station (27.3 x 6.7 mm) would not
#     admit the fork at all.
#   pre-insert +0.141277 puts the fork's lowest point exactly on the slot mouth
#     (block top +0.0425, fork's own zlo -0.098777): touching, zero penetration.
#   final +0.063277 is 78 mm deeper. That stop is FLOOR-limited, not
#     jam-limited -- the handle reaches the block floor before its widening
#     section reaches the slot -- so the depth does not vary with clearance.
#
# Handle-first is a deliberate choice, not a geometric necessity: real
# dishwashers are loaded both ways. It is the harder entry, since the handle is
# a blunt 15.4 x 18.7 mm end with no taper to self-align.
_HOLE_Z_OFFSET = 0.0425
_PRE_INSERT_Z = 0.141277
_FINAL_Z = 0.063277
_TOLERANCES = (("0p5mm", 0.5), ("1mm", 1.0), ("2mm", 2.0))

# --- plug -> power board socket -------------------------------------------
#   Socket block 120 x 56 x 30 mm, origin at centroid -> hole_z_offset = 0.015.
#   pre-insert +0.044400  blade tips exactly on the socket face, zero
#                         penetration (the plug origin sits 29.40 mm above its
#                         blade tips).
#   final      +0.028500  body face FLUSH with the socket face, blades 15.9 mm
#                         into a 26 mm cavity. NOT flush-minus-blade-length:
#                         the blades already hang below the body, and allowing
#                         for them twice sinks the body 15.9 mm into the block.
_SOCKET_Z_OFFSET = 0.0150
_PLUG_PRE_INSERT_Z = 0.044400
_PLUG_FINAL_Z = 0.028500
# 1 mm is the bring-up variant: on a 1.5 mm blade it opens the slot to 3.6 mm,
# which is generous, but it lets the policy learn the approach before the
# clearance is the binding constraint. 0.1 mm is NEMA spec (1.5 mm blade in a
# 1.6 mm slot) and is the hard end of the sweep.
_PLUG_TOLERANCES = (("0p1mm", 0.1), ("0p3mm", 0.3), ("0p5mm", 0.5), ("1mm", 1.0))


def _register_fork() -> None:
    for tag, _clearance_mm in _TOLERANCES:
        name = f"fork.tol{tag}"
        PROBLEM_REGISTRY[name] = Problem(
            name=name,
            insertion_object_name="ycb_fork",
            receptive_urdf=f"urdf/plug_fork/holder/holder_tol{tag}.urdf",
            insert_pose_rel_receptive=(
                (0.0, 0.0, _PRE_INSERT_Z, *_INSERT_QUAT),
                (0.0, 0.0, _FINAL_Z, *_INSERT_QUAT),
            ),
            hole_z_offset=_HOLE_Z_OFFSET,
            insertion_direction=(0.0, 0.0, -1.0),
        )


def _register_plug() -> None:
    for tag, _clearance_mm in _PLUG_TOLERANCES:
        name = f"plug.tol{tag}"
        PROBLEM_REGISTRY[name] = Problem(
            name=name,
            insertion_object_name="apple_20w_plug",
            receptive_urdf=f"urdf/plug_fork/socket/socket_tol{tag}.urdf",
            insert_pose_rel_receptive=(
                (0.0, 0.0, _PLUG_PRE_INSERT_Z, *_INSERT_QUAT),
                (0.0, 0.0, _PLUG_FINAL_Z, *_INSERT_QUAT),
            ),
            hole_z_offset=_SOCKET_Z_OFFSET,
            insertion_direction=(0.0, 0.0, -1.0),
        )


_register_fork()
_register_plug()

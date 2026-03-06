from dextoolbench.objects import Object, rescale_by_factor, NAME_TO_OBJECT

from isaacgymenvs.utils.utils import get_repo_root_dir


BEAM_NAME_TO_OBJECT = {
    "beam_0": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/beam/0/beam_0.urdf"
        ),
        scale=rescale_by_factor((0.013, 0.076, 0.015), factor=25),
        need_vhacd=True,
    ),
    "beam_1": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/beam/1/beam_1.urdf"
        ),
        scale=rescale_by_factor((0.013, 0.076, 0.015), factor=25),
        need_vhacd=True,
    ),
    "beam_2": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/beam/2/beam_2.urdf"
        ),
        scale=rescale_by_factor((0.012, 0.012, 0.076), factor=25),
        need_vhacd=True,
    ),
    "beam_3": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/beam/3/beam_3.urdf"
        ),
        scale=rescale_by_factor((0.012, 0.012, 0.076), factor=25),
        need_vhacd=True,
    ),
    "beam_6": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/beam/6/beam_6.urdf"
        ),
        scale=rescale_by_factor((0.152, 0.018, 0.013), factor=25),
        need_vhacd=True,
    ),
}

CAR_NAME_TO_OBJECT = {
    "car_0": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/car/0/car_0.urdf"
        ),
        scale=rescale_by_factor((0.0350, 0.0350, 0.0349), factor=25),
        need_vhacd=True,
    ),
    "car_1": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/car/1/car_1.urdf"
        ),
        scale=rescale_by_factor((0.1710, 0.0500, 0.0619), factor=25),
        need_vhacd=True,
    ),
    "car_2": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/car/2/car_2.urdf"
        ),
        scale=rescale_by_factor((0.1584, 0.0450, 0.0600), factor=25),
        need_vhacd=True,
    ),
    "car_3": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/car/3/car_3.urdf"
        ),
        scale=rescale_by_factor((0.0350, 0.0350, 0.0349), factor=25),
        need_vhacd=True,
    ),
    "car_4": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/car/4/car_4.urdf"
        ),
        scale=rescale_by_factor((0.0350, 0.0350, 0.0349), factor=25),
        need_vhacd=True,
    ),
    "car_5": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/car/5/car_5.urdf"
        ),
        scale=rescale_by_factor((0.0350, 0.0350, 0.0349), factor=25),
        need_vhacd=True,
    ),
}

COOLING_MANIFOLD_NAME_TO_OBJECT = {
    "cooling_manifold_0": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/cooling_manifold/0/cooling_manifold_0.urdf"
        ),
        scale=rescale_by_factor((0.0177, 0.0175, 0.0465), factor=25),
        need_vhacd=True,
    ),
    "cooling_manifold_1": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/cooling_manifold/1/cooling_manifold_1.urdf"
        ),
        scale=rescale_by_factor((0.1200, 0.1200, 0.0350), factor=25),
        need_vhacd=True,
    ),
    "cooling_manifold_2": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/cooling_manifold/2/cooling_manifold_2.urdf"
        ),
        scale=rescale_by_factor((0.0201, 0.0201, 0.0350), factor=25),
        need_vhacd=True,
    ),
    "cooling_manifold_3": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/cooling_manifold/3/cooling_manifold_3.urdf"
        ),
        scale=rescale_by_factor((0.0203, 0.0198, 0.0350), factor=25),
        need_vhacd=True,
    ),
    "cooling_manifold_4": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/cooling_manifold/4/cooling_manifold_4.urdf"
        ),
        scale=rescale_by_factor((0.0177, 0.0175, 0.0465), factor=25),
        need_vhacd=True,
    ),
    "cooling_manifold_5": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/cooling_manifold/5/cooling_manifold_5.urdf"
        ),
        scale=rescale_by_factor((0.0195, 0.0205, 0.0350), factor=25),
        need_vhacd=True,
    ),
    "cooling_manifold_6": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/cooling_manifold/6/cooling_manifold_6.urdf"
        ),
        scale=rescale_by_factor((0.0201, 0.0201, 0.0350), factor=25),
        need_vhacd=True,
    ),
}

DUCT_NAME_TO_OBJECT = {
    "duct_0": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/duct/0/duct_0.urdf"
        ),
        scale=rescale_by_factor((0.1089, 0.1200, 0.1437), factor=25),
        need_vhacd=True,
    ),
    "duct_1": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/duct/1/duct_1.urdf"
        ),
        scale=rescale_by_factor((0.1064, 0.1200, 0.0709), factor=25),
        need_vhacd=True,
    ),
    "duct_2": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/duct/2/duct_2.urdf"
        ),
        scale=rescale_by_factor((0.0321, 0.0231, 0.0398), factor=25),
        need_vhacd=True,
    ),
    "duct_3": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/duct/3/duct_3.urdf"
        ),
        scale=rescale_by_factor((0.0321, 0.0231, 0.0398), factor=25),
        need_vhacd=True,
    ),
    "duct_4": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/duct/4/duct_4.urdf"
        ),
        scale=rescale_by_factor((0.0323, 0.0230, 0.0399), factor=25),
        need_vhacd=True,
    ),
    "duct_5": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/duct/5/duct_5.urdf"
        ),
        scale=rescale_by_factor((0.0323, 0.0231, 0.0398), factor=25),
        need_vhacd=True,
    ),
    "duct_6": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/duct/6/duct_6.urdf"
        ),
        scale=rescale_by_factor((0.0321, 0.0231, 0.0398), factor=25),
        need_vhacd=True,
    ),
    "duct_7": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/duct/7/duct_7.urdf"
        ),
        scale=rescale_by_factor((0.0323, 0.0231, 0.0398), factor=25),
        need_vhacd=True,
    ),
}

GAMEPAD_NAME_TO_OBJECT = {
    "gamepad_0": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/gamepad/0/gamepad_0.urdf"
        ),
        scale=rescale_by_factor((0.0120, 0.0120, 0.0285), factor=25),
        need_vhacd=True,
    ),
    "gamepad_1": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/gamepad/1/gamepad_1.urdf"
        ),
        scale=rescale_by_factor((0.0230, 0.0230, 0.0300), factor=25),
        need_vhacd=True,
    ),
    "gamepad_2": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/gamepad/2/gamepad_2.urdf"
        ),
        scale=rescale_by_factor((0.0230, 0.0230, 0.0300), factor=25),
        need_vhacd=True,
    ),
    "gamepad_3": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/gamepad/3/gamepad_3.urdf"
        ),
        scale=rescale_by_factor((0.1352, 0.0756, 0.0250), factor=25),
        need_vhacd=True,
    ),
    "gamepad_4": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/gamepad/4/gamepad_4.urdf"
        ),
        scale=rescale_by_factor((0.0140, 0.0600, 0.0120), factor=25),
        need_vhacd=True,
    ),
    "gamepad_5": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/gamepad/5/gamepad_5.urdf"
        ),
        scale=rescale_by_factor((0.0120, 0.0120, 0.0285), factor=25),
        need_vhacd=True,
    ),
}

PLUMBERS_BLOCK_NAME_TO_OBJECT = {
    "plumbers_block_0": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/plumbers_block/0/plumbers_block_0.urdf"
        ),
        scale=rescale_by_factor((0.0427, 0.0800, 0.0495), factor=25),
        need_vhacd=True,
    ),
    "plumbers_block_1": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/plumbers_block/1/plumbers_block_1.urdf"
        ),
        scale=rescale_by_factor((0.0288, 0.0250, 0.0700), factor=25),
        need_vhacd=True,
    ),
    "plumbers_block_2": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/plumbers_block/2/plumbers_block_2.urdf"
        ),
        scale=rescale_by_factor((0.1600, 0.0400, 0.0650), factor=25),
        need_vhacd=True,
    ),
    "plumbers_block_3": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/plumbers_block/3/plumbers_block_3.urdf"
        ),
        scale=rescale_by_factor((0.1000, 0.0400, 0.0261), factor=25),
        need_vhacd=True,
    ),
    "plumbers_block_4": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/plumbers_block/4/plumbers_block_4.urdf"
        ),
        scale=rescale_by_factor((0.0288, 0.0250, 0.0700), factor=25),
        need_vhacd=True,
    ),
}

STOOL_CIRCULAR_NAME_TO_OBJECT = {
    "stool_circular_0": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/stool_circular/0/stool_circular_0.urdf"
        ),
        scale=rescale_by_factor((0.0180, 0.0180, 0.0300), factor=25),
        need_vhacd=True,
    ),
    "stool_circular_1": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/stool_circular/1/stool_circular_1.urdf"
        ),
        scale=rescale_by_factor((0.0208, 0.0180, 0.0950), factor=25),
        need_vhacd=True,
    ),
    "stool_circular_2": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/stool_circular/2/stool_circular_2.urdf"
        ),
        scale=rescale_by_factor((0.0750, 0.0750, 0.0165), factor=25),
        need_vhacd=True,
    ),
    "stool_circular_3": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/stool_circular/3/stool_circular_3.urdf"
        ),
        scale=rescale_by_factor((0.0208, 0.0180, 0.0950), factor=25),
        need_vhacd=True,
    ),
    "stool_circular_4": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/stool_circular/4/stool_circular_4.urdf"
        ),
        scale=rescale_by_factor((0.0180, 0.0180, 0.0300), factor=25),
        need_vhacd=True,
    ),
    "stool_circular_5": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/stool_circular/5/stool_circular_5.urdf"
        ),
        scale=rescale_by_factor((0.0208, 0.0180, 0.0950), factor=25),
        need_vhacd=True,
    ),
    "stool_circular_6": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/stool_circular/6/stool_circular_6.urdf"
        ),
        scale=rescale_by_factor((0.0180, 0.0180, 0.0300), factor=25),
        need_vhacd=True,
    ),
    "stool_circular_7": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/stool_circular/7/stool_circular_7.urdf"
        ),
        scale=rescale_by_factor((0.0208, 0.0180, 0.0950), factor=25),
        need_vhacd=True,
    ),
    "stool_circular_8": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/fabrica/stool_circular/8/stool_circular_8.urdf"
        ),
        scale=rescale_by_factor((0.0180, 0.0180, 0.0300), factor=25),
        need_vhacd=True,
    ),
}

FABRICA_NAME_TO_OBJECT = {}
FABRICA_NAME_TO_OBJECT.update(BEAM_NAME_TO_OBJECT)
FABRICA_NAME_TO_OBJECT.update(CAR_NAME_TO_OBJECT)
FABRICA_NAME_TO_OBJECT.update(COOLING_MANIFOLD_NAME_TO_OBJECT)
FABRICA_NAME_TO_OBJECT.update(DUCT_NAME_TO_OBJECT)
FABRICA_NAME_TO_OBJECT.update(GAMEPAD_NAME_TO_OBJECT)
FABRICA_NAME_TO_OBJECT.update(PLUMBERS_BLOCK_NAME_TO_OBJECT)
FABRICA_NAME_TO_OBJECT.update(STOOL_CIRCULAR_NAME_TO_OBJECT)

# Register into the global object registry so the sim can find them
NAME_TO_OBJECT.update(FABRICA_NAME_TO_OBJECT)

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

FABRICA_NAME_TO_OBJECT = {}
FABRICA_NAME_TO_OBJECT.update(BEAM_NAME_TO_OBJECT)

# Register into the global object registry so the sim can find them
NAME_TO_OBJECT.update(FABRICA_NAME_TO_OBJECT)

"""Register the peg in the global NAME_TO_OBJECT registry.

Mirrors fabrica/objects.py. Imported by peg_eval.py so that FabricaEnv
can resolve "peg" via NAME_TO_OBJECT[object_name].urdf_path.
"""

from dextoolbench.objects import Object, rescale_by_factor, NAME_TO_OBJECT

from isaacgymenvs.utils.utils import get_repo_root_dir


ASSETS_DIR = get_repo_root_dir() / "assets" / "urdf" / "peg_in_hole"

PEG_NAME_TO_OBJECT = {
    "peg": Object(
        urdf_path=ASSETS_DIR / "peg" / "peg.urdf",
        scale=rescale_by_factor((0.25, 0.03, 0.02), factor=25),
        need_vhacd=False,
    ),
}

NAME_TO_OBJECT.update(PEG_NAME_TO_OBJECT)

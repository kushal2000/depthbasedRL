"""Register plug_fork insertion objects in the global NAME_TO_OBJECT registry.

Mirrors peg/objects.py. Imported by this package's __init__ so the env can
resolve the object via NAME_TO_OBJECT[problem.insertion_object_name].

`scale` is the object's metric bounding box in metres, passed through
rescale_by_factor(..., 25) -- the same convention the peg uses with its literal
250 x 30 x 20 mm handle. It is fed to the policy as an object-size observation,
not used as a geometric scale on the mesh.

NOTE this module must only .update() NAME_TO_OBJECT, never rebind it. The env
does `from dextoolbench.objects import NAME_TO_OBJECT` BEFORE importing
peg_in_hole_dynamic, so it holds a reference to that dict; rebinding here would
leave the env looking at an empty registry.
"""

from dextoolbench.objects import NAME_TO_OBJECT, Object, rescale_by_factor

from isaacgymenvs.utils.utils import get_repo_root_dir

ASSETS_DIR = get_repo_root_dir() / "assets" / "urdf" / "plug_fork"

PLUG_FORK_NAME_TO_OBJECT = {
    # YCB 030_fork, a real scanned everyday object at true scale
    # (197.6 x 27.1 x 15.9 mm). Added because reviewers noted the existing
    # task suite uses enlarged geometry -- beam_3x is literally 3x scale --
    # rather than real-world object sizes.
    #
    # Visual is the scan with its texture; collision is 9 CoACD hulls. The
    # geometry is taken from YCB's nontextured.ply, which is watertight and a
    # single component; textured.obj is the same scan but splits vertices at
    # UV seams, which presents as 172 disconnected fragments.
    #
    # need_vhacd=False: already convex-decomposed by CoACD at bake time.
    "ycb_fork": Object(
        urdf_path=ASSETS_DIR / "fork" / "fork.urdf",
        scale=rescale_by_factor((0.198, 0.027, 0.016), factor=25),
        need_vhacd=False,
    ),
    # Apple 20 W USB-C adapter A2940, built to published dimensions
    # (41.5 x 42.5 x 27 mm, 58.5 g). The task reviewers named explicitly: a
    # naturally sized electric plug. Its 1.5 mm NEMA 1-15P blades are an order
    # of magnitude below any feature in the existing suite, and it is the only
    # TWO-feature insertion here -- both blades enter together, so contact
    # constrains yaw rather than the reward having to.
    #
    # Unlike the fork it is compact rather than elongated, which is the other
    # half of the reviewers' point: the existing objects are all long and thin.
    #
    # need_vhacd=False: collision is 3 exact boxes, not a decomposition.
    "apple_20w_plug": Object(
        urdf_path=ASSETS_DIR / "plug" / "plug.urdf",
        scale=rescale_by_factor((0.0429, 0.0425, 0.0415), factor=25),
        need_vhacd=False,
    ),
}

NAME_TO_OBJECT.update(PLUG_FORK_NAME_TO_OBJECT)

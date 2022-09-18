from pathlib import Path

import numpy as np

from partnet_mobility_utils.articulate import articulate_joint

TESTDATA_DIR = Path(__file__).parent / "testdata"
from partnet_mobility_utils.data import PMObject
from partnet_mobility_utils.render.pybullet import PybulletRenderer


def test_articulation():

    obj = PMObject(TESTDATA_DIR / "7179")
    assert obj.obj_id == "7179"
    assert obj.category == "Oven"
    assert obj.well_formed

    # Get some points from the object.
    renderer = PybulletRenderer()
    render = renderer.render(pm_obj=obj, joints="random")

    jas = renderer._render_env.get_joint_angles()

    # Rotating by 2pi should be perfectly equal.
    new_pos = articulate_joint(
        obj,
        jas,
        "link_3",
        np.pi * 2,
        render["pos"],
        render["seg"],
        render["labelmap"],
        render["T_world_base"],
    )

    assert np.isclose(new_pos, render["pos"]).all()

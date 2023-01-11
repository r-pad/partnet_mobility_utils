import importlib
from pathlib import Path

import numpy as np
import pytest

PYBULLET_INSTALLED = importlib.util.find_spec("pybullet")

if PYBULLET_INSTALLED:
    from rpad.partnet_mobility_utils.data import PMObject
    from rpad.partnet_mobility_utils.render.pybullet import PybulletRenderer


TESTDATA_DIR = Path(__file__).parent / "testdata"


@pytest.mark.skipif(not PYBULLET_INSTALLED, reason="pybullet not installed")
def test_render():
    obj = PMObject(TESTDATA_DIR / "100809")
    renderer = PybulletRenderer()
    res = renderer.render(obj, joints="random", camera_xyz="random")


@pytest.mark.skipif(not PYBULLET_INSTALLED, reason="pybullet not installed")
def test_render_reproducibility():
    obj = PMObject(TESTDATA_DIR / "100809")
    renderer = PybulletRenderer()
    res1 = renderer.render(obj, joints="random", camera_xyz="random", seed=12345)
    res2 = renderer.render(obj, joints="random", camera_xyz="random", seed=12345)

    # Make sure everything is byte-equal with the same seed.
    assert res1["id"] == res2["id"]
    assert np.array_equal(res1["pos"], res2["pos"])
    assert np.array_equal(res1["seg"], res2["seg"])
    assert res1["frame"] == res2["frame"]
    assert np.array_equal(res1["T_world_cam"], res2["T_world_cam"])
    assert np.array_equal(res1["T_world_base"], res2["T_world_base"])
    assert res1["labelmap"] == res2["labelmap"]
    assert res1["angles"] == res2["angles"]

    # Make sure when we change the seed we get something entirely different.
    res3 = renderer.render(obj, joints="random", camera_xyz="random", seed=54321)
    assert res1["id"] == res3["id"]
    assert not np.array_equal(res1["pos"], res3["pos"])
    assert not np.array_equal(res1["seg"], res3["seg"])
    assert res1["frame"] == res3["frame"]
    assert not np.array_equal(res1["T_world_cam"], res3["T_world_cam"])
    # assert not np.array_equal(res1["T_world_base"], res3["T_world_base"])
    # assert res1["labelmap"] == res3["labelmap"]
    assert not res1["angles"] == res3["angles"]


if __name__ == "__main__":
    test_render()

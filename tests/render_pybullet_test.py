from pathlib import Path

from partnet_mobility_utils.data import PMObject
from partnet_mobility_utils.render.pybullet import PybulletRenderer

TESTDATA_DIR = Path(__file__).parent / "testdata"


def test_render():
    obj = PMObject(TESTDATA_DIR / "100809")
    renderer = PybulletRenderer()
    res = renderer.render(obj, joints="random", camera_xyz="random")


if __name__ == "__main__":
    test_render()

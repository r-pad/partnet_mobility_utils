from pathlib import Path

from partnet_mobility_utils.pm_data import PMObject
from partnet_mobility_utils.render.trimesh import TrimeshRenderer

TESTDATA_DIR = Path(__file__).parent / "testdata"


def test_render():
    obj = PMObject(TESTDATA_DIR / "100809")
    renderer = TrimeshRenderer()
    res = renderer.render(obj, 1000)


if __name__ == "__main__":
    test_render()

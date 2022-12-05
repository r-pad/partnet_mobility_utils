from pathlib import Path

from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.render.trimesh import TrimeshRenderer

TESTDATA_DIR = Path(__file__).parent / "testdata"


def test_render():
    obj = PMObject(TESTDATA_DIR / "100809")
    renderer = TrimeshRenderer()
    res = renderer.render(obj, 1000)


if __name__ == "__main__":
    test_render()

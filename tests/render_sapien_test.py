import sys
from pathlib import Path

import pytest

from partnet_mobility_utils.pm_data import PMObject

TESTDATA_DIR = Path(__file__).parent / "testdata"

PLATFORM = sys.platform


# @pytest.mark.skipif(
#     PLATFORM != "linux", reason="SAPIEN doesn't run on non-Linux machines"
# )
@pytest.mark.skip()
def test_render():
    from partnet_mobility_utils.render.sapien import SAPIENRenderer

    obj = PMObject(TESTDATA_DIR / "100809")
    renderer = SAPIENRenderer()
    res = renderer.render(obj)


if __name__ == "__main__":
    test_render()

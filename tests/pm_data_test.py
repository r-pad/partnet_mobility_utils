from pathlib import Path

from partnet_mobility_utils.data import PMObject

TESTDATA_DIR = Path(__file__).parent / "testdata"


def test_simple_parse():
    obj = PMObject(TESTDATA_DIR / "100809")
    assert obj.obj_id == "100809"
    assert obj.category == "Remote"
    assert obj.well_formed

    # Test some semantics properties.
    assert obj.semantics.by_name("link_55").type == "slider"
    assert len(obj.semantics.by_label("button")) == 57
    assert len(obj.semantics.by_type("free")) == 1

    # Test some kinematic properties.
    assert len(obj.obj.get_chain("link_55")) == 2

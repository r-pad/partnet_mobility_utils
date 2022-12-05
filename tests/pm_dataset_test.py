import os.path
import random
from pathlib import Path

import pytest

from rpad.partnet_mobility_utils.dataset import PCDataset

PM_DATASET_EXISTS = os.path.exists(
    os.path.expanduser("~/datasets/partnet-mobility/raw")
)


@pytest.mark.skipif(not PM_DATASET_EXISTS, reason="pm not found")
def test_full_dataset_creation():
    dset = PCDataset(
        root=os.path.expanduser("~/datasets/partnet-mobility/raw"), split="all"
    )
    random.seed(0)
    for i in range(10):
        data = dset[random.randint(0, len(dset))]


@pytest.mark.skipif(not PM_DATASET_EXISTS, reason="pm not found")
def test_umpnet_creation():
    dset = PCDataset(
        root=os.path.expanduser("~/datasets/partnet-mobility/raw"),
        split="umpnet-test",
    )
    random.seed(0)
    for i in range(10):
        data = dset[random.randint(0, len(dset) - 1)]


def test_single_dataset():
    dset = PCDataset(root=Path(__file__).parent / "testdata", split=["100809"])
    pc = dset[0]

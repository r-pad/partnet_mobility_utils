import copy
import logging
import typing
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.render import PartialPC, PMRenderer
from rpad.partnet_mobility_utils.render.pybullet import PybulletRenderer


def write_ids(path: Union[str, Path], objs: Iterable[Tuple[str, str]]):
    path = Path(path)
    with path.open("w") as f:
        for oid, cat in objs:
            f.write(f"{oid},{cat}\n")


def read_ids(path: Path) -> List[Tuple[str, str]]:
    with path.open("r") as f:
        contents = f.read()
    lines = contents.split("\n")[:-1]  # Not the last line.
    objs = [tuple(line.split(",")) for line in lines]
    return typing.cast(List[Tuple[str, str]], objs)


__parent = Path(__file__).parent
WELL_FORMED = {obj[0] for obj in read_ids(__parent / "data_lists" / "well_formed.txt")}
SUFFICIENT = {obj[0] for obj in read_ids(__parent / "data_lists" / "extra_objs.txt")}
DEFAULT_OBJS = list(WELL_FORMED.union(SUFFICIENT))

UMPNET_DIR = Path(__file__).parent / "data_lists" / "umpnet"
UMPNET_TRAIN_TRAIN_OBJS = read_ids(UMPNET_DIR / "train_train.txt")
UMPNET_TRAIN_TEST_OBJS = read_ids(UMPNET_DIR / "train_test.txt")
UMPNET_TEST_OBJS = read_ids(UMPNET_DIR / "test.txt")
UMPNET_TRAIN_TRAIN_OBJ_IDS = [obj[0] for obj in UMPNET_TRAIN_TRAIN_OBJS]
UMPNET_TRAIN_TEST_OBJ_IDS = [obj[0] for obj in UMPNET_TRAIN_TEST_OBJS]
UMPNET_TEST_OBJ_IDS = [obj[0] for obj in UMPNET_TEST_OBJS]


def get_ids_by_class(class_name: str, filter_valid: bool = True) -> List[str]:
    ids = read_ids(Path(__file__).parent / "data_lists" / f"{class_name}.txt")
    return [id[0] for id in ids if ((id in DEFAULT_OBJS) or not filter_valid)]


AVAILABLE_DATASET = Literal[
    "all", "umpnet-train-train", "umpnet-train-test", "umpnet-test"
]


class PCDataset:
    """Partnet-Mobility Point Cloud Dataset"""

    def __init__(
        self,
        root: Union[str, Path],
        split: Union[AVAILABLE_DATASET, List[str]],
        renderer: Literal["pybullet", "sapien", "trimesh"] = "pybullet",
        use_egl: bool = False,
    ):
        if isinstance(split, str):
            if split == "all":
                self._ids = DEFAULT_OBJS
            else:
                ids = read_ids(
                    {
                        "umpnet-train-train": UMPNET_DIR / "train_train.txt",
                        "umpnet-train-test": UMPNET_DIR / "train_test.txt",
                        "umpnet-test": UMPNET_DIR / "test.txt",
                    }[split]
                )
                self._ids = [id[0] for id in ids]
        else:
            self._ids = copy.deepcopy(split)

        new_ids = []
        def_objs = set(DEFAULT_OBJS)
        for id in self._ids:
            if id not in def_objs:
                logging.warning(f"{id} is not well-formed, excluding...")
                raise ValueError("BDADAD")
            else:
                new_ids.append(id)

        self._ids = new_ids

        self.pm_objs: Dict[str, PMObject] = {
            id: PMObject(Path(root) / id) for id in self._ids
        }
        self.renderers: Dict[str, PMRenderer] = {}
        self.renderer_type = renderer
        self.use_egl = use_egl

    def get(
        self,
        obj_id: str,
        joints: Union[
            Literal["random"],
            Dict[str, Union[float, Literal["random", "random-oc"]]],
            None,
        ] = None,
        camera_xyz: Union[
            Literal["random"],
            Tuple[float, float, float],
            None,
        ] = None,
        seed: Optional[int] = None,
    ) -> PartialPC:
        if obj_id not in self.renderers:
            if self.renderer_type == "pybullet":
                new_renderer = PybulletRenderer()
            else:
                raise NotImplementedError("not yet implemented")
            self.renderers[obj_id] = new_renderer
        renderer = self.renderers[obj_id]

        pc_render = renderer.render(
            pm_obj=self.pm_objs[obj_id],
            joints=joints,
            camera_xyz=camera_xyz,
            seed=seed,
            use_egl=self.use_egl,
        )

        return pc_render

    def __getitem__(self, item: Union[int, str]) -> PartialPC:
        if isinstance(item, str):
            obj_id = item
        else:
            obj_id = self._ids[item]

        return self.get(obj_id, joints=None, camera_xyz=None)

    def __len__(self):
        return len(self._ids)

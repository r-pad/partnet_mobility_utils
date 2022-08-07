import typing
from pathlib import Path
from typing import Iterable, List, Tuple, Union


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


def get_ids_by_class(class_name: str, filter_valid: bool = True) -> List[str]:
    ids = read_ids(Path(__file__).parent / "data_lists" / f"{class_name}.txt")
    return [id[0] for id in ids if ((id in DEFAULT_OBJS) or not filter_valid)]

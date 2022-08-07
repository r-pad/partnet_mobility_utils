from typing import Dict, List, Literal, TypedDict

import numpy as np
import numpy.typing as npt


class PartialPC(TypedDict):
    """A Partial PointCloud

    Attributes:
        pos: Position
        seg: segmentation
    """

    pos: npt.NDArray[np.float32]
    seg: npt.NDArray[np.uint]
    frame: Literal["world", "camera"]
    T_world_cam: npt.NDArray[np.float32]
    T_world_base: npt.NDArray[np.float32]
    # proj_matrix: npt.NDArray[np.float32]
    labelmap: Dict[str, int]
    angles: Dict[str, float]


class FullPCData(TypedDict):
    pos: npt.NDArray[np.float32]
    norm: npt.NDArray[np.float32]
    seg: npt.NDArray[np.uint]
    ins: List
    sem: List
    art: List

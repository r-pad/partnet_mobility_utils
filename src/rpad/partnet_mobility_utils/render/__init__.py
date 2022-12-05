import abc
from abc import abstractmethod
from typing import Dict, List, Literal, Tuple, TypedDict, Union

import numpy as np
import numpy.typing as npt

from rpad.partnet_mobility_utils.data import PMObject


class PartialPC(TypedDict):
    """A Partial PointCloud

    Attributes:
        pos: Position
        seg: segmentation
    """

    id: str
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
    # seg: npt.NDArray[np.uint]
    ins: List
    sem: List
    art: List


class PMRenderer(abc.ABC):
    @abstractmethod
    def render(
        self,
        pm_obj: PMObject,
        joints: Union[
            Literal["random", "random-oc", "open", "closed"],
            Dict[str, Union[float, Literal["random", "random-oc"]]],
            None,
        ] = None,
        camera_xyz: Union[
            Literal["random"],
            Tuple[float, float, float],
            None,
        ] = None,
    ) -> PartialPC:
        ...

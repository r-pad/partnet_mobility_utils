import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union, cast

from partnet_mobility_utils.urdf import PMTree

JOINT_TYPES = {"slider", "free", "hinge", "heavy", "static", "slider+"}
JointType = Literal["slider", "free", "hinge", "heavy", "static", "slider+"]


@dataclass
class JointSemantic:
    """Joint Semantics

    Attributes:
        name: it's a name
        type: it's a type
        label: it's a mfing label
    """

    name: str
    type: JointType
    label: str


@dataclass
class Semantics:
    """Dis a thing"""

    sems: List[JointSemantic]

    def by_name(self, name: str) -> JointSemantic:
        """Does this do somthing?

        Args:
            name (str): Desc

        Returns:
            JointSemantic: Desc
        """
        return {semantic.name: semantic for semantic in self.sems}[name]

    def by_type(
        self, joint_type: Union[JointType, Sequence[JointType]]
    ) -> List[JointSemantic]:
        """By type

        Args:
            joint_type (Union[JointType, Sequence[JointType]]): Whether it's a sequence or not.

        Returns:
            List[JointSemantic]: _description_
        """
        if isinstance(joint_type, str):
            joint_types = {joint_type}
        else:
            joint_types = set(joint_type)
        return [sem for sem in self.sems if sem.type in joint_types]

    def by_label(self, label: str) -> List[JointSemantic]:
        return [sem for sem in self.sems if sem.label == label]

    @staticmethod
    def from_file(fn: Union[str, Path]) -> "Semantics":
        path = Path(fn)
        with path.open("r") as f:
            lines = f.read().split("\n")

        # Remove all empty lines.
        lines = [line for line in lines if line.strip()]
        semantics = []
        for line in lines:
            name, jt, sem = line.split(" ")
            if jt not in JOINT_TYPES:
                raise ValueError("bad file for parsing semantics...")
            jt = cast(JointType, jt)  # it passes parsing
            semantics.append(JointSemantic(name, jt, sem))

        # assert no duplicates.
        names = {semantic.name for semantic in semantics}
        assert len(names) == len(semantics)

        return Semantics(semantics)


@dataclass
class Metadata:
    """This represents the metadata file

    Attributes:
        model_cat: The model categorty. Will be something like "Chair".
    """

    model_cat: str

    @staticmethod
    def from_file(fn: Union[str, Path]) -> "Metadata":
        """Parse the metadata file.

        Args:
            fn (Union[str, Path]): The metadata filename to parse.

        Returns:
            Metadata
        """
        path = Path(fn)
        with path.open("r") as f:
            raw_metadata = json.load(f)
        return Metadata(model_cat=raw_metadata["model_cat"])


class PMObject:
    """This class describes the grouping of files for each object."""

    def __init__(self, obj_dir: Union[str, Path]):
        self.obj_dir = Path(obj_dir)

        # Load the data in.
        self.semantics = Semantics.from_file(self.semantics_fn)
        self.metadata = Metadata.from_file(self.meta_fn)
        self.obj = PMTree.parse_urdf(self.urdf_fn)

        self.__issubset: Optional[bool] = None
        self.__issame: Optional[bool] = None

    @property
    def obj_id(self) -> str:
        return self.obj_dir.name

    @property
    def category(self) -> str:
        return self.metadata.model_cat

    @property
    def semantics_fn(self) -> Path:
        """The semantics file"""
        return self.obj_dir / "semantics.txt"

    @property
    def urdf_fn(self) -> Path:
        return self.obj_dir / "mobility.urdf"

    @property
    def meta_fn(self) -> Path:
        return self.obj_dir / "meta.json"

    @property
    def original_dataset(self) -> Literal["partnet", "shapenet"]:
        raise NotImplementedError("not implemented yet")

    def _evaluate_meshes(self) -> Tuple[bool, bool]:
        """Get information about whether or not the URDF agrees with the textured_objs directory."""
        meshes = self.obj.meshes

        objs = list((self.obj_dir / "textured_objs").glob("*.obj"))
        obj_bases = set([f"textured_objs/{path.name}" for path in objs])

        # Is same == are the meshes claimed in the urdf the same as are included
        # in the dataset?
        issame = meshes == obj_bases

        # issubset == are teh meshes claimed in the urdf at least a subset of the dataset?
        issubset = meshes.issubset(obj_bases)

        return issubset, issame

    @property
    def has_extra_objs(self) -> bool:
        """Some of the items in the dataset have extra obj files in their textured_objs directory."""
        if self.__issubset is None or self.__issame is None:
            self.__issubset, self.__issame = self._evaluate_meshes()

        return self.__issubset and not self.__issame  # type: ignore

    @property
    def well_formed(self) -> bool:
        """Does the set of objs in the textured_objs folder exactly match those described in the URDF?"""
        if self.__issubset is None or self.__issame is None:
            self.__issubset, self.__issame = self._evaluate_meshes()
        return self.__issame  # type: ignore

    @property
    def usable(self) -> bool:
        """Are all the objects specified in the URDF contianed in the textured_objs folder?"""
        if self.__issubset is None or self.__issame is None:
            self.__issubset, self.__issame = self._evaluate_meshes()
        return self.__issubset  # type: ignore

    def __repr__(self) -> str:
        return f'PMRawData(id="{self.obj_id}")'

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union, cast

from rpad.partnet_mobility_utils.urdf import PMTree

JOINT_TYPES = {"slider", "free", "hinge", "heavy", "static", "slider+"}
JointType = Literal["slider", "free", "hinge", "heavy", "static", "slider+"]


@dataclass
class JointSemantic:
    """Describes semantic attributes of a joint.

    Attributes:
        name: The name of the joint, as defined in URDF.
        type: The type of joint, one of: {"slider", "free", "hinge", "heavy", "static", "slider+"}
        label: The semantic label of the joint (i.e. button, door, etc.). Shared across objects.
    """

    name: str
    type: JointType
    label: str


@dataclass
class Semantics:
    """Each object includes a semantics.txt file, which describes the semantics of each joint.

    This class wraps the semantic file, and provides functions to filter down to specific joints by
    type, name, or label.
    """

    sems: List[JointSemantic]

    def by_name(self, name: str) -> JointSemantic:
        """Get the semantics for a specific joint.

        Args:
            name (str): Joint name.

        Returns:
            JointSemantic
        """
        return {semantic.name: semantic for semantic in self.sems}[name]

    def by_type(
        self, joint_type: Union[JointType, Sequence[JointType]]
    ) -> List[JointSemantic]:
        """Filter down all semantics in the object by joint type.

        Args:
            joint_type (Union[JointType, Sequence[JointType]]): One or more joint types

        Returns:
            List[JointSemantic]: A list of JointSemantics matching the joint_type.
        """
        if isinstance(joint_type, str):
            joint_types = {joint_type}
        else:
            joint_types = set(joint_type)
        return [sem for sem in self.sems if sem.type in joint_types]

    def by_label(self, label: str) -> List[JointSemantic]:
        """Filter down all semantics in the object by label name.

        Args:
            label (str): The semantic joint label (i.e. "button")

        Returns:
            List[JointSemantic]: Joints matching this label.
        """
        return [sem for sem in self.sems if sem.label == label]

    @staticmethod
    def from_file(fn: Union[str, Path]) -> "Semantics":
        """Parse a semantics file.

        Args:
            fn (Union[str, Path]): Semantics filename (should end in semantics.txt)

        Returns:
            Semantics: The parsed semantics file.
        """
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
    """This represents provides access to the metadata for the object, found in metadata.txt.

    Attributes:
        model_cat: The model category. Will be something like "Chair".
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
    "This class describes the grouping of files for each object."

    def __init__(self, obj_dir: Union[str, Path]):
        """Initializer.

        Args:
            obj_dir (Union[str, Path]): The object-level directory. For instance, "pm/111000/". This file should contain a urdf file, semantics.txt, and metadata.txt.
        """
        self.obj_dir = Path(obj_dir)

        # Load the data in.
        self.semantics = Semantics.from_file(self.semantics_fn)
        self.metadata = Metadata.from_file(self.meta_fn)
        self.obj = PMTree.parse_urdf(self.urdf_fn)

        self.__issubset: Optional[bool] = None
        self.__issame: Optional[bool] = None

    @property
    def obj_id(self) -> str:
        """The object ID."""
        return self.obj_dir.name

    @property
    def category(self) -> str:
        """Object category."""
        return self.metadata.model_cat

    @property
    def semantics_fn(self) -> Path:
        """The semantics file"""
        return self.obj_dir / "semantics.txt"

    @property
    def urdf_fn(self) -> Path:
        """The URDF file"""
        return self.obj_dir / "mobility.urdf"

    @property
    def meta_fn(self) -> Path:
        """The metadata file"""
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

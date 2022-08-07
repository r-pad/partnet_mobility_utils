import copy
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from xml.etree import ElementTree as ET

import numpy as np


@dataclass
class Joint:
    """A URDF Joint. See http://wiki.ros.org/urdf/XML/joint for descriptions of each field. Not all fields are supported."""

    name: str
    type: str
    parent: str
    child: str
    origin: Optional[Tuple[np.ndarray, np.ndarray]] = None
    axis: Optional[np.ndarray] = None
    limit: Optional[Tuple[float, float]] = None


@dataclass
class Link:
    """A URDF Link. See http://wiki.ros.org/urdf/XML/link for more details. Not all fields are supported."""

    name: str

    # Since the collision geometries and the visual geometries are the same
    # for all the data, we just call it meshes. They're also always at 0 0 0.
    mesh_names: Set[str]


class PMTree:
    """The parsed URDF Tree."""

    ROOT_LINK = "base"  # specific to partnet-mobility, all objs have a base obj

    def __init__(self, links: List[Link], joints: List[Joint]):
        """Initializer, based on a flattened representation of the tree. The best way to instantiate the object is from `parse_urdf`"""

        self.__links = links
        self.__joints = joints
        self.__linkmap = {link.name: ix for ix, link in enumerate(self.__links)}
        self.__childmap = {joint.child: ix for ix, joint in enumerate(self.__joints)}
        self.__jointmap = {joint.name: ix for ix, joint in enumerate(self.__joints)}

    def get_chain(self, link_name: str) -> List[Joint]:
        """Get the kinematic chain, starting from the base of the object, and ending with the desired link.

        Args:
            link_name (str): The child node in the chain.

        Returns:
            List[Joint]: All links in the kinematic chain.
        """
        parent_dict: Dict[str, Union[Tuple[str, Joint], Tuple[None, None]]]
        parent_dict = {joint.child: (joint.parent, joint) for joint in self.joints}
        parent_dict["base"] = None, None

        parent_name, parent_link = parent_dict[link_name]
        parents: List[Joint] = []
        while parent_name is not None and parent_link is not None:
            parents = [parent_link] + parents
            parent_name, parent_link = parent_dict[parent_name]

        return parents

    @property
    def links(self) -> List[Link]:
        """All links in the object."""
        return copy.deepcopy(self.__links)

    @property
    def joints(self) -> List[Joint]:
        """All joints in the object."""
        return copy.deepcopy(self.__joints)

    def get_joint(self, name: str) -> Joint:
        """Get joints by name.

        Args:
            name (str): The name of the joint.

        Raises:
            ValueError: Joint not found.

        Returns:
            Joint: The desired joint.
        """
        if name not in self.__jointmap:
            raise ValueError(f"invalid joint name: {name}")
        return copy.deepcopy(self.__joints[self.__jointmap[name]])

    def get_joint_by_child(self, child_name: str) -> Joint:
        """Get a joint by its child name. Each child only has one parent.

        Args:
            child_name (str): The child name.

        Raises:
            ValueError: Joint not found

        Returns:
            Joint: The joint connecting to the child.
        """
        if child_name not in self.__childmap:
            raise ValueError(f"invalid child name: {child_name}")
        return copy.deepcopy(self.__joints[self.__childmap[child_name]])

    def get_link(self, name: str) -> Link:
        """Get link by name.

        Args:
            name (str): The name of the link.

        Raises:
            ValueError: Link not found.

        Returns:
            Link: The desired link.
        """
        if name not in self.__linkmap:
            raise ValueError(f"invalid link name: {name}")
        return copy.deepcopy(self.__links[self.__linkmap[name]])

    @property
    def children(self) -> Dict[str, List[str]]:
        """The child dictionary for the tree."""
        children: Dict[str, List[str]] = defaultdict(list)
        for link in self.__links:
            children[link.name] = []
        for joint in self.__joints:
            children[joint.parent].append(joint.child)
        return dict(children)

    @property
    def descendants(self) -> Dict[str, List[str]]:
        """Get all descendants for each node."""
        descendents = defaultdict(list)
        children = self.children
        link_dict = {link.name: link for link in self.__links}

        def dfs(link, ancestor_keys):
            for key in ancestor_keys:
                descendents[key].append(link.name)

            descendents[link.name] = []
            for child in children[link.name]:
                dfs(link_dict[child], ancestor_keys + [link.name])

        dfs(link_dict[self.ROOT_LINK], [])
        return dict(descendents)

    def __str__(self) -> str:
        return f"PMObject(links={len(self.__links)}, joints={len(self.__joints)})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def meshes(self) -> Set[str]:
        """Get all meshes for the object."""
        return reduce(lambda a, b: a.union(b.mesh_names), self.links, set())

    @staticmethod
    def parse_urdf_from_string(urdf_string: str) -> "PMTree":
        """Parses a given URDF string into a PMTree. This is necessary because
        many of the partnet-mobility objects are malformed, and violate the URDF schema.
        Otherwise I'd just have used a urdf parsing library.

        Args:
            urdf_string (str): A string of the URDF.

        Returns:
            PMTree: A parsed URDF tree.
        """
        robot = ET.fromstring(urdf_string)

        def parse_pose(element: ET.Element) -> Tuple[np.ndarray, np.ndarray]:
            xyz = (
                np.asarray(element.attrib["xyz"].split(" "), dtype=float)
                if "xyz" in element.attrib
                else np.asarray([0.0, 0.0, 0.0])
            )
            rpy = (
                np.asarray(element.attrib["rpy"].split(" "), dtype=float)
                if "rpy" in element.attrib
                else np.asarray([0.0, 0.0, 0.0])
            )
            return xyz, rpy

        def parse_link(link_et: ET.Element) -> Link:
            link_name = link_et.attrib["name"]
            # Recursively (via iter()) grab the meshes.
            meshes = {
                it.attrib["filename"] for it in link_et.iter() if it.tag == "mesh"
            }
            return Link(name=link_name, mesh_names=meshes)

        def parse_joint(joint_et: ET.Element) -> Joint:
            joint_name = joint_et.attrib["name"]
            joint_type = joint_et.attrib["type"]
            child = joint_et.find("child").attrib["link"]  # type: ignore
            parent = joint_et.find("parent").attrib["link"]  # type: ignore

            # Parse the optional fields.
            origin_et = joint_et.find("origin")
            origin = parse_pose(origin_et) if origin_et is not None else None
            axis_et = joint_et.find("axis")

            # There are a number of malformed entries (i.e. joint_0 in 103252/mobility.urdf)
            # where we need to replace None -> 0
            axis: Optional[np.ndarray]
            if axis_et is not None:
                xyzstrs = axis_et.attrib["xyz"].split(" ")
                xyzstrs = [xyzstr if xyzstr != "None" else "0" for xyzstr in xyzstrs]
                axis = np.asarray(xyzstrs, dtype=float)
            else:
                axis = None

            limit_et = joint_et.find("limit")
            limit = (
                (float(limit_et.attrib["lower"]), float(limit_et.attrib["upper"]))
                if limit_et is not None
                else None
            )

            return Joint(
                name=joint_name,
                type=joint_type,
                parent=parent,
                child=child,
                origin=origin,
                axis=axis,
                limit=limit,
            )

        link_ets = robot.findall("link")
        joint_ets = robot.findall("joint")

        links = [parse_link(link_et) for link_et in link_ets]
        joints = [parse_joint(joint_et) for joint_et in joint_ets]

        return PMTree(links, joints)

    @staticmethod
    def parse_urdf(urdf_fn: Union[str, Path]) -> "PMTree":
        """Parses a given URDF string into a PMTree. This is necessary because
        many of the partnet-mobility objects are malformed, and violate the URDF schema.
        Otherwise I'd just have used a urdf parsing library.

        Args:
            urdf_fn (Union[str, Path]): URDF file to parse.

        Raises:
            ValueError: URDF File not found.

        Returns:
            PMTree: the parsed PMTree.
        """
        urdf_path = Path(urdf_fn)
        if not (urdf_path.exists() and urdf_path.suffix == ".urdf"):
            raise ValueError(f"{urdf_path} is not a URDF file")

        with urdf_path.open("r") as f:
            contents = f.read()
        return PMTree.parse_urdf_from_string(contents)

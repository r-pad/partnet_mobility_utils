import copy
from functools import reduce
from typing import Dict, List, Sequence

import numpy as np
import numpy.typing as npt
import trimesh
from scipy.spatial.transform import Rotation

from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.urdf import Joint


def fk(chain: List[Joint], joint_angles: Sequence[float]) -> np.ndarray:
    def compute_T_link_childnew(joint: Joint, angle: float) -> np.ndarray:
        T_link_child: np.ndarray = np.eye(4)
        if joint.origin is not None:
            xyz, rpy = joint.origin
            T_link_child[:3, 3] = xyz
            T_link_child[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()

        T_articulation: np.ndarray = np.eye(4)
        if joint.type == "revolute" or joint.type == "continuous":
            theta = angle
            if theta != 0.0:
                axis = (
                    joint.axis
                    if joint.axis is not None
                    else np.asarray([1.0, 0.0, 0.0])
                )
                R = trimesh.transformations.rotation_matrix(theta, axis)
                T_articulation[:3, :3] = R[:3, :3]

        elif joint.type == "prismatic":
            theta = angle
            axis = joint.axis if joint.axis is not None else np.asarray([1.0, 0.0, 0.0])
            axis = axis / np.linalg.norm(axis)

            T_articulation[:3, 3] = axis * theta

        T_link_childnew: np.ndarray = T_link_child @ T_articulation
        return T_link_childnew

    tforms = [compute_T_link_childnew(j, a) for j, a in zip(chain, joint_angles)]

    # Compose all transforms into a single one. Basically, we left-apply each transform
    # down the chain.
    T_base_endlink = reduce(lambda T_gp_p, T_p_c: T_gp_p @ T_p_c, tforms, np.eye(4))  # type: ignore

    return T_base_endlink


def articulate_points(
    P_world_pts: np.ndarray,
    T_world_base: np.ndarray,
    kinematic_chain: List[Joint],
    current_ja: Sequence[float],
    target_ja: Sequence[float],
) -> np.ndarray:
    if P_world_pts.shape == (3,):
        P_world_pts = np.expand_dims(P_world_pts, 0)

    # Validation.
    assert len(P_world_pts.shape) == 2
    assert P_world_pts.shape[1] == 3
    assert len(kinematic_chain) == len(current_ja) == len(target_ja)
    assert T_world_base.shape == (4, 4)

    N = len(P_world_pts)
    Ph_world_pts = np.concatenate([P_world_pts, np.ones((N, 1))], axis=1)

    ################## STEP 1. ####################
    # Put all the points in the frame of the final joint.

    # Do forward kinematics to get the position of the final link.
    T_base_endlink = fk(kinematic_chain, current_ja)
    T_world_endlink = T_world_base @ T_base_endlink
    T_endlink_world = np.linalg.inv(T_world_endlink)

    # Points in the final link frame.
    Ph_endlink_pts = (T_endlink_world @ Ph_world_pts.T).T
    assert Ph_endlink_pts.shape == (N, 4)

    ################## STEP 2. ####################
    # Compute the frame of the final joint for the new joint angles,
    # and find the relative transform from the original frame.

    T_base_endlinknew = fk(kinematic_chain, target_ja)
    T_endlink_endlinknew = np.linalg.inv(T_base_endlink) @ T_base_endlinknew

    ################## STEP 3. ####################
    # Apply that relative transform to the points in the end link frame,
    # and put it all back in the world frame.
    Ph_endlink_ptsnew = (T_endlink_endlinknew @ Ph_endlink_pts.T).T
    Ph_world_ptsnew = (T_world_endlink @ Ph_endlink_ptsnew.T).T
    assert Ph_world_ptsnew.shape == (N, 4)
    P_world_ptsnew: np.ndarray = Ph_world_ptsnew[:, :3]  # not homogenous anymore.

    return P_world_ptsnew


def articulate_joint(
    obj: PMObject,
    current_jas: Dict[str, float],
    link_to_actuate: str,
    amount_to_actuate: float,
    pos: npt.NDArray[np.float32],
    seg: npt.NDArray[np.uint],
    labelmap: Dict[str, int],
    T_world_base: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    chain = obj.obj.get_chain(link_to_actuate)

    c_jas = [current_jas[joint.name] for joint in chain]
    target_jas = copy.deepcopy(current_jas)
    target_jas[obj.obj.get_joint_by_child(link_to_actuate).name] += amount_to_actuate
    t_jas = [target_jas[joint.name] for joint in chain]

    links_to_include = [link_to_actuate] + obj.obj.descendants[link_to_actuate]
    seg_ids = np.asarray([labelmap[id] for id in links_to_include])

    to_act = (seg.reshape(-1, 1) == seg_ids.reshape(1, -1)).any(axis=-1)
    P_world_pts = pos[to_act]

    P_world_pts_new = articulate_points(P_world_pts, T_world_base, chain, c_jas, t_jas)

    pos_new = np.copy(pos)
    pos_new[to_act] = P_world_pts_new

    return pos_new

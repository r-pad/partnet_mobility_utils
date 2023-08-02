import os
from typing import Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
from rpad.core.distributed import NPSeed
from rpad.pybullet_libs.camera import Camera, Render
from rpad.pybullet_libs.utils import get_obj_z_offset, isnotebook, suppress_stdout
from scipy.spatial.transform import Rotation as R

from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.render import PartialPC, PMRenderer

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:
    print("pybullet is not installed. Please install")
    raise ImportError from exc


def sample_az_ele(radius, az_lo, az_hi, ele_lo, ele_hi, seed=None):
    """Sample random azimuth elevation pair and convert to cartesian."""

    rng = np.random.default_rng(seed)

    azimuth = rng.uniform(az_lo, az_hi)
    elevation = rng.uniform(ele_lo, ele_hi)

    x = -radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation) * np.sin(azimuth)
    y = radius * np.cos(azimuth)

    return x, y, z, azimuth, elevation


def calc_rot(v1, v2):
    angle = np.arccos(np.dot(v1, v2))
    axis = np.cross(v1, v2)
    axis = axis / np.linalg.norm(axis)
    qx = axis[0] * np.sin(angle / 2)
    qy = axis[1] * np.sin(angle / 2)
    qz = axis[2] * np.sin(angle / 2)
    qw = np.cos(angle / 2)
    quat = np.array([qx, qy, qz, qw])
    quat = quat / np.linalg.norm(quat)
    return quat


def randomize_camera(env, seed=None):
    """Randomize random camera viewpoints"""
    target = env.cabinet.get_pose().p
    can_cam_loc = env.cameras[1].sub_cameras[0].get_pose().p
    radius = np.linalg.norm(can_cam_loc - target)
    # Can change the bounds of sampling too
    x, y, z, az, ele = sample_az_ele(
        radius, np.deg2rad(70), np.deg2rad(110), np.deg2rad(30), np.deg2rad(60)
    )
    v1 = target - can_cam_loc
    v1 = v1 / np.linalg.norm(v1)
    new_loc = np.array([x, y, z])
    v2 = target - new_loc
    v2 = v2 / np.linalg.norm(v2)
    trans_quat = calc_rot(v1, v2)
    for i in range(3):
        cam = env.cameras[1].sub_cameras[i]
        new_mat = (
            R.from_quat(trans_quat).as_matrix()
            @ R.from_quat(env.cameras[1].sub_cameras[i].get_pose().q).as_matrix()
        )
        new_quat = R.from_matrix(new_mat).as_quat()
        new_pose = Pose(new_loc, new_quat)
        cam.set_initial_pose(new_pose)
    return


class PMRenderEnv:
    def __init__(
        self,
        obj_id: str,
        dataset_path: str,
        camera_pos: List = [-2, 0, 2],
        gui: bool = False,
        with_plane: bool = True,
    ):
        self.with_plane = with_plane
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)

        # Add in a plane.
        if with_plane:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        # Add in gravity.
        p.setGravity(0, 0, 0, self.client_id)

        # Add in the object.
        self.obj_id_str = obj_id
        obj_urdf = os.path.join(dataset_path, obj_id, "mobility.urdf")

        if isnotebook() or "PYTEST_CURRENT_TEST" in os.environ:
            self.obj_id = p.loadURDF(
                obj_urdf,
                useFixedBase=True,
                # flags=p.URDF_MAINTAIN_LINK_ORDER,
                physicsClientId=self.client_id,
            )

        else:
            with suppress_stdout():
                self.obj_id = p.loadURDF(
                    obj_urdf,
                    useFixedBase=True,
                    # flags=p.URDF_MAINTAIN_LINK_ORDER,
                    physicsClientId=self.client_id,
                )

        # The object isn't placed at the bottom of the scene.
        self.minz = get_obj_z_offset(self.obj_id, self.client_id)
        p.resetBasePositionAndOrientation(
            self.obj_id,
            posObj=[0, 0, -self.minz],
            ornObj=[0, 0, 0, 1],
            physicsClientId=self.client_id,
        )
        self.T_world_base = np.eye(4)
        self.T_world_base[2, 3] = -self.minz

        # Create a camera.
        self.camera = Camera(pos=camera_pos, znear=0.01, zfar=10)

        # From https://pybullet.org/Bullet/phpBB3/viewtopic.php?f=24&t=12728&p=42293&hilit=linkIndex#p42293
        self.link_name_to_index = {
            p.getBodyInfo(self.obj_id, physicsClientId=self.client_id)[0].decode(
                "UTF-8"
            ): -1,
        }
        self.jn_to_ix = {}

        # Get the segmentation.
        for _id in range(p.getNumJoints(self.obj_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.obj_id, _id, physicsClientId=self.client_id)
            joint_name = info[1].decode("UTF-8")
            link_name = info[12].decode("UTF-8")
            self.link_name_to_index[link_name] = _id

            # Only store if the joint is one we can control.
            if info[2] == p.JOINT_REVOLUTE or info[2] == p.JOINT_PRISMATIC:
                self.jn_to_ix[joint_name] = _id

    def render(self, return_prgb=False, link_seg=True) -> Render:
        return self.camera.render(self.client_id, self.with_plane, link_seg)

    def set_camera(
        self,
        camera_xyz: Union[Literal["random"], Tuple[float, float, float]],
        seed: Optional[NPSeed] = None,
    ):
        if camera_xyz == "random":
            x, y, z, az, el = sample_az_ele(
                np.sqrt(8),
                np.deg2rad(30),
                np.deg2rad(150),
                np.deg2rad(30),
                np.deg2rad(60),
                seed=seed,
            )
            camera_xyz = (x, y, z)

        self.camera.set_camera_position(camera_xyz)

    def _get_random_joint_value(
        self, joint_name: str, openclose=False, seed=None
    ) -> float:
        rng = np.random.default_rng(seed)

        i = self.jn_to_ix[joint_name]
        jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
        lower, upper = jinfo[8], jinfo[9]
        if openclose:
            angle: float = [lower, upper][rng.choice(2)]
        else:
            angle = rng.random() * (upper - lower) + lower
        return angle

    def _get_random_joint_values(self, openclose=False, seed=None) -> Dict[str, float]:
        return {
            k: self._get_random_joint_value(k, openclose, seed)
            for k in self.jn_to_ix.keys()
        }

    def set_joint_angles(
        self,
        joints: Union[
            Literal["random", "random-oc", "open", "closed"],
            Mapping[str, Union[float, Literal["random", "random-oc"]]],
            None,
        ] = None,
        seed=None,
    ) -> None:
        if joints is None:
            joint_dict = {jn: 0.0 for jn in self.jn_to_ix.keys()}
        elif joints == "random":
            joint_dict = self._get_random_joint_values(openclose=False, seed=seed)
        elif joints == "random-oc":
            joint_dict = self._get_random_joint_values(openclose=True, seed=seed)
        elif joints == "open":
            raise NotImplementedError
        elif joints == "closed":
            raise NotImplementedError
        else:
            # Default zeros.
            new_joints = {jn: 0.0 for jn in self.jn_to_ix.keys()}

            # Set the values to either a fixed value or a random one.
            for k, v in joints.items():
                if not k in self.jn_to_ix:
                    raise ValueError(f"invalid joint {k}")

                if isinstance(v, float):
                    new_joints[k] = v
                else:
                    if v == "open" or v == "closed":
                        raise NotImplementedError
                    # Randomize, and determine if open/close
                    new_joints[k] = self._get_random_joint_value(
                        k, v == "random-oc", seed=seed
                    )
            joint_dict = new_joints  # type: ignore

        # Reset the joints.
        for jn, ja in joint_dict.items():
            jix = self.jn_to_ix[jn]
            jv = 0  # joint velocity
            p.resetJointState(self.obj_id, jix, ja, jv, self.client_id)

    def get_joint_angles(self) -> Dict[str, float]:
        angles = {}
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
            jstate = p.getJointState(self.obj_id, i, self.client_id)
            angles[jinfo[1].decode("UTF-8")] = jstate[0]
        return angles

    def get_joint_ranges(self) -> Dict[str, Tuple[float, float]]:
        ranges = {}
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
            lower, upper = jinfo[8], jinfo[9]
            ranges[jinfo[1].decode("UTF-8")] = lower, upper
        return ranges

    def close(self):
        p.disconnect(self.client_id)


class PybulletRenderer(PMRenderer):
    def __init__(self):
        self._render_env = None

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
        seed: Optional[int] = None,
    ) -> PartialPC:
        """Sample a partial pointcloud using the Pybullet GL renderer. Currently only supports
        randomized parameters.

        Args:
            randomize_joints: Decide whether and how to randomize joints. Defaults to False.
                False -> no randomization
                "all" -> Randomize all joints on the object
                [list] -> Randomize just these joints
            randomize_camera (bool, optional): Randomize the camera position. Defaults to False.
                Only occurs in a window.
            set_joints: Decide whether and how to set the joints. Can't also randomize the joints.

        Returns:
            PartialPC: A big dictionary of things. See PartialPC above for what you get.
        """
        if self._render_env is None:
            self._render_env = PMRenderEnv(
                pm_obj.obj_dir.name, str(pm_obj.obj_dir.parent)
            )

        rng = np.random.default_rng(seed)
        seed1, seed2 = rng.bit_generator._seed_seq.spawn(2)  # type: ignore

        self._render_env.set_joint_angles(joints, seed=seed1)

        if camera_xyz is not None:
            self._render_env.set_camera(camera_xyz, seed=seed2)

        obs = self._render_env.render()
        rgb = obs["rgb"]
        depth = obs["depth"]
        seg = obs["seg"]
        P_cam = obs["P_cam"]
        P_world = obs["P_world"]
        pc_seg = obs["pc_seg"]
        segmap = obs["segmap"]

        # Reindex the segmentation.
        pc_seg_obj = np.ones_like(pc_seg) * -1
        for k, (body, link) in segmap.items():
            if body == self._render_env.obj_id:
                ixs = pc_seg == k
                pc_seg_obj[ixs] = link

        return {
            "id": pm_obj.obj_id,
            "pos": P_world,
            "seg": pc_seg_obj,
            "frame": "world",
            "T_world_cam": self._render_env.camera.T_world2cam,
            "T_world_base": np.copy(self._render_env.T_world_base),
            # "proj_matrix": None,
            "labelmap": self._render_env.link_name_to_index,
            "angles": self._render_env.get_joint_angles(),
        }

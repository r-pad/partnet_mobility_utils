import functools
import os
import sys
from contextlib import contextmanager
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from partnet_mobility_utils.data import PMObject
from partnet_mobility_utils.render import PartialPC

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:
    print("pybullet is not installed. Please install")
    raise ImportError from exc


def sample_az_ele(radius, az_lo, az_hi, ele_lo, ele_hi):
    """Sample random azimuth elevation pair and convert to cartesian."""
    azimuth = np.random.uniform(az_lo, az_hi)
    elevation = np.random.uniform(ele_lo, ele_hi)

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


def randomize_camera(env):
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


def get_obj_z_offset(object_id, sim_id, starting_min=0.0):
    bboxes = [p.getAABB(object_id, physicsClientId=sim_id)]  # Add the base one.
    for i in range(p.getNumJoints(object_id, physicsClientId=sim_id)):
        aabb = p.getAABB(object_id, i, physicsClientId=sim_id)  # Add the links.
        bboxes.append(aabb)
    minz = functools.reduce(lambda a, b: min(a, b[0][2]), bboxes, starting_min)
    return minz


def get_obj_bbox_xy(object_id, sim_id):
    bboxes = [p.getAABB(object_id, physicsClientId=sim_id)]  # Add the base one.
    for i in range(p.getNumJoints(object_id, physicsClientId=sim_id)):
        aabb = p.getAABB(object_id, i, physicsClientId=sim_id)  # Add the links.
        bboxes.append(aabb)
    # xmin = np.array(bboxes)[:, :, 0].min()
    xmin = functools.reduce(lambda a, b: min(a, b[0][0]), bboxes, 0.0)
    # xmax = np.array(bboxes)[:, :, 0].max()
    xmax = functools.reduce(lambda a, b: max(a, b[1][0]), bboxes, 0.0)
    # ymin = np.array(bboxes)[:, :, 1].min()
    ymin = functools.reduce(lambda a, b: min(a, b[0][1]), bboxes, 0.0)
    # ymax = np.array(bboxes)[:, :, 1].max()
    ymax = functools.reduce(lambda a, b: max(a, b[1][1]), bboxes, 0.0)
    return [[xmin, xmax], [ymin, ymax]]


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


@contextmanager
def suppress_stdout():

    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


RENDER_WIDTH = 640
RENDER_HEIGHT = 480
CAMERA_INTRINSICS = np.array(
    [
        [450, 0, RENDER_WIDTH / 2],
        [0, 450, RENDER_HEIGHT / 2],
        [0, 0, 1],
    ]
)

T_CAMGL_2_CAM = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]
)


def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.
    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


class Camera:
    def __init__(
        self,
        pos,
        render_height=RENDER_HEIGHT,
        render_width=RENDER_WIDTH,
        znear=0.01,
        zfar=6,
        intrinsics=CAMERA_INTRINSICS,
        target=None,
    ):
        #######################################
        # First, compute the projection matrix.
        #######################################
        self.intrinsics = intrinsics
        focal_length = intrinsics[0][0]
        self.znear, self.zfar = znear, zfar
        self.fovh = (np.arctan((render_height / 2) / focal_length) * 2 / np.pi) * 180
        self.render_width = render_width
        self.render_height = render_height

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = render_width / render_height
        self.proj_list = p.computeProjectionMatrixFOV(
            self.fovh, aspect_ratio, self.znear, self.zfar
        )

        #######################################
        # Next, compute the view matrix.
        #######################################
        if target is None:
            target = [0, 0, 0.5]
        self.target = target
        self.view_list = self.__view_list(pos, target)

    @property
    def view_list(self):
        return self._view_list

    @view_list.setter
    def view_list(self, value):
        self._view_list = value
        self.T_camgl2world = np.asarray(value).reshape(4, 4).T
        self.T_world2camgl = np.linalg.inv(self.T_camgl2world)
        self.T_world2cam = self.T_world2camgl @ T_CAMGL_2_CAM

    @staticmethod
    def __view_list(eye, target):
        up = [0.0, 0.0, 1.0]
        target = target
        view_list = p.computeViewMatrix(eye, target, up)
        return view_list

    def set_camera_position(self, pos):
        self.view_list = self.__view_list(pos, self.target)

    def render(
        self, client_id, return_prgb=False, has_plane=True, link_seg=True
    ) -> Tuple[np.ndarray, ...]:
        if link_seg:
            _, _, rgb, zbuffer, seg = p.getCameraImage(
                RENDER_WIDTH,
                RENDER_HEIGHT,
                self.view_list,
                self.proj_list,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=client_id,
            )
        else:
            _, _, rgb, zbuffer, seg = p.getCameraImage(
                RENDER_WIDTH,
                RENDER_HEIGHT,
                self.view_list,
                self.proj_list,
                physicsClientId=client_id,
            )

        # Sometimes on mac things get weird.
        if isinstance(rgb, tuple):
            rgb = np.asarray(rgb).reshape(RENDER_HEIGHT, RENDER_WIDTH, 4)
            zbuffer = np.asarray(zbuffer).reshape(RENDER_HEIGHT, RENDER_WIDTH)
            seg = np.asarray(seg).reshape(RENDER_HEIGHT, RENDER_WIDTH)

        zfar, znear = self.zfar, self.znear
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth

        P_cam = get_pointcloud(depth, self.intrinsics)
        foreground_ixs = seg > 0 if has_plane else seg > -1
        pc_seg = seg[foreground_ixs].flatten()
        P_cam = P_cam[foreground_ixs]
        P_cam = P_cam.reshape(-1, 3)
        P_rgb = rgb[foreground_ixs]
        P_rgb = P_rgb[:, :3].reshape(-1, 3)

        Ph_cam = np.concatenate([P_cam, np.ones((len(P_cam), 1))], axis=1)
        Ph_world = (self.T_world2cam @ Ph_cam.T).T
        P_world = Ph_world[:, :3]

        # Undoing the bitmask so we can get the obj_id, link_index
        segmap: Optional[Dict]
        if link_seg:
            segmap = {
                label: ((label & ((1 << 24) - 1)), (label >> 24) - 1)
                for label in np.unique(seg)
            }
        else:
            segmap = None

        if return_prgb:
            return rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap  # type: ignore

        return rgb, depth, seg, P_cam, P_world, pc_seg, segmap  # type: ignore


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

        # Get the segmentation.
        for _id in range(p.getNumJoints(self.obj_id, physicsClientId=self.client_id)):
            _name = p.getJointInfo(self.obj_id, _id, physicsClientId=self.client_id)[
                12
            ].decode("UTF-8")
            self.link_name_to_index[_name] = _id

    def render(self, return_prgb=False, link_seg=True):
        if not return_prgb:
            rgb, depth, seg, P_cam, P_world, pc_seg, segmap = self.camera.render(
                self.client_id, return_prgb, self.with_plane, link_seg
            )
            return rgb, depth, seg, P_cam, P_world, pc_seg, segmap
        else:
            rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = self.camera.render(
                self.client_id, return_prgb, self.with_plane, link_seg
            )
            return rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap

    def set_camera(self, camera_pos: List[int]):
        self.camera.set_camera_position(camera_pos)

    def randomize_joints(self, joints: Optional[Sequence[str]] = None):
        if joints is not None:
            joints_set = set(joints)
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)

            # Skip joint if we are filtering and it's not selected.
            if joints is not None and jinfo[1].decode("UTF-8") not in joints_set:
                continue

            if jinfo[2] == p.JOINT_REVOLUTE or jinfo[2] == p.JOINT_PRISMATIC:
                lower, upper = jinfo[8], jinfo[9]
                angle = np.random.random() * (upper - lower) + lower
                p.resetJointState(self.obj_id, i, angle, 0, self.client_id)

    def randomize_specific_joints(self, joint_list):
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
            if jinfo[12].decode("UTF-8") in joint_list:
                lower, upper = jinfo[8], jinfo[9]
                angle = np.random.random() * (upper - lower) + lower
                p.resetJointState(self.obj_id, i, angle, 0, self.client_id)

    def articulate_specific_joints(self, joint_list, amount):
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
            if jinfo[12].decode("UTF-8") in joint_list:
                lower, upper = jinfo[8], jinfo[9]
                angle = amount * (upper - lower) + lower
                p.resetJointState(self.obj_id, i, angle, 0, self.client_id)

    def randomize_joints_openclose(self, joint_list):
        randind = np.random.choice([0, 1])
        # Close: 0
        # Open: 1
        self.close_or_open = randind
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
            if jinfo[12].decode("UTF-8") in joint_list:
                lower, upper = jinfo[8], jinfo[9]
                angles = [lower, upper]
                angle = angles[randind]
                p.resetJointState(self.obj_id, i, angle, 0, self.client_id)

    def set_joint_angles(self, joints: Dict[str, float]):
        """Sets joint angles, interpolating between lower and upper."""
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
            joint_name = jinfo[1].decode("UTF-8")
            # Skip joint if we are filtering and it's not selected.
            if joint_name not in joints:
                continue

            # Interpolate between upper and lower.
            assert 0.0 <= joints[joint_name] <= 1.0
            lower, upper = jinfo[8], jinfo[9]
            angle = joints[joint_name] * (upper - lower) + lower
            p.resetJointState(self.obj_id, i, angle, 0, self.client_id)

    def get_joint_angles(self) -> Dict[str, float]:
        angles = {}
        for i in range(p.getNumJoints(self.obj_id, self.client_id)):
            jinfo = p.getJointInfo(self.obj_id, i, self.client_id)
            jstate = p.getJointState(self.obj_id, i, self.client_id)
            angles[jinfo[1].decode("UTF-8")] = jstate[0]
        return angles

    def randomize_camera(self):
        x, y, z, az, el = sample_az_ele(
            np.sqrt(8), np.deg2rad(30), np.deg2rad(150), np.deg2rad(30), np.deg2rad(60)
        )
        self.camera.set_camera_position((x, y, z))


class PybulletRenderer:
    def __init__(self):
        self.__render_env = None

    def render(
        self,
        obj: PMObject,
        randomize_joints: Union[Literal[False], Literal["all"], Sequence[str]] = False,
        randomize_camera: bool = False,
        set_joints: Union[Literal[False], Sequence[Tuple[str, float]]] = False,
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
        if self.__render_env is None:
            self.__render_env = PMRenderEnv(obj.obj_dir.name, str(obj.obj_dir.parent))

        if randomize_joints and set_joints:
            raise ValueError("unable to randomize and set joints")
        if randomize_joints:
            if randomize_joints == "all":
                self.__render_env.randomize_joints()
            else:
                self.__render_env.randomize_joints(randomize_joints)
        if set_joints:
            self.__render_env.set_joint_angles({jn: ja for jn, ja in set_joints})
        if randomize_camera:
            self.__render_env.randomize_camera()

        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = self.__render_env.render()

        # Reindex the segmentation.
        pc_seg_obj = np.ones_like(pc_seg) * -1
        for k, (body, link) in segmap.items():
            if body == self.__render_env.obj_id:
                ixs = pc_seg == k
                pc_seg_obj[ixs] = link

        return {
            "pos": P_world,
            "seg": pc_seg_obj,
            "frame": "world",
            "T_world_cam": self.__render_env.camera.T_world2cam,
            "T_world_base": np.copy(self.__render_env.T_world_base),
            # "proj_matrix": None,
            "labelmap": self.__render_env.link_name_to_index,
            "angles": self.__render_env.get_joint_angles(),
        }

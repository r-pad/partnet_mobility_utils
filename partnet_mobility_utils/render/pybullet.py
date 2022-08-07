from typing import Literal, Sequence, Tuple, Union

import numpy as np

from partnet_mobility_utils.render import PartialPC

try:
    pass
except ImportError as exc:
    print("pybullet is not installed. Please install")
    raise ImportError from exc


class PybulletRenderer:
    def __init__(self):
        pass

    def render(
        self,
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
            self.__render_env = PMRenderEnv(self.obj_dir.name, str(self.obj_dir.parent))

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

import numpy as np

from partnet_mobility_utils.pm_data import PMObject
from partnet_mobility_utils.render import PartialPC

try:
    import sapien
except ImportError as exc:
    print("sapien not installed. please install, although only Linux is supported")
    raise ImportError from exc


class SAPIENRenderer:
    def __init__(self):
        self._engine = sapien.core.Engine()
        self._renderer = sapien.core.VulkanRenderer(offscreen_only=True)
        self._engine.set_renderer(self._renderer)

    def render(self, obj: PMObject) -> PartialPC:
        scene = self._engine.create_scene()
        scene.set_timestep(1 / 100.0)

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        # load as a kinematic articulation
        asset = loader.load_kinematic(obj.urdf_fn)
        assert asset, "URDF not loaded."

        rscene = scene.get_renderer_scene()
        rscene.set_ambient_light([0.5, 0.5, 0.5])
        rscene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        rscene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        rscene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        rscene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

        # ---------------------------------------------------------------------------- #
        # Camera
        # ---------------------------------------------------------------------------- #
        near, far = 0.1, 100
        width, height = 640, 480
        camera_mount_actor = scene.create_actor_builder().build_kinematic()
        camera = scene.add_mounted_camera(
            name="camera",
            actor=camera_mount_actor,
            pose=sapien.core.Pose(),  # relative to the mounted actor
            width=width,
            height=height,
            fovx=np.deg2rad(35),
            fovy=np.deg2rad(35),
            near=near,
            far=far,
        )

        # Compute the camera pose by specifying forward(x), left(y) and up(z)
        cam_pos = np.array([-2, -2, 3])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_mount_actor.set_pose(sapien.core.Pose.from_transformation_matrix(mat44))

        scene.step()  # make everything set
        scene.update_render()
        camera.take_picture()

        # ---------------------------------------------------------------------------- #
        # RGBA
        # ---------------------------------------------------------------------------- #
        rgba = camera.get_float_texture("Color")  # [H, W, 4]
        # An alias is also provided
        # rgba = camera.get_color_rgba()  # [H, W, 4]
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")

        # ---------------------------------------------------------------------------- #
        # XYZ position in the camera space
        # ---------------------------------------------------------------------------- #
        # Each pixel is (x, y, z, is_valid) in camera space (OpenGL/Blender)
        position = camera.get_float_texture("Position")  # [H, W, 4]

        # OpenGL/Blender: y up and -z forward
        obj_pts = position[..., 3] < 1.0
        points_opengl = position[..., :3][np.where(obj_pts)]
        # breakpoint()
        points_color = rgba[np.where(obj_pts)][..., :3]
        # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
        # camera.get_model_matrix() must be called after scene.update_render()!
        model_matrix = camera.get_model_matrix()
        points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]

        # SAPIEN CAMERA: z up and x forward
        points_camera = points_opengl[..., [2, 0, 1]] * [-1, -1, 1]

        # Depth
        depth = -position[..., 2]
        depth_image = (depth * 1000.0).astype(np.uint16)

        # ---------------------------------------------------------------------------- #
        # Segmentation labels
        # ---------------------------------------------------------------------------- #
        # Each pixel is (visual_id, actor_id/link_id, 0, 0)
        # visual_id is the unique id of each visual shape
        seg_labels = camera.get_uint32_texture("Segmentation")  # [H, W, 4]

        mesh_level_segmentation = seg_labels[..., 0].astype(np.uint8)  # mesh-level
        actor_level_segmentation = seg_labels[..., 1].astype(np.uint8)  # actor-level

        points_mesh_segmentation = mesh_level_segmentation[np.where(obj_pts)]
        points_actor_segmentation = actor_level_segmentation[np.where(obj_pts)]

        return (
            rgba_img,
            depth_image,
            mesh_level_segmentation,
            actor_level_segmentation,
            points_camera,
            points_world,
            points_color,
            points_mesh_segmentation,
            points_actor_segmentation,
        )

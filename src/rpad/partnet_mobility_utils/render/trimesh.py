import os
from typing import List, Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.render import FullPCData


class TrimeshRenderer:
    def render(self, obj: PMObject, n_points: int) -> FullPCData:
        # First, load all the meshes for the object.
        # link_name, link_mesh (there may be many)
        # Also, they might be scenes, so we should flatten.
        link_meshes_unflattened = []
        for link in obj.obj.links:
            obj_files = [os.path.join(obj.obj_dir, f) for f in link.mesh_names]
            link_meshes_unflattened.extend(
                [(link.name, trimesh.load(obj_file)) for obj_file in obj_files]
            )

        # We need to flatten.
        link_meshes: List[Tuple[str, trimesh.Trimesh]] = []
        for link_name, link_mesh_or_scene in link_meshes_unflattened:
            if isinstance(link_mesh_or_scene, trimesh.Scene):
                scene_meshes = list(link_mesh_or_scene.geometry.values())
                link_meshes.extend([(link_name, mesh) for mesh in scene_meshes])
            elif isinstance(link_mesh_or_scene, trimesh.Trimesh):
                link_meshes.append((link_name, link_mesh_or_scene))
            else:
                raise ValueError("we are getting a mesh type we don't understand")

        # Next, compute the relative areas, and buckets to know how many points
        # to sample per mesh.
        mesh_areas = np.asarray([mtup[1].area for mtup in link_meshes])
        ratios = mesh_areas / mesh_areas.sum()
        buckets = np.floor((n_points * ratios)).astype(int)
        while buckets.sum() < n_points:
            buckets[-1] += 1

        # Finally, sample each mesh recursively.
        points = []
        # colors = []
        normals = []
        ins = []
        sem = []
        art: List[str] = []

        for bucket, (link_name, mesh) in zip(buckets, link_meshes):
            mesh_points, face_indices = trimesh.sample.sample_surface(mesh, bucket)
            face_normals = mesh.face_normals[face_indices]

            # Extract colors. Doesn't seem to work right now.
            # visual = mesh.visual
            # if isinstance(visual, trimesh.visual.TextureVisuals):
            #     color_visual: trimesh.visual.ColorVisuals = visual.to_color()
            # else:
            #     color_visual = visual
            # breakpoint()
            # face_colors = color_visual.face_colors[face_indices]

            normals.append(face_normals)
            points.append(mesh_points)
            # colors.append(face_colors)
            ins.extend([link_name] * len(mesh_points))
            sem.extend([obj.semantics.by_name(link_name).label] * len(mesh_points))
            art.extend([obj.semantics.by_name(link_name).type] * len(mesh_points))

        pos: np.ndarray = np.concatenate(points, axis=0)
        norm: np.ndarray = np.concatenate(normals, axis=0)

        # Lastly, apply the base transformation.
        # All the points are in the same frame, except that there's a base transform.
        bj = [j for j in obj.obj.joints if j.parent == "base"]
        assert len(bj) == 1
        base_joint = bj[0]

        # Extract a transform.
        assert base_joint.origin is not None
        xyz, rpy = base_joint.origin
        T_base_obj = np.eye(4)
        T_base_obj[:3, 3] = xyz
        T_base_obj[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()

        # Apply.
        pos = pos @ T_base_obj[:3, :3].T + xyz.reshape((1, 3))
        norm = norm @ T_base_obj[:3, :3].T

        return {
            "pos": pos,
            "norm": norm,
            "ins": ins,
            "sem": sem,
            "art": art,
        }

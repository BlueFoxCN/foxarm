import trimesh
import numpy as np

from foxarm.common.viewsphere_discretizer import ViewsphereDiscretizer
from foxarm.common.render_mode import RenderMode
from foxarm.common.camera_intrinsics import CameraIntrinsics
from foxarm.cfgs.config import cfg

obj_path = "mini_dexnet/vase.obj"

mesh = trimesh.load_mesh(obj_path)

mesh.vertices -= mesh.center_mass

[tf_list, prob_list] = mesh.compute_stable_poses(n_samples = 1)


stable_pose_meshes = []
for tf in tf_list:
    copied = mesh.copy()
    copied.apply_transform(tf)
    stable_pose_meshes.append(copied)

vp = ViewsphereDiscretizer(min_radius=cfg.min_radius,
                           max_radius=cfg.max_radius,
                           num_radii=cfg.num_radii,
                           min_elev=cfg.min_elev*np.pi,
                           max_elev=cfg.max_elev*np.pi,
                           num_elev=cfg.num_elev,
                           num_az=cfg.num_az,
                           num_roll=cfg.num_roll)


ci = CameraIntrinsics('camera', fx=cfg.focal, fy=cfg.focal, cx=cfg.width/2, cy=cfg.height/2,
                      height=cfg.height, width=cfg.width)

render_modes = [RenderMode.SEGMASK, RenderMode.DEPTH, RenderMode.SCALED_DEPTH]

for stable_pose_mesh in stable_pose_meshes:

    vertices = stable_pose_mesh.vertices
    triangles = stable_pose_mesh.faces
    normals = stable_pose_mesh.vertex_normals

    camera_poses = vp.object_to_camera_poses()

    for tf in camera_poses:
        R = tf.rotation
        t = tf.translation
        P = ci.proj_matrix.dot(np.c_[R, t])

        stable_pose_mesh.show()

        break

    break

        # import pdb
        # pdb.set_trace()

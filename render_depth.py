import trimesh
import numpy as np
import ctypes
from ctypes import *
from numpy.ctypeslib import ndpointer
from mayavi import mlab
from scipy import misc

from foxarm.common.viewsphere_discretizer import ViewsphereDiscretizer
from foxarm.common.render_mode import RenderMode
from foxarm.common.camera_intrinsics import CameraIntrinsics
from foxarm.common.visualizer import Vis
from foxarm.cfgs.config import cfg

obj_path = "mini_dexnet/bar_clamp.obj"

im_height = 227
im_width = 227

mesh = trimesh.load_mesh(obj_path)

mesh.vertices -= mesh.center_mass

render_lib = cdll.LoadLibrary("./render_lib.so")

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

        # Vis.plot_mesh(stable_pose_mesh)
        # mlab.show()

        render_lib.render.restype = ndpointer(dtype=ctypes.c_float, shape=(im_height * im_width,))

        projection = P.reshape(12)
        print("projection: %f %f %f" % (projection[0], projection[1], projection[2]))
        projection = (ctypes.c_double * 12)(*projection)

        num_verts = stable_pose_mesh.vertices.shape[0]
        num_tris = stable_pose_mesh.faces.shape[0]
        num_norms = stable_pose_mesh.vertex_normals.shape[0]

        verts_buffer = stable_pose_mesh.vertices.reshape(-1)
        print("verts_buffer: %f %f %f" % (verts_buffer[0], verts_buffer[1], verts_buffer[2]))
        verts_buffer = (ctypes.c_double * (num_verts * 3))(*verts_buffer)

        tris_buffer = stable_pose_mesh.faces.reshape(-1)
        print("verts_buffer: %d %d %d" % (tris_buffer[0], tris_buffer[1], tris_buffer[2]))
        tris_buffer = (ctypes.c_int * (num_tris * 3))(*tris_buffer)

        norm_buffer = stable_pose_mesh.vertex_normals.reshape(-1)
        print("norm_buffer: %f %f %f" % (norm_buffer[0], norm_buffer[1], norm_buffer[2]))
        norm_buffer = (ctypes.c_double * (num_norms * 3))(*norm_buffer)

        ret = render_lib.render(projection,
                                im_height,
                                im_width,
                                verts_buffer,
                                tris_buffer,
                                norm_buffer,
                                num_verts,
                                num_tris,
                                num_norms)


        ret = ret.reshape(im_height, im_width)

        misc.imsave('depth.jpg', ret)

        break

    break


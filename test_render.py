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

obj_path = "table.obj"

im_height = 227
im_width = 227

mesh = trimesh.load_mesh(obj_path)

# mesh.vertices -= mesh.center_mass

render_lib = cdll.LoadLibrary("./render_lib.so")

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

vertices = mesh.vertices
triangles = mesh.faces
normals = mesh.vertex_normals

camera_poses = vp.object_to_camera_poses()
tf = camera_poses[0]

R = tf.rotation
t = tf.translation
P = ci.proj_matrix.dot(np.c_[R, t])

# Vis.plot_mesh(mesh)
# mlab.show()

render_lib.render.restype = ndpointer(dtype=ctypes.c_float, shape=(im_height * im_width,))

projection = P.reshape(12)
print("projection: %f %f %f" % (projection[0], projection[1], projection[2]))
projection = (ctypes.c_double * 12)(*projection)

num_verts = mesh.vertices.shape[0]
num_tris = mesh.faces.shape[0]
num_norms = mesh.vertex_normals.shape[0]

verts_buffer = mesh.vertices.reshape(-1)
print("verts_buffer: %f %f %f" % (verts_buffer[0], verts_buffer[1], verts_buffer[2]))
verts_buffer = (ctypes.c_double * (num_verts * 3))(*verts_buffer)

tris_buffer = mesh.faces.reshape(-1)
print("verts_buffer: %d %d %d" % (tris_buffer[0], tris_buffer[1], tris_buffer[2]))
tris_buffer = (ctypes.c_int * (num_tris * 3))(*tris_buffer)

norm_buffer = mesh.vertex_normals.reshape(-1)
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

import pdb
pdb.set_trace()

misc.imsave('depth.jpg', ret)

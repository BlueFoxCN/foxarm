import trimesh
import numpy as np
from foxarm.common.visualizer import Vis
from mayavi import mlab
from autolab_core import RigidTransform
'''
camera_path = "camera.obj"
camera_mesh = trimesh.load_mesh(camera_path)

rot1 = RigidTransform.x_axis_rotation(-np.pi / 2)
t1 = RigidTransform(rotation=rot1)
rot2 = RigidTransform.z_axis_rotation(np.pi)
t2 = RigidTransform(rotation=rot2)

camera_mesh.apply_transform(t1.matrix)
camera_mesh.apply_transform(t2.matrix)

t3 = RigidTransform(translation=-camera_mesh.center_mass)

camera_mesh.apply_transform(t3.matrix)

camera_mesh.apply_scale(1 / 200)

'''

camera_path = "new_camera.obj"
camera_mesh = trimesh.load_mesh(camera_path)

Vis.plot_mesh(camera_mesh)

mag = 2 * float(np.max(np.abs(camera_mesh.vertices)))
Vis.plot_mesh(camera_mesh)
Vis.plot_frame(mag)

print(np.max(camera_mesh.vertices, axis=0))
print(np.min(camera_mesh.vertices, axis=0))

obj_path = "mini_dexnet/vase.obj"
obj_mesh = trimesh.load_mesh(obj_path)

print(np.max(obj_mesh.vertices, axis=0))
print(np.min(obj_mesh.vertices, axis=0))

trimesh.io.export.export_mesh(camera_mesh, 'new_camera.obj')

mlab.show()

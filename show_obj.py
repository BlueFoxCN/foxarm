import trimesh
import numpy as np
from foxarm.common.visualizer import Vis
from mayavi import mlab
from autolab_core import RigidTransform

mesh_path = "sample_objs/c6442bba98ad3dcf5e912b2d2934c0b6.obj"
mesh = trimesh.load_mesh(mesh_path)

Vis.plot_mesh(mesh)

mag = 2 * float(np.max(np.abs(mesh.vertices)))
Vis.plot_mesh(mesh)
Vis.plot_frame(mag)

print(np.max(mesh.vertices, axis=0))
print(np.min(mesh.vertices, axis=0))

mlab.show()

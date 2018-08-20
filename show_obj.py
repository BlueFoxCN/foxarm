import trimesh
from foxarm.common.visualizer import Vis
from mayavi import mlab

# obj_path = "mini_dexnet/vase.obj"
obj_path = "table.obj"
mesh = trimesh.load_mesh(obj_path)

Vis.plot_mesh(mesh)
mlab.show()

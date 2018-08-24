

class SceneObject(object):
    """ Struct to encapsulate objects added to a scene """
    def __init__(self, mesh, T_mesh_world):
        self.mesh = mesh
        self.T_mesh_world = T_mesh_world

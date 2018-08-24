"""
Wrapper for rendered images of an object on a table using Maya or our stable pose
Author: Jeff Mahler
"""
from autolab_core import RigidTransform

from .image import BinaryImage, ColorImage, DepthImage
from .render_mode import RenderMode

class ObjectRender(object):
    """Class to encapsulate images of an object rendered from a virtual camera.
    Note
    ----
        In this class, the table's frame of reference is the 'world' frame for
        the renderer.
    """

    def __init__(self, image, T_camera_world=RigidTransform(from_frame='camera', to_frame='table'),
                 obj_key = None, stable_pose=None):
        """Create an ObjectRender.
        Parameters
        ----------
        image : :obj:`Image`
            The image to be encapsulated.
        T_camera_world : :obj:`autolab_core.RigidTransform`
            A rigid transform from camera to world coordinates (positions the
            camera in the world). TODO -- this should be renamed.
        obj_key : :obj:`str`, optional
            A string identifier for the object being rendered.
        stable_pose : :obj:`meshpy.StablePose`
            The object's stable pose.
        """
        self.image = image
        self.T_camera_world = T_camera_world
        self.obj_key = obj_key
        self.stable_pose = stable_pose

    @property
    def T_obj_camera(self):
        """Returns the transformation from camera to object when the object is in the given stable pose.
        Returns
        -------
        :obj:`autolab_core.RigidTransform`
            The desired transform.
        """
        if self.stable_pose is None:
            T_obj_world = RigidTransform(from_frame='obj', to_frame='world')
        else:
            T_obj_world = self.stable_pose.T_obj_table.as_frames('obj', 'world')
        T_camera_obj = T_obj_world.inverse() * self.T_camera_world
        return T_camera_obj

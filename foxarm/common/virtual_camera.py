import numpy as np
import random
import string
import copy
import time
import logging
from scipy import misc

import ctypes
from ctypes import *
from numpy.ctypeslib import ndpointer

from autolab_core import RigidTransform

from foxarm.constants import *
from foxarm.common.render_mode import RenderMode
from .image import *
from .camera_intrinsics import CameraIntrinsics
from .object_render import ObjectRender

render_lib = cdll.LoadLibrary(RENDER_LIB_PATH)

class VirtualCamera(object):
    """A virtualized camera for rendering virtual color and depth images of meshes.

    Rendering is performed by using OSMesa offscreen rendering and boost_numpy.
    """
    def __init__(self, camera_intr):
        """Initialize a virtual camera.

        Parameters
        ----------
        camera_intr : :obj:`CameraIntrinsics`
            The CameraIntrinsics object used to parametrize the virtual camera.

        Raises
        ------
        ValueError
            When camera_intr is not a CameraIntrinsics object.
        """
        if not isinstance(camera_intr, CameraIntrinsics):
            raise ValueError('Must provide camera intrinsics as a CameraIntrinsics object')
        self._camera_intr = camera_intr
        self._scene = {} 

    def add_to_scene(self, name, scene_object):
        """ Add an object to the scene.

        Parameters
        ---------
        name : :obj:`str`
            name of object in the scene
        scene_object : :obj:`SceneObject`
            object to add to the scene
        """
        self._scene[name] = scene_object

    def remove_from_scene(self, name):
        """ Remove an object to a from the scene.

        Parameters
        ---------
        name : :obj:`str`
            name of object to remove
        """
        self._scene[name] = None

    def images(self, mesh, object_to_camera_poses, debug=False):
        """Render images of the given mesh at the list of object to camera poses.

        Parameters
        ----------
        mesh : :obj:`Trimesh`
            The mesh to be rendered.
        object_to_camera_poses : :obj:`list` of :obj:`RigidTransform`
            A list of object to camera transforms to render from.
        debug : bool
            Whether or not to debug the C++ meshrendering code.

        Returns
        -------
        :obj:`list` of `numpy.ndarray`
            A list of ndarrays. The first, which represents the color image.
            The second, which represents the depth image,
            contains floats and is of shape (height, width). Each pixel is a
            single float that represents the depth of the image.
        """
        # get mesh spec as numpy arrays
        vertex_arr = mesh.vertices
        tri_arr = mesh.faces.astype(np.int32)
        norms_arr = mesh.vertex_normals

        # render for each object to camera pose
        depth_ims = []
        render_start = time.time()
        for T_obj_camera in object_to_camera_poses:
            # form projection matrix
            R = T_obj_camera.rotation
            t = T_obj_camera.translation
            P = self._camera_intr.proj_matrix.dot(np.c_[R, t])

            render_lib.render.restype = ndpointer(dtype=ctypes.c_float, 
                                                  shape=(self._camera_intr.height * self._camera_intr.width,))

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
                                    self._camera_intr.height,
                                    self._camera_intr.width,
                                    verts_buffer,
                                    tris_buffer,
                                    norm_buffer,
                                    num_verts,
                                    num_tris,
                                    num_norms)

            ret = ret.reshape(self._camera_intr.height, self._camera_intr.width)

            depth_ims.append(ret)
        render_stop = time.time()
        logging.debug('Rendering took %.3f sec' %(render_stop - render_start))

        return depth_ims

    def wrapped_images(self, mesh, object_to_camera_poses,
                       render_mode, stable_pose=None, debug=False):
        """Create ObjectRender objects of the given mesh at the list of object to camera poses.

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The mesh to be rendered.
        object_to_camera_poses : :obj:`list` of :obj:`RigidTransform`
            A list of object to camera transforms to render from.
        render_mode : int
            One of RenderMode.COLOR, RenderMode.DEPTH, or
            RenderMode.SCALED_DEPTH.
        stable_pose : :obj:`RigidTransformation`
            A rigid transformation that transforms the object to a stable pose
        debug : bool
            Whether or not to debug the C++ meshrendering code.

        Returns
        -------
        :obj:`list` of :obj:`ObjectRender`
            A list of ObjectRender objects generated from the given parameters.
        """
        # pre-multiply the stable pose
        world_to_camera_poses = [T_obj_camera.as_frames('obj', 'camera') for T_obj_camera in object_to_camera_poses]
        if stable_pose is not None:
            '''
            t_obj_stp = np.array([0,0,-stable_pose.rotation.dot(stable_pose.translation)[2]])
            T_obj_stp = RigidTransform(rotation=stable_pose.rotation,
                                       translation=t_obj_stp,
                                       from_frame='obj',
                                       to_frame='stp')            
            '''
            T_obj_stp = stable_pose.copy()
            T_obj_stp.frame_frame = 'obj'
            T_obj_stp.to_frame = 'stp'
            stp_to_camera_poses = copy.copy(object_to_camera_poses)
            object_to_camera_poses = []
            for T_stp_camera in stp_to_camera_poses:
                T_stp_camera.from_frame = 'stp'
                object_to_camera_poses.append(T_stp_camera.dot(T_obj_stp))

        depth_ims = self.images(mesh, object_to_camera_poses, debug=debug)

        # convert to image wrapper classes
        images = []
        if render_mode == RenderMode.SEGMASK:
            # wrap binary images
            for binary_im in color_ims:
                images.append(BinaryImage(binary_im[:,:,0], frame=self._camera_intr.frame, threshold=0))

        elif render_mode == RenderMode.DEPTH:
            # render depth image
            for depth_im in depth_ims:
                images.append(DepthImage(depth_im, frame=self._camera_intr.frame))

        elif render_mode == RenderMode.DEPTH_SCENE:
            # create empty depth images
            for depth_im in depth_ims:
                images.append(DepthImage(depth_im, frame=self._camera_intr.frame))

            rnd_strs = []
            for _ in images:
                rnd_strs.append(''.join(random.choices(string.ascii_uppercase + string.digits, k=5)))

            for img_idx, img in enumerate(images):
                misc.imsave('depth_imgs/depth_%s.jpg' % rnd_strs[img_idx], img.data)
                np.save('depth_imgs/depth_%s' % rnd_strs[img_idx], img.data)

            # render images of scene objects
            depth_scene_ims = {}
            for name, scene_obj in self._scene.items():
                scene_object_to_camera_poses = []
                for world_to_camera_pose in world_to_camera_poses:
                    scene_object_to_camera_poses.append(world_to_camera_pose * scene_obj.T_mesh_world)
                depth_scene_ims[name] = self.wrapped_images(scene_obj.mesh, scene_object_to_camera_poses, RenderMode.DEPTH)

            # combine with scene images
            for i in range(len(images)):
                for j, name in enumerate(depth_scene_ims.keys()):
                    images[i] = images[i].combine_with(depth_scene_ims[name][j].image)

            for img_idx, img in enumerate(images):
                misc.imsave('depth_imgs/depth_%s_with_scene.jpg' % rnd_strs[img_idx], img.data)
                np.save('depth_imgs/depth_%s_with_scene' % rnd_strs[img_idx], img.data)

        # create object renders
        if stable_pose is not None:
            object_to_camera_poses = copy.copy(stp_to_camera_poses)
        rendered_images = []
        for image, T_obj_camera in zip(images, object_to_camera_poses):
            T_camera_obj = T_obj_camera.inverse()
            rendered_images.append(ObjectRender(image, T_camera_obj))

        return rendered_images

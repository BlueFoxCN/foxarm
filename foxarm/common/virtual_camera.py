
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

    def images(self, mesh, object_to_camera_poses,
               mat_props=None, light_props=None, enable_lighting=True, debug=False):
        """Render images of the given mesh at the list of object to camera poses.

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The mesh to be rendered.
        object_to_camera_poses : :obj:`list` of :obj:`RigidTransform`
            A list of object to camera transforms to render from.
        mat_props : :obj:`MaterialProperties`
            Material properties for the mesh
        light_props : :obj:`MaterialProperties`
            Lighting properties for the scene
        enable_lighting : bool
            Whether or not to enable lighting
        debug : bool
            Whether or not to debug the C++ meshrendering code.

        Returns
        -------
        :obj:`tuple` of `numpy.ndarray`
            A 2-tuple of ndarrays. The first, which represents the color image,
            contains ints (0 to 255) and is of shape (height, width, 3). 
            Each pixel is a 3-ndarray (red, green, blue) associated with a given
            y and x value. The second, which represents the depth image,
            contains floats and is of shape (height, width). Each pixel is a
            single float that represents the depth of the image.
        """
        # get mesh spec as numpy arrays
        vertex_arr = mesh.vertices
        tri_arr = mesh.triangles.astype(np.int32)
        if mesh.normals is None:
            mesh.compute_vertex_normals()
        norms_arr = mesh.normals

        # set default material properties
        if mat_props is None:
            mat_props = MaterialProperties()
        mat_props_arr = mat_props.arr

        # set default light properties
        if light_props is None:
            light_props = LightingProperties()

        # render for each object to camera pose
        # TODO: clean up interface, use modelview matrix!!!!
        color_ims = []
        depth_ims = []
        render_start = time.time()
        for T_obj_camera in object_to_camera_poses:
            # form projection matrix
            R = T_obj_camera.rotation
            t = T_obj_camera.translation
            P = self._camera_intr.proj_matrix.dot(np.c_[R, t])

            # form light props
            light_props.set_pose(T_obj_camera)
            light_props_arr = light_props.arr

            # render images for each
            c, d = meshrender.render_mesh([P],
                                          self._camera_intr.height,
                                          self._camera_intr.width,
                                          vertex_arr,
                                          tri_arr,
                                          norms_arr,
                                          mat_props_arr,
                                          light_props_arr,
                                          enable_lighting,
                                          debug)
            color_ims.extend(c)
            depth_ims.extend(d)
        render_stop = time.time()
        logging.debug('Rendering took %.3f sec' %(render_stop - render_start))

        return color_ims, depth_ims

    def images_viewsphere(self, mesh, vs_disc, mat_props=None, light_props=None):
        """Render images of the given mesh around a view sphere.

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The mesh to be rendered.
        vs_disc : :obj:`ViewsphereDiscretizer`
            A discrete viewsphere from which we draw object to camera
            transforms.
        mat_props : :obj:`MaterialProperties`
            Material properties for the mesh
        light_props : :obj:`MaterialProperties`
            Lighting properties for the scene

        Returns
        -------
        :obj:`tuple` of `numpy.ndarray`
            A 2-tuple of ndarrays. The first, which represents the color image,
            contains ints (0 to 255) and is of shape (height, width, 3). 
            Each pixel is a 3-ndarray (red, green, blue) associated with a given
            y and x value. The second, which represents the depth image,
            contains floats and is of shape (height, width). Each pixel is a
            single float that represents the depth of the image.
        """
        return self.images(mesh, vs_disc.object_to_camera_poses(),
                           mat_props=mat_props, light_props=light_props)

    def wrapped_images(self, mesh, object_to_camera_poses,
                       render_mode, stable_pose=None, mat_props=None,
                       light_props=None,debug=False):
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
        stable_pose : :obj:`StablePose`
            A stable pose to render the object in.
        mat_props : :obj:`MaterialProperties`
            Material properties for the mesh
        light_props : :obj:`MaterialProperties`
            Lighting properties for the scene
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
            t_obj_stp = np.array([0,0,-stable_pose.r.dot(stable_pose.x0)[2]])
            T_obj_stp = RigidTransform(rotation=stable_pose.r,
                                       translation=t_obj_stp,
                                       from_frame='obj',
                                       to_frame='stp')            
            stp_to_camera_poses = copy.copy(object_to_camera_poses)
            object_to_camera_poses = []
            for T_stp_camera in stp_to_camera_poses:
                T_stp_camera.from_frame = 'stp'
                object_to_camera_poses.append(T_stp_camera.dot(T_obj_stp))

        # set lighting mode
        enable_lighting = True
        if render_mode == RenderMode.SEGMASK or render_mode == RenderMode.DEPTH or \
           render_mode == RenderMode.DEPTH_SCENE:
            enable_lighting = False

        # render both image types (doesn't really cost any time)
        color_ims, depth_ims = self.images(mesh, object_to_camera_poses,
                                           mat_props=mat_props,
                                           light_props=light_props,
                                           enable_lighting=enable_lighting,
                                           debug=debug)

        # convert to image wrapper classes
        images = []
        if render_mode == RenderMode.SEGMASK:
            # wrap binary images
            for binary_im in color_ims:
                images.append(BinaryImage(binary_im[:,:,0], frame=self._camera_intr.frame, threshold=0))

        elif render_mode == RenderMode.COLOR:
            # wrap color images
            for color_im in color_ims:
                images.append(ColorImage(color_im, frame=self._camera_intr.frame))

        elif render_mode == RenderMode.COLOR_SCENE:
            # wrap color and depth images
            for color_im in color_ims:
                images.append(ColorImage(color_im, frame=self._camera_intr.frame))

            # render images of scene objects
            color_scene_ims = {}
            for name, scene_obj in self._scene.iteritems():
                scene_object_to_camera_poses = []
                for world_to_camera_pose in world_to_camera_poses:
                    scene_object_to_camera_poses.append(world_to_camera_pose * scene_obj.T_mesh_world)

                color_scene_ims[name] = self.wrapped_images(scene_obj.mesh, scene_object_to_camera_poses, RenderMode.COLOR, mat_props=scene_obj.mat_props, light_props=light_props, debug=debug)

            # combine with scene images
            # TODO: re-add farther
            for i in range(len(images)):
                for j, name in enumerate(color_scene_ims.keys()):
                    zero_px = images[i].zero_pixels()
                    images[i].data[zero_px[:,0], zero_px[:,1], :] = color_scene_ims[name][i].image.data[zero_px[:,0], zero_px[:,1], :]

        elif render_mode == RenderMode.DEPTH:
            # render depth image
            for depth_im in depth_ims:
                images.append(DepthImage(depth_im, frame=self._camera_intr.frame))

        elif render_mode == RenderMode.DEPTH_SCENE:
            # create empty depth images
            for depth_im in depth_ims:
                images.append(DepthImage(depth_im, frame=self._camera_intr.frame))

            # render images of scene objects
            depth_scene_ims = {}
            for name, scene_obj in self._scene.iteritems():
                scene_object_to_camera_poses = []
                for world_to_camera_pose in world_to_camera_poses:
                    scene_object_to_camera_poses.append(world_to_camera_pose * scene_obj.T_mesh_world)
                depth_scene_ims[name] = self.wrapped_images(scene_obj.mesh, scene_object_to_camera_poses, RenderMode.DEPTH, mat_props=scene_obj.mat_props, light_props=light_props)

            # combine with scene images
            for i in range(len(images)):
                for j, name in enumerate(depth_scene_ims.keys()):
                    images[i] = images[i].combine_with(depth_scene_ims[name][i].image)

        elif render_mode == RenderMode.RGBD:
            # create RGB-D images
            for color_im, depth_im in zip(color_ims, depth_ims):
                c = ColorImage(color_im, frame=self._camera_intr.frame)
                d = DepthImage(depth_im, frame=self._camera_intr.frame)
                images.append(RgbdImage.from_color_and_depth(c, d))

        elif render_mode == RenderMode.RGBD_SCENE:
            # create RGB-D images
            for color_im, depth_im in zip(color_ims, depth_ims):
                c = ColorImage(color_im, frame=self._camera_intr.frame)
                d = DepthImage(depth_im, frame=self._camera_intr.frame)
                images.append(RgbdImage.from_color_and_depth(c, d))

            # render images of scene objects
            rgbd_scene_ims = {}
            for name, scene_obj in self._scene.iteritems():
                scene_object_to_camera_poses = []
                for world_to_camera_pose in world_to_camera_poses:
                    scene_object_to_camera_poses.append(world_to_camera_pose * scene_obj.T_mesh_world)

                rgbd_scene_ims[name] = self.wrapped_images(scene_obj.mesh, scene_object_to_camera_poses, RenderMode.RGBD, mat_props=scene_obj.mat_props, light_props=light_props, debug=debug)

           # combine with scene images
            for i in range(len(images)):
                for j, name in enumerate(rgbd_scene_ims.keys()):
                    images[i] = images[i].combine_with(rgbd_scene_ims[name][i].image)

        elif render_mode == RenderMode.SCALED_DEPTH:
            # convert to color image
            for depth_im in depth_ims:
                d = DepthImage(depth_im, frame=self._camera_intr.frame)
                images.append(d.to_color())
        else:
            raise ValueError('Render mode %s not supported' %(render_mode))

        # create object renders
        if stable_pose is not None:
            object_to_camera_poses = copy.copy(stp_to_camera_poses)
        rendered_images = []
        for image, T_obj_camera in zip(images, object_to_camera_poses):
            T_camera_obj = T_obj_camera.inverse()
            rendered_images.append(ObjectRender(image, T_camera_obj))

        return rendered_images

    def wrapped_images_viewsphere(self, mesh, vs_disc, render_mode, stable_pose=None, mat_props=None, light_props=None):
        """Create ObjectRender objects of the given mesh around a viewsphere.

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The mesh to be rendered.
        vs_disc : :obj:`ViewsphereDiscretizer`
            A discrete viewsphere from which we draw object to camera
            transforms.
        render_mode : int
            One of RenderMode.COLOR, RenderMode.DEPTH, or
            RenderMode.SCALED_DEPTH.
        stable_pose : :obj:`StablePose`
            A stable pose to render the object in.
        mat_props : :obj:`MaterialProperties`
            Material properties for the mesh
        light_props : :obj:`MaterialProperties`
            Lighting properties for the scene

        Returns
        -------
        :obj:`list` of :obj:`ObjectRender`
            A list of ObjectRender objects generated from the given parameters.
        """
        return self.wrapped_images(mesh, vs_disc.object_to_camera_poses(), render_mode, stable_pose=stable_pose, mat_props=mat_props, light_props=light_props)

    def wrapped_images_planar_worksurface(self, mesh, ws_disc, render_mode, stable_pose=None, mat_props=None, light_props=None):
        """ Create ObjectRender objects of the given mesh around a viewsphere and 
        a planar worksurface, where translated objects project into the center of
        the  camera.

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The mesh to be rendered.
        ws_disc : :obj:`PlanarWorksurfaceDiscretizer`
            A discrete viewsphere and translations in plane from which we draw
            object to camera transforms.
        render_mode : int
            One of RenderMode.COLOR, RenderMode.DEPTH, or
            RenderMode.SCALED_DEPTH.
        stable_pose : :obj:`StablePose`
            A stable pose to render the object in.
        mat_props : :obj:`MaterialProperties`
            Material properties for the mesh
        light_props : :obj:`MaterialProperties`
            Lighting properties for the scene

        Returns
        -------
        :obj:`list` of :obj:`ObjectRender`
            A list of ObjectRender objects generated from the given parameters.
        :obj:`list` of :obj:`RigidTransform`
            A list of the transformations from object frame to camera frame used
            for rendering the images.
        :obj:`list` of :obj:`CameraIntrinsics`
            A list of the camera intrinsics used for rendering the images.
        """
        # save original intrinsics
        old_camera_intr = copy.deepcopy(self._camera_intr)
        object_to_camera_poses, object_to_camera_normalized_poses, shifted_camera_intrinsics = ws_disc.object_to_camera_poses(old_camera_intr)
        logging.info('Rendering %d images' %(len(object_to_camera_poses)))

        images = []
        for T_obj_camera, shifted_camera_intr in zip(object_to_camera_poses, shifted_camera_intrinsics):
            self._camera_intr = shifted_camera_intr
            images.extend(self.wrapped_images(mesh, [T_obj_camera], render_mode, stable_pose=stable_pose, mat_props=mat_props, light_props=light_props))

        self._camera_intr = old_camera_intr
        return images, object_to_camera_poses, shifted_camera_intrinsics

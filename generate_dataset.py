import copy
from scipy import misc
import random
import logging
import pickle
import numpy as np
import os
import trimesh
import time
from autolab_core import RigidTransform, YamlConfig, PointCloud
from mayavi import mlab

from foxarm.grasping.grasp import ParallelJawPtGrasp3D
from foxarm.grasping.gripper import RobotGripper
from foxarm.grasping.grasp_sampler import GraspSampler, AntipodalGraspSampler
from foxarm.grasping.graspable_object import GraspableObject, GraspableObject3D
from foxarm.grasping.contacts import Contact, Contact3D
from foxarm.grasping.quality import PointGraspMetrics3D
from foxarm.grasping.grasp_quality_config import GraspQualityConfigFactory
from foxarm.constants import *
from foxarm.common import constants
from foxarm.common.keys import *
from foxarm.common.sdf_file import SdfFile
from foxarm.common import Vis, SceneObject
from foxarm.common.render_mode import RenderMode
from foxarm.common import *

CONFIG = YamlConfig(TEST_CONFIG_NAME)
gripper = RobotGripper.load(GRIPPER_NAME, os.path.join(WORK_DIR, "foxarm/common"))
# render_modes = [RenderMode.SEGMASK, RenderMode.DEPTH_SCENE]
render_modes = [RenderMode.DEPTH_SCENE]

SEED = 197561

def generate_dataset(config, debug):
    obj_paths = ["mini_dexnet/bar_clamp.obj"]

    vis_params = config["vis"]
    camera_filename = vis_params["camera_filename"]
    save_dir = vis_params["save_dir"]

    coll_check_params = config['collision_checking']
    approach_dist = coll_check_params['approach_dist']
    delta_approach = coll_check_params['delta_approach']
    table_offset = coll_check_params['table_offset']

    # read gqcnn params
    gqcnn_params = config['gqcnn']
    im_crop_height = gqcnn_params['crop_height']
    im_crop_width = gqcnn_params['crop_width']
    im_final_height = gqcnn_params['final_height']
    im_final_width = gqcnn_params['final_width']
    cx_crop = float(im_crop_width) / 2
    cy_crop = float(im_crop_height) / 2

    image_samples_per_stable_pose = config['images_per_stable_pose']

    table_mesh_filename = coll_check_params['table_mesh_filename']
    table_mesh = trimesh.load_mesh(table_mesh_filename)


    T_table_obj = RigidTransform(from_frame='table', to_frame='obj')
    scene_objs = {'table': SceneObject(table_mesh, T_table_obj)}

    table_alignment_params = config['table_alignment']
    env_rv_params = config['env_rv_params']

    max_grasp_approach_table_angle = np.deg2rad(table_alignment_params['max_approach_table_angle'])

    img_idx = 0

    for obj_path in obj_paths:

        # 1. construct the object
        sdf_path = obj_path[:-3] + "sdf"
        mesh = trimesh.load_mesh(obj_path)
        sdf = SdfFile(sdf_path).read()
        plot_scale = 2 * float(np.max(np.abs(mesh.vertices)))
        obj = GraspableObject3D(sdf, mesh)

        '''
        # 2. sample grasps
        ags = AntipodalGraspSampler(gripper, CONFIG)
        unaligned_grasps = ags.generate_grasps(obj, target_num_grasps=100)

        # 3. compute stable poses
        stp_mats, stp_probs = mesh.compute_stable_poses(n_samples = 1)
        stps = []
        for stp_mat in stp_mats:
            r, t = RigidTransform.rotation_and_translation_from_matrix(stp_mat)
            stps.append(RigidTransform(rotation=r, translation=t))
        '''

        # 4. process and filter the grasps with following steps:
        #   A. align the grasp
        #   B. check angle with table plan and only keep perpendicular ones
        #   C. check whether collision free
        # collision_checker = GraspCollisionChecker(gripper)
        # collision_checker.set_graspable_object(obj)

        grasps_f = open('grasps.pkl', 'rb')
        unaligned_grasps = pickle.load(grasps_f)

        stps_f = open('stps.pkl', 'rb')
        stps = pickle.load(stps_f)

        grasps = {}
        for stp_idx, stp in enumerate(stps):

            # T_obj_stp = stable_pose.as_frames('obj', 'stp')
            # T_obj_table = obj.mesh.get_T_surface_obj(T_obj_stp, delta=table_offset).as_frames('obj', 'table')
            # T_table_obj = T_obj_table.inverse()
            # collision_checker.set_table(table_mesh_filename, T_table_obj)

            grasps[stp_idx] = []
            for grasp in unaligned_grasps:
                # align the grasp
                aligned_grasp = grasp.perpendicular_table(stp)

                # check perpendicular with table
                _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_stp_z(stp)
                perpendicular_table = (np.abs(grasp_approach_table_angle) < max_grasp_approach_table_angle)
                if not perpendicular_table: 
                    continue

                # check collision
                # collision_free = not collision_checker.collides_along_approach(aligned_grasp, approach_dist, delta_approach)

                grasps[stp_idx].append(aligned_grasp)


            # render depth images
            urv = UniformPlanarWorksurfaceImageRandomVariable(obj.mesh,
                                                              render_modes,
                                                              'camera',
                                                              env_rv_params,
                                                              stable_pose=stp,
                                                              scene_objs=scene_objs)
            

            render_start = time.time()
            # image_samples_per_stable_pose = 100
            render_samples = urv.rvs(size=image_samples_per_stable_pose)
            render_stop = time.time()
            logging.info('Rendering images took %.3f sec' % (render_stop - render_start))

            if not isinstance(render_samples, (list,)):
                render_samples = [render_samples]

            if debug:
                for render_sample in render_samples:
                    mlab.clf()
                    camera_t = render_sample.camera.object_to_camera_pose
                    Vis.plot_camera(camera_filename, camera_t)

                    t_obj_stp = np.array([0,0,-stp.rotation.dot(stp.translation)[2]])
                    T_obj_stp = RigidTransform(rotation=stp.rotation,
                                               translation=stp.translation,
                                               from_frame='obj',
                                               to_frame='stp')

                    stable_mesh = obj.mesh.copy()
                    stable_mesh.apply_transform(T_obj_stp.matrix)

                    # obj.mesh.apply_transform(T_obj_stp.matrix)
                    mag = 2 * float(np.max(np.abs(stable_mesh.vertices)))
                    Vis.plot_mesh(stable_mesh)
                    Vis.plot_plane(4 * mag)
                    Vis.plot_frame(mag)

                    mlab.view(0, 0, 0.85)

                    # https://stackoverflow.com/questions/16543634/mayavi-mlab-savefig-gives-an-empty-image
                    mlab.savefig('%s/%d.jpg' % (save_dir, img_idx))
                    img_idx += 1

                    # mlab.show()

            for render_sample in render_samples:
                depth_im_table = render_sample.renders[RenderMode.DEPTH_SCENE].image
                T_stp_camera = render_sample.camera.object_to_camera_pose
                shifted_camera_intr = render_sample.camera.camera_intr

                # read pixel offsets
                cx = depth_im_table.center[1]
                cy = depth_im_table.center[0]

                # compute intrinsics for virtual camera of the final
                # cropped and rescaled images
                camera_intr_scale = float(im_final_height) / float(im_crop_height)
                cropped_camera_intr = shifted_camera_intr.crop(im_crop_height, im_crop_width, cy, cx)
                final_camera_intr = cropped_camera_intr.resize(camera_intr_scale)

                T_obj_camera = T_stp_camera * stp.as_frames('obj', T_stp_camera.from_frame)

                for grasp in grasps[stp_idx]:

                    # get the gripper pose
                    grasp_2d = grasp.project_camera(T_obj_camera, shifted_camera_intr)


                    # center images on the grasp, rotate to image x axis
                    dx = cx - grasp_2d.center.x
                    dy = cy - grasp_2d.center.y
                    translation = np.array([dy, dx])

                    depth_im_tf_table = depth_im_table.transform(translation, grasp_2d.angle)

                    # crop to image size
                    depth_im_tf_table = depth_im_tf_table.crop(im_crop_height, im_crop_width)

                    # resize to image size
                    depth_im_tf_table = depth_im_tf_table.resize((im_final_height, im_final_width))

                    import pdb
                    pdb.set_trace()

                    misc.imsave('final.jpg', depth_im_tf_table.data)


            if debug:
                break


if __name__ == '__main__':

    dataset_config = YamlConfig(GENERATE_DATASET_CONFIG_NAME)
    debug = dataset_config['debug']
    if debug:
        random.seed(SEED)
        np.random.seed(SEED)
        
    # target_object_keys = config['target_objects']
    # env_rv_params = config['env_rv_params']
    # gripper_name = config['gripper']

    # generate the dataset
    generate_dataset(dataset_config, debug)

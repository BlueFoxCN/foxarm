import copy
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

    coll_check_params = config['collision_checking']
    approach_dist = coll_check_params['approach_dist']
    delta_approach = coll_check_params['delta_approach']
    table_offset = coll_check_params['table_offset']

    image_samples_per_stable_pose = config['images_per_stable_pose']

    table_mesh_filename = coll_check_params['table_mesh_filename']
    table_mesh = trimesh.load_mesh(table_mesh_filename)


    T_table_obj = RigidTransform(from_frame='table', to_frame='obj')
    scene_objs = {'table': SceneObject(table_mesh, T_table_obj)}

    table_alignment_params = config['table_alignment']
    env_rv_params = config['env_rv_params']

    max_grasp_approach_table_angle = np.deg2rad(table_alignment_params['max_approach_table_angle'])

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
            if debug:
                image_samples_per_stable_pose = 1
            render_samples = urv.rvs(size=image_samples_per_stable_pose)
            render_stop = time.time()
            logging.info('Rendering images took %.3f sec' %(render_stop - render_start))

            if debug:
                camera_path = "new_camera.obj"
                camera_mesh = trimesh.load_mesh(camera_path)

                camera_t = render_samples.camera.object_to_camera_pose
                camera_mesh.apply_transform(camera_t.matrix)

                Vis.plot_mesh(camera_mesh)

                t_obj_stp = np.array([0,0,-stp.rotation.dot(stp.translation)[2]])
                T_obj_stp = RigidTransform(rotation=stp.rotation,
                                           translation=stp.translation,
                                           from_frame='obj',
                                           to_frame='stp')

                obj.mesh.apply_transform(T_obj_stp.matrix)
                mag = 2 * float(np.max(np.abs(obj.mesh.vertices)))
                Vis.plot_mesh(obj.mesh)
                Vis.plot_plane(mag)
                Vis.plot_frame(mag)
                mlab.show()

            if debug:
                break


if __name__ == '__main__':

    dataset_config = YamlConfig(GENERATE_DATASET_CONFIG_NAME)
    debug = dataset_config['debug']
    '''
    if debug:
        random.seed(SEED)
        np.random.seed(SEED)
    '''
        
    # target_object_keys = config['target_objects']
    # env_rv_params = config['env_rv_params']
    # gripper_name = config['gripper']

    # generate the dataset
    generate_dataset(dataset_config, debug)

import argparse
import copy
from time import sleep
from threading import Thread
from queue import Queue
import h5py
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
from foxarm.grasping.random_variables import GraspableObjectPoseGaussianRV, ParallelJawGraspPoseGaussianRV, ParamsGaussianRV
from foxarm.grasping.robust_grasp_quality import RobustPointGraspMetrics3D
from foxarm.constants import *
from foxarm.common import constants
from foxarm.common.keys import *
from foxarm.common.sdf_file import SdfFile
from foxarm.common import Vis, SceneObject
from foxarm.common.render_mode import RenderMode
from foxarm.common import *

CONFIG = YamlConfig(TEST_CONFIG_NAME)
gripper = RobotGripper.load(GRIPPER_NAME, os.path.join(WORK_DIR, "foxarm/common"))
render_modes = [RenderMode.DEPTH_SCENE]

SEED = 197561

class Generator:
    def __init__(self, obj_ids, obj_paths, config, debug, result_queue):
        self.obj_ids = obj_ids
        self.obj_paths = obj_paths
        self.config = config
        self.debug = debug
        self.result_queue = result_queue

def generate_dataset(obj_ids, obj_paths, config, debug, result_queue):
    '''
    obj_ids = ["bar_clamp"]
    obj_dir = "mini_dexnet"
    obj_paths = ["%s/%s.obj" % (obj_dir, e) for e in obj_ids]
    '''

    # read parameters
    vis_params = config["vis"]
    camera_filename = vis_params["camera_filename"]
    save_dir = vis_params["save_dir"]
    coll_check_params = config['collision_checking']
    approach_dist = coll_check_params['approach_dist']
    delta_approach = coll_check_params['delta_approach']
    table_offset = coll_check_params['table_offset']
    gqcnn_params = config['gqcnn']
    im_crop_height = gqcnn_params['crop_height']
    im_crop_width = gqcnn_params['crop_width']
    im_final_height = gqcnn_params['final_height']
    im_final_width = gqcnn_params['final_width']
    cx_crop = float(im_crop_width) / 2
    cy_crop = float(im_crop_height) / 2
    image_samples_per_stable_pose = config['images_per_stable_pose']
    table_mesh_filename = coll_check_params['table_mesh_filename']
    table_alignment_params = config['table_alignment']
    env_rv_params = config['env_rv_params']
    max_grasp_approach_table_angle = np.deg2rad(table_alignment_params['max_approach_table_angle'])


    table_mesh = trimesh.load_mesh(table_mesh_filename)
    T_table_obj = RigidTransform(from_frame='table', to_frame='obj')
    scene_objs = {'table': SceneObject(table_mesh, T_table_obj)}

    img_idx = 0

    quality_config = GraspQualityConfigFactory.create_config(CONFIG['metrics']['robust_ferrari_canny'])

    # rv: friction core
    params_rv = ParamsGaussianRV(quality_config,
                                 quality_config.params_uncertainty)

    for obj_idx, obj_path in enumerate(obj_paths):

        obj_id = obj_ids[obj_idx]
        result = {}
        result[obj_id] = {}

        # 1. construct the object
        sdf_path = obj_path[:-3] + "sdf"
        mesh = trimesh.load_mesh(obj_path)
        sdf = SdfFile(sdf_path).read()
        plot_scale = 2 * float(np.max(np.abs(mesh.vertices)))
        obj = GraspableObject3D(sdf, mesh)

        T_obj_world = RigidTransform(from_frame='obj', to_frame='world')
        graspable_rv = GraspableObjectPoseGaussianRV(obj,
                                                     T_obj_world,
                                                     quality_config.obj_uncertainty)

        result[obj_id]['mesh'] = {}
        result[obj_id]['mesh']['vertices'] = mesh.vertices
        result[obj_id]['mesh']['triangles'] = mesh.faces

        # 2. sample force closure grasps
        unaligned_fc_grasps = []
        ags = AntipodalGraspSampler(gripper, CONFIG)
        unaligned_grasps = ags.generate_grasps(obj, target_num_grasps=100)
        for i, grasp in enumerate(unaligned_grasps):
            success, c = grasp.close_fingers(obj, check_approach=False)
            if success:
                c1, c2 = c
                if_force_closure = bool(PointGraspMetrics3D.force_closure(c1, c2, CONFIG['sampling_friction_coef']))
                if if_force_closure:
                    unaligned_fc_grasps.append(grasp)

        # 3. compute stable poses
        stp_mats, stp_probs = mesh.compute_stable_poses(n_samples = 1)
        stps = []
        for stp_mat in stp_mats:
            r, t = RigidTransform.rotation_and_translation_from_matrix(stp_mat)
            stps.append(RigidTransform(rotation=r, translation=t))
        result[obj_id]['stable_poses'] = {}

        # for each stable pose
        grasps = {}
        for stp_idx, stp in enumerate(stps):
            result[obj_id]['stable_poses']['stp_0'] = {}
            grasp_idx = 0

            # filter grasps and calculate quality
            grasps[stp_idx] = []
            for grasp in unaligned_fc_grasps:
                # align the grasp
                aligned_grasp = grasp.perpendicular_table(stp)

                # check perpendicular with table
                _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_stp_z(stp)
                perpendicular_table = (np.abs(grasp_approach_table_angle) < max_grasp_approach_table_angle)
                if not perpendicular_table: 
                    continue

                # check collision
                success, c = aligned_grasp.close_fingers(obj)
                if not success:
                    continue

                # calculate grasp quality
                grasp_rv = ParallelJawGraspPoseGaussianRV(aligned_grasp,
                                                          quality_config.grasp_uncertainty)
                mean_q, std_q = RobustPointGraspMetrics3D.expected_quality(grasp_rv,
                                                                           graspable_rv,
                                                                           params_rv,
                                                                           quality_config)
                grasps[stp_idx].append((aligned_grasp, mean_q))

            # render depth images
            urv = UniformPlanarWorksurfaceImageRandomVariable(obj.mesh,
                                                              render_modes,
                                                              'camera',
                                                              env_rv_params,
                                                              stable_pose=stp,
                                                              scene_objs=scene_objs)
            image_samples_per_stable_pose = 1
            render_samples = urv.rvs(size=image_samples_per_stable_pose)
            render_samples = [render_samples]

            # if debug:
            if False:
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

                for grasp_info in grasps[stp_idx]:

                    grasp = grasp_info[0]
                    grasp_quality = grasp_info[1]

                    # get the gripper pose
                    grasp_2d = grasp.project_camera(T_obj_camera, shifted_camera_intr)
                    grasp_depth = grasp_2d.depth

                    # center images on the grasp, rotate to image x axis
                    dx = cx - grasp_2d.center.x
                    dy = cy - grasp_2d.center.y
                    translation = np.array([dy, dx])

                    # rotate, crop and resize
                    depth_im_tf_table = depth_im_table.transform(translation, grasp_2d.angle)
                    depth_im_tf_table = depth_im_tf_table.crop(im_crop_height, im_crop_width)
                    depth_im_tf_table = depth_im_tf_table.resize((im_final_height, im_final_width))

                    # one sample consists of depth_im_tf_table, grasp_quality, and grasp_depth
                    # misc.imsave('generated_data/%f.jpg' % grasp_quality, depth_im_tf_table.data)

                    grasp_key = 'grasp_%d' % grasp_idx
                    result[obj_id]['stable_poses']['stp_0'][grasp_key] = {}
                    result[obj_id]['stable_poses']['stp_0'][grasp_key]['depth_img'] = depth_im_tf_table.data
                    result[obj_id]['stable_poses']['stp_0'][grasp_key]['grasp_depth'] = grasp_depth
                    result[obj_id]['stable_poses']['stp_0'][grasp_key]['quality'] = grasp_quality

                    grasp_idx += 1

            if debug:
                break

        result_queue.put(result)
        break

class FileWriter:
    def __init__(self, file_path):
        self.result_queue = Queue(maxsize=100)
        self.f = h5py.File(file_path, 'a')
        self.dataset = self.f.create_group('dataset')

    def write(self):
        while True:
            if self.result_queue.empty():
                sleep(0.01)
                continue
            result = self.result_queue.get()
            if isinstance(result, str) and result == "stop":
                break

            def construct(group, data):
                for key in data.keys():
                    if isinstance(data[key], dict):
                        sub_group = group.create_group(key)
                        construct(sub_group, data[key])
                    elif isinstance(data[key], np.ndarray):
                        group.create_dataset(key, data=data[key])
                    else:
                        group.create_dataset(key, data=np.array(data[key]))

            construct(self.dataset, result)


    def start(self):
        self.t = Thread(target=self.write)
        self.t.daemon = True
        self.t.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', help='name of the database file')
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--seed', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()


    dataset_config = YamlConfig(GENERATE_DATASET_CONFIG_NAME)
    if args.seed:
        random.seed(SEED)
        np.random.seed(SEED)

    writer = FileWriter(file_path='a.hdf5')
    writer.start()

    obj_dir = 'objs'
    obj_sub_dirs = ['3dnet', 'kit']
    obj_dir_paths = [os.path.join(obj_dir, e) for e in obj_sub_dirs]

    obj_paths = []
    obj_ids = []
    for obj_dir_path in obj_dir_paths:
        obj_files = os.listdir(obj_dir_path)
        for obj_file in obj_files:
            if not obj_file.endswith('.obj'):
                continue
            obj_paths.append(os.path.join(obj_dir_path, obj_file))
            obj_ids.append(obj_file[:-4])

    '''
    obj_ids = ["bar_clamp"]
    obj_dir = "mini_dexnet"
    obj_paths = ["%s/%s.obj" % (obj_dir, e) for e in obj_ids]
    '''

    # generate the dataset
    generate_dataset(obj_ids, obj_paths, dataset_config, args.debug, writer.result_queue)

    writer.result_queue.put('stop')
    writer.t.join()

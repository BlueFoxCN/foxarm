import copy
import pickle
import numpy as np
import random
import os
import trimesh
from autolab_core import RigidTransform, YamlConfig, PointCloud
from mayavi import mlab

from foxarm.grasping.grasp import ParallelJawPtGrasp3D
from foxarm.grasping.gripper import RobotGripper
from foxarm.grasping.grasp_sampler import GraspSampler, AntipodalGraspSampler
from foxarm.grasping.graspable_object import GraspableObject, GraspableObject3D
from foxarm.constants import *
from foxarm.grasping.contacts import Contact, Contact3D
from foxarm.grasping.quality import PointGraspMetrics3D
from foxarm.grasping.grasp_quality_config import GraspQualityConfigFactory
from foxarm.common import constants
from foxarm.common.keys import *
from foxarm.common.sdf_file import SdfFile
from foxarm.common import Vis

from foxarm.grasping.random_variables import GraspableObjectPoseGaussianRV, ParallelJawGraspPoseGaussianRV, ParamsGaussianRV
from foxarm.grasping.robust_grasp_quality import RobustPointGraspMetrics3D

obj_path = "mini_dexnet/bar_clamp.obj"
sdf_path = "mini_dexnet/bar_clamp.sdf"
mesh = trimesh.load_mesh(obj_path)
sdf = SdfFile(sdf_path).read()

mag = 2 * float(np.max(np.abs(mesh.vertices)))

CONFIG = YamlConfig(TEST_CONFIG_NAME)

obj = GraspableObject3D(sdf, mesh)

# SEED = 197561
# random.seed(SEED)
# np.random.seed(SEED)

stp_mats, stp_probs = mesh.compute_stable_poses(n_samples = 1)
stps = []
for stp_mat in stp_mats:
    r, t = RigidTransform.rotation_and_translation_from_matrix(stp_mat)
    stps.append(RigidTransform(rotation=r, translation=t))
print('### Generated %d stable poses! ###' % len(stps))

'''
gripper = RobotGripper.load(GRIPPER_NAME, os.path.join(WORK_DIR, "foxarm/common"))
ags = AntipodalGraspSampler(gripper, CONFIG)

##### Generate Grasps #####
unaligned_grasps = ags.generate_grasps(obj, target_num_grasps=100)
print('### Generated %d unaligned grasps! ###' % len(unaligned_grasps))

grasps = []
for grasp in unaligned_grasps:
    aligned_grasp = grasp.perpendicular_table(stps[0])
    grasps.append(copy.deepcopy(aligned_grasp))
print('\tStable pose has %d grasps!' % len(grasps))

fc_grasps = []
for i, grasp in enumerate(grasps):
    success, c = grasp.close_fingers(obj)
    if success:
        c1, c2 = c
        if_force_closure = bool(PointGraspMetrics3D.force_closure(c1, c2, CONFIG['sampling_friction_coef']))
        if if_force_closure:
            fc_grasps.append(grasp)
'''

f = open('fc_grasps.pkl', 'rb')
fc_grasps = pickle.load(f)

# calculate quality
quality_config = GraspQualityConfigFactory.create_config(CONFIG['metrics']
                                                               ['robust_ferrari_canny'])
T_obj_world = RigidTransform(from_frame='obj', to_frame='world')
graspable_rv = GraspableObjectPoseGaussianRV(obj,
                                             T_obj_world,
                                             quality_config.obj_uncertainty)
params_rv = ParamsGaussianRV(quality_config,
                             quality_config.params_uncertainty)

# for debug, only take 5 grasps
# fc_grasps = fc_grasps[:5]

qualities = []
mean_qualities = []

for grasp in fc_grasps:
    grasp_rv = ParallelJawGraspPoseGaussianRV(grasp,
                                              quality_config.grasp_uncertainty)
    quality = PointGraspMetrics3D.grasp_quality(grasp, obj, quality_config)
    qualities.append(quality)

    q_samples = RobustPointGraspMetrics3D.expected_quality(grasp_rv,
                                                           graspable_rv,
                                                           params_rv,
                                                           quality_config)
    mean_q = np.mean(q_samples)
    mean_qualities.append(mean_q)

print(qualities)
print(mean_qualities)

import pdb
pdb.set_trace()


# visualize results
stable_mesh = obj.mesh.copy()
stable_mesh.apply_transform(stps[0].matrix)
Vis.plot_mesh(stable_mesh)
# Vis.plot_plane(mag)
# Vis.plot_frame(mag)

max_q = 0.002
min_q = 0.0005
for i, grasp in enumerate(fc_grasps):
    quality = mean_qualities[i]
    score = (quality - min_q) / (max_q - min_q)
    green = np.clip(score, 0, 1)
    red = 1 - green
    color = (red, green, 0)
    Vis.plot_grasp(grasp, obj, mag=0.1*mag, transform=stps[0], color=color)

mlab.show()

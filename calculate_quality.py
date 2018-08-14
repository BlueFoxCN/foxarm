import copy
import os
import trimesh
from autolab_core import RigidTransform, YamlConfig

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

from foxarm.grasping.random_variables import GraspableObjectPoseGaussianRV, ParallelJawGraspPoseGaussianRV, ParamsGaussianRV
from foxarm.grasping.robust_grasp_quality import RobustPointGraspMetrics3D
# from autolab_core import RigidTransform

obj_path = "mini_dexnet/bar_clamp.obj"
sdf_path = "mini_dexnet/bar_clamp.sdf"
mesh = trimesh.load_mesh(obj_path)
sdf = SdfFile(sdf_path).read()

CONFIG = YamlConfig(TEST_CONFIG_NAME)

obj = GraspableObject3D(sdf, mesh)

gripper = RobotGripper.load(GRIPPER_NAME, os.path.join(WORK_DIR, "foxarm/common"))
ags = AntipodalGraspSampler(gripper, CONFIG)

##### Generate Grasps #####
unaligned_grasps = ags.generate_grasps(obj, target_num_grasps=100)
print('### Generated %d unaligned grasps! ###' % len(unaligned_grasps))
grasps = {}

# calculate quality
quality_config = GraspQualityConfigFactory.create_config(CONFIG['metrics']
                                                               ['robust_ferrari_canny'])
'''
quality = PointGraspMetrics3D.grasp_quality(unaligned_grasps[0], obj, quality_config)
print(quality)
'''
# robust quality
T_obj_world = RigidTransform(from_frame='obj', to_frame='world')
graspable_rv_ = GraspableObjectPoseGaussianRV(obj,
                                              T_obj_world,
                                              quality_config.obj_uncertainty)
params_rv_ = ParamsGaussianRV(quality_config,
                              quality_config.params_uncertainty)
grasp_rv = ParallelJawGraspPoseGaussianRV(unaligned_grasps[0],
                                          quality_config.grasp_uncertainty)
mean_q, std_q = RobustPointGraspMetrics3D.expected_quality(grasp_rv,
                                                           graspable_rv_,
                                                           params_rv_,
                                                           quality_config)
print(mean_q)
print(std_q)
# x = RobustPointGraspMetrics3D.expected_quality(grasp_rv,
#                                                graspable_rv_,
#                                                params_rv_,
#                                                quality_config)
# print(x)
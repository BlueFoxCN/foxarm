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
quality = PointGraspMetrics3D.grasp_quality(unaligned_grasps[0], obj, quality_config)
print(quality)



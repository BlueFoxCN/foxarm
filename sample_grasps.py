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

stp_mats, stp_probs = mesh.compute_stable_poses(n_samples = 1)
stps = []
for stp_mat in stp_mats:
    r, t = RigidTransform.rotation_and_translation_from_matrix(stp_mat)
    stps.append(RigidTransform(rotation=r, translation=t))
print('### Generated %d stable poses! ###' % len(stps))

for stp_idx, stp in enumerate(stps):
    grasps[stp_idx] = []
    for grasp in unaligned_grasps:
        aligned_grasp = grasp.perpendicular_table(stp)
        grasps[stp_idx].append(copy.deepcopy(aligned_grasp))
    print('\tStable poses %s has %d grasps!' % (stp_idx, len(grasps[stp_idx])))

print('\n### Grasp Quality ###')
for stp_idx, stp in enumerate(stps):
    grasps_test = grasps[stp_idx]
    print('### Stable pose %s (%-5.3f) ###' % (stp_idx, stp_probs[stp_idx]))
    l_fa1, l_fa2 = [], []
    for i, grasp_t in enumerate(grasps_test):
        success, c = grasp_t.close_fingers(obj)
        if success:
            c1, c2 = c
            if_force_closure = bool(PointGraspMetrics3D.force_closure(c1, c2, CONFIG['sampling_friction_coef']))
            if not if_force_closure:
                l_fa2.append(i)
        else:
            if_force_closure = False
            l_fa1.append(i)
            l_fa2.append(i)
    print('\t%d grasps satisfy force closure' % (len(unaligned_grasps) - len(l_fa2)))
    print('\t%d grasps do not satisfy force closure' % len(l_fa2))
    print('\t%d grasps can not found contacts\n' % len(l_fa1))

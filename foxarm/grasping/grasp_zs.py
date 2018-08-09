import copy
import os
import grasp
from grasp import ParallelJawPtGrasp3D
from gripper import RobotGripper
from grasp_sampler import GraspSampler, AntipodalGraspSampler
from graspable_object import GraspableObject, GraspableObject3D

from foxarm.constants import *
from contacts import Contact, Contact3D
from quality import PointGraspMetrics3D

from autolab_core import RigidTransform, YamlConfig
from foxarm.common import constants
from foxarm.common import sdf_file, obj_file
from foxarm.common.keys import *
from foxarm.common.sdf_file import SdfFile
from foxarm.common.obj_file import ObjFile
from foxarm.common.database import Hdf5Database
def prn_obj(obj):
	print('\n'.join(['*** %s ***\n%s\n' % item for item in obj.__dict__.items()]))

CONFIG = YamlConfig(TEST_CONFIG_NAME)

of = ObjFile(OBJ_FILENAME)
sf = SdfFile(SDF_FILENAME)
mesh  = of.read()
sdf   = sf.read()

obj = GraspableObject3D(sdf, mesh)
gripper = RobotGripper.load(GRIPPER_NAME, os.path.join(WORK_DIR, "foxarm/common"))
ags = AntipodalGraspSampler(gripper, CONFIG)

database = Hdf5Database(TEST_DB_NAME, access_level=READ_WRITE_ACCESS)
dataset = database.dataset(TEST_DS_NAME)

##### Generate Grasps #####
print('\n\n*****\tObject:  %s\t*****' % dataset.object_keys[0])
unaligned_grasps = ags.generate_grasps(obj, target_num_grasps=100)
print('### Generated %d unaligned grasps! ###' % len(unaligned_grasps))
grasps = {}

stps = dataset.stable_poses(dataset.object_keys[0])
print('### Generated %d stable poses! ###' % len(stps))

for stp in stps:
	grasps[stp.id] = []
	for grasp in unaligned_grasps:
		aligned_grasp = grasp.perpendicular_table(stp)
		grasps[stp.id].append(copy.deepcopy(aligned_grasp))
	print('\tStable poses %s has %d grasps!' % (stp.id, len(grasps[stp.id])))

print('\n### Grasp Quality ###')
for stp in stps:
	grasps_test = grasps[stp.id]
	print('### Stable pose %s (%-5.3f) ###' % (stp.id, stp.p))
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
	# print('\t%s' % l_fa1)
		# print('Grasp%-3d    Contacts found: %s    Force closure: %s'%(i, success, if_force_closure))

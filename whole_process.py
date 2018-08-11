import copy
import numpy as np
import os
import trimesh
from autolab_core import RigidTransform, YamlConfig
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

obj_path = "mini_dexnet/bar_clamp.obj"
sdf_path = "mini_dexnet/bar_clamp.sdf"
mesh = trimesh.load_mesh(obj_path)
sdf = SdfFile(sdf_path).read()

mag = 2 * float(np.max(np.abs(mesh.vertices)))

CONFIG = YamlConfig(TEST_CONFIG_NAME)

obj = GraspableObject3D(sdf, mesh)

gripper = RobotGripper.load(GRIPPER_NAME, os.path.join(WORK_DIR, "foxarm/common"))
ags = AntipodalGraspSampler(gripper, CONFIG)

##### Generate Grasps #####
unaligned_grasps = ags.generate_grasps(obj, target_num_grasps=100)
print('### Generated %d unaligned grasps! ###' % len(unaligned_grasps))

stp_mats, stp_probs = mesh.compute_stable_poses(n_samples = 1)
stps = []
for stp_mat in stp_mats:
    r, t = RigidTransform.rotation_and_translation_from_matrix(stp_mat)
    stps.append(RigidTransform(rotation=r, translation=t))
print('### Generated %d stable poses! ###' % len(stps))

grasps = []
for grasp in unaligned_grasps:
    aligned_grasp = grasp.perpendicular_table(stps[0])
    grasps.append(copy.deepcopy(aligned_grasp))
print('\tStable pose has %d grasps!' % len(grasps))

# plot the result with mayavi.mlab
def plot_mesh(mesh):
    mesh.vertices
    x = mesh.vertices[:, 0]
    y = mesh.vertices[:, 1]
    z = mesh.vertices[:, 2]
    mlab.triangular_mesh(x, y, z, mesh.faces, color=(0.5, 0.5, 1))

def plot_plane(mag):
    x = np.asarray([mag, -mag, -mag, mag])
    y = np.asarray([mag, mag, -mag, -mag])
    z = np.asarray([0, 0, 0, 0])
    faces = [(0, 1, 2), (0, 2, 3)]
    mlab.triangular_mesh(x, y, z, faces, color=(1, 1, 1))
    # s = np.ones((10, 10))
    # mlab.imshow(s, color=(1, 1, 1), extent=[-mag, mag, -mag, mag, -mag, mag])

def plot_frame(length=1):
    x = np.linspace(0, length, num=10)
    y = np.zeros(10)
    z = np.zeros(10)
    l = mlab.plot3d(x, y, z, color=(1,0,0), tube_radius=mag / 40)

    x = np.zeros(10)
    y = np.linspace(0, length, num=10)
    z = np.zeros(10)
    l = mlab.plot3d(x, y, z, color=(0,1,0), tube_radius=mag / 40)

    x = np.zeros(10)
    y = np.zeros(10)
    z = np.linspace(0, length, num=10)
    l = mlab.plot3d(x, y, z, color=(0,0,1), tube_radius=mag / 40)

stable_mesh = obj.mesh.copy()
stable_mesh.apply_transform(stps[0].matrix)
plot_mesh(stable_mesh)
plot_plane(mag)
plot_frame(mag)
mlab.show()

'''
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


# calculate quality
quality_config = GraspQualityConfigFactory.create_config(CONFIG['metrics']
                                                               ['robust_ferrari_canny'])
quality = PointGraspMetrics3D.grasp_quality(unaligned_grasps[0], obj, quality_config)
print(quality)
'''


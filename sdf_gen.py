import os
import subprocess

obj_path_list_1 = [os.path.join('objs/3dnet', e) for e in os.listdir('objs/3dnet')]
obj_path_list_2 = [os.path.join('objs/kit', e) for e in os.listdir('objs/kit')]

obj_path_list = obj_path_list_1 + obj_path_list_2

for idx, obj_path in enumerate(obj_path_list):
    sdf_path = obj_path[:-4] + ".sdf"
    if os.path.isfile(sdf_path):
        continue
    cmd = 'SDFGen %s 100 5' % obj_path
    if idx % 10 == 0:
        print(idx)
    os.system(cmd)
    # subprocess.run(cmd)


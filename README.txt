The foxarm Python package.

Installation:
trimesh: sudo pip3 install trimesh
autolab_core: sudo pip3 install autolab_core

Add the following line to the end of ".bashrc" script and run the "source ~/.bashrc" command.
export PYTHONPATH=$PYTHONPATH:`readlink -f {path_of_foxarm}`

3D object data can be downloaded from the server 10.8.0.22 (192.168.1.120), and the path is ~/Datasets/mini_dexnet.zip. The data should be unzipped and put in the foxarm directory.

After installation and data preparation, run the render_depth.py and a 3D object window should be seen.


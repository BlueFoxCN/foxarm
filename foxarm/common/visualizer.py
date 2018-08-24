from mayavi import mlab
import trimesh
from autolab_core import PointCloud
import numpy as np

class Vis:
    @staticmethod
    def plot_mesh(mesh):
        mesh.vertices
        x = mesh.vertices[:, 0]
        y = mesh.vertices[:, 1]
        z = mesh.vertices[:, 2]
        mlab.triangular_mesh(x, y, z, mesh.faces, color=(0.5, 0.5, 1))

    @staticmethod
    def plot_plane(mag):
        x = np.asarray([mag, -mag, -mag, mag])
        y = np.asarray([mag, mag, -mag, -mag])
        z = np.asarray([0, 0, 0, 0])
        faces = [(0, 1, 2), (0, 2, 3)]
        mlab.triangular_mesh(x, y, z, faces, color=(0,0,1))

    @staticmethod
    def plot_grasp(grasp, obj, mag, transform=None, color=(1, 0, 0)):
        _, c = grasp.close_fingers(obj)
        c1, c2 = c
        c1_start = c1.point
        c2_start = c2.point
        d1 = (c1_start - grasp.center)
        d1 = mag * d1 / np.linalg.norm(d1)
        d2 = (c2_start - grasp.center)
        d2 = mag * d2 / np.linalg.norm(d2)
        c1_end = c1_start + d1
        c2_end = c2_start + d2

        if transform is not None:
            points = PointCloud(np.asarray([c1_start, c1_end, c2_start, c2_end]).T, frame=transform.from_frame)
            points = transform.apply(points).data.T
            c1_start = points[0]
            c1_end = points[1]
            c2_start = points[2]
            c2_end = points[3]

        x = np.linspace(c1_start[0], c1_end[0], num=10)
        y = np.linspace(c1_start[1], c1_end[1], num=10)
        z = np.linspace(c1_start[2], c1_end[2], num=10)
        l = mlab.plot3d(x, y, z, color=color, tube_radius=mag / 40)

        x = np.linspace(c2_start[0], c2_end[0], num=10)
        y = np.linspace(c2_start[1], c2_end[1], num=10)
        z = np.linspace(c2_start[2], c2_end[2], num=10)
        l = mlab.plot3d(x, y, z, color=color, tube_radius=mag / 40)

    @staticmethod
    def plot_frame(length=1, transform=None):
        x = np.linspace(0, length, num=10)
        y = np.zeros(10)
        z = np.zeros(10)
        if transform is not None:
            points = PointCloud(np.stack([x,y,z]), frame=transform.from_frame)
            points = transform.apply(points).data
            x = points[0,:]
            y = points[1,:]
            z = points[2,:]
        l = mlab.plot3d(x, y, z, color=(1,0,0), tube_radius=length / 40)

        x = np.zeros(10)
        y = np.linspace(0, length, num=10)
        z = np.zeros(10)
        if transform is not None:
            points = PointCloud(np.stack([x,y,z]), frame=transform.from_frame)
            points = transform.apply(points).data
            x = points[0,:]
            y = points[1,:]
            z = points[2,:]
        l = mlab.plot3d(x, y, z, color=(0,1,0), tube_radius=length / 40)

        x = np.zeros(10)
        y = np.zeros(10)
        z = np.linspace(0, length, num=10)
        if transform is not None:
            points = PointCloud(np.stack([x,y,z]), frame=transform.from_frame)
            points = transform.apply(points).data
            x = points[0,:]
            y = points[1,:]
            z = points[2,:]
        l = mlab.plot3d(x, y, z, color=(0,0,1), tube_radius=length / 40)

    @staticmethod
    def plot_camera(filename, transform, with_axis=True):
        camera_mesh = trimesh.load_mesh(filename)
        camera_mesh.apply_transform(transform.matrix)
        Vis.plot_mesh(camera_mesh)

        if with_axis:
            # mag = 2 * float(np.max(np.abs(camera_mesh.vertices)))
            mag = np.max(np.max(camera_mesh.vertices, axis=0) - np.min(camera_mesh.vertices, axis=0))
            Vis.plot_frame(mag, transform)

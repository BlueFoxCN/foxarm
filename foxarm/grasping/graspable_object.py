from abc import ABCMeta, abstractmethod

import copy
import logging
import numpy as np

# import meshpy.mesh as m
# import meshpy.sdf as s
from foxarm.common.sdf import Sdf3D
from trimesh import Trimesh

import IPython
import matplotlib.pyplot as plt

from autolab_core import RigidTransform, SimilarityTransform

class GraspableObject:
    """ Encapsulates geometric structures for computing contact in grasping.
    
    Attributes
    ----------
    sdf : :obj:`Sdf3D`
        signed distance field, for quickly computing contact points
    mesh : :obj:`Trimesh`
        3D triangular mesh to specify object geometry, should match SDF
    key : :obj:`str`
        object identifier, usually given from the database
    model_name : :obj:`str`
        name of the object mesh as a .obj file, for use in collision checking
    mass : float
        mass of the object
    convex_pieces : :obj:`list` of :obj:`Trimesh`
        convex decomposition of the object geom for collision checking
    """
    __metaclass__ = ABCMeta

    def __init__(self, sdf, mesh, key='', model_name='', mass=1.0, convex_pieces=None):
        self.sdf_ = sdf
        self.mesh_ = mesh

        self.key_ = key
        self.model_name_ = model_name # for OpenRave usage, gross!
        self.mass_ = mass
        self.convex_pieces_ = convex_pieces

    @property
    def sdf(self):
        return self.sdf_

    @property
    def mesh(self):
        return self.mesh_

    @property
    def mass(self):
        return self.mass_

    @property
    def key(self):
        return self.key_

    @property
    def model_name(self):
        return self.model_name_

    @property
    def convex_pieces(self):
        return self.convex_pieces_

class GraspableObject3D(GraspableObject):
    """ 3D Graspable object for computing contact in grasping.
    
    Attributes
    ----------
    sdf : :obj:`Sdf3D`
        signed distance field, for quickly computing contact points
    mesh : :obj:`Trimesh`
        3D triangular mesh to specify object geometry, should match SDF
    key : :obj:`str`
        object identifier, usually given from the database
    model_name : :obj:`str`
        name of the object mesh as a .obj file, for use in collision checking
    mass : float
        mass of the object
    convex_pieces : :obj:`list` of :obj:`Trimesh`
        convex decomposition of the object geom for collision checking
    """
    def __init__(self, sdf, mesh, key='',
                 model_name='', mass=1.0,
                 convex_pieces=None):
        if not isinstance(sdf, Sdf3D):
            raise ValueError('Must initialize 3D graspable object with 3D sdf')
        if not isinstance(mesh, Trimesh):
            raise ValueError('Must initialize 3D graspable object with 3D mesh')

        GraspableObject.__init__(self, sdf, mesh, key=key,
                                 model_name=model_name, mass=mass,
                                 convex_pieces=convex_pieces)

    def moment_arm(self, x):
        """ Computes the moment arm to a point x.

        Parameters
        ----------
        x : 3x1 :obj:`numpy.ndarray`
            point to get moment arm for
        
        Returns
        -------
        3x1 :obj:`numpy.ndarray`
        """
        return x - self.mesh.center_mass

    def rescale(self, scale):
        """ Rescales uniformly by a given factor.

        Parameters
        ----------
        scale : float
            the amount to scale the object

        Returns
        -------
        :obj:`GraspableObject3D`
            the graspable object rescaled by the given factor
        """
        stf = SimilarityTransform(scale=scale)
        sdf_rescaled = self.sdf_.rescale(scale)
        mesh_rescaled = self.mesh_.transform(stf)
        convex_pieces_rescaled = None
        if self.convex_pieces_ is not None:
            convex_pieces_rescaled = []
            for convex_piece in self.convex_pieces_:
                convex_piece_rescaled = convex_piece.apply_transform(stf.matrix)
                convex_pieces_rescaled.append(convex_piece_rescaled)
        return GraspableObject3D(sdf_rescaled, mesh_rescaled, key=self.key,
                                 model_name=self.model_name, mass=self.mass,
                                 convex_pieces=convex_pieces_rescaled)

    def transform(self, delta_T):
        """ Transform by a delta transform.


        Parameters
        ----------
        delta_T : :obj:`RigidTransform`
            the transformation from the current reference frame to the alternate reference frame
        
        Returns
        -------
        :obj:`GraspableObject3D`
             graspable object trasnformed by the delta
        """
        sdf_tf = self.sdf_.transform(delta_T)
        mesh_tf = self.mesh_.apply_transform(delta_T.matrix)
        convex_pieces_tf = None
        if self.convex_pieces_ is not None:
            convex_pieces_tf = []
            for convex_piece in self.convex_pieces_:
                convex_piece_tf = convex_piece.apply_transform(delta_T.matrix)
                convex_pieces_tf.append(convex_piece_tf)
        return GraspableObject3D(sdf_tf, mesh_tf, key=self.key,
                                 model_name=self.model_name, mass=self.mass,
                                 convex_pieces=convex_pieces_tf)

    def surface_information(self, grasp, width, num_steps, plot=False, direction1=None, direction2=None):
        """ Returns the patches on this object for a given grasp.

        Parameters
        ----------
        grasp : :obj:`ParallelJawPtGrasp3D`
            grasp to get the patch information for
        width : float
            width of jaw opening
        num_steps : int
            number of steps
        plot : bool
            whether to plot the intermediate computation, for debugging
        direction1 : normalized 3x1 :obj:`numpy.ndarray`
            direction along which to compute the surface information for the first jaw, if None then defaults to grasp axis
        direction2 : normalized 3x1 :obj:`numpy.ndarray`
            direction along which to compute the surface information for the second jaw, if None then defaults to grasp axis
       
        Returns
        -------
        :obj:`list` of :obj:`SurfaceWindow`
             surface patches, one for each contact
       """
        contacts_found, contacts = grasp.close_fingers(self)#, vis=True)
        if not contacts_found:
            raise ValueError('Failed to find contacts')
        contact1, contact2 = contacts

        if plot:
            plt.figure()
            contact1.plot_friction_cone()
            contact2.plot_friction_cone()

            ax = plt.gca(projection = '3d')
            ax.set_xlim3d(0, self.sdf.dims_[0])
            ax.set_ylim3d(0, self.sdf.dims_[1])
            ax.set_zlim3d(0, self.sdf.dims_[2])

        window1 = contact1.surface_information(width, num_steps, direction=direction1)
        window2 = contact2.surface_information(width, num_steps, direction=direction2)
        return window1, window2, contact1, contact2


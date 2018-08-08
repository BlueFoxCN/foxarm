import logging
import numpy as np
try:
    import pyhull.convex_hull as cvh
except:
    logging.warning('Failed to import pyhull')
try:
    import cvxopt as cvx
except:
    logging.warning('Failed to import cvx')
import os
import scipy.spatial as ss
import sys
import time

'''
	from dexnet.grasping import PointGrasp, GraspableObject3D, GraspQualityConfig
'''
from foxarm.grasping.grasp import PointGrasp
from foxarm.grasping.graspable_object import GraspableObject3D
from foxarm.grasping.grasp_quality_config import GraspQualityConfig

'''
import meshpy.obj_file as obj_file
import meshpy.sdf_file as sdf_file
'''
import foxarm.common.obj_file as obj_file
import foxarm.common.sdf_file as sdf_file

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import mayavi.mlab as mv
except:
    logging.warning('Failed to import mayavi')
import IPython

# turn off output logging
cvx.solvers.options['show_progress'] = False

class PointGraspMetrics3D:
    """ Class to wrap functions for quasistatic point grasp quality metrics.
    """

	@staticmethod
	def grasp_quality(grasp, obj, params, vis=False):
        """
        Computes the quality of a two-finger point grasps on a given object using a quasi-static model.
        Parameters
        ----------
        grasp : :obj:`ParallelJawPtGrasp3D`
            grasp to evaluate
        obj : :obj:`GraspableObject3D`
            object to evaluate quality on
        params : :obj:`GraspQualityConfig`
            parameters of grasp quality function
        """
		start = time.time()
		if not isinstance(grasp, PointGrasp):
            raise ValueError('Must provide a point grasp object')
        if not isinstance(obj, GraspableObject3D):
            raise ValueError('Must provide a 3D graspable object')
        if not isinstance(params, GraspQualityConfig):
            raise ValueError('Must provide GraspQualityConfig')

        method = params.quality_method
        friction_coef = params.friction_coef
        num_cone_faces = params.num_cone_faces
        soft_fingers = params.soft_fingers
        check_approach = params.check_approach
        if not hasattr(PointGraspMetrics3D, method):
            raise ValueError('Illegal point grasp metric %s specified' %(method))

        # get point grasp contacts
        contacts_start = time.time()
        contacts_found, contacts = grasp.close_fingers(obj,
                                                       check_approach=check_approach,
                                                       vis=vis)
        if not contacts_found:
            logging.debug('Contacts not found')
            return 0

        if method == 'force_closure':
            # Use fast force closure test (Nguyen 1988) if possible.
            if len(contacts) == 2:
                c1, c2 = contacts
                return PointGraspMetrics3D.force_closure(c1, c2, friction_coef)

            # Default to QP force closure test.
            method = 'force_closure_qp'

        # add the forces, torques, etc at each contact point
        forces_start = time.time()
        num_contacts = len(contacts)
        forces = np.zeros([3,0])
        torques = np.zeros([3,0])
        normals = np.zeros([3,0])
        for i in range(num_contacts):
            contact = contacts[i]
            if vis:
                if i == 0:
                    contact.plot_friction_cone(color='y')
                else:
                    contact.plot_friction_cone(color='c')

            # get contact forces
            force_success, contact_forces, contact_outward_normal = contact.friction_cone(num_cone_faces, friction_coef)

            if not force_success:
                logging.debug('Force computation failed')
                if params.all_contacts_required:
                    return 0              
            # get contact torques
            torque_success, contact_torques = contact.torques(contact_forces)
            if not torque_success:
                logging.debug('Torque computation failed')
                if params.all_contacts_required:
                    return 0

            # get the magnitude of the normal force that the contacts could apply
            n = contact.normal_force_magnitude()

            forces = np.c_[forces, n * contact_forces]
            torques = np.c_[torques, n * contact_torques]
            normals = np.c_[normals, n * -contact_outward_normal] # store inward pointing normals


        if normals.shape[1] == 0:
            logging.debug('No normals')
            return 0

        # normalize torques
        if 'torque_scaling' not in params.keys():
            torque_scaling = 1.0
            if method == 'ferrari_canny_L1':
                mn, mx = obj.mesh.bounding_box()
                torque_scaling = 1.0 / np.median(mx)
            params.torque_scaling = torque_scaling 
        if vis:
            ax = plt.gca()
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.show()

        # evaluate the desired quality metric
        quality_start = time.time()
        Q_func = getattr(PointGraspMetrics3D, method)
        quality = Q_func(forces, torques, normals,
                         soft_fingers=soft_fingers,
                         params=params)

        end = time.time()
        logging.debug('Contacts took %.3f sec' %(forces_start - contacts_start))
        logging.debug('Forces took %.3f sec' %(quality_start - forces_start))
        logging.debug('Quality eval took %.3f sec' %(end - quality_start))
        logging.debug('Everything took %.3f sec' %(end - start))

        return quality

    @staticmethod
    def grasp_matrix(forces, torques, normals, soft_fingers=False,
                     finger_radius=0.005, params=None):
        """ Computes the grasp map between contact forces and wrenchs on the object in its reference frame.
        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        finger_radius : float
            the radius of the fingers to use
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model
        Returns
        -------
        G : 6xM :obj:`numpy.ndarray`
            grasp map
        """
        if params is not None and 'finger_radius' in params.keys():
            finger_radius = params.finger_radius
        num_forces = forces.shape[1]
        num_torques = torques.shape[1]
        if num_forces != num_torques:
            raise ValueError('Need same number of forces and torques')

        num_cols = num_forces
        if soft_fingers:
            num_normals = 2
            if normals.ndim > 1:
                num_normals = 2*normals.shape[1]
            num_cols = num_cols + num_normals

        G = np.zeros([6, num_cols])
        for i in range(num_forces):
            G[:3,i] = forces[:,i]
            G[3:,i] = params.torque_scaling * torques[:,i]

        if soft_fingers:
            torsion = np.pi * finger_radius**2 * params.friction_coef * normals * params.torque_scaling
            pos_normal_i = -num_normals
            neg_normal_i = -num_normals + num_normals / 2
            G[3:,pos_normal_i:neg_normal_i] = torsion
            G[3:,neg_normal_i:] = -torsion

        return G

    @staticmethod
    def ferrari_canny_L1(forces, torques, normals, soft_fingers=False, params=None,
                         wrench_norm_thresh=1e-3,
                         wrench_regularizer=1e-10):
        """ Ferrari & Canny's L1 metric. Also known as the epsilon metric.
        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        Returns
        -------
        float : value of metric
        """
        if params is not None and 'wrench_norm_thresh' in params.keys():
            wrench_norm_thresh = params.wrench_norm_thresh
        if params is not None and 'wrench_regularizer' in params.keys():
            wrench_regularizer = params.wrench_regularizer

        # create grasp matrix
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals,
                                             soft_fingers, params=params)
        s = time.time()
        # center grasp matrix for better convex hull comp
        hull = cvh.ConvexHull(G.T)
        # TODO: suppress ridiculous amount of output for perfectly valid input to qhull
        e = time.time()
        logging.debug('CVH took %.3f sec' %(e - s))
        
        debug = False
        if debug:
            fig = plt.figure()
            torques = G[3:,:].T
            ax = Axes3D(fig)
            ax.scatter(torques[:,0], torques[:,1], torques[:,2], c='b', s=50)
            ax.scatter(0, 0, 0, c='k', s=80)
            ax.set_xlim3d(-1.5, 1.5)
            ax.set_ylim3d(-1.5, 1.5)
            ax.set_zlim3d(-1.5, 1.5)
            ax.set_xlabel('tx')
            ax.set_ylabel('ty')
            ax.set_zlabel('tz')
            plt.show()

        if len(hull.vertices) == 0:
            logging.warning('Convex hull could not be computed')
            return 0.0

        # determine whether or not zero is in the convex hull
        s = time.time()
        min_norm_in_hull, v = PointGraspMetrics3D.min_norm_vector_in_facet(G, wrench_regularizer=wrench_regularizer)
        e = time.time()
        logging.debug('Min norm took %.3f sec' %(e - s))

        # if norm is greater than 0 then forces are outside of hull
        if min_norm_in_hull > wrench_norm_thresh:
            logging.debug('Zero not in convex hull')
            return 0.0

        # if there are fewer nonzeros than D-1 (dim of space minus one)
        # then zero is on the boundary and therefore we do not have
        # force closure
        if np.sum(v > 1e-4) <= G.shape[0]-1:
            logging.debug('Zero not in interior of convex hull')
            return 0.0

        # find minimum norm vector across all facets of convex hull
        s = time.time()
        min_dist = sys.float_info.max
        closest_facet = None
        for v in hull.vertices:
            if np.max(np.array(v)) < G.shape[1]: # because of some occasional odd behavior from pyhull
                facet = G[:, v]
                dist, _ = PointGraspMetrics3D.min_norm_vector_in_facet(facet, wrench_regularizer=wrench_regularizer)
                if dist < min_dist:
                    min_dist = dist
                    closest_facet = v
        e = time.time()
        logging.debug('Min dist took %.3f sec for %d vertices' %(e - s, len(hull.vertices)))

        return min_dist

    @staticmethod
    def min_norm_vector_in_facet(facet, wrench_regularizer=1e-10):
        """ Finds the minimum norm point in the convex hull of a given facet (aka simplex) by solving a QP.
        Parameters
        ----------
        facet : 6xN :obj:`numpy.ndarray`
            vectors forming the facet
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        Returns
        -------
        float
            minimum norm of any point in the convex hull of the facet
        Nx1 :obj:`numpy.ndarray`
            vector of coefficients that achieves the minimum
        """
        dim = facet.shape[1] # num vertices in facet

        # create alpha weights for vertices of facet
        G = facet.T.dot(facet)
        grasp_matrix = G + wrench_regularizer * np.eye(G.shape[0])

        # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h, Ax = b
        P = cvx.matrix(2 * grasp_matrix)   # quadratic cost for Euclidean dist
        q = cvx.matrix(np.zeros((dim, 1)))
        G = cvx.matrix(-np.eye(dim))       # greater than zero constraint
        h = cvx.matrix(np.zeros((dim, 1)))
        A = cvx.matrix(np.ones((1, dim)))  # sum constraint to enforce convex
        b = cvx.matrix(np.ones(1))         # combinations of vertices

        sol = cvx.solvers.qp(P, q, G, h, A, b)
        v = np.array(sol['x'])
        min_norm = np.sqrt(sol['primal objective'])

        return abs(min_norm), v
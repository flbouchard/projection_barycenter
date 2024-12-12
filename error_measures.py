import numpy as np
import numpy.linalg as la

from scipy.linalg import logm


def dist2_eucl(point1, point2):
    """
    Squared Euclidean distance.

    Parameters
    ----------
    point1 : ndarray
        matrix
    point2 : ndarray
        matrix

    Returns
    -------
    float
        squared Euclidean distance
    """
    return la.norm(point1-point2, 'fro')**2

def err_st(point1, point2):
    """
    Similarity measure for Stiefel matrices, i.e., it measures
    how much point1.T @ point2 deviates from identity.

    Parameters
    ----------
    point1 : ndarray
        matrix in Stiefel
    point2 : ndarray
        matrix in Stiefel

    Returns
    -------
    float
        error measure
    """
    return la.norm(point1.T@point2 - np.eye(point1.shape[1]),'fro')**2


def dist2_gr(point1, point2):
    """
    Squared Grassmann distance.

    Parameters
    ----------
    point1 : ndarray
        matrix
    point2 : ndarray
        matrix

    Returns
    -------
    float
        squared Euclidean distance
    """
    Omega = 0.5* logm( (np.eye(point1.shape[0]) - 2*point2) @ (np.eye(point1.shape[0]) - 2*point1) )
    return la.norm(Omega,'fro')**2
import numpy as np
import numpy.linalg as la

from projection_retraction import (
    st_projection_polar, st_projection_qr,
    gr_projection_evd
)


def st_projected_arithmetic_mean_polar(points):
    """
    Barycenter on Stiefel resulting from the projection of
    the arithmetic mean through the polar decomposition.

    Parameters
    ----------
    points : ndarray, shape (npoints, nfeatures, nrank)
        set of orthogonal matrices in Stiefel
    
    Returns
    -------
    ndarray, shape (nfeatures, nrank)
        orthogonal matrix in Stiefel
    """
    return st_projection_polar(np.mean(points, axis=0))


def st_projected_arithmetic_mean_qr(points):
    """
    Barycenter on Stiefel resulting from the projection of
    the arithmetic mean through the QR decomposition.

    Parameters
    ----------
    points : ndarray, shape (npoints, nfeatures, nrank)
        set of orthogonal matrices in Stiefel
    
    Returns
    -------
    ndarray, shape (nfeatures, nrank)
        orthogonal matrix in Stiefel
    """
    return st_projection_qr(np.mean(points, axis=0))


def gr_projected_arithmetic_mean_evd(points, nrank):
    """
    Barycenter on Grassmann resulting from the projection of
    the arithmetic mean through the eigenvalue decomposition.

    Parameters
    ----------
    points : ndarray, shape (npoints, nfeatures, nfeatures)
        set of projector matrices in Grassmann
    
    Returns
    -------
    ndarray, shape (nfeatures, nfeatures)
        projector matrix in Grassmann
    """
    return gr_projection_evd(np.mean(points, axis=0),nrank)


def R_barycenter(points, retr, inv_retr, init, stepsize=1, max_it=100, tol=1e-6, verbosity=True):
    """
    R-barycenter .

    Parameters
    ----------
    points : ndarray, shape (npoints, ...)
        set of matrices
    retr : function
        retraction
    inv_retr : function
        inverse retraction of retr
    init : ndarray, shape (...)
        initial guess for barycenter
    stepsize : float, optional
        constant step size for the fixed-point algorithm, by default 1
    max_it : int, optional
        maximum number of iterations, by default 100
    tol : float, optional
        tolerance for the stopping criterion, by default 1e-6
    verbosity : bool, optional
        whether to print information on the optimization process or not, by default True

    Returns
    -------
    ndarray, shape (...)
        barycenter
    list
        list of iterates
    """
    npoints = points.shape[0]
    G = init
    iterates = []
    iterates.append(G)
    it = 1
    err = 1
    while (it < max_it) and (err > tol):
        tangent_vecs = np.zeros(points.shape)
        for i in range(npoints):
            tangent_vecs[i] = inv_retr(G,points[i])
        G_new = retr(G, stepsize * np.mean(tangent_vecs, axis=0))
        err = la.norm(G_new-G, 'fro') / la.norm(G,'fro')
        if verbosity:
            print(f"it: {it} \t err: {err}")
        it = it+1
        G = G_new
        iterates.append(G)
    return G, iterates

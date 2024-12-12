import numpy as np
import numpy.linalg as la
from scipy.linalg import polar, qr, solve_continuous_lyapunov, solve_continuous_are, expm, logm


from tools import sym


def st_projection_polar(point):
    """
    Projection from the Euclidean (ambient) space of nfeatures x nrank matrices
    onto the Stiefel manifold according to the polar decomposition.

    Parameters
    ----------
    point : ndarray, shape (nfeatures, nrank)
        matrix

    Returns
    -------
    ndarray, shape (nfeatures, nrank)
        orthogonal matrix in Stiefel
    """
    return polar(point)[0]


def st_projection_qr(point):
    """
    Projection from the Euclidean (ambient) space of nfeatures x nrank matrices
    onto the Stiefel manifold according to the QR decomposition.

    Parameters
    ----------
    point : ndarray, shape (nfeatures, nrank)
        matrix

    Returns
    -------
    ndarray, shape (nfeatures, nrank)
        orthogonal matrix in Stiefel
    """
    Q,R = qr(point, mode='economic')
    signs = 2 * (np.diag(R) >= 0) - 1
    Q = Q * signs[np.newaxis, :]
    return Q


def st_projection_tangent(point, vector):
    """
    Projection from the Euclidean (ambient) space of nfeatures x nrank matrices
    onto the tangent space of the Stiefel manifold at point.

    Parameters
    ----------
    point : ndarray, shape (nfeatures, nrank)
        orthogonal matrix in Stiefel
    vector : ndarray, shape (nfeatures, nrank)
        matrix
    Returns
    -------
    ndarray, shape (nfeatures, nrank)
        tangent vector at point
    """
    return vector - point @ sym(point.T @ vector)


def st_retr_polar(point, vector):
    """
    Retraction on Stiefel based on the polar decomposition.

    Parameters
    ----------
    point : ndarray, shape (nfeatures, nrank)
        matrix in Stiefel
    vector : ndarray, shape (nfeatures, nrank)
        tangent vector at point in Stiefel 

    Returns
    -------
    ndarray, shape (nfeatures, nrank)
        matrix in Stiefel
    """
    return st_projection_polar(point+vector)


def st_inv_retr_polar(point1, point2):
    """
    Inverse retraction on Stiefel of the retraction based on the polar decomposition.

    Parameters
    ----------
    point1 : ndarray, shape (nfeatures, nrank)
        matrix in Stiefel
    point2 : ndarray, shape (nfeatures, nrank)
        matrix in Stiefel

    Returns
    -------
    ndarray, shape (nfeatures, nrank)
        tangent vector at point1 in Stiefel 
    """
    M = point1.T @ point2
    S = solve_continuous_lyapunov(a=M, q=2*np.eye(M.shape[0]))
    return point2 @ S - point1


def st_retr_qr(point, vector):
    """
    Retraction on Stiefel based on the QR decomposition.

    Parameters
    ----------
    point : ndarray, shape (nfeatures, nrank)
        matrix in Stiefel
    vector : ndarray, shape (nfeatures, nrank)
        tangent vector at point in Stiefel 

    Returns
    -------
    ndarray, shape (nfeatures, nrank)
        matrix in Stiefel
    """
    return st_projection_qr(point+vector)


def st_inv_retr_qr(point1, point2):
    """
    Inverse retraction on Stiefel of the retraction based on the QR decomposition.

    Parameters
    ----------
    point1 : ndarray, shape (nfeatures, nrank)
        matrix in Stiefel
    point2 : ndarray, shape (nfeatures, nrank)
        matrix in Stiefel

    Returns
    -------
    ndarray, shape (nfeatures, nrank)
        tangent vector at point1 in Stiefel 
    """
    M = point1.T @ point2
    R = np.zeros(shape=M.shape)

    R[0,0] = 1 / M[0,0]
    # if M[0,0]>0:
    #     R[0,0] = 1 / M[0,0]
    # else:
    #     raise ValueError('st_inv_retr_qr: inverse retraction not defined in this case')
    
    if M.shape[0]>1:
        for i in range(1,M.shape[0]):
            Mi = M[np.ix_(range(i+1),range(i+1))]
            bi = np.ones(i+1)
            bi[:i] = -Mi[-1][:i] @ R[np.ix_(range(i),range(i))]
            ri = la.solve(Mi,bi)
            # if ri[i]<=0:
            #     raise ValueError('st_inv_retr_qr: inverse retraction not defined in this case')
            R[np.ix_(range(i+1),[i])]=np.reshape(ri,(i+1,1))

    return point2 @ R - point1


def st_retr_orthographic(point, vector):
    """
    Orthographic retraction on Stiefel.

    Parameters
    ----------
    point : ndarray, shape (nfeatures, nrank)
        matrix in Stiefel
    vector : ndarray, shape (nfeatures, nrank)
        tangent vector at point in Stiefel 

    Returns
    -------
    ndarray, shape (nfeatures, nrank)
        matrix in Stiefel
    """
    M = point.T @ vector + np.eye(point.shape[1])
    S = solve_continuous_are(a=-M, b=np.eye(point.shape[1]), q=-vector.T@vector, r=np.eye(point.shape[1]))
    return point + vector + point @ S


def st_inv_retr_orthographic(point1, point2):
    """
    Inverse retraction on Stiefel of the orthographic retraction.

    Parameters
    ----------
    point1 : ndarray, shape (nfeatures, nrank)
        matrix in Stiefel
    point2 : ndarray, shape (nfeatures, nrank)
        matrix in Stiefel

    Returns
    -------
    ndarray, shape (nfeatures, nrank)
        tangent vector at point1 in Stiefel 
    """
    return st_projection_tangent(point1,point2)




def gr_projection_evd(point, nrank):
    """
    Projection from the Euclidean (ambient) space of nfeatures x nfeatures symmetric matrices
    onto the Grassmann manifold identified as the set of projectors of rank nrank.
    It relies on the eigenvalue decomposition of point.

    Parameters
    ----------
    point : ndarray, shape (nfeatures, nfeatures)
        matrix
    nrank : int
        rank of the desired Grassmann projector matrix (nrank < nfeatures)

    Returns
    -------
    ndarray, shape (nfeatures, nfeatures)
        projector matrix of rank nrank
    """
    _, U = la.eigh(point)
    return U[:,-nrank:] @ U[:,-nrank:].T


def gr_projection_tangent(point, vector):
    """
    Projection from the Euclidean (ambient) space of nfeatures x nfeatures matrices
    onto the tangent space of the Grassmann manifold identified as the set of projectors at point.

    Parameters
    ----------
    point : ndarray, shape (nfeatures, nfeatures)
        projector matrix in Grassmann
    vector : ndarray, shape (nfeatures, nfeatures)
        matrix
    Returns
    -------
    ndarray, shape (nfeatures, nfeatures)
        tangent vector at point
    """
    return 2 * sym( (np.eye(point.shape[0]) - point) @ sym(vector) @ point )


def gr_exp(point, vector):
    """
    Riemannian exponential mapping of the Grassmann manifold
    identified as the set of projectors.

    Parameters
    ----------
    point : ndarray, shape (nfeatures, nfeatures)
        projector matrix in Grassmann
    vector : ndarray, shape (nfeatures, nfeatures)
        tangent vector at point in Grassmann

    Returns
    -------
    ndarray, shape (nfeatures, nfeatures)
        projector matrix in Grassmann
    """
    exp_Omega = expm(vector @ point - point @ vector)
    return exp_Omega @ point @ exp_Omega.T


def gr_log(point1, point2):
    """
    Riemannian logarithm mapping of the Grassmann manifold
    identified as the set of projectors.

    Parameters
    ----------
    point1 : ndarray, shape (nfeatures, nfeatures)
        projector matrix in Grassmann
    point2 : ndarray, shape (nfeatures, nfeatures)
        projector matrix in Grassmann

    Returns
    -------
    ndarray, shape (nfeatures, nfeatures)
        tangent vector at point1 in Grassmann 
    """
    Omega = 0.5* logm( (np.eye(point1.shape[0]) - 2*point2) @ (np.eye(point1.shape[0]) - 2*point1) )
    return Omega @ point1 - point1 @ Omega
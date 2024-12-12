from scipy.linalg import expm
from scipy.stats import ortho_group, norm

from tools import skew


def st_random_center(nfeatures,nrank):
    """
    Computes a random orthogonal matrix on Stiefel
    along with its orthogonal complement by leveraging
    the uniform distribution on the orthogonal group 

    Parameters
    ----------
    nfeatures : int
        size
    nrank : int
        size (nfeatures < nrank)

    Returns
    -------
    ndarray, ndarray, shapes (nfeatures, nrank) and (nfeatures, nfeatures-nrank)
        random orthogonal matrix and its orthogonal complement
    """
    O = ortho_group.rvs(nfeatures)
    return O[:,:nrank], O[:,nrank:]

def st_random_sample_from_center(center, scale=0.5):
    """
    Computes a random orthogonal matrix sample on Stiefel
    around a center according to
        sample = expm(scale * Omega) @ center,
    where Omega is a random skew-symmetric matrix obtained by
    taking the skew-symmetrical part of a nfeatures x nfeatures
    matrix whose elements are drawn from the normal distribution


    Parameters
    ----------
    center : ndarray, shape (nfeatures, nrank)
        center on the Stiefel manifold
    scale : float, optional
        "distance" of sample to the center, by default 0.5

    Returns
    -------
    ndarray, shape (nfeatures, nrank)
        random orthogonal matrix on Stiefel
    """
    return expm(scale* skew(norm.rvs(size=(center.shape[0],center.shape[0])))) @ center


def st_to_gr(point):
    """
    Mapping from the Stiefel manifold to the Grassmann manifold,
    represented by the space of projectors.

    Parameters
    ----------
    point : ndarray, shape (nfeatures, nrank)
        orthogonal manifold on Stiefel

    Returns
    -------
    ndarray, shape (nfeatures, nfeatures)
        projector in Grassmann
    """
    return point @ point.T
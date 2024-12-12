

def skew(point):
    """
    Returns the skew-symmetrical part of point

    Parameters
    ----------
    point : ndarray, shape (nfeatures, nfeatures)
        square matrix

    Returns
    -------
    ndarray, shape (nfeatures, nfeatures)
        skew-symmetrical part of point
    """
    return 0.5*(point - point.T)


def sym(point):
    """
    Returns the symmetrical part of point.

    Parameters
    ----------
    point : ndarray, shape (nfeatures, nfeatures)
        square matrix

    Returns
    -------
    ndarray, shape (nfeatures, nfeatures)
        symmetrical part of point
    """
    return 0.5*(point + point.T)
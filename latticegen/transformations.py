"""Common code for geometrical transformations"""
import numpy as np


def rotation_matrix(angle):
    """Create a rotation matrix

    an array corresponding to the 2D transformation matrix
    of a rotation over `angle`.

    Parameters
    ----------
    angle : float
        rotation angle in radians

    Returns
    -------
    ndarray (2x2)
        2D transformation matrix corresponding
        to the rotation
    """
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def scaling_matrix(kappa, dims=2):
    """Create a scaling matrix

    Creates a numpy array containing the `dims`-dimensional
    scaling matrix scaling the first dimension by `kappa`.

    Parameters
    ----------
    kappa : float
        scaling factor of first dimension
    dims : int, default: 2
        number of dimensions
    Returns
    -------
    ndarray (dims x dims)
        scaling matrix corresponding
        to scaling the first dimension by a factor `kappa`
    """
    return np.diag([kappa]+[1]*(dims-1))


def apply_transformation_matrix(vecs, matrix):
    """Apply transformation matrix to a list of vectors.

    Apply transformation matrix `matrix` to a list of vectors `vecs`.
    `vecs` can either be a list of vectors, or a NxM array, where N is
    the number of M-dimensional vectors.

    Parameters
    ----------
    vecs : 2D array or iterable
        list of vectors to be transformed
    matrix : 2D array
        Array corresponding to the transformation matrix

    Returns
    -------
    2D array
        Transformed vectors
    """
    avecs = np.asanyarray(vecs)
    return avecs @ matrix.T


def rotate(vecs, angle):
    "Rotate 2D vectors `vecs` by `angle` radians around the origin."
    return apply_transformation_matrix(vecs, rotation_matrix(angle))


def wrapToPi(x):
    """Wrap all values of `x` to the interval [-pi,pi)"""
    r = (x + np.pi) % (2*np.pi) - np.pi
    return r

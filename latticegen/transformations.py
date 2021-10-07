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


def strain_matrix(epsilon, delta=0.16, axis=0, diff=True):
    """Create a scaling matrix corresponding to uniaxial strain

    Only works for the 2D case.

    Parameters
    ----------
    epsilon : float
        applied strain
    delta : float, default 0.16
        Poisson ratio. default value corresponds to graphene
    axis : {0, 1}
        Axis along which to apply the strain.
    diff : bool, default True
        Whether to apply in diffraction or real space.

    Returns
    -------
    ndarray (2 x 2)
        scaling matrix corresponding
        to `epsilon` strain along `axis`
    """
    scaling = np.full(2, 1 - delta*epsilon)
    scaling[axis] = 1 + epsilon
    if diff:
        scaling = 1 / scaling
    return np.diag(scaling)


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


def a_0_to_r_k(a_0, symmetry=6):
    """Transform realspace lattice constant to r_k

    Where $r_k = (a_0\\sin(2\\pi/symmetry))^{-1}$.
    i.e. r_k is 1/(2\\pi) the reciprocal lattice constant.

    Parameters
    ----------
    a_0 : float
        realspace lattice constant
    symmetry : integer
        symmetry of the lattice

    Returns
    -------
    r_k : float
        length of k-vector in frequency space
    """
    crossprodnorm = np.sin(2*np.pi / symmetry)
    r_k = 1 / a_0 / crossprodnorm
    return r_k


def r_k_to_a_0(r_k, symmetry=6):
    """Transform r_k to a realspace lattice constant a_0

    Parameters
    ----------
    r_k : float
        length of k-vector in frequency space
        as used by lattigeneration functions
    symmetry : integer
        symmetry of the lattice

    Returns
    -------
    a_0 : float
        realspace lattice constant
    """
    crossprodnorm = np.sin(2*np.pi / symmetry)
    a_0 = 1 / r_k / crossprodnorm
    return a_0


def epsilon_to_kappa(r_k, epsilon, delta=0.16):
    """Convert frequency r_k and strain epsilon
    to corresponding r_k and kappa as consumed by
    functions in `latticegeneration`.

    Returns
    -------
    r_k2 : float
    kappa : float

    See also
    --------
    latticegeneration.generate_ks
    """
    return r_k / (1 - delta*epsilon), (1+epsilon) / (1 - delta*epsilon)

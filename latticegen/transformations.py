import numpy as np

def rotation_matrix(angle):
    """Create an array corresponding to the 2D transformation matrix 
    of a rotation over `angle`.
    """
    return np.array([[np.cos(angle), -np.sin(angle)], 
                  [np.sin(angle), np.cos(angle)]])

def scaling_matrix(kappa, dims=2):
    """Create a numpy array containing 
    the `dims`-dimensional scaling matrix scaling the first dimension by kappa."""
    return np.diag([kappa]+[1]*(dims-1))

def apply_transformation_matrix(vecs, matrix):
    """Apply transformation matrix `matrix` to a list of vectors `vecs`.
    `vecs` can either be a list of vectors, or a NxM array, where N is 
    the number of M-dimensional vectors.
    """
    avecs = np.asanyarray(vecs)
    return avecs @ matrix.T

def rotate(vecs, angle):
    "Rotate `vecs` by `angle` around the origin."
    return apply_transformation_matrix(vecs, rotation_matrix(angle))

def wrapToPi(x):
    """Wrap all values of x to the interval -pi,pi"""
    r = (x+np.pi)  % (2*np.pi) - np.pi
    return r
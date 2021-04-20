"""Code to generate edge dislocations/singularities in lattices"""
import numpy as np
import dask.array as da
import itertools as itert

from latticegen.transformations import (
    rotate,
    rotation_matrix,
    apply_transformation_matrix,
    wrapToPi,
)
from latticegen.latticegeneration import hexlattice_gen


def hexlattice_gen_singularity(r_k, theta, order, size=250,
                               position=[0, 0], shift=np.array([0, 0]),
                               **kwargs):
    """Generate a hexagonal lattice with a singularity.

    Singularity is shifted `position` from the center.
    Not yet equivalent to hexlattice_gen_singularity_legacy.

    Parameters
    ----------
    r_k : float
        length of lattice vectors in k-space. Larger `r_k` correspond
        to smaller real space lattice constants.
    theta : float
        Angle of the first lattice vector with respect to positive
        horizontal.
    order : int
        Order upto which to generate higher frequency components
        by combining lattice vectors
    size: int, or pair of int, default: 500
        Size of the resulting lattice in pixels. if int, the
        returned lattice will be square.
    kappa : float, default: 1
        strain/deformation magnitude. 1 corresponds to no strain.
    position : iterable, default [0, 0]
        [x, y] position of the singularity in pixels with
        respect to the center.
    shift : iterable or array, optional
        shift of the lattice. Either a pair (x,y) global shift,
        or an (2xNxM) array where (NxM) corresponds to `size`.
    **kwargs : dict
        Keyword arguments to be passed to `hexlattice_gen`

    Returns
    -------
    lattice : Dask array
        The generated lattice with singularity

    See Also
    --------
    hexlattice_gen, singularity_shift
    """
    shift2 = shift + singularity_shift(r_k, theta, size, position)
    return hexlattice_gen(r_k, theta, order, size, shift=shift2, **kwargs)


def singularity_shift(r_k, theta, size=250, position=[0, 0],
                      alpha=0., symmetry=6):
    """Generate the shift of an edge dislocation.

    Generate the shift / displacement field
    for a hexagonal lattice of `size`, with the Burgers vector
    at angle `alpha` (radians) w.r.t. `theta` (in degrees).
    Burgers vector has length `np.sin(2*np.pi/symmetry) / r_k`,
    to create a first order edge dislocation.

    Parameters
    ----------
    r_k : float
        length of lattice vectors in k-space. Larger `r_k` correspond
        to smaller real space lattice constants.
    theta : float
        Angle of the first lattice vector with respect to positive
        horizontal.
    order : int
        Order upto which to generate higher frequency components
        by combining lattice vectors
    size: int, or pair of int, default: 500
        Size of the resulting lattice in pixels. if int, the
        returned lattice will be square.
    position : iterable, default: [0, 0]
        [x, y] position of the singularity in pixels with
        respect to the center.
    alpha : float, default 0.0
        angle in radians of the Burgers vector with respect
        to the first lattice vector.
    symmetry : int, default: 6
        symmetry of the lattice.

    Returns
    -------
    shift : ndarray
        The generated shift corresponding to a singularity,
        which can be passed to anylattice_gen

    See Also
    --------
    anylattice_gen
    """
    if not isinstance(size, tuple):
        size = (size, size)
    xx = da.arange(-size[0]/2, size[0]/2)[:, None] - position[0]
    yy = da.arange(-size[1]/2, size[1]/2)[None] - position[1]
    phi = np.arctan2(yy, xx)
    phiprime = phi / (2*np.pi)
    a_0 = 1 / np.sin(2*np.pi / symmetry) / r_k
    xp = a_0*np.sin(alpha - np.deg2rad(theta)) * phiprime
    yp = a_0*np.cos(alpha - np.deg2rad(theta)) * phiprime
    shift = np.array([xp, yp])
    return shift


def hexlattice_gen_singularity_legacy(r_k, theta, order, size=250):
    """Generate a regular hexagonal lattice of `2*size` times `2*size`.
    The lattice is generated from the six 60 degree rotations of `k0`,
    further rotated by `theta` degrees. With higher order frequency
    components upto order `order` and containing a singularity in the center.
    `theta != 0` does not yield a correct lattice.

    The generated lattice gets returned as a dask array.
    """
    if not isinstance(size, tuple):
        size = (size, size)
    xx = da.arange(-size[0]/2, size[0]/2)[:, None]
    yy = da.arange(-size[1]/2, size[1]/2)[None]
    ks = np.stack(
        [rotate(np.array([r_k, 0]), np.pi / 3 * i) for i in range(6)] + [(0, 0)]
    )
    W = rotation_matrix(np.deg2rad(theta))
    ks = apply_transformation_matrix(ks, W)
    # Yes, this is probably not the smartest way to do this
    tks = np.array([np.sum(ksp, axis=0) for ksp in itert.product(*[ks] * order)])
    rks, k_c = np.unique(tks, return_counts=True, axis=0)
    rks = da.from_array(rks, chunks=(13, 2))
    x = np.array([ks[1], -ks[2]])
    shift = (np.linalg.inv(x).T / 3).sum(axis=0)  # Don't ask, this works
    xp = xx + wrapToPi(-np.deg2rad(120 + theta) + np.arctan2(yy, xx)) / 2 / np.pi / r_k
    yp = (yy + 1/np.sqrt(3) * (wrapToPi(-np.deg2rad(120 + theta) + np.arctan2(yy, xx)))
          / 2/np.pi/r_k)
    iterated = k_c[:, None, None] * np.exp(np.pi*2j
               * ((xp - 1 * shift[0]) * rks[:, 0, None, None]
                  + (yp - 1 * shift[1]) * rks[:, 1, None, None]))
    iterated = iterated.sum(axis=0)
    # Now add the second shifted sublattice lattice to get a hexagonal lattice
    iterated += (k_c[:, None, None] * np.exp(np.pi * 2j
                 * ((xp + shift[0]) * rks[:, 0, None, None]
                    + (yp + shift[1]) * rks[:, 1, None, None]))
                 ).sum(axis=0)
    return iterated.real

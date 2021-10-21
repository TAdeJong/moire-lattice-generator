"""Code for generating (quasi-)lattices and the corresponding k-vectors"""

import numpy as np
import dask.array as da
import warnings
import itertools as itert

from latticegen.transformations import (rotate, rotation_matrix, scaling_matrix,
                                        apply_transformation_matrix, a_0_to_r_k,
                                        epsilon_to_kappa)


def generate_ks(r_k, theta, kappa=1., psi=0., sym=6):
    """Generate k-vectors from given parameters.

    Parameters
    ----------
    r_k : float
        length of lattice vectors in k-space. Larger `r_k` correspond
        to smaller real space lattice constants. 1/r_k is the line
        spacing in pixels in the resulting image.
    theta : float
        Angle of the first lattice vector with respect to positive
        horizontal.
    kappa : float, default: 1
        strain/deformation magnitude. 1 corresponds to no strain.
        Larger values corresponds to stretching along the `psi` direction
        in real space, so compression along the same direction in k-space.
    psi : float, default: 0
        Principal strain direction with respect to horizontal
        in degrees.
    sym : int, default 6
        Rotational symmetry of the unstrained lattice.
    Returns
    -------
    ks : np.array (2x(`sym` + 1))
    """
    W = rotation_matrix(np.deg2rad(theta))
    V = rotation_matrix(np.deg2rad(psi))
    D = scaling_matrix(1 / kappa)
    ks = np.stack([rotate(np.array([r_k, 0]), 2*np.pi/sym*i) for i in range(sym)]
                  + [(0, 0)])
    ks = apply_transformation_matrix(ks,  V.T @ D @ V @ W)
    return ks


def combine_ks(kvecs, order=1, return_counts=False):
    """Generate all possible different sums of kvecs upto order.

    Parameters
    ----------
    kvecs : array-like 2xN
        k vectors to combine.
    order: int, default: 1
        Number of different `kvecs` to combine for each resulting `tks`.
    return_counts : bool, default False
        if True, also return the number of possible combinations
        for each different combination.

    Returns
    -------
    tks : (2xM) array of float
        All possible unique combinations of `kvecs` upto `order`
    counts : (1xM) array of int, optional
        Number of combinations for each vectors in tks.
    """
    # Yes, this is probably not the smartest way to do this
    tks = np.array([np.sum(k_prod, axis=0) for k_prod in itert.product(*[kvecs]*order)])
    return np.unique(tks, return_counts=return_counts, axis=0)


def hexlattice_gen_fast(r_k, theta, order, size=500,
                        kappa=1., psi=0., shift=np.array((0, 0))):
    """Generate a regular hexagonal lattice.

    Speed optimized version, losing some mathematical precision.
    Tested to be accurate down to max(1e-10*r_k, 1e-10) w.r.t. the regular function.
    The lattice is generated from the six 60 degree rotated k-vectors
    of length r_k, further rotated by `theta` degrees.
    Size either an int for a size*size lattice, or tuple (N,M) for a rectangle N*M.
    Optionally, the lattice can be strained by a factor kappa in direction psi[1].

    With higher order frequency components upto order `order`

    The generated lattice gets returned as a dask array.

    See Also
    --------
    anylattice_gen
    """
    if not isinstance(size, tuple):
        size = (size, size)
    xx = da.arange(-1*size[0]/2, size[0]/2)[:, None]
    yy = da.arange(-1*size[1]/2, size[1]/2)[None]
    ks = generate_ks(r_k, theta, kappa, psi, sym=6)
    rks, k_c = combine_ks(ks, order=order, return_counts=True)
    rks = da.from_array(rks, chunks=(32, 2))
    phases = (xx + shift[0])[..., None]*rks[:, 0], (yy + shift[1])[..., None]*rks[:, 1]
    iterated = k_c * (np.exp(np.pi*2*1j * phases[0]) * np.exp(np.pi*2*1j * phases[1]))
    iterated = iterated.real.sum(axis=-1)
    # Now add the second shifted sublattice lattice to get a hexagonal lattice
    x = np.array([ks[1], -ks[2]])
    shift2 = (shift.T + (np.linalg.inv(x).T/3).sum(axis=0)).T  # Don't ask, this works
    phases2 = (xx + shift2[0])[..., None]*rks[:, 0], (yy + shift2[1])[..., None]*rks[:, 1]
    iterated += (k_c*np.exp(np.pi*2*1j * phases2[0])*np.exp(np.pi*2*1j * phases2[1])).real.sum(axis=-1)
    return iterated


def hexlattice_gen(r_k, theta, order, size=500,
                   kappa=1., psi=0., shift=np.array((0, 0)), **kwargs):
    """Generate a regular hexagonal lattice.
    The lattice is generated from the six 60 degree rotated k-vectors
    of length r_k, further rotated by `theta` degrees.
    Optionally, the lattice can be strained by a factor kappa in direction psi [1].
    Rendered with higher order frequency components upto order `order`.

    Parameters
    ----------
    r_k : float
        length of lattice vectors in k-space. Larger `r_k` correspond
        to smaller real space lattice constants. 1/r_k is the line
        spacing in pixels in the resulting image.
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
    psi : float, default: 0
        Principal strain direction with respect to horizontal
        in degrees.
    shift : iterable or array, optional
        shift of the lattice in pixels. Either a pair (x,y) global shift,
        or an (2xNxM) array where (NxM) corresponds to `size`.
    **kwargs : dict
        Keyword arguments to be passed to `anylattice_gen`

    Returns
    -------
    lattice : Dask array
        The generated lattice.

    See Also
    --------
    anylattice_gen
    """
    sublattice_a = anylattice_gen(r_k, theta, order, symmetry=6, size=size,
                                  kappa=kappa, psi=psi, shift=shift, **kwargs)
    # Now add the second shifted sublattice lattice to get a hexagonal lattice
    ks = generate_ks(r_k, theta, kappa, psi, sym=6)
    x = np.array([ks[1], -ks[2]])
    if r_k < 1e-10:
        shift2 = (shift.T + (np.linalg.inv(x).T/3).sum(axis=0)).T  # Don't ask, this works
    else:
        shift2 = (shift.T + (np.linalg.inv(x / r_k).T/(3*r_k)).sum(axis=0)).T
    sublattice_b = anylattice_gen(r_k, theta, order, symmetry=6, size=size,
                                  kappa=kappa, psi=psi, shift=shift2, **kwargs)
    return sublattice_a + sublattice_b


def squarelattice_gen(r_k, theta, order, size=500,
                      kappa=1., psi=0., shift=np.array((0, 0)), **kwargs):
    """Generate a regular square lattice.

    The lattice is generated from the four 90 degree rotated k-vectors
    of length `r_k`, further rotated by `theta` degrees.
    Optionally, the lattice can be strained by a factor `kappa` in direction `psi`[1].
    Rendered with higher order frequency components upto order `order`.

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
    psi : float, default: 0
        Principal strain direction with respect to horizontal.
    shift : iterable or array, optional
        shift of the lattice in pixels. Either a pair (x,y) global shift,
        or an (2xNxM) array where (NxM) corresponds to `size`.
    **kwargs : dict
        Keyword arguments to be passed to `anylattice_gen`

    Returns
    -------
    lattice : Dask array
        The generated lattice.

    See Also
    --------
    anylattice_gen
    """
    return anylattice_gen(r_k, theta, order,
                          symmetry=4, size=size,
                          kappa=kappa, psi=psi,
                          shift=shift, **kwargs)


def trilattice_gen(r_k, theta, order, size=500,
                   kappa=1., psi=0., shift=np.array((0, 0)), **kwargs):
    """Generate a regular trigonal lattice.

    The lattice is generated from the six 60 degree rotated k-vectors
    of length `r_k`, further rotated by `theta` degrees.
    Optionally, the lattice can be strained by a factor `kappa` in direction `psi`[1].
    Rendered with higher order frequency components upto order `order`.

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
    psi : float, default: 0
        Principal strain direction with respect to horizontal.
    shift : iterable or array, optional
        shift of the lattice in pixels. Either a pair (x,y) global shift,
        or an (2xNxM) array where (NxM) corresponds to `size`.
    **kwargs : dict
        Keyword arguments to be passed to `anylattice_gen`

    Returns
    -------
    lattice : Dask array
        The generated lattice.

    See Also
    --------
    anylattice_gen, hexlattice_gen
    """
    return anylattice_gen(r_k, theta, order,
                          symmetry=6, size=size,
                          kappa=kappa, psi=psi,
                          shift=shift, **kwargs)


def anylattice_gen(r_k, theta, order, symmetry=6, size=500,
                   kappa=1., psi=0., shift=np.array([0, 0]),
                   normalize=False, chunks=(-1, -1)):
    """Generate a regular lattice of any symmetry.
    The lattice is generated from the `symmetry` `360/symmetry` degree rotated k-vectors
    of length `r_k`, further rotated by `theta` degrees.
    Size either an int for a size*size lattice, or tuple (N,M) for a rectangle N*M.
    Optionally, the lattice can be strained by a factor kappa in direction psi[1].

    With higher order frequency components upto order `order`

    Parameters
    ----------
    r_k : float
        length of lattice vectors in k-space. Larger `r_k` correspond
        to smaller real space lattice constants. 1/r_k is the line
        spacing in pixels in the resulting image.
    theta : float
        Angle of the first lattice vector with respect to positive
        horizontal.
    order : int
        Order upto which to generate higher frequency components
        by combining lattice vectors
    symmetry : int
        symmetry of the lattice.
    size: int, or pair of int, default: 500
        Size of the resulting lattice in pixels. if int, the
        returned lattice will be square.
    kappa : float, default: 1
        strain/deformation magnitude. 1 corresponds to no strain.
    psi : float, default: 0
        Principal strain direction with respect to horizontal.
    shift : iterable or array, optional
        shift of the lattice in pixels. Either a pair (x,y) global shift,
        or an (2xNxM) array where (NxM) corresponds to `size`.
    normalize : bool, default: False
        if true, normalize the output values to the interval [0,1].
    chunks : int or pair of int, optional
        dask chunks in which to divide the returned `lattice`.

    Returns
    -------
    lattice : Dask array
        The generated lattice.

    See Also
    --------
    generate_ks
    physical_lattice_gen

    References
    ----------
    [1] T. Benschop et al., 2020, https://doi.org/10.1103/PhysRevResearch.3.013153
    """
    if not isinstance(size, tuple):
        size = (size, size)
    if not isinstance(chunks, tuple):
        chunks = (chunks, chunks)
    xx = da.arange(-1*size[0]/2, size[0]/2, chunks=chunks[0])[:, None]
    yy = da.arange(-1*size[1]/2, size[1]/2, chunks=chunks[1])[None]
    ks = generate_ks(r_k, theta, kappa, psi, sym=symmetry)
    rks, k_c = combine_ks(ks, order=order, return_counts=True)
    rks = da.from_array(rks, chunks=(13, 2))
    phases = (xx + shift[0])*rks[:, 0, None, None] + (yy + shift[1])*rks[:, 1, None, None]
    iterated = k_c[:, None, None] * np.exp(np.pi*2*1j * phases)
    iterated = iterated.sum(axis=0)
    iterated = iterated.real
    if normalize:
        iterated -= iterated.min()
        iterated = iterated / iterated.max()
        # Prevent pathetic max = min case.
        iterated = np.nan_to_num(iterated)
    return iterated


def anylattice_gen_np(r_k, theta, order=1, symmetry=6, size=50,
                      kappa=1., psi=0., shift=np.array((0, 0))):
    """Generate a regular lattice of any symmetry in pure numpy.

    The lattice is generated from the `symmetry` `360/symmetry` degree rotated k-vectors
    of length r_k, further rotated by `theta` degrees.
    Size either an int for a size*size lattice, or tuple (N,M) for a rectangle N*M.
    Optionally, the lattice can be strained by a factor kappa in direction psi[1].

    With higher order frequency components upto order `order`

    The generated lattice gets returned as a numpy array.
    Only usable for small values of size and order.

    See Also
    --------
    anylattice_gen
    """
    if not isinstance(size, tuple):
        size = (size, size)
    xx = np.arange(-1*size[0]/2, size[0]/2)[:, None]
    yy = np.arange(-1*size[1]/2, size[1]/2)[None]
    ks = generate_ks(r_k, theta, kappa, psi, sym=symmetry)
    rks, k_c = combine_ks(ks, order=order, return_counts=True)
    if not np.isfinite(2*np.pi * r_k * (max(size) + max(shift)) / 2):
        print("Warning, using more than float-max periods in a single lattice.")
    phases = (xx + shift[0])*rks[:, 0, None, None] + (yy + shift[1])*rks[:, 1, None, None]
    iterated = k_c[:, None, None]*np.exp(np.pi*2*1j * phases)
    iterated = iterated.sum(axis=0)
    return iterated.real


def physical_lattice_gen(a_0, theta, order, pixelspernm=10, symmetry='hexagonal',
                         size=500, epsilon=None, delta=0.16, **kwargs):
    """Generate a physical lattice

    Wraps anylattice_gen.
    Using a lattice constant `a_0` in nm and a resolution in pixels per nm,
    generate a rendering of a lattice of `size` pixels.
    Optionally, the lattice can be strained by a factor kappa in direction psi[1].

    With higher order frequency components upto order `order`

    Parameters
    ----------
    a_0 : float
        lattice constant in nm
    theta : float
        Angle of the first lattice vector with respect to positive
        horizontal.
    order : int
        Order upto which to generate higher frequency components
        by combining lattice vectors
    pixelspernm : float, default: 10
        number of pixels per nanometer.
    symmetry : {'hexagonal', 'trigonal', 'square'}
        symmetry of the lattice.
    size: int, or pair of int, default: 500
        Size of the resulting lattice in pixels. if int, the
        returned lattice will be square.
    epsilon : float
        Lattice strain
    delta : float, default=0.16
        Poisson ratio for lattice strain.
        Default value corresponds to graphene.
    **kwargs : dict
        Keyword arguments to be passed to `anylattice_gen`

    Returns
    -------
    lattice : Dask array
        The generated lattice.

    See Also
    --------
    anylattice_gen

    """
    if symmetry not in ['square', 'hexagonal', 'trigonal']:
        raise Exception("Symmetry {} is unknown".format(symmetry))
    if symmetry == 'square':
        sym = 4
    else:
        sym = 6
    r_k = a_0_to_r_k(a_0 * pixelspernm, sym)
    if epsilon is not None:
        if 'kappa' in kwargs.keys():
            warnings.warn("Both kappa and epsilon specified, ignoring kappa value")
        r_k, kappa = epsilon_to_kappa(r_k, epsilon, delta)
        kwargs['kappa'] = kappa
    if symmetry == 'hexagonal':
        return hexlattice_gen(r_k, theta, order, size=size,
                              **kwargs)
    else:
        return anylattice_gen(r_k, theta, order,
                              symmetry=sym, size=size, **kwargs)

import numpy as np
import dask.array as da
import itertools as itert

from latticegen.transformations import rotate, rotation_matrix, scaling_matrix, apply_transformation_matrix


def generate_ks(r_k, theta, kappa=1., psi=0., sym=6):
    """Generate k-vectors from given parameters."""
    W = rotation_matrix(np.deg2rad(theta))
    V = rotation_matrix(np.deg2rad(psi))
    D = scaling_matrix(kappa)
    ks = np.stack([rotate(np.array([r_k, 0]), 2*np.pi/sym*i) for i in range(sym)]
                  + [(0, 0)])
    ks = apply_transformation_matrix(ks, W @ V.T @ D @ V)
    return ks


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

    [1] T. Benschop et al., 2020, https://arxiv.org/abs/2008.13766
    """
    if not isinstance(size, tuple):
        size = (size, size)
    xx = da.arange(-1*size[0]/2, size[0]/2)[:, None]
    yy = da.arange(-1*size[1]/2, size[1]/2)[None]
    ks = generate_ks(r_k, theta, kappa, psi, sym=6)
    # Yes, this is probably not the smartest way to do this
    tks = np.array([np.sum(ksp, axis=0) for ksp in itert.product(*[ks]*order)])
    rks, k_c = np.unique(tks, return_counts=True, axis=0)
    rks = da.from_array(rks, chunks=(32, 2))
    phases = (xx + shift[0])[..., None]*rks[:, 0], (yy + shift[1])[..., None]*rks[:, 1]
    iterated = k_c * (np.exp(np.pi*2*1j * phases[0]) * np.exp(np.pi*2*1j * phases[1]))
    iterated = iterated.real.sum(axis=-1)
    # Now add the second shifted sublattice lattice to get a hexagonal lattice
    x = np.array([ks[1], -ks[2]])
    shift2 = (shift.T + (np.linalg.inv(x).T/3).sum(axis=0)).T # Don't ask, this works
    phases2 = (xx + shift2[0])[..., None]*rks[:, 0], (yy + shift2[1])[..., None]*rks[:, 1]
    iterated += (k_c*np.exp(np.pi*2*1j * phases2[0])*np.exp(np.pi*2*1j * phases2[1])).real.sum(axis=-1)
    return iterated

def hexlattice_gen(r_k, theta, order, size=500,
                   kappa=1., psi=0., shift=np.array((0, 0))):
    """Generate a regular hexagonal lattice.
    The lattice is generated from the six 60 degree rotated k-vectors
    of length r_k, further rotated by `theta` degrees.
    Size either an int for a size*size lattice, or tuple (N,M) for a rectangle N*M.
    Optionally, the lattice can be strained by a factor kappa in direction psi[1].

    With higher order frequency components upto order `order`

    The generated lattice gets returned as a dask array.

    [1] T. Benschop et al., 2020, https://arxiv.org/abs/2008.13766
    """
    sublattice_a = anylattice_gen(r_k, theta, order, symmetry=6, size=size,
                   kappa=kappa, psi=psi, shift=shift)
    # Now add the second shifted sublattice lattice to get a hexagonal lattice
    ks = generate_ks(r_k, theta, kappa, psi, sym=6)
    x = np.array([ks[1], -ks[2]])
    if r_k < 1e-10:
        shift2 = (shift.T + (np.linalg.inv(x).T/3).sum(axis=0)).T # Don't ask, this works
    else:
        shift2 = (shift.T + (np.linalg.inv(x / r_k).T/(3*r_k)).sum(axis=0)).T
    sublattice_b = anylattice_gen(r_k, theta, order, symmetry=6, size=size,
                   kappa=kappa, psi=psi, shift=shift2)
    return sublattice_a + sublattice_b


def squarelattice_gen(r_k, theta, order, size=500,
                      kappa=1., psi=0., shift=np.array((0, 0))):
    """Generate a regular square lattice.
    The lattice is generated from the six 90 degree rotated k-vectors
    of length r_k, further rotated by `theta` degrees.
    Size either an int for a size*size lattice, or tuple (N,M) for a rectangle N*M.
    Optionally, the lattice can be strained by a factor kappa in direction psi[1].

    With higher order frequency components upto order `order`

    The generated lattice gets returned as a dask array.
    """
    return anylattice_gen(r_k, theta, order,
                          symmetry=4, size=size,
                          kappa=kappa, psi=psi,
                          shift=shift)


def trilattice_gen(r_k, theta, order, size=500,
                   kappa=1., psi=0., shift=np.array((0, 0))):
    """Generate a regular trigonal lattice.
    The lattice is generated from the six 60 degree rotated k-vectors
    of length r_k, further rotated by `theta` degrees.
    Size either an int for a size*size lattice, or tuple (N,M) for a rectangle N*M.
    Optionally, the lattice can be strained by a factor kappa in direction psi[1].

    With higher order frequency components upto order `order`

    The generated lattice gets returned as a dask array.

    [1] T. Benschop et al., 2020, https://arxiv.org/abs/2008.13766
    """
    return anylattice_gen(r_k, theta, order,
                          symmetry=6, size=size,
                          kappa=kappa, psi=psi,
                          shift=shift)


def anylattice_gen(r_k, theta, order, symmetry=6, size=500,
                   kappa=1., psi=0., shift=np.array((0, 0))):
    """Generate a regular lattice of any symmetry.
    The lattice is generated from the `symmetry` `360/symmetry` degree rotated k-vectors
    of length r_k, further rotated by `theta` degrees.
    Size either an int for a size*size lattice, or tuple (N,M) for a rectangle N*M.
    Optionally, the lattice can be strained by a factor kappa in direction psi[1].

    With higher order frequency components upto order `order`

    The generated lattice gets returned as a dask array.

    [1] T. Benschop et al., 2020, https://arxiv.org/abs/2008.13766
    """
    if not isinstance(size, tuple):
        size = (size, size)
    xx = da.arange(-1*size[0]/2, size[0]/2)[:, None]
    yy = da.arange(-1*size[1]/2, size[1]/2)[None]
    ks = generate_ks(r_k, theta, kappa, psi, sym=symmetry)
    # Yes, this is probably not the smartest way to do this
    tks = np.array([np.sum(ksp, axis=0) for ksp in itert.product(*[ks]*order)])
    rks, k_c = np.unique(tks, return_counts=True, axis=0)
    rks = da.from_array(rks, chunks=(13, 2))
    phases = (xx + shift[0])*rks[:, 0, None, None] + (yy + shift[1])*rks[:, 1, None, None]
    iterated = k_c[:, None, None]*np.exp(np.pi*2*1j * phases)
    iterated = iterated.sum(axis=0)
    return iterated.real

def anylattice_gen_np(r_k, theta, order=1, symmetry=6, size=50,
                   kappa=1., psi=0., shift=np.array((0, 0))):
    """Generate a regular lattice of any symmetry.
    The lattice is generated from the `symmetry` `360/symmetry` degree rotated k-vectors
    of length r_k, further rotated by `theta` degrees.
    Size either an int for a size*size lattice, or tuple (N,M) for a rectangle N*M.
    Optionally, the lattice can be strained by a factor kappa in direction psi[1].

    With higher order frequency components upto order `order`

    The generated lattice gets returned as a numpy array.
    Only usable for small values of size and order.

    [1] T. Benschop et al., 2020, https://arxiv.org/abs/2008.13766
    """
    if not isinstance(size, tuple):
        size = (size, size)
    xx = np.arange(-1*size[0]/2, size[0]/2)[:, None]
    yy = np.arange(-1*size[1]/2, size[1]/2)[None]
    ks = generate_ks(r_k, theta, kappa, psi, sym=symmetry)
    # Yes, this is probably not the smartest way to do this
    tks = np.array([np.sum(ksp, axis=0) for ksp in itert.product(*[ks]*order)])
    rks, k_c = np.unique(tks, return_counts=True, axis=0)
    if not np.isfinite(2*np.pi*r_k*(max(size)+max(shift))/2):
        print("Warning, using more than float-max periods in a single lattice.")
    phases = (xx + shift[0])*rks[:, 0, None, None] + (yy + shift[1])*rks[:, 1, None, None]
    iterated = k_c[:, None, None]*np.exp(np.pi*2*1j * phases)
    iterated = iterated.sum(axis=0)
    return iterated.real
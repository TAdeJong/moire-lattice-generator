import numpy as np
import dask.array as da
import itertools as itert

from transformations import rotate, rotation_matrix, scaling_matrix, apply_transformation_matrix

def hexlattice_gen(r_k, theta, order, size=250, kappa=1., psi=0., shift=np.array((0,0))):
    """Generate a regular hexagonal lattice of `2*size` times `2*size`.
    The lattice is generated from the six 60 degree rotated k-vectors 
    of length r_k, further rotated by `theta` degrees. 
    Optionally, the lattice can be strained by a factor kappa in direction psi.[1]
    
    With higher order frequency
    components upto order `order`
    
    The generated lattice gets returned as a dask array.
    
    [1] T. Benschop et al., 2020, https://arxiv.org/abs/2008.13766
    """
    xx, yy = da.meshgrid(np.arange(-size,size), np.arange(-size,size), indexing='ij')
    ks = np.stack([rotate(np.array([r_k,0]), np.pi/3*i) for i in range(6)]+[(0,0)])
    W = rotation_matrix(np.deg2rad(theta))
    V = rotation_matrix(np.deg2rad(psi))
    D = scaling_matrix(kappa)
    ks = apply_transformation_matrix(ks, W @ V.T @ D @ V)
    # Yes, this is probably not the smartest way to do this
    tks = np.array([np.sum(ksp, axis=0) for ksp in itert.product(*[ks]*order)])
    rks, k_c = np.unique(tks, return_counts=True, axis=0)
    rks = da.from_array(rks, chunks=(13,2))
    phases = (xx + shift[0])*rks[:,0,None,None] + (yy + shift[1])*rks[:,1,None,None]
    iterated = k_c[:,None,None]*np.exp(np.pi*2*1j * phases)
    iterated = iterated.sum(axis=0)
    # Now add the second shifted sublattice lattice to get a hexagonal lattice
    x = np.array([ks[1], -ks[2]])
    shift2 = shift + (np.linalg.inv(x).T/3).sum(axis=0)  # Don't ask, this works
    phases2 = ((xx + shift2[0])*rks[:,0,None,None] + (yy + shift2[1])*rks[:,1,None,None])
    iterated += (k_c[:,None,None]*np.exp(np.pi*2*1j * phases2)).sum(axis=0)
    return iterated.real


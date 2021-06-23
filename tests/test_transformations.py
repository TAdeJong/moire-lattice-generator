# Test with pytest
import numpy as np
import pytest
from hypothesis import given
import hypothesis.strategies as st

from latticegen.transformations import *


@pytest.mark.parametrize("A", [
    np.random.random((5, 5))*10,
    np.full(4, np.pi),
    np.arange(-10, 10, 0.1),
    ])
def test_wrapToPi(A):
    res = wrapToPi(A)
    assert res.shape == A.shape
    assert np.all(res <= np.pi)
    assert np.all(res >= -np.pi)


def test_identities():
    assert np.allclose(rotation_matrix(0.0), np.eye(2))
    assert np.allclose(rotation_matrix(2*np.pi), np.eye(2))
    assert np.allclose(rotation_matrix(np.pi), -1*np.eye(2))


@given(st.floats(0., exclude_min=True, allow_infinity=False))
def test_scaling_matrix(kappa):
    res = scaling_matrix(kappa)
    assert res[0, 0] == kappa
    assert res.shape == (2, 2)
    res_restore = res.copy()
    res_restore[0, 0] = 1.
    assert np.allclose(res_restore, np.eye(2))

    
def test_outputshapes():
    A = np.random.random((2, 2))
    vecs = np.random.random((3, 2))
    res = apply_transformation_matrix(vecs, A)
    assert res.shape == vecs.shape
    for a in A.flatten():
        assert rotate(vecs, a).shape == vecs.shape


@given(a_0=st.floats(0., 1e300, exclude_min=True, allow_infinity=False),
      sym=st.integers(4, 7))
def test_r_k_to_a_0_identity(a_0, sym):
    assert np.isclose(r_k_to_a_0(a_0_to_r_k(a_0, sym), sym), a_0)

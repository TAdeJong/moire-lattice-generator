from hypothesis import given, assume, settings
import hypothesis.strategies as st
import pytest
import numpy as np

from latticegen.latticegeneration import *


@given(st.floats(0., exclude_min=True, allow_infinity=False),
       st.floats(0, np.pi),
       st.floats(1e-200, 10, exclude_min=True),
       st.floats(0, np.pi),
       st.integers(4, 7),
       )
def test_generate_ks(r, t, k, p, sym):
    assume(np.isfinite(r*np.pi*2))
    assume(r / (np.pi*2) > 0.0)
    ks = generate_ks(r, t, k, p, sym)
    assert ks.shape == (sym + 1, 2)
    assert ks.max() <= max(r / k, r)


@given(r=st.floats(1e-20, 1e8, exclude_min=True, allow_infinity=False),
       t=st.floats(0, np.pi),
       o=st.integers(1, 3),
       k=st.floats(0.001, 10, exclude_min=True),
       p=st.floats(0, np.pi),
       size=st.one_of([st.integers(2, 50),
                       st.tuples(st.integers(2, 50), st.integers(2, 50))]),
       )
def test_fast_gen(r, t, o, k, p, size):
    # Don't use more than float max periods.
    assume(np.isfinite(r*np.max(size)*np.pi*2))
    ref = hexlattice_gen(r, t, o, size, k, p)
    fast = hexlattice_gen_fast(r, t, o, size, k, p)
    assert fast.shape == ref.shape
    ref = ref.compute()
    fast = fast.compute()
    atol = max(4e-10*r, 1e-10)
    assert np.abs(ref - fast).max() < atol
    assert np.allclose(ref, fast, rtol=1e-5, atol=atol)


# @given(r=st.floats(1e-250, 1e250, allow_infinity=False),
#        t=st.floats(0, np.pi),
#        o=st.integers(1, 3),
#        k=st.floats(1e-10, 1e10, exclude_min=True),
#        p=st.floats(0, np.pi),
#        size=st.tuples(st.integers(2, 50), st.integers(2, 50)),
#        )
# def test_new_hex_gen(r, t, o, k, p, size):
#     # Don't use more than float max periods.
#     assume(np.isfinite(r*np.max(size)*np.pi*2))
#     ref = hexlattice_gen(r, t, o, size, k, p)
#     new = hexlattice_gen2(r, t, o, size, k, p)
#     assert new.shape == ref.shape
#     ref = ref.compute()
#     new = new.compute()
#     assert np.allclose(ref, new)


@given(r=st.floats(0.0, exclude_min=True, allow_infinity=False),
       t=st.floats(0, np.pi),
       o=st.integers(1, 3),
       k=st.floats(1e-6, 10, exclude_min=True),
       p=st.floats(0, np.pi),
       size=st.tuples(st.integers(2, 70),
                      st.integers(2, 70)),
       sym=st.integers(4, 7),
       norm=st.booleans()
       )
@pytest.mark.filterwarnings("ignore:invalid value")
def test_gen(r, t, o, sym, k, p, size, norm):
    # Don't use more than float max periods.
    assume(np.isfinite(r*max(size)*np.pi*2))
    ref = anylattice_gen(r, t, o, sym, size, k, p, normalize=norm)
    assert ref.shape == size
    ref = ref.compute()
    assert np.all(~np.isnan(ref))
    if norm:
        assert np.all(ref <= 1.0)
        assert np.all(ref >= 0.0)


@given(st.integers(3, 500))
def test_shapes_square(s):
    assert hexlattice_gen(0.1, 0., 1, s).shape == (s, s)
    assert anylattice_gen(0.1, 0., 1, 4, s).shape == (s, s)

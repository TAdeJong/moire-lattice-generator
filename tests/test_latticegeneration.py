from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from latticegen.latticegeneration import *


@given(st.floats(0., exclude_min=True, allow_infinity=False), 
       st.floats(0, np.pi), 
       st.floats(0., 10, exclude_min=True),
       st.floats(0, np.pi), 
       st.sampled_from([4,6]),
       )
def test_generate_ks(r, t, k, p, sym):
    ks = generate_ks(r, t, k, p, sym)
    assert ks.shape == (sym + 1,2)
    assert ks.max() <= max(r * k, r)


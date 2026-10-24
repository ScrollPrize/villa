"""Property test: chunked edt_dilate equals one whole-volume scipy oracle, for both EDT backends."""
import itertools

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

scipy_ndimage = pytest.importorskip("scipy.ndimage")

from vesuvius.image_proc.run.zarr_tasks.tasks import edt_dilate as m

from .test_edt_dilate_halo import _oracle as _halo_oracle
from .test_edt_dilate_halo import _run_chunked

BACKENDS = ["scipy"] + (["edt"] if m.HAS_EDT else [])


@pytest.fixture(params=BACKENDS)
def backend(request, monkeypatch):
    if request.param == "scipy":
        monkeypatch.setattr(m, "HAS_EDT", False)
        monkeypatch.setattr(m, "scipy_edt", scipy_ndimage.distance_transform_edt, raising=False)
    return request.param


def _oracle(vol, dist, black_border, backend):
    # One backend-independent reference for both backends.
    return _halo_oracle(vol, dist, black_border)


@st.composite
def cases(draw):
    shape = tuple(draw(st.integers(1, 12)) for _ in range(3))
    ch = tuple(draw(st.integers(1, s)) for s in shape)
    dist = draw(st.one_of(st.integers(0, 6).map(float), st.floats(0, 6, allow_nan=False)))
    n = draw(st.integers(0, 4))
    seeds = draw(st.lists(st.tuples(*(st.integers(0, s - 1) for s in shape)), min_size=n, max_size=n))
    vol = np.zeros(shape, np.uint8)
    for s in seeds:
        vol[s] = draw(st.integers(1, 255))
    return vol, ch, dist, draw(st.booleans())


@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(case=cases())
def test_chunked_equals_whole_volume(tmp_path_factory, backend, case):
    vol, ch, dist, bb = case
    tmp = tmp_path_factory.mktemp("p")
    got = _run_chunked(tmp, vol, ch, dist, bb)
    np.testing.assert_array_equal(got, _oracle(vol, dist, bb, backend))


@pytest.mark.parametrize("black_border", [False, True])
def test_corner_where_eight_chunks_meet(tmp_path, backend, black_border):
    shape, ch, dist = (16, 16, 16), (8, 8, 8), 3.0
    vol = np.zeros(shape, np.uint8)
    vol[7, 7, 7] = 1  # last voxel of chunk (0,0,0); corner shared by 8 chunks
    got = _run_chunked(tmp_path, vol, ch, dist, black_border)
    for c in itertools.product((7, 8), repeat=3):
        assert got[c], f"chunk containing {c} not reached"
    assert not got[9, 9, 9]  # distance sqrt(12) > 3
    np.testing.assert_array_equal(got, _oracle(vol, dist, black_border, backend))

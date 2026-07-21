"""Regression tests for label expansion in ``vesuvius.scripts.labels_to_zarr``.

A chunked EDT label expansion must produce exactly the same result as a single
global (unchunked) EDT expansion, including for labels that sit on or span chunk
boundaries. Two bugs made this fail:

* with the default ``halo=0`` the expansion could not cross chunk boundaries
  (seams / under-expansion), and
* with a non-zero halo the phased passes read each other's freshly-written
  labels, so expansions cascaded and over-grew near chunk boundaries.

These tests pin the corrected behaviour by comparing against a global EDT.
"""
import numpy as np
import pytest
import zarr
from scipy.ndimage import distance_transform_edt

from vesuvius.scripts.labels_to_zarr import expand_labels_in_zarr


def _ground_truth(shape, seeds, dist, label=1):
    a = np.zeros(shape, np.uint8)
    for s in seeds:
        a[s] = label
    edt = distance_transform_edt(~(a == label))
    a[(edt > 0) & (edt <= dist) & (a == 0)] = label
    return a


def _chunked(path, shape, chunk, seeds, dist, workers=1, label=1):
    a = zarr.open(str(path), mode="w", shape=shape, chunks=(chunk, chunk, chunk),
                  dtype="uint8", zarr_format=2, dimension_separator=".")
    a[:] = 0
    for s in seeds:
        a[s] = label
    expand_labels_in_zarr(str(path), dist, label, fill_value=0,
                          num_workers=workers, halo=0)
    return zarr.open(str(path), mode="r")[:]


@pytest.mark.parametrize("seeds", [
    [(16, 16, 16)],                              # interior of one chunk
    [(31, 16, 16)],                              # on a 2-chunk face boundary
    [(31, 31, 31)],                              # on the 8-chunk corner
    [(31, 31, 31), (31, 16, 45), (45, 45, 16)],  # several boundary-spanning labels
])
def test_expansion_matches_global_edt(tmp_path, seeds):
    shape, chunk, dist = (64, 64, 64), 32, 5
    expected = _ground_truth(shape, seeds, dist)
    got = _chunked(tmp_path / "a.zarr", shape, chunk, seeds, dist)
    assert np.array_equal(got, expected)


def test_expansion_larger_than_chunk_is_race_free_and_correct(tmp_path):
    # expansion distance (40) exceeds the chunk size (32): the phase stride must
    # widen so parallel processing stays race-free and still matches the truth.
    shape, chunk, dist = (64, 64, 64), 32, 40
    seeds = [(z, y, x) for z in (10, 45) for y in (10, 45) for x in (10, 45)]
    expected = _ground_truth(shape, seeds, dist)
    serial = _chunked(tmp_path / "s.zarr", shape, chunk, seeds, dist, workers=1)
    parallel = _chunked(tmp_path / "p.zarr", shape, chunk, seeds, dist, workers=4)
    assert np.array_equal(serial, expected)
    assert np.array_equal(parallel, serial)

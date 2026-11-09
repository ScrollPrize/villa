"""A tissue map cut from a larger one is the map read on its own."""

import numpy as np
import zarr

from volreg.cores import dense_correlation as dense


def levels():
    rng = np.random.default_rng(0)
    low = rng.integers(0, 3, (24, 12, 14), dtype=np.uint8)      # zeros are background
    level = zarr.array(low, chunks=(5, 5, 5))
    return [(zarr.zeros((24 * 32, 12 * 32, 14 * 32), chunks=(128, 128, 128), dtype="u1"), 1.0),
            (level, 32.0)]


def test_a_cut_is_the_map_read_on_its_own():
    pyramid = levels()
    within = dense.tissue_map(pyramid, np.array([64.0, 40.0, 30.0]), np.array([700.0, 330.0, 420.0]))
    rng = np.random.default_rng(1)
    cut_some = 0
    for _ in range(200):
        a = rng.uniform(0, [760, 380, 440])
        b = a + rng.uniform(1, [400, 200, 200])
        read = dense.tissue_map(pyramid, a, b)
        cut = dense.tissue_map(pyramid, a, b, within=within)
        if read is None:
            assert cut is None
            continue
        assert np.array_equal(read.data, cut.data) and np.array_equal(read.low, cut.low)
        cut_some += cut.data.base is not None
        for origin in rng.uniform(a, b, (5, 3)):
            assert read.holds_tissue(origin) == cut.holds_tissue(origin)
    assert cut_some > 20          # the cut path was taken, not only the read one

"""vc_tomography.mosaic.tile_reconstructor: the checks of its arguments made before any frame is read."""
import numpy as np
import pytest

from vc_tomography import mosaic as M


@pytest.mark.parametrize("reduce", [-1, 4, 32])
def test_tile_rejects_invalid_reduction(reduce):
    with pytest.raises(ValueError, match="reduce is 0, 1, 2 or 3"):
        M.tile_reconstructor("unused", {}, "T0", reduce=reduce, device="cpu")


@pytest.mark.parametrize("dark_shape, flat_shape, error", [
    ((2, 3), (2, 2, 3), "detector dimensions"),
    ((4, 6), (1, 4, 6), "flat series count"),
    ((4, 6), (2, 6, 4), "flat dimensions"),
])
def test_tile_checks_supplied_calibration(tmp_path, dark_shape, flat_shape, error):
    from test_scan import _write_scan

    master = _write_scan(tmp_path, "sample_HA")
    reduced = np.ones(dark_shape, np.float32), np.ones(flat_shape, np.float32)
    with pytest.raises(ValueError, match=error):
        M.tile_reconstructor(master, {"layout": {}}, "T0", reduced=reduced, device="cpu")

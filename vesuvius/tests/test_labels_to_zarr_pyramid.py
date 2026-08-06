"""Regression test for the OME-Zarr output-creation path in ``labels_to_zarr``.

The output arrays -- level 0 in ``main()`` and every downsampled level in
``generate_pyramid_levels`` -- are created with the Zarr 3 array API
(``root.create_array(..., config={"write_empty_chunks": False})`` on a
``zarr_format=2`` group). The existing expand tests exercise
``expand_labels_in_zarr`` but never touch this creation path, so a regression in
the API contract (an unsupported kwarg, the wrong compressor argument, or a
zarr-format drift) would slip through CI.

This test drives ``generate_pyramid_levels`` end to end -- with level 0 created
exactly the way ``main()`` creates it -- and asserts the result is a valid
zarr-format-2 pyramid that round-trips.
"""
import numpy as np
import zarr
from numcodecs import Blosc

from vesuvius.scripts.labels_to_zarr import generate_pyramid_levels


def _make_level0(path, shape, chunks, compressor, dtype, fill_value=0):
    # Mirror labels_to_zarr.main(): a zarr_format=2 group whose level "0" array
    # is created via the Zarr 3 create_array API with the same kwargs.
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    arr = root.create_array(
        "0",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        fill_value=fill_value,
        config={"write_empty_chunks": False},
    )
    return arr


def test_generate_pyramid_levels_creates_valid_zarr_v2_pyramid(tmp_path):
    path = tmp_path / "labels.ome.zarr"
    shape, chunks = (16, 32, 32), (8, 16, 16)
    dtype = np.dtype("uint8")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    level0 = _make_level0(path, shape, chunks, compressor, dtype)
    # Two solid label blocks split on an even z boundary, so the stride-2
    # nearest-neighbour downsampling maps them cleanly with no boundary mixing.
    level0[:8] = 1
    level0[8:] = 2

    generate_pyramid_levels(str(path), num_levels=3, chunks=chunks,
                            compressor=compressor, dtype=dtype, num_workers=1)

    reopened = zarr.open_group(str(path), mode="r")

    # Every level must exist, open, keep zarr format 2, and halve each axis.
    prev = shape
    for level in range(3):
        arr = reopened[str(level)]
        assert int(arr.metadata.zarr_format) == 2, level
        expected = shape if level == 0 else tuple(max(1, s // 2) for s in prev)
        assert arr.shape == expected, (level, arr.shape, expected)
        prev = arr.shape

    # Nearest-neighbour (stride-2) downsampling preserves the two label blocks.
    lvl1 = reopened["1"][:]
    assert lvl1.shape == (8, 16, 16)
    assert np.all(lvl1[:4] == 1) and np.all(lvl1[4:] == 2)
    lvl2 = reopened["2"][:]
    assert np.all(lvl2[:2] == 1) and np.all(lvl2[2:] == 2)

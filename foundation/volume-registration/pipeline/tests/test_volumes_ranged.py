"""A box read by z-range is the box zarr reads, on a raw store with missing chunks."""

import numpy as np
import zarr

from volreg.cores import volumes


def raw_store(tmp_path, separator):
    store = zarr.storage.FSStore(f"file://{tmp_path}/v.zarr", key_separator=separator,
                                 dimension_separator=separator)
    array = zarr.create(shape=(70, 50, 45), chunks=(16, 16, 16), dtype="u1", compressor=None,
                        fill_value=7, store=store, write_empty_chunks=False)
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, array.shape, dtype=np.uint8)
    data[16:32, 16:32, :] = 7                     # chunks equal to the fill value: not written
    array[:] = data
    return zarr.open(store, mode="r"), data


def test_a_box_by_z_range_is_the_box_zarr_reads(tmp_path):
    for separator in ("/", "."):
        array, data = raw_store(tmp_path / {"/": "slash", ".": "dot"}[separator], separator)
        assert volumes.by_z_range(array)
        rng = np.random.default_rng(1)
        for _ in range(40):
            lo = rng.integers(0, np.array(array.shape) - 1)
            hi = lo + rng.integers(1, np.array(array.shape) - lo + 1)
            assert np.array_equal(volumes.read_z_ranges(array, lo, hi), data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        whole = volumes.read_patch(array, (0, 0, 0), array.shape)
        assert np.array_equal(whole, data)
        # Past the edge: through zarr, which clips as numpy does.
        assert volumes.read_patch(array, (60, 40, 40), (80, 60, 60)).shape == (10, 10, 5)


def test_without_fsstore_a_box_is_read_through_zarr(tmp_path, monkeypatch):
    # zarr 3 has no zarr.storage.FSStore: by_z_range must say no instead of raising.
    array, data = raw_store(tmp_path, "/")
    monkeypatch.delattr(zarr.storage, "FSStore")
    assert not volumes.by_z_range(array)
    assert np.array_equal(volumes.read_patch(array, (3, 5, 7), (40, 30, 20)), data[3:40, 5:30, 7:20])


def test_a_compressed_store_is_read_through_zarr(tmp_path):
    array = zarr.create(shape=(8, 8, 8), chunks=(4, 4, 4), dtype="u1",
                        store=zarr.storage.FSStore(f"file://{tmp_path}/c.zarr"))
    assert not volumes.by_z_range(array)

"""Optional local-volume scheduling regression (no GPU or downloads).

Set LASAGNA_TEST_VOLUME to a local Zarr-v2 array and LASAGNA_TEST_ORIGIN_ZYX
to a comma-separated voxel origin. An identity model isolates scheduling and
storage from model quality; this is not an end-to-end GPU speed benchmark.
"""
import itertools
import os
import time

import numpy as np
import pytest
import zarr

from lasagna import tiled_predict3d as tiled
from lasagna.demand_predict3d import DemandGeometry, DemandStore, DemandWorkerPool
from lasagna.tests.test_demand_predict3d import outputs, tensor_result


def test_local_volume_serial_worker_and_resume(tmp_path):
    path = os.environ.get("LASAGNA_TEST_VOLUME")
    if not path:
        pytest.skip("set LASAGNA_TEST_VOLUME and LASAGNA_TEST_ORIGIN_ZYX")
    origin = tuple(map(int, os.environ["LASAGNA_TEST_ORIGIN_ZYX"].split(",")))
    assert len(origin) == 3
    source = zarr.open_array(path, mode="r")
    data = np.asarray(source[tuple(slice(v, v + 32) for v in origin)])
    assert data.shape == (32, 32, 32) and data.dtype == np.uint8
    assert np.ptp(data) > 0, "test crop must contain actual varied CT data"
    geometry = DemandGeometry(data.shape, 16, 0, 0, 1, 16)
    indices = list(itertools.product(range(2), repeat=3))
    results = []
    identity = {"validation": "local CT crop"}

    def reader(tile):
        spec = tiled._tile_read_spec(data.shape, (0, 0, 0), *tile, 16, 0)
        return tiled._finalize_tile_read(data[spec.slices_zyx], spec)

    for mode in ("serial", "worker"):
        root = tmp_path / mode
        adapter, output = outputs(root, data.shape, 16)
        with DemandStore(root, identity, geometry, adapter, output,
                         accumulator_dtype="float32") as store:
            for index in indices:
                store.request(index)
            if mode == "serial":
                while (tile := store.next_tile()) is not None:
                    store.complete(tile, tensor_result(geometry, tile, data))
            else:
                with DemandWorkerPool({"ct": store}, {"ct": reader}, ["cpu"],
                                      input_workers=2, slots_per_gpu=4,
                                      max_batch_size=3) as pool:
                    deadline = time.monotonic() + 60
                    while pool.busy and time.monotonic() < deadline:
                        pool.pump()
                        time.sleep(.005)
                    assert not pool.busy
                    assert pool.stats["tiles"] == 8
            assert all(store.ready(geometry.origin(index)) for index in indices)
        results.append(zarr.open_array(str(root / "value.zarr/0"), mode="r")[:])
        with DemandStore(root, identity, geometry, adapter, output,
                         accumulator_dtype="float32") as resumed:
            for index in indices:
                assert resumed.ready(resumed.request(index))
            assert resumed.next_tile() is None
    np.testing.assert_array_equal(*results)
    # This test explicitly selects FP32 accumulation; input normalization is
    # also FP32. Do not introduce the default arena's FP16 rounding here.
    expected = data.astype(np.float32) / 255
    np.testing.assert_array_equal(results[0], (expected * 255).astype(np.uint8))

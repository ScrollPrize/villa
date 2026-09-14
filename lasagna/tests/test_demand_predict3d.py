"""Tiny CPU identity-model tests of inference scheduling, not CPU training."""
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from lasagna import tiled_predict3d as tiled
from lasagna.demand_predict3d import DemandGeometry, DemandStore, DemandWorkerPool
import time

LAYOUT_CASES = [("flat", False), ("chunk-major", False),
    pytest.param("chunk-major", True, marks=pytest.mark.skipif(
        not tiled.sys.platform.startswith("linux"), reason="Linux writable-prefault experiment"))]


def test_queue_snapshot_units_and_capacity_accounting():
    from concurrent.futures import Future
    from types import SimpleNamespace
    from lasagna.demand_predict3d import format_queue_snapshot
    queued, running, done = Future(), Future(), Future()
    running.set_running_or_notify_cancel()
    done.set_result(None)
    pool = DemandWorkerPool.__new__(DemandWorkerPool)
    pool.loading = {i: ("source", (i, 0, 0), i, future, 0)
                    for i, future in enumerate((queued, running, done))}
    store = SimpleNamespace(priorities={i: (i % 2, 0) for i in range(110)}, tiles={0: {}},
                            chunks={i: {"ready": False, "requested": i < 105} for i in range(110)},
                            publications=[1, 2])
    pool.stores = {"source": store}
    pool.active, pool.input_owners = {4: None, 5: None}, {4: 4}
    pool.free_inputs, pool.free = list(range(4)), list(range(125))
    pool.stats = {"input_slots": 8, "result_slots": 128}
    pool.accumulating, pool.accumulation_tasks = {6: None}, {1: None, 2: None, 3: None}
    pool.pending_accumulation, pool.publishing = [[1], []], {1: None}
    row = pool.queue_snapshot()
    assert (row["read_queued"], row["read_running"], row["read_done"]) == (1, 1, 1)
    assert row["tiles_queued"] == 109 and row["requested_chunks_pending"] == 105
    assert row["tiles_queued_foreground"] + row["tiles_queued_background"] == 109
    assert row["input_slots_free"] + row["input_slots_owned"] + row["reads_submitted"] == 8
    assert row["result_slots_free"] + row["gpu_submitted_uncollected"] + row["tiles_accumulating"] == 128
    assert row["accumulation_tasks_outstanding"] == 3 and row["accumulation_tasks_not_submitted"] == 1
    assert "reads(q/run/done)=1/1/1" in format_queue_snapshot(row)


class IdentityAdapter:
    native_output_stride = 1

    def __init__(self, root, chunk, scale):
        self.output_products = (tiled.OutputProductSpec(name="identity", level=0, scaledown=scale,
            inference_scaledown=scale, channels=(tiled.OutputChannelSpec("value", relative_path=str(root / "value.zarr")),),
            chunk_size=chunk, accumulator_channel_count=1),)

    def run_tile_inference(self, model, tile, *, device):
        return tile

    def load_model(self, *, device):
        return torch.nn.Identity().to(device)

    def product_tensors_from_output(self, value):
        return {"identity": value}

    def finalize_product_slab(self, product, raw):
        return {"value": np.clip(raw[0] * 255, 0, 255).astype(np.uint8)}


def test_deep_input_window_starts_128_readers_and_preserves_exact_values(tmp_path):
    import threading
    geometry = DemandGeometry((24, 24, 24), 4, 0, 0, 1, 4)
    adapter, output = outputs(tmp_path, geometry.shape, geometry.chunk)
    release = threading.Event()
    lock = threading.Lock()
    entered = set()
    volume = np.full(geometry.shape, 85, np.uint8)
    def reader(tile):
        with lock:
            entered.add(tile)
        if not release.wait(20):
            raise RuntimeError("concurrent-reader test timed out")
        return np.full((4, 4, 4), 85, np.uint8)
    with DemandStore(tmp_path, {"test": "deep-input"}, geometry, adapter, output,
                     accumulator_layout="chunk-major") as store:
        for index in itertools.product(range(6), repeat=3):
            store.request(index, background=True)
        with DemandWorkerPool({"test": store}, {"test": reader}, ["cpu"],
                              input_slots=256, input_workers=128, result_slots=7) as pool:
            try:
                deadline = time.monotonic() + 15
                while len(entered) < 128 and time.monotonic() < deadline:
                    before = len(pool.loading)
                    pool.pump()
                    assert len(pool.loading) - before <= 16
                    time.sleep(.005)
                row = pool.queue_snapshot()
                assert len(entered) == row["read_running"] == 128
                assert row["reads_submitted"] == 128  # no deep hidden executor FIFO
                assert row["input_slots_free"] == 128
                assert row["tiles_queued_background"] == 88
                assert row["result_slots_capacity"] == 7
            finally:
                release.set()
            deadline = time.monotonic() + 30
            while pool.busy and time.monotonic() < deadline:
                pool.pump()
                time.sleep(.005)
            assert not pool.busy
            assert pool.stats["tiles"] == 216
            assert len(pool.free_inputs) == 256 and len(pool.free) == 7
        # Compare with the existing numerical path, including its fp16 input
        # conversion and truncating identity finalizer, not original integers.
        control_root = tmp_path / "control"
        control_adapter, control_output = outputs(control_root, geometry.shape, geometry.chunk)
        with DemandStore(control_root, {"test": "serial"}, geometry, control_adapter, control_output) as control:
            for index in itertools.product(range(6), repeat=3):
                control.request(index)
            drain_reverse(control, volume)
        np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "value.zarr/0"), mode="r")[:],
                                      zarr.open_array(str(control_root / "value.zarr/0"), mode="r")[:])


@pytest.mark.parametrize("slots", [0, -1, 1.5, True])
def test_invalid_explicit_input_slots(tmp_path, slots):
    from types import SimpleNamespace
    geometry = DemandGeometry((4, 4, 4), 4, 0, 0, 1, 4)
    adapter, output = outputs(tmp_path, geometry.shape, geometry.chunk)
    store = SimpleNamespace(model_adapter=adapter, geometry=geometry, products=adapter.output_products)
    with pytest.raises(ValueError, match="input_slots"):
        DemandWorkerPool({"test": store}, {}, ["cpu"], input_slots=slots)


def outputs(root, shape, chunk, scale=1):
    adapter = IdentityAdapter(root, chunk, scale)
    # The logical output here is already on its final lattice (level zero in
    # this test). Production products use their physical scaledown/level.
    tiled.create_product_omezarr_groups(products=adapter.output_products,
        base_shape_zyx=shape, n_levels=1, ome_chunk=chunk, ome_compressor="none")
    return adapter, tiled.OmeZarrOutputAdapter(products=adapter.output_products, n_levels=1)


def tensor_result(geometry, tile, volume):
    torch.set_num_threads(1)
    spec = tiled._tile_read_spec(geometry.shape, (0, 0, 0), *tile, geometry.tile, geometry.border)
    raw = tiled._finalize_tile_read(None if spec.slices_zyx is None else volume[spec.slices_zyx], spec)
    normalized = tiled._input_tile_to_device(raw, torch.device("cpu"))[0, 0]
    ramp = tiled.blend_ramp(geometry.tile, geometry.border, geometry.overlap)
    weight = torch.from_numpy(ramp[:, None, None] * ramp[None, :, None] * ramp[None, None, :])
    weighted = normalized * weight
    if geometry.scale > 1:
        weight = tiled._pyrdown3d(weight[None], factor=geometry.scale)[0]
        weighted = tiled._pyrdown3d(weighted[None], factor=geometry.scale)[0]
    return {"weight": weight.numpy(), "identity": weighted[None].numpy()}


def drain_reverse(store, volume):
    tiles = []
    while (tile := store.next_tile()) is not None:
        tiles.append(tile)
    for tile in reversed(tiles):
        store.complete(tile, tensor_result(store.geometry, tile, volume))
    return tiles


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("layout,prefault", LAYOUT_CASES)
def test_demand_late_chunks_reverse_completion_matches_sequential(tmp_path, dtype, scale, layout, prefault):
    geometry = DemandGeometry((9, 11, 13), 6 if scale == 1 else 8, 2 if scale == 1 else 4,
                              scale, scale, 4)
    volume = np.random.default_rng(42).integers(0, 256, geometry.shape, dtype=np.uint8)
    options = {"zarr_format": 2} if int(zarr.__version__.split(".")[0]) >= 3 else {"zarr_version": 2}
    source = zarr.open_array(str(tmp_path / "input.zarr"), mode="w", shape=geometry.shape,
                            chunks=(4, 4, 4), dtype="u1", **options)
    source[:] = volume
    sequential_adapter, sequential_output = outputs(tmp_path / "sequential", geometry.output_shape, geometry.chunk, scale)
    torch.set_num_threads(1)
    tiled.run_tiled_inference_3d(object(), volume,
        crop_slices=(0, 9, 0, 11, 0, 13), device=torch.device("cpu"),
        model_adapter=sequential_adapter, output_adapter=sequential_output,
        products=sequential_adapter.output_products,
        output_regions_zyx={"identity": (0, 0, 0, *geometry.output_shape)},
        full_output_shapes_zyx={"identity": geometry.output_shape},
        input_zarr_path=str(tmp_path / "input.zarr"),
        output_scaledown_base={"identity": scale}, tile_size=geometry.tile, overlap=geometry.overlap, border=geometry.border,
        tmp_dir=str(tmp_path), input_reader="python-zarr", flush_workers=0,
        accumulator_workers=0, product_accumulator_dtype=dtype)
    root = tmp_path / "demand"
    adapter, output = outputs(root, geometry.output_shape, geometry.chunk, scale)
    with DemandStore(root, {"test": 1, "dtype": dtype}, geometry, adapter, output,
                     accumulator_dtype=dtype, accumulator_layout=layout, accumulator_prefault=prefault) as store:
        origin = store.request((0, 0, 0), remaining=100)
        first = drain_reverse(store, volume)
        assert store.ready(origin)
        # Registering touched-but-unrequested neighbors must not infer them.
        assert store.next_tile() is None
        indices = tuple(itertools.product(*(range((n + 3) // 4) for n in geometry.output_shape)))
        for index in indices:
            store.request(index, remaining=1)
        second = drain_reverse(store, volume)
        assert not set(first) & set(second)
        assert all(store.ready(geometry.origin(index)) for index in indices)
        assert not list(root.rglob("*.npz"))
        assert all(not e.get("pending") for e in store.chunks.values())
        # Independent full-volume arithmetic oracle in actual arrival order.
        raw_oracle = np.zeros(geometry.output_shape, dtype=dtype)
        weight_oracle = np.zeros(geometry.output_shape, dtype=np.float32)
        for tile in [*reversed(first), *reversed(second)]:
            values = tensor_result(geometry, tile, volume)
            for touched in geometry.touched(tile):
                dst, src = geometry.intersection(touched, tile)
                np.add(raw_oracle[dst], values["identity"][(0, *src)], out=raw_oracle[dst], casting="unsafe")
                np.add(weight_oracle[dst], values["weight"][src], out=weight_oracle[dst])
        cache = {}
        try:
            nz, ny, nx = geometry.output_shape
            actual_raw = tiled._read_mmap_band_chunk(store._descriptor(store.accumulators["identity"]),
                                                     0, 0, nz, 0, ny, 0, nx, cache)
            actual_weight = tiled._read_mmap_band_chunk(store._descriptor((store.weight,)),
                                                        0, 0, nz, 0, ny, 0, nx, cache)
            np.testing.assert_array_equal(actual_raw, raw_oracle)
            np.testing.assert_array_equal(actual_weight, weight_oracle)
        finally:
            for a in cache.values():
                a._mmap.close()
        actual = zarr.open_array(str(root / "value.zarr/0"), mode="r")[:]
        expected_arrival = np.clip(raw_oracle.astype(np.float32) / np.maximum(weight_oracle, 1e-7) * 255, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(actual, expected_arrival)
        expected = zarr.open_array(str(tmp_path / "sequential/value.zarr/0"), mode="r")[:]
        # Arrival-order additions need not reproduce sequential f16 rounding.
        # This identity fixture bounds the resulting uint8 difference to 1.
        assert np.max(np.abs(actual.astype(int) - expected.astype(int))) <= 1
    assert not list((root / "value.zarr/.live_tmp").iterdir())
    with DemandStore(root, {"test": 1, "dtype": dtype}, geometry, adapter, output,
                     accumulator_dtype=dtype, accumulator_layout=layout, accumulator_prefault=prefault) as store:
        for index in indices:
            store.request(index)
        assert store.next_tile() is None  # complete warm cache needs zero inference
        assert store.stats["warm_chunks"] == len(indices)


def test_atomic_writer_persists_computed_fill_chunk(tmp_path):
    adapter, output = outputs(tmp_path, (4, 4, 4), 4)
    product = adapter.output_products[0]
    output.write_product_chunk(product, chunk_origin_zyx=(0, 0, 0), data={"value": np.zeros((4, 4, 4), np.uint8)})
    assert output.product_chunk_complete(product, chunk_origin_zyx=(0, 0, 0))


def test_dependency_geometry_matches_exhaustive_footprint():
    with pytest.raises(ValueError, match="alignment"):
        DemandGeometry((15, 17, 19), 8, 3, 0, 2, 3)
    geometry = DemandGeometry((15, 17, 19), 8, 4, 2, 2, 3)
    positions = [tiled._build_tile_positions(tiled._round_up_to_multiple(n + 4, 2), 8, 4) for n in geometry.shape]
    all_tiles = tuple(itertools.product(*positions))
    for index in itertools.product(*(range((n + 2) // 3) for n in geometry.output_shape)):
        origin = geometry.origin(index)
        oracle = tuple(tile for tile in all_tiles if origin in set(geometry.touched(tile)))
        assert geometry.contributors(origin) == oracle


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_chunk_major_shared_views_cross_bricks_and_preserve_padding(tmp_path, dtype):
    shape, b = (5, 7, 9), 4
    path = tmp_path / "bricks.mmap"
    descriptor = tiled._MmapBandDescriptor((str(path),), shape, dtype, b)
    assert descriptor.storage_shape == (2, 2, 3, 4, 4, 4)
    storage = np.memmap(path, mode="w+", dtype=dtype, shape=descriptor.storage_shape)
    cache = {str(path): storage}
    oracle = np.zeros(shape, dtype=dtype)
    try:
        for bounds in ((1, 5, 2, 7, 3, 9), (0, 5, 0, 7, 0, 9)):
            slices = tuple(slice(a, z) for a, z in zip(bounds[::2], bounds[1::2]))
            source = np.random.default_rng(8).normal(size=oracle[slices].shape).astype(np.float32)
            tiled._add_mmap_band_chunk(descriptor, 0, *bounds, source, cache)
            np.add(oracle[slices], source, out=oracle[slices], casting="unsafe")
            result = tiled._read_mmap_band_chunk(descriptor, 0, 0, 5, 0, 7, 0, 9, cache)
            np.testing.assert_array_equal(result, oracle)
        for index in itertools.product(*(range(n) for n in descriptor.storage_shape[:3])):
            brick = storage[index]
            assert brick.flags.c_contiguous
            for local in itertools.product(range(b), repeat=3):
                coord = tuple(i*b+j for i, j in zip(index, local))
                if any(v >= n for v, n in zip(coord, shape)):
                    assert brick[local] == 0
        with pytest.raises(ValueError, match="region"):
            list(tiled._chunk_major_views(storage, descriptor, (0, 6, 0, 7, 0, 9)))
    finally:
        storage._mmap.close()


@pytest.mark.parametrize("b,dtype", [(2, "float16"), (16, "float32"), (32, "float16")])
def test_prefault_full_brick_page_range_once_and_failure(tmp_path, monkeypatch, b, dtype):
    path = tmp_path / "prefault.mmap"
    descriptor = tiled._MmapBandDescriptor((str(path),), (b, b, 2*b), dtype, b, True)
    storage = np.memmap(path, mode="w+", dtype=dtype, shape=descriptor.storage_shape)
    actual_mapping = storage._mmap
    calls = []
    class AdviceSpy:
        fail = False
        def madvise(self, advice, start, length):
            calls.append((advice, start, length))
            if self.fail:
                raise OSError("injected failure")
    spy = AdviceSpy()
    storage._mmap = spy
    monkeypatch.setattr(tiled, "_mmap_populate_write_advice", lambda: 23)
    try:
        brick_bytes = b**3 * storage.dtype.itemsize
        page = tiled.mmap.PAGESIZE
        start = brick_bytes // page * page
        end = min(storage.nbytes, ((2*brick_bytes + page-1)//page)*page)
        # Only the last voxel is requested, but the entire second brick is advised.
        bounds = (b-1, b, b-1, b, 2*b-1, 2*b)
        cache = {str(path): storage}
        value = np.ones((1, 1, 1), np.float32)
        spy.fail = True
        with pytest.raises(RuntimeError, match="POPULATE_WRITE"):
            tiled._add_mmap_band_chunk(descriptor, 0, *bounds, value, cache)
        assert not storage._demand_prefaulted_offsets
        assert not np.any(storage)
        spy.fail = False
        for _ in range(2):
            tiled._add_mmap_band_chunk(descriptor, 0, *bounds, value, cache)
        assert calls == [(23, start, end-start)]*2  # failed call, one success, no repeat
        assert storage.sum() == 2
        assert storage[0, 0, 1, b-1, b-1, b-1] == 2
        tiled._read_mmap_band_chunk(descriptor, 0, *bounds, cache)
        assert len(calls) == 2  # Never prefault during finalization reads.
    finally:
        storage._mmap = actual_mapping
        actual_mapping.close()


def test_prefault_rejects_flat_and_unsupported_platform():
    from unittest.mock import patch
    with pytest.raises(ValueError, match="chunk-major"):
        tiled._MmapBandDescriptor(("unused",), (4, 4, 4), prefault=True)
    with patch.object(tiled.sys, "platform", "darwin"):
        with pytest.raises(RuntimeError, match="Linux"):
            tiled._mmap_populate_write_advice()


def test_flat_descriptor_keeps_circular_z_wrap(tmp_path):
    path = tmp_path / "ring.mmap"
    descriptor = tiled._MmapBandDescriptor((str(path),), (3, 5, 7))
    storage = np.memmap(path, mode="w+", dtype=np.float32, shape=descriptor.storage_shape)
    cache = {str(path): storage}
    source = np.arange(3*5*7, dtype=np.float32).reshape(3, 5, 7)
    try:
        tiled._add_mmap_band_chunk(descriptor, 0, 2, 5, 0, 5, 0, 7, source, cache)
        np.testing.assert_array_equal(tiled._read_mmap_band_chunk(descriptor, 0, 2, 5, 0, 5, 0, 7, cache), source)
        np.testing.assert_array_equal(storage[2], source[0])
        np.testing.assert_array_equal(storage[:2], source[1:])
    finally:
        storage._mmap.close()


def test_chunk_major_padded_edge_has_no_budget_and_preserves_values(tmp_path):
    geometry = DemandGeometry((3, 3, 3), 4, 0, 0, 1, 4)
    adapter, output = outputs(tmp_path, geometry.output_shape, 4)
    with DemandStore(tmp_path, {}, geometry, adapter, output, accumulator_layout="chunk-major") as store:
        store.request((0, 0, 0))
        tile = store.next_tile()
        store.complete(tile, tensor_result(geometry, tile, np.full(geometry.shape, 85, np.uint8)))
        assert store.ready((0, 0, 0))
    # Preserve the existing fp16 input/accumulator and truncating finalizer,
    # rather than assuming byte-perfect identity through that numerical path.
    expected = np.uint8(np.float32(np.float16(85 / 255)) * 255)
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "value.zarr/0"), mode="r")[:], expected)


@pytest.mark.parametrize("profile_pipeline", [False, True])
@pytest.mark.parametrize("max_batch_size", [1, 3])
def test_shared_spawned_worker_demand_pipeline(tmp_path, profile_pipeline, max_batch_size):
    geometry = DemandGeometry((5, 5, 5), 4, 2, 1, 1, 4)
    volume = np.full(geometry.shape, 180, dtype=np.uint8)
    adapter, output = outputs(tmp_path, geometry.output_shape, geometry.chunk)

    def reader(tile):
        spec = tiled._tile_read_spec(geometry.shape, (0, 0, 0), *tile, geometry.tile, geometry.border)
        return tiled._finalize_tile_read(volume[spec.slices_zyx], spec)

    with DemandStore(tmp_path, {"test": "worker"}, geometry, adapter, output) as store:
        origin = store.request((0, 0, 0), remaining=10)
        # CPU is only an identity-model smoke fixture; production runner
        # requires explicitly selected CUDA devices.
        with DemandWorkerPool({"test": store}, {"test": reader}, ["cpu"], slots_per_gpu=5, input_workers=5,
                              profile_pipeline=profile_pipeline, max_batch_size=max_batch_size) as pool:
            deadline = time.monotonic() + 30
            while pool.busy and time.monotonic() < deadline:
                pool.pump()
                time.sleep(0.005)
            assert not pool.busy
            assert store.ready(origin)
            assert pool.stats["tiles"] > 0
            assert pool.worker_ids == {0}
            assert sum(int(size) * count for size, count in pool.stats["batch_size_counts"].items()) == pool.stats["tiles"]
            assert max(map(int, pool.stats["batch_size_counts"])) <= max_batch_size
            assert sum(pool.stats["batch_seconds"]) == pytest.approx(pool.stats["worker_seconds"])
            assert len(pool.free_inputs) == 5 and len(pool.free) == 5
            assert bool(pool.profile_samples) == profile_pipeline
            assert bool(pool.stage_samples) == profile_pipeline
            if profile_pipeline:
                stages = {row["stage"] for row in pool.stage_samples}
                assert {"admit.global_priority_selection", "admit.store_state_update",
                        "admit.executor_submit", "reader.shared_slot_copy"} <= stages
                assert len(pool.profile_samples) == pool.stats["tiles"]
                assert all(s["worker_total_wall_s"] >= 0 and s["coordinator_receive_delay_s"] >= 0
                           for s in pool.profile_samples)
        values = zarr.open_array(str(tmp_path / "value.zarr/0"), mode="r")[:4, :4, :4]
        control_root = tmp_path / "control"
        control_adapter, control_output = outputs(control_root, geometry.output_shape, geometry.chunk)
        with DemandStore(control_root, {"test": "control"}, geometry, control_adapter, control_output) as control:
            control.request((0, 0, 0))
            drain_reverse(control, volume)
        expected = zarr.open_array(str(control_root / "value.zarr/0"), mode="r")[:4, :4, :4]
        np.testing.assert_array_equal(values, expected)


def test_cache_writer_identity_and_corrupt_ready_recovery(tmp_path):
    geometry = DemandGeometry((4, 4, 4), 4, 0, 0, 1, 4)
    volume = np.full(geometry.shape, 127, np.uint8)
    adapter, output = outputs(tmp_path, geometry.shape, 4)
    with DemandStore(tmp_path, {"version": 1}, geometry, adapter, output) as store:
        with pytest.raises(RuntimeError, match="writer"):
            DemandStore(tmp_path, {"version": 1}, geometry, adapter, output)
        store.request((0, 0, 0))
        drain_reverse(store, volume)
    with pytest.raises(ValueError, match="identity"):
        DemandStore(tmp_path, {"version": 2}, geometry, adapter, output)
    member = tmp_path / "value.zarr/0/0/0/0"
    member.write_bytes(b"corrupt synthetic fixture")
    with DemandStore(tmp_path, {"version": 1}, geometry, adapter, output) as store:
        origin = store.request((0, 0, 0))
        assert not store.ready(origin)
        assert drain_reverse(store, volume)
        assert store.ready(origin)
    assert np.all(zarr.open_array(str(tmp_path / "value.zarr/0"), mode="r")[:] == 127)


@pytest.mark.parametrize("shape,value", [((4, 4, 4), 0), ((5, 6, 7), 127)])
def test_ram_only_state_reuses_published_chunks_without_sqlite(tmp_path, monkeypatch, shape, value):
    import sqlite3
    def forbidden(*args, **kwargs):
        raise AssertionError("SQLite must never be opened")
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    geometry = DemandGeometry(shape, 4, 0, 0, 1, 4)
    adapter, output = outputs(tmp_path, shape, 4)
    indices = list(itertools.product(*(range((n + 3) // 4) for n in shape)))
    with DemandStore(tmp_path, {}, geometry, adapter, output) as store:
        for index in indices:
            store.request(index)
        drain_reverse(store, np.full(shape, value, np.uint8))
        assert not hasattr(store, "db")
    with DemandStore(tmp_path, {}, geometry, adapter, output) as store:
        assert not store.tiles and not store.chunks and not store.priorities
        for index in indices:
            assert store.ready(store.request(index))
        assert store.next_tile() is None
        assert store.stats["warm_chunks"] == len(indices)
    assert not list(tmp_path.glob("*.sqlite*"))


@pytest.mark.parametrize("layout", ["flat", "chunk-major"])
def test_publication_borrows_existing_maps_without_remapping(tmp_path, monkeypatch, layout):
    from concurrent.futures import ThreadPoolExecutor
    geometry = DemandGeometry((8, 8, 8), 4, 0, 0, 1, 4)
    adapter, output = outputs(tmp_path, geometry.shape, 4)
    with DemandStore(tmp_path, {}, geometry, adapter, output, accumulator_layout=layout) as store:
        for arrays in store.accumulators.values():
            for array in arrays:
                array[:] = 0.5  # exactly representable normalized model output
        store.weight[:] = 1
        def forbidden(*args, **kwargs):
            raise AssertionError("publication must not remap full-volume accumulators")
        with monkeypatch.context() as patch:
            patch.setattr(np, "memmap", forbidden)
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(store._publish, itertools.product((0, 4), repeat=3)))
        assert all(not a._mmap.closed for a in store.arrays)
        assert np.all(zarr.open_array(str(tmp_path / "value.zarr/0"), mode="r")[:] == 127)


def test_legacy_database_is_not_opened_or_deleted(tmp_path):
    geometry = DemandGeometry((4, 4, 4), 4, 0, 0, 1, 4)
    adapter, output = outputs(tmp_path, geometry.shape, 4)
    legacy = tmp_path / "live.sqlite"
    legacy.write_bytes(b"must remain untouched, not opened")
    with pytest.raises(ValueError, match="legacy database-only"):
        DemandStore(tmp_path, {}, geometry, adapter, output)
    assert legacy.read_bytes() == b"must remain untouched, not opened"
    assert not (tmp_path / "cache_identity.json").exists()


def test_unidentified_payloads_are_not_adopted(tmp_path):
    geometry = DemandGeometry((4, 4, 4), 4, 0, 0, 1, 4)
    adapter, output = outputs(tmp_path, geometry.shape, 4)
    array = zarr.open_array(str(tmp_path / "value.zarr/0"), mode="r+")
    array[:] = 42
    with pytest.raises(ValueError, match="unidentified cache payloads"):
        DemandStore(tmp_path, {}, geometry, adapter, output)
    assert not (tmp_path / "cache_identity.json").exists()
    assert np.all(array[:] == 42)


def test_publication_rejects_nested_external_symlink_before_any_unlink(tmp_path):
    root = tmp_path / "cache"
    geometry = DemandGeometry((4, 4, 4), 4, 0, 0, 1, 4)
    adapter = SiblingIdentityAdapter(root)
    tiled.create_product_omezarr_groups(products=adapter.output_products,
        base_shape_zyx=geometry.shape, n_levels=1, ome_chunk=4, ome_compressor="none")
    output = tiled.OmeZarrOutputAdapter(products=adapter.output_products, n_levels=1)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "0").write_bytes(b"preserve")
    with DemandStore(root, {}, geometry, adapter, output) as store:
        a = root / "a.zarr/0/0/0/0"
        a.parent.mkdir(parents=True)
        a.write_bytes(b"old sibling")
        b_parent = root / "b.zarr/0/0"
        b_parent.mkdir()
        (b_parent / "0").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="outside cache"):
            store._publish((0, 0, 0))
        assert a.read_bytes() == b"old sibling"
        assert (outside / "0").read_bytes() == b"preserve"


def test_stale_scratch_cleanup_validates_all_before_deleting(tmp_path):
    geometry = DemandGeometry((4, 4, 4), 4, 0, 0, 1, 4)
    adapter, output = outputs(tmp_path, geometry.shape, 4)
    with DemandStore(tmp_path, {}, geometry, adapter, output):
        pass
    parent = tmp_path / "value.zarr/.live_tmp"
    good, bad = parent / "session-good", parent / "session-bad"
    good.mkdir()
    bad.mkdir()
    (good / "raw-0.mmap").write_bytes(b"temporary")
    unrelated = bad / "not-scratch.txt"
    unrelated.write_text("keep")
    with pytest.raises(ValueError, match="unsafe stale scratch contents"):
        DemandStore(tmp_path, {}, geometry, adapter, output)
    assert (good / "raw-0.mmap").exists() and unrelated.read_text() == "keep"
    unrelated.unlink()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = bad / "raw-0.mmap"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe stale scratch contents"):
        DemandStore(tmp_path, {}, geometry, adapter, output)
    assert good.exists() and outside.is_dir()
    link.unlink()
    with DemandStore(tmp_path, {}, geometry, adapter, output):
        assert not good.exists() and not bad.exists()


class SiblingIdentityAdapter(IdentityAdapter):
    def __init__(self, root):
        super().__init__(root, 4, 1)
        from dataclasses import replace
        self.output_products = (replace(self.output_products[0], channels=(
            tiled.OutputChannelSpec("a", relative_path=str(root / "a.zarr")),
            tiled.OutputChannelSpec("b", relative_path=str(root / "b.zarr")))),)

    def finalize_product_slab(self, product, raw):
        values = super().finalize_product_slab(product, raw)["value"]
        return {"a": values, "b": values}


def test_interrupted_sibling_replacement_cannot_reuse_old_generation(tmp_path, monkeypatch):
    geometry = DemandGeometry((4, 4, 4), 4, 0, 0, 1, 4)
    adapter = SiblingIdentityAdapter(tmp_path)
    tiled.create_product_omezarr_groups(products=adapter.output_products,
        base_shape_zyx=geometry.shape, n_levels=1, ome_chunk=4, ome_compressor="none")
    output = tiled.OmeZarrOutputAdapter(products=adapter.output_products, n_levels=1)
    with DemandStore(tmp_path, {}, geometry, adapter, output):
        pass
    for name in ("a", "b"):
        zarr.open_array(str(tmp_path / f"{name}.zarr/0"), mode="r+")[:] = 22
    a = tmp_path / "a.zarr/0/0/0/0"
    b = tmp_path / "b.zarr/0/0/0/0"
    a.write_bytes(b"corrupt")
    def interrupted(*args):
        assert not a.exists() and not b.exists()
        zarr.open_array(str(tmp_path / "a.zarr/0"), mode="r+")[:] = 77
        raise RuntimeError("publication interrupted")
    with DemandStore(tmp_path, {}, geometry, adapter, output) as store:
        origin = store.request((0, 0, 0))
        assert not store.ready(origin)
        with monkeypatch.context() as patch:
            patch.setattr(tiled, "_execute_flush_process_task", interrupted)
            with pytest.raises(RuntimeError, match="publication interrupted"):
                drain_reverse(store, np.full(geometry.shape, 127, np.uint8))
        assert not store.ready(origin)
    assert a.exists() and not b.exists()
    with DemandStore(tmp_path, {}, geometry, adapter, output) as store:
        origin = store.request((0, 0, 0))
        assert not store.ready(origin)
        drain_reverse(store, np.full(geometry.shape, 127, np.uint8))
        assert store.ready(origin)
    assert np.all(zarr.open_array(str(tmp_path / "a.zarr/0"), mode="r")[:] == 127)
    assert np.all(zarr.open_array(str(tmp_path / "b.zarr/0"), mode="r")[:] == 127)


def test_scratch_plans_touched_brick_above_old_cap_without_reservation(tmp_path, monkeypatch):
    # Old chunk-major accounting rejected this single padded brick (>64GiB).
    # Plan only: never allocate/write a giant tensor or publish a giant chunk.
    geometry = DemandGeometry((3, 3, 3), 4, 0, 0, 1, 4096)
    adapter, output = outputs(tmp_path, geometry.shape, geometry.chunk)
    with DemandStore(tmp_path, {}, geometry, adapter, output, accumulator_layout="chunk-major") as store:
        assert sum(a.nbytes for a in store.arrays) > 64 << 30
        def forbidden(*args, **kwargs):
            raise AssertionError("no per-tile filesystem-space polling")
        monkeypatch.setattr("shutil.disk_usage", forbidden)
        store.request((0, 0, 0))
        tile = store.next_tile()
        tasks = store.begin_complete(tile, None, 0, 0)
        assert len(tasks) == 1
        assert tasks[0].origin == (0, 0, 0)
        assert all(Path(a.filename).stat().st_blocks == 0 for a in store.arrays)
        assert not store.ready((0, 0, 0))


def test_scratch_mapping_failure_propagates_and_cleans_up(tmp_path, monkeypatch):
    import errno
    geometry = DemandGeometry((4, 4, 4), 4, 0, 0, 1, 4)
    adapter, output = outputs(tmp_path, geometry.shape, 4)
    original = np.memmap
    opened = []
    def fail_second_mapping(*args, **kwargs):
        if opened:
            raise OSError(errno.ENOSPC, "injected mapping failure")
        opened.append(original(*args, **kwargs))
        return opened[-1]
    with monkeypatch.context() as patch:
        patch.setattr(np, "memmap", fail_second_mapping)
        with pytest.raises(OSError) as error:
            DemandStore(tmp_path, {}, geometry, adapter, output)
    assert error.value.errno == errno.ENOSPC
    assert opened[0]._mmap.closed
    assert not list(tmp_path.glob("value.zarr/.live_tmp/session-*"))
    # Constructor failure released the writer lock too.
    with DemandStore(tmp_path, {}, geometry, adapter, output):
        pass


@pytest.mark.parametrize("stage", ["_execute_accumulate_process_task", "_execute_flush_process_task"])
def test_actual_storage_error_propagates_without_readiness(tmp_path, monkeypatch, stage):
    import errno
    geometry = DemandGeometry((4, 4, 4), 4, 0, 0, 1, 4)
    adapter, output = outputs(tmp_path, geometry.shape, 4)
    def fail(*args, **kwargs):
        raise OSError(errno.ENOSPC, "injected storage failure")
    monkeypatch.setattr(tiled, stage, fail)
    with DemandStore(tmp_path, {}, geometry, adapter, output) as store:
        store.request((0, 0, 0))
        tile = store.next_tile()
        with pytest.raises(OSError) as error:
            store.complete(tile, tensor_result(geometry, tile, np.ones(geometry.shape, np.uint8)))
        assert error.value.errno == errno.ENOSPC
        assert not store.ready((0, 0, 0))
        assert not output.product_chunk_complete(adapter.output_products[0], chunk_origin_zyx=(0, 0, 0))
    assert not list(tmp_path.glob("value.zarr/.live_tmp/session-*"))


def test_demand_priority_promotes_reference_prefetch(tmp_path):
    geometry = DemandGeometry((12, 4, 4), 4, 0, 0, 1, 4)
    adapter, output = outputs(tmp_path, geometry.shape, 4)
    with DemandStore(tmp_path, {}, geometry, adapter, output) as store:
        store.request((0, 0, 0), background=True)
        with pytest.raises(ValueError, match="result_slots"):
            DemandWorkerPool({"s": store}, {}, ["cpu"], result_slots=0)
        store.request((1, 0, 0), remaining=5)
        store.request((2, 0, 0), remaining=50)
        assert store.next_tile() == (8, 0, 0)
        store.request((0, 0, 0), remaining=100)
        assert store.next_tile() == (0, 0, 0)
        assert store.next_tile() == (4, 0, 0)


def test_priority_heap_matches_full_scan_with_promotions_and_source_ties(tmp_path):
    from contextlib import ExitStack
    import random
    rng = random.Random(20260911)
    geometry = DemandGeometry((64, 4, 4), 4, 0, 0, 1, 4)
    with ExitStack() as cleanup:
        pool = DemandWorkerPool.__new__(DemandWorkerPool)
        pool.stores = {}
        priorities, admitted = {}, set()
        for name in ("z", "a", "m"):
            root = tmp_path / name
            root.mkdir()
            adapter, output = outputs(root, geometry.shape, 4)
            pool.stores[name] = cleanup.enter_context(DemandStore(root, {}, geometry, adapter, output))

        def request(name, index, remaining, background):
            pool.stores[name].request((index, 0, 0), remaining=remaining, background=background)
            tile = (index * 4, 0, 0)
            if (name, tile) not in admitted:
                priorities[name, tile] = max((0 if background else 1, remaining),
                                            priorities.get((name, tile), (-1, 0)))

        def take():
            expected = min(((-p[0], -p[1], name, tile)
                            for (name, tile), p in priorities.items() if (name, tile) not in admitted),
                           default=None)
            assert pool._next_admission() == expected
            if expected is not None:
                _, _, name, tile = expected
                assert pool.stores[name].next_tile() == tile
                assert pool.stores[name].tiles[tile]["state"] == "inflight"
                admitted.add((name, tile))
            return expected

        for name in pool.stores:
            for index in range(16):
                request(name, index, 0, True)
        take()  # Equal priorities: source name precedes tile coordinates.
        for _ in range(300):
            name, index = rng.choice(tuple(pool.stores)), rng.randrange(16)
            request(name, index, rng.choice((0, 5, 5, 50, 100)), rng.choice((False, True)))
            if rng.random() < .2:
                take()
        while take() is not None:
            pass
        assert len(admitted) == 48
        for store in pool.stores.values():
            assert store.pending_tile_key() is None
            assert store.next_tile() is None
            assert not store._pending_priorities  # Stale promotions fully drained.


@pytest.mark.parametrize("layout,prefault", LAYOUT_CASES)
def test_arrival_applied_immediately_without_npz_reservation_or_sync(tmp_path, monkeypatch, layout, prefault):
    geometry = DemandGeometry((8, 8, 8), 6, 2, 1, 1, 4)
    adapter, output = outputs(tmp_path, geometry.shape, geometry.chunk)
    with DemandStore(tmp_path, {}, geometry, adapter, output, accumulator_layout=layout,
                     accumulator_prefault=prefault) as store:
        origin = store.request((0, 0, 0))
        tiles = []
        while (tile := store.next_tile()) is not None:
            tiles.append(tile)
        assert len(tiles) > 1
        tile = tiles[-1]  # Earlier contributors are deliberately not finished.
        values = tensor_result(geometry, tile, np.full(geometry.shape, 100, np.uint8))
        def forbidden(*args, **kwargs):
            raise AssertionError("scratch disk staging/reservation/sync is forbidden")
        with monkeypatch.context() as patch:
            patch.setattr(np, "savez", forbidden)
            patch.setattr(np, "load", forbidden)
            patch.setattr("os.pwrite", forbidden)
            patch.setattr("os.fsync", forbidden)
            store.complete(tile, values)
        assert tile in store.chunks[origin]["applied"]
        assert np.any(store.weight > 0)
        assert not store.ready(origin)
        assert store.tiles[tile]["state"] == "consumed"
        with pytest.raises(ValueError, match="duplicate"):
            store.complete(tile, values)
        with pytest.raises(ValueError, match="duplicate"):
            store.contribution_done(tile, origin)


@pytest.mark.parametrize("result_slots", [1, 7])
@pytest.mark.parametrize("layout,prefault", LAYOUT_CASES)
def test_parallel_sources_backpressure_and_publication_readiness(tmp_path, monkeypatch, result_slots, layout, prefault):
    from contextlib import ExitStack
    import threading
    geometry = DemandGeometry((9, 9, 9), 8, 4, 2, 1, 2)
    entered, release = threading.Event(), threading.Event()
    with ExitStack() as cleanup:
        stores, readers = {}, {}
        for name, value in (("a", 80), ("b", 180)):
            root = tmp_path / name
            adapter, output = outputs(root, geometry.shape, geometry.chunk)
            store = cleanup.enter_context(DemandStore(root, {"source": name}, geometry, adapter, output,
                                                     accumulator_layout=layout, accumulator_prefault=prefault))
            store.request((1, 1, 1))
            volume = np.full(geometry.shape, value, np.uint8)
            def reader(tile, volume=volume):
                spec = tiled._tile_read_spec(geometry.shape, (0, 0, 0), *tile, geometry.tile, geometry.border)
                return tiled._finalize_tile_read(volume[spec.slices_zyx], spec)
            stores[name], readers[name] = store, reader
        original = stores["a"]._publish
        def delayed(origin):
            entered.set()
            if not release.wait(15):
                raise RuntimeError("test failed to release publication")
            return original(origin)
        monkeypatch.setattr(stores["a"], "_publish", delayed)
        pool = cleanup.enter_context(DemandWorkerPool(stores, readers, ["cpu", "cpu"],
            slots_per_gpu=2, result_slots=result_slots, accumulator_workers=2, flush_workers=2))
        assert len(pool.inputs) == 4
        assert len(pool.results) == result_slots
        assert pool.stats["shared_slot_bytes"] == 4 * pool.stats["input_slot_bytes"] + result_slots * pool.stats["result_slot_bytes"]
        try:
            deadline = time.monotonic() + 30
            while not entered.is_set() and time.monotonic() < deadline:
                pool.pump()
                time.sleep(.005)
            assert entered.is_set()
            assert pool.busy
            assert not stores["a"].ready((2, 2, 2))
        finally:
            release.set()
        while pool.busy and time.monotonic() < deadline:
            pool.pump()
            time.sleep(.005)
        assert not pool.busy
        assert not pool.accumulation_tasks and not pool.input_owners
        assert len(pool.free) == len(pool.results)
        assert len(pool.free_inputs) == len(pool.inputs)
        assert all(count > 0 for count in pool.accumulator_counts)
        for name, value in (("a", 80), ("b", 180)):
            store = stores[name]
            assert store.ready((2, 2, 2))
            assert all(not e.get("pending") for e in store.chunks.values())
            actual = zarr.open_array(str(tmp_path / name / "value.zarr/0"), mode="r")[2:4, 2:4, 2:4]
            assert np.max(np.abs(actual.astype(int) - value)) <= 1

        # A neighboring complete incidental chunk needs publication, not another
        # tile or contribution. busy must include this work even with no GPU work.
        store = stores["a"]
        late = next(o for o, e in store.chunks.items()
                    if not e["requested"] and len(e["applied"]) == len(e["dependencies"]))
        before = pool.stats["tiles"]
        store.request(tuple(o // geometry.chunk for o in late))
        assert pool.busy and not store.ready(late)
        while pool.busy and time.monotonic() < deadline:
            pool.pump()
            time.sleep(.005)
        assert store.ready(late)
        assert pool.stats["tiles"] == before


@pytest.mark.parametrize("failure", ["accumulator", "publication"])
def test_async_failure_never_marks_ready(tmp_path, monkeypatch, failure):
    geometry = DemandGeometry((4, 4, 4), 4, 0, 0, 1, 4)
    adapter, output = outputs(tmp_path, geometry.shape, 4)
    with DemandStore(tmp_path, {}, geometry, adapter, output) as store:
        origin = store.request((0, 0, 0))
        def fail_publish(origin):
            raise RuntimeError("synthetic publication failure")
        if failure == "publication":
            monkeypatch.setattr(store, "_publish", fail_publish)
        with DemandWorkerPool({"s": store}, {"s": lambda tile: np.ones(geometry.shape, np.uint8)},
                              ["cpu"], slots_per_gpu=1) as pool:
            if failure == "accumulator":
                pool.accumulator_processes[0].terminate()
                pool.accumulator_processes[0].join(5)
            deadline = time.monotonic() + 20
            with pytest.raises(RuntimeError, match="exited unexpectedly|synthetic publication"):
                while pool.busy and time.monotonic() < deadline:
                    pool.pump()
                    time.sleep(.005)
            assert not store.ready(origin)

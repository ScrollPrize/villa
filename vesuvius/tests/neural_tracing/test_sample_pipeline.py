"""CPU loading regressions extracted from ea829d388, plus shared-source integration."""
from dataclasses import fields, replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch

from tests.neural_tracing.oriented_blur_reference import _anisotropic_blur_3d
from tests.neural_tracing.test_fiber_trace_3d import _loader, _loader_with_mapping, _straight_fiber
from vesuvius.neural_tracing.fiber_trace_2d.sampling import Vc3dCoordinateSampler
from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    FiberTrace3DConfig, FiberTrace3DLoader, FiberTrace3DSharedTensorSlots, config_from_mapping, load_config,
    _gaussian_kernel1d, _normalize_image,
    collate_fiber_trace_3d_samples,
)
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import (
    DistributedSampleIndex, EvaluationSampleIndex, SampleQueueSource, FiberTrace3DSampleQueue,
    FiberTrace3DTrainingBatchPipeline,
)
from vesuvius.neural_tracing.fiber_trace_3d.queued_batches import QueuedBatchView


def test_resumed_rank_local_pipeline_preserves_samples_and_lease_boundary():
    loader = _loader_with_mapping({"augment_enabled": False, "batch_size": 2,
                                   "patch_shape_zyx": [8, 8, 8]})
    mapping = DistributedSampleIndex(batch_size=2, world_size=2, rank=1, start_step=3)
    pipeline = FiberTrace3DTrainingBatchPipeline(
        config_path=loader.config, index_map=mapping, batch_size=2,
        patch_shape_zyx=(8, 8, 8), batch_count=2, workers=2,
        load_affinity_workers=1, slot_count=4, initial_in_flight=2,
        sample_index_limit=None, sample_mode="random", profile=False)
    try:
        for ordinal in range(2):
            actual = next(pipeline)
            _assert_same_batch(actual, loader.load_batch(mapping(ordinal * 2)))
            assert actual.volume.data_ptr() == pipeline.transport.volumes[(ordinal * 2) % 4].data_ptr()
            with pytest.raises(RuntimeError, match="not released"):
                next(pipeline)
            pipeline.release_current()
        with pytest.raises(StopIteration):
            next(pipeline)
    finally:
        pipeline.close()


def test_evaluation_mapping_keeps_explicit_indices_and_unaligned_offset():
    mapping = EvaluationSampleIndex(3, (1, 4, 9), 17)
    assert [mapping(i) for i in range(9)] == [20, 21, 22, 29, 30, 31, 44, 45, 46]


def _assert_same_batch(actual, expected):
    for field in fields(expected):
        if field.name == "profile_timings_ms":
            continue
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(b, torch.Tensor):
            assert torch.equal(a, b), field.name
        else:
            assert a == b, field.name


def test_shared_views_reuse_workers_across_tiny_train_test_train_windows():
    owner, loader = _queue(slots=3)
    pids = [p.pid for p in owner.pipeline.processes]
    train = QueuedBatchView(owner, "train", ((31, 32), (33, 34)), limit=1)
    try:
        _assert_same_batch(next(train), loader.load_batch(31))
        train.release_current()
        # The final partial test batch wraps the physical ring. The labels,
        # masks and stream indices still belong to their canonical samples.
        test = QueuedBatchView(owner, "test", ((70, 71), (72,)))
        try:
            _assert_same_batch(next(test), loader.load_batch(70))
            partial = next(test)
            expected = collate_fiber_trace_3d_samples([loader.load_sample(72)])
            _assert_same_batch(partial, expected)
        finally:
            test.close()
        train.set_limit(2)
        _assert_same_batch(next(train), loader.load_batch(33))
        assert [p.pid for p in owner.pipeline.processes] == pids
    finally:
        train.close()
        owner.close()


def test_failed_sample_surfaces_and_owner_stops_without_leaking_slots():
    loader = _loader_with_mapping({"patch_shape_zyx": [8, 8, 8]})
    owner = FiberTrace3DSampleQueue(
        sources={"train": SampleQueueSource(loader.config, sample_mode="invalid")},
        workers=2, slot_count=2, initial_in_flight=1)
    try:
        ticket = owner.submit("train", (0,))
        with pytest.raises(RuntimeError, match="sample_mode"):
            _await(owner, lambda: owner.batch(ticket))
    finally:
        owner.close()
    assert owner.pipeline.closed


def _real_paris4_loader(tmp_path):
    volume = os.environ.get("VC_QUEUE_TEST_VOLUME")
    fiber = os.environ.get("VC_QUEUE_TEST_FIBER")
    if not volume or not fiber:
        pytest.skip("set VC_QUEUE_TEST_VOLUME and VC_QUEUE_TEST_FIBER for local Paris4")
    import vesuvius.neural_tracing.fiber_trace_3d.loader as module
    assert Path(module.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2] / "src")
    if os.environ.get("FIBER_COOPERATIVE_NATIVE_BUILD"):
        import vc.volume
        assert Path(vc.volume.__file__).resolve().is_relative_to(
            Path(os.environ["FIBER_COOPERATIVE_NATIVE_BUILD"]).resolve())
    # The local cache has one concrete array; the legacy dataset opener expects
    # a group with a numeric level. Expose the same array read-only as level0,
    # without fetching the cache's absent advertised higher pyramid levels.
    root = tmp_path / "ct.zarr"
    root.mkdir()
    (root / "0").symlink_to(Path(volume), target_is_directory=True)
    shape = json.loads((Path(volume) / ".zarray").read_text())["shape"]
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (root / ".zattrs").write_text(json.dumps({"multiscales": [{
        "version": "0.4", "axes": [{"name": a, "type": "space"} for a in "zyx"],
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]}]
    }]}))
    (root / "metadata.json").write_text(json.dumps({
        "type": "vol", "uuid": "paris4-queue-fixture", "name": "Paris4 level1",
        "width": shape[2], "height": shape[1], "slices": shape[0], "voxelsize": 4.8,
        "scan": {"voxelsize": 4.8}}))
    config = config_from_mapping({
        "datasets": [{"base_volume_path": str(root), "base_volume_scale": 0,
                      "fiber_paths": [fiber], "fiber_transform": np.diag([.5, .5, .5, 1.]).tolist()}],
        "batch_size": 2, "patch_shape_zyx": [32, 32, 32], "cp_margin_voxels": 3,
        "geometry_backend": "native", "value_augmentation_backend": "native",
        "volume_cache_memory_mib": 64, "volume_io_threads": 1,
        "volume_decode_threads": 1, "volume_sample_threads": 1,
        "volume_cache_offline": True, "augment_enabled": True,
        "augment_rotation_degrees": 10., "augment_scale_min": .95, "augment_scale_max": 1.05,
        "augment_contrast_min": .8, "augment_contrast_max": 1.2,
        "augment_gamma_min": .8, "augment_gamma_max": 1.2,
        "augment_blur_sigma": .5, "augment_noise_std": .01,
        "augment_smooth_displacement_mode": "3d",
        "augment_smooth_displacement_amplitude_zyx": [1., .75, .5],
        "augment_smooth_displacement_probability": 1.,
    })
    return FiberTrace3DLoader(config)


def test_real_paris4_loader_queue_matches_serial_native_path(tmp_path):
    loader = _real_paris4_loader(tmp_path)
    config = loader.config
    # Test uses the actual annotation at the selected local pyramid lattice,
    # not a translated synthetic line. No normals/manifests are fetched.
    raw = loader.load_raw_sample(0, sample_mode="flat", profile=True)
    assert torch.count_nonzero(raw.sample.volume), "real annotated crop is absent from local CT"
    assert raw.sample.profile_timings_ms["geometry_native_used"] == 1
    assert raw.sample.profile_timings_ms["vc_sample_workers"] == 1
    expected = loader.load_batch(0, sample_mode="flat", profile=True)
    assert expected.profile_timings_ms["value_native_transform_used"] == 2
    owner = FiberTrace3DSampleQueue(sources={
        "train": SampleQueueSource(config, sample_mode="flat"),
        "test": SampleQueueSource(replace(config, augment_enabled=False), sample_mode="flat")},
        workers=2, slot_count=4, initial_in_flight=2, profile=True)
    try:
        for source in ("train", "test", "train"):
            ticket = owner.submit(source, (0, 1))
            actual = _await(owner, lambda: owner.batch(ticket))
            serial = expected if source == "train" else FiberTrace3DLoader(
                replace(config, augment_enabled=False)).load_batch(0, sample_mode="flat")
            _assert_same_batch(actual, serial)
            owner.release(ticket)
    finally:
        owner.close()
    print(f"real Paris4 queue: {os.environ['VC_QUEUE_TEST_FIBER']}; "
          f"{os.environ['VC_QUEUE_TEST_VOLUME']}; exact train/test/train batches")


@pytest.mark.parametrize("selection", (None, "native", "auto"))
def test_preprocessing_defaults_and_explicit_selection(tmp_path, selection):
    raw = {"datasets": [{"base_volume_path": "unused", "fiber_paths": []}]}
    if selection is not None:
        raw.update(geometry_backend=selection, value_augmentation_backend=selection)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    configs = [config_from_mapping(raw), load_config(path)]
    if selection is None:
        configs.append(FiberTrace3DConfig(datasets=tuple(raw["datasets"])))
    for config in configs:
        assert config.geometry_backend == (selection or "auto")
        assert config.value_augmentation_backend == (selection or "auto")


def test_real_paris4_default_preprocessing_ignores_native_availability(tmp_path, monkeypatch):
    import vc.volume

    native_loader = _real_paris4_loader(tmp_path)
    config = replace(native_loader.config, geometry_backend="eager", value_augmentation_backend="torch")
    loader = FiberTrace3DLoader(config)
    expected = loader.load_batch(0, sample_mode="flat", profile=True)
    for name in ("generate_fiber3d_coords", "fiber3d_value_transform", "anisotropic_blur_3d"):
        monkeypatch.delattr(vc.volume, name, raising=False)
    actual = FiberTrace3DLoader(config).load_batch(0, sample_mode="flat", profile=True)
    _assert_same_batch(actual, expected)
    assert actual.profile_timings_ms["geometry_native_used"] == 0
    assert actual.profile_timings_ms["value_native_transform_used"] == 0


@pytest.mark.parametrize("url,expected", (
    ("http://example.invalid/ct.zarr", "http://example.invalid/ct.zarr"),
    ("https://example.invalid/ct.zarr", "https://example.invalid/ct.zarr"),
    ("s3://example-bucket/ct.zarr", "https://example-bucket.s3.amazonaws.com/ct.zarr"),
))
@pytest.mark.parametrize("cache_root", (None, "legacy-cache-suggestion"))
def test_remote_sampler_uses_one_argument_open_url(monkeypatch, url, expected, cache_root):
    import vc.volume

    calls = []
    sentinel = object()

    class Volume:
        @staticmethod
        def open_url(url):
            calls.append(url)
            return sentinel

    monkeypatch.setattr(vc.volume, "Volume", Volume)
    sampler = Vc3dCoordinateSampler(url, level=0, cache_root=cache_root)
    assert sampler.volume is sentinel
    assert calls == [expected]


def test_real_paris4_http_sampler_matches_local_binding(tmp_path):
    # Serve actual cached CT; no public network or AWS authentication required.
    loader = _real_paris4_loader(tmp_path)
    root = Path(loader.config.datasets[0]["base_volume_path"])
    cache_config = tmp_path / "vc-settings"
    cache_config.mkdir()
    cache = tmp_path / "remote-cache"
    (cache_config / "cache.json").write_text(json.dumps({"version": 1, "root": str(cache)}))

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(root.parent)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Isolate the process-wide cache before the native runtime opens it.
        result = subprocess.run([sys.executable, "-c", """
import json, sys
import numpy as np
from pathlib import Path
from vesuvius.neural_tracing.fiber_trace_2d.sampling import Vc3dCoordinateSampler
root, url, fiber, cache = sys.argv[1:]
point = np.asarray(json.loads(Path(fiber).read_text())["control_points"][0]["position"], np.float32)
coords = (point[::-1] * .5).reshape(1, 1, 3)
valid = np.ones((1, 1), dtype=bool)
options = dict(level=0, cache_root="legacy-unused", sample_threads=1,
               io_threads=1, decode_threads=1, cache_budget_bytes=64 * 1024**2)
local = Vc3dCoordinateSampler(root, **options).sample_coords(coords, valid)
remote = Vc3dCoordinateSampler(url, **options)
actual = remote.sample_coords(coords, valid)
assert Path(remote.volume.remote_cache_root) == Path(cache)
assert local.valid_mask.all() and np.count_nonzero(local.image)
np.testing.assert_array_equal(actual.valid_mask, local.valid_mask)
np.testing.assert_array_equal(actual.image, local.image)
print("actual Paris4 HTTP/local samples identical; cache isolated")
""", str(root), f"http://127.0.0.1:{server.server_port}/{root.name}",
            os.environ["VC_QUEUE_TEST_FIBER"], str(cache)],
            env={**os.environ, "VC3D_CONFIG_DIR": str(cache_config), "AWS_EC2_METADATA_DISABLED": "true"},
            capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        print(result.stdout.strip())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

def _queue(slots=4):
    loader = _loader_with_mapping({"augment_enabled": False, "batch_size": 2,
                                   "patch_shape_zyx": [8, 8, 8]})
    owner = FiberTrace3DSampleQueue(sources={
        "train": SampleQueueSource(loader.config),
        "historical": SampleQueueSource(loader.config),
        "test": SampleQueueSource(loader.config)}, workers=2,
        slot_count=slots, initial_in_flight=2, load_affinity_workers=1)
    return owner, loader


def _await(owner, function):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        owner.poll()
        value = function()
        if value is not None:
            return value
        time.sleep(0.005)
    raise AssertionError("sample queue stalled")


def test_shared_pool_switches_sources_without_recreating_workers_or_changing_seeds():
    owner, loader = _queue()
    pids = [process.pid for process in owner.pipeline.processes]
    try:
        for source, start in (("test", 11), ("train", 0), ("test", 11), ("train", 2)):
            ticket = owner.submit(source, range(start, start + 2))
            actual = _await(owner, lambda: owner.batch(ticket))
            expected = loader.load_batch(start, sample_mode="random", device="cpu")
            assert actual.stream_indices.tolist() == [start, start + 1]
            torch.testing.assert_close(actual.volume, expected.volume)
            owner.release(ticket)
        assert [process.pid for process in owner.pipeline.processes] == pids
    finally:
        owner.close()


def test_split_raw_and_value_augmentation_matches_composed_sample() -> None:
    loader = _loader(augment_enabled=True)
    combined = loader.load_sample(5, profile=False)
    staged = loader.augment_raw_sample(loader.load_raw_sample(5, profile=False), profile=False)

    for name in (
        "stream_index",
        "data_index",
        "record_index",
        "fiber_path",
        "control_point_index",
        "target_mode",
    ):
        assert getattr(staged, name) == getattr(combined, name)
    for name in (
        "volume",
        "cp_local_zyx",
        "crop_origin_zyx",
        "target_segment_starts_zyx",
        "target_segment_ends_zyx",
        "target_segment_bbox_lo_zyx",
        "target_segment_bbox_hi_zyx",
        "target_tangent_zyx",
    ):
        assert torch.equal(getattr(staged, name), getattr(combined, name)), name
    assert staged.valid_mask is combined.valid_mask is None
    assert staged.profile_timings_ms is None


def test_shared_tensor_slots_round_trip_and_reject_stale_generation() -> None:
    loader = _loader_with_mapping({"patch_shape_zyx": [8, 8, 8]})
    slots = FiberTrace3DSharedTensorSlots(
        start_index=5,
        slot_count=2,
        patch_shape_zyx=(8, 8, 8),
    )
    raw = loader.load_raw_sample(5)
    payload = slots.encode(5, raw, final=False)
    restored = slots.decode(5, payload, final=False)
    assert torch.equal(restored.sample.volume, raw.sample.volume)
    assert restored.sample.valid_mask is raw.sample.valid_mask is None
    assert restored.sample.volume.data_ptr() == slots.volumes[0].data_ptr()
    assert torch.equal(restored.sample.target_tangent_zyx, raw.sample.target_tangent_zyx)

    replacement = loader.load_raw_sample(7)
    slots.encode(7, replacement, final=False)
    with pytest.raises(RuntimeError, match="generation mismatch"):
        slots.decode(5, payload, final=False)


def test_shared_tensor_slots_expose_zero_copy_contiguous_batch() -> None:
    loader = _loader_with_mapping({"patch_shape_zyx": [8, 8, 8]})
    slots = FiberTrace3DSharedTensorSlots(
        start_index=3,
        slot_count=4,
        patch_shape_zyx=(8, 8, 8),
    )
    expected = []
    for index in (3, 4):
        raw = loader.load_raw_sample(index)
        slots.encode(index, raw, final=False)
        expected.append(raw.sample.volume)
    volumes = slots.batch_tensors([3, 4])
    assert volumes.shape == (2, 1, 8, 8, 8)
    assert volumes.data_ptr() == slots.volumes[0].data_ptr()
    assert torch.equal(volumes[0], expected[0])
    assert torch.equal(volumes[1], expected[1])
    with pytest.raises(RuntimeError, match="crosses or is not contiguous"):
        slots.batch_tensors([4, 6])


def test_distributed_sample_index_maps_rank_local_batches() -> None:
    index_map = DistributedSampleIndex(
        batch_size=8,
        world_size=2,
        rank=1,
        start_step=3,
    )
    assert [index_map(index) for index in (0, 7, 8, 15)] == [56, 63, 72, 79]


def test_shared_tensor_slots_validate_mapped_stream_index() -> None:
    loader = _loader_with_mapping({"patch_shape_zyx": [8, 8, 8]})
    index_map = DistributedSampleIndex(batch_size=1, world_size=2, rank=1)
    slots = FiberTrace3DSharedTensorSlots(
        start_index=0,
        slot_count=2,
        patch_shape_zyx=(8, 8, 8),
        stream_index_map=index_map,
    )
    raw = loader.load_raw_sample(index_map(0))
    payload = slots.encode(0, raw, final=False)
    restored = slots.decode(0, payload, final=False)
    assert restored.sample.stream_index == 1
    with pytest.raises(RuntimeError, match="sample/index mismatch"):
        slots.encode(1, raw, final=False)


def test_3d_config_preserves_volume_io_decode_and_sample_threads() -> None:
    config = config_from_mapping(
        {
            "batch_size": 1,
            "patch_shape_zyx": [16, 16, 16],
            "volume_io_threads": 2,
            "volume_decode_threads": 3,
            "volume_sample_threads": 1,
            "_array_records": [
                {
                    "volume": np.zeros((16, 16, 16), dtype=np.float32),
                    "fiber": _straight_fiber(),
                    "volume_path": "synthetic",
                }
            ],
        }
    )

    assert config.volume_io_threads == 2
    assert config.volume_decode_threads == 3
    assert config.volume_sample_threads == 1


@pytest.mark.parametrize("mode", ("none", "1d", "2d", "3d"))
@pytest.mark.parametrize("tile_depth", (1, 3, 8))
def test_tiled_geometry_is_bitwise_identical_to_eager_geometry(
    mode: str, tile_depth: int
) -> None:
    common = {
        "geometry_backend": "eager",
        "augment_smooth_displacement_mode": mode,
        "augment_smooth_displacement_amplitude_zyx": [1.0, 0.75, 0.5],
        "augment_smooth_displacement_control_spacing_zyx": [8.0, 8.0, 8.0],
        "augment_smooth_displacement_probability": 1.0,
    }
    eager = _loader_with_mapping(common)
    tiled = _loader_with_mapping({**common, "geometry_tile_depth": tile_depth})
    record, _record_index, cp_index = eager.descriptor_for_sample_index(2)
    params = eager._sample_augment_params(record, cp_index, 2)

    eager_geometry = eager._build_geometry_maps(params, device=torch.device("cpu"))
    tiled_geometry = tiled._build_geometry_maps(params, device=torch.device("cpu"))

    assert torch.equal(
        tiled_geometry.backward_source_zyx,
        eager_geometry.backward_source_zyx,
    )

    fused_geometry = tiled._build_geometry_maps(
        params,
        device=torch.device("cpu"),
        volume_spacing_base=float(record.volume_spacing_base),
        base_shape_zyx=record.base_shape_zyx,
    )
    assert eager_geometry.backward_source_zyx is not None
    expected_base = np.ascontiguousarray(
        eager_geometry.backward_source_zyx.numpy() * float(record.volume_spacing_base)
    )
    base_shape = np.asarray(record.base_shape_zyx, dtype=np.float32)
    expected_valid = np.isfinite(expected_base).all(axis=-1)
    for axis in range(3):
        expected_valid &= (expected_base[..., axis] >= 0.0) & (
            expected_base[..., axis] <= float(base_shape[axis] - 1.0)
        )
    assert fused_geometry.backward_source_zyx is None
    assert np.array_equal(fused_geometry.backward_source_base_zyx, expected_base)
    assert np.array_equal(fused_geometry.valid_base, expected_valid)


def test_tiled_geometry_is_bitwise_identical_for_training_patch_shape() -> None:
    common = {
        "geometry_backend": "eager",
        "patch_shape_zyx": [128, 128, 128],
        "augment_smooth_displacement_mode": "3d",
        "augment_smooth_displacement_amplitude_zyx": [2.0, 2.0, 2.0],
        "augment_smooth_displacement_control_spacing_zyx": [16.0, 16.0, 16.0],
        "augment_smooth_displacement_probability": 1.0,
    }
    eager = _loader_with_mapping(common)
    tiled = _loader_with_mapping({**common, "geometry_tile_depth": 8})
    record, _record_index, cp_index = eager.descriptor_for_sample_index(2)
    params = eager._sample_augment_params(record, cp_index, 2)

    eager_geometry = eager._build_geometry_maps(params, device=torch.device("cpu"))
    tiled_geometry = tiled._build_geometry_maps(params, device=torch.device("cpu"))

    assert torch.equal(
        tiled_geometry.backward_source_zyx,
        eager_geometry.backward_source_zyx,
    )


@pytest.mark.parametrize("mode", ("none", "1d", "2d", "3d"))
def test_native_geometry_matches_eager_coordinates(mode: str) -> None:
    native_module = pytest.importorskip("vc.volume")
    if not hasattr(native_module, "generate_fiber3d_coords"):
        pytest.skip("volume-cartographer binding lacks native Fiber3D geometry")
    common = {
        "patch_shape_zyx": [17, 15, 13],
        "augment_smooth_displacement_mode": mode,
        "augment_smooth_displacement_amplitude_zyx": [1.0, 0.75, 0.5],
        "augment_smooth_displacement_control_spacing_zyx": [8.0, 8.0, 8.0],
        "augment_smooth_displacement_probability": 1.0,
    }
    eager = _loader_with_mapping({**common, "geometry_backend": "eager"})
    native = _loader_with_mapping({**common, "geometry_backend": "native"})
    record, _record_index, cp_index = eager.descriptor_for_sample_index(2)
    params = eager._sample_augment_params(record, cp_index, 2)

    eager_geometry = eager._build_geometry_maps(
        params,
        device=torch.device("cpu"),
        volume_spacing_base=float(record.volume_spacing_base),
        base_shape_zyx=record.base_shape_zyx,
    )
    timings: dict[str, float] = {}
    native_geometry = native._build_geometry_maps(
        params,
        device=torch.device("cpu"),
        volume_spacing_base=float(record.volume_spacing_base),
        base_shape_zyx=record.base_shape_zyx,
        profile_timings_ms=timings,
    )
    assert eager_geometry.backward_source_zyx is not None
    expected = np.ascontiguousarray(
        eager_geometry.backward_source_zyx.numpy()
        * float(record.volume_spacing_base)
    )
    assert native_geometry.backward_source_base_zyx is not None
    error = np.abs(native_geometry.backward_source_base_zyx - expected)
    assert float(error.max()) <= 1.0e-3
    assert float(error.mean()) <= 1.0e-5
    assert native_geometry.valid_base is not None
    base_shape = np.asarray(record.base_shape_zyx, dtype=np.float32)
    expected_valid = np.isfinite(expected).all(axis=-1)
    for axis in range(3):
        expected_valid &= (expected[..., axis] >= 0.0) & (
            expected[..., axis] <= float(base_shape[axis] - 1.0)
        )
    assert np.array_equal(native_geometry.valid_base, expected_valid)
    assert timings["geometry_native_used"] == 1.0


def test_native_geometry_rejects_stale_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    import vc.volume

    monkeypatch.delattr(vc.volume, "generate_fiber3d_coords", raising=False)
    loader = _loader_with_mapping({"geometry_backend": "native"})
    record, _record_index, cp_index = loader.descriptor_for_sample_index(0)
    params = loader._sample_augment_params(record, cp_index, 0)
    with pytest.raises(RuntimeError, match="requires a rebuilt volume-cartographer"):
        loader._build_geometry_maps(
            params,
            device=torch.device("cpu"),
            volume_spacing_base=float(record.volume_spacing_base),
            base_shape_zyx=record.base_shape_zyx,
        )


def test_geometry_auto_prefers_native_and_falls_back_for_stale_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vc.volume

    if not hasattr(vc.volume, "generate_fiber3d_coords"):
        pytest.skip("volume-cartographer binding lacks native Fiber3D geometry")
    loader = _loader_with_mapping({"geometry_backend": "auto"})
    record, _record_index, cp_index = loader.descriptor_for_sample_index(0)
    params = loader._sample_augment_params(record, cp_index, 0)
    native_timings: dict[str, float] = {}
    native_geometry = loader._build_geometry_maps(
        params,
        device=torch.device("cpu"),
        volume_spacing_base=float(record.volume_spacing_base),
        base_shape_zyx=record.base_shape_zyx,
        profile_timings_ms=native_timings,
    )
    assert native_geometry.backward_source_base_zyx is not None
    assert native_timings["geometry_native_used"] == 1.0

    monkeypatch.delattr(vc.volume, "generate_fiber3d_coords")
    eager_timings: dict[str, float] = {}
    eager_geometry = loader._build_geometry_maps(
        params,
        device=torch.device("cpu"),
        volume_spacing_base=float(record.volume_spacing_base),
        base_shape_zyx=record.base_shape_zyx,
        profile_timings_ms=eager_timings,
    )
    assert eager_geometry.backward_source_zyx is not None
    assert eager_timings["geometry_native_used"] == 0.0


def test_native_geometry_preserves_sampled_patch() -> None:
    native_module = pytest.importorskip("vc.volume")
    if not hasattr(native_module, "generate_fiber3d_coords"):
        pytest.skip("volume-cartographer binding lacks native Fiber3D geometry")
    common = {
        "patch_shape_zyx": [17, 15, 13],
        "augment_smooth_displacement_mode": "3d",
        "augment_smooth_displacement_amplitude_zyx": [1.0, 0.75, 0.5],
        "augment_smooth_displacement_control_spacing_zyx": [8.0, 8.0, 8.0],
        "augment_smooth_displacement_probability": 1.0,
        "volume_sample_threads": 1,
    }
    eager = _loader_with_mapping({**common, "geometry_backend": "eager"})
    native = _loader_with_mapping({**common, "geometry_backend": "native"})
    eager_raw = eager.load_raw_sample(2, profile=True)
    native_raw = native.load_raw_sample(2, profile=True)
    assert native_raw.sample.valid_mask is eager_raw.sample.valid_mask is None
    difference = torch.abs(
        native_raw.sample.volume.to(torch.int16)
        - eager_raw.sample.volume.to(torch.int16)
    )
    mismatch = float(torch.count_nonzero(difference)) / float(difference.numel())
    assert mismatch <= 1.0e-4
    assert int(difference.max()) <= 1


@pytest.mark.parametrize(
    ("shape", "axis"),
    (
        ((17, 19, 23), np.asarray([0.31, 0.83, -0.46], dtype=np.float32)),
        ((1, 13, 11), np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
        ((11, 13, 17), np.asarray([0.8999, 0.436096, 0.0], dtype=np.float32)),
        ((11, 13, 17), np.asarray([0.9001, 0.435683, 0.0], dtype=np.float32)),
    ),
)
def test_native_anisotropic_blur_matches_torch_reference(
    shape: tuple[int, int, int], axis: np.ndarray
) -> None:
    native_module = pytest.importorskip("vc.volume")
    blur = getattr(native_module, "anisotropic_blur_3d", None)
    if blur is None:
        pytest.skip("volume-cartographer binding lacks native anisotropic blur")
    source = torch.randn(shape, generator=torch.Generator().manual_seed(23))
    source_before = source.clone()
    expected = _anisotropic_blur_3d(
        source,
        axis_zyx=axis,
        sigma_along=1.5,
        sigma_across=0.25,
    )
    actual_array = blur(source.numpy(), axis, 1.5, 0.25)
    actual = torch.from_numpy(actual_array)
    scale = max(1.0, float(torch.max(torch.abs(source))))
    error = torch.abs(actual - expected)
    assert float(error.max()) <= 1.0e-4 * scale
    assert float(error.mean()) <= 3.0e-6 * scale
    assert actual.shape == source.shape
    assert actual.dtype == torch.float32
    assert actual_array.flags.c_contiguous
    assert torch.equal(source, source_before)


def test_native_anisotropic_blur_matches_full_training_cube() -> None:
    native_module = pytest.importorskip("vc.volume")
    blur = getattr(native_module, "anisotropic_blur_3d", None)
    if blur is None:
        pytest.skip("volume-cartographer binding lacks native anisotropic blur")
    source = torch.randn((128, 128, 128), generator=torch.Generator().manual_seed(41))
    axis = np.asarray([0.31, 0.83, -0.46], dtype=np.float32)
    expected = _anisotropic_blur_3d(
        source,
        axis_zyx=axis,
        sigma_along=1.5,
        sigma_across=0.25,
    )
    actual = torch.from_numpy(blur(source.numpy(), axis, 1.5, 0.25))
    scale = max(1.0, float(torch.max(torch.abs(source))))
    error = torch.abs(actual - expected)
    assert float(error.max()) <= 1.0e-4 * scale
    assert float(error.mean()) <= 3.0e-6 * scale


def test_native_axis_aligned_blur_matches_torch_isotropic_reference() -> None:
    native_module = pytest.importorskip("vc.volume")
    blur = getattr(native_module, "anisotropic_blur_3d", None)
    if blur is None:
        pytest.skip("volume-cartographer binding lacks native anisotropic blur")
    source = torch.randn((33, 35, 37), generator=torch.Generator().manual_seed(43))
    sigma = 0.5
    kernel = _gaussian_kernel1d(sigma, device=source.device)
    radius = int((kernel.numel() - 1) // 2)
    data = source.view(1, 1, *source.shape)
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = int(kernel.numel())
        padding = [0, 0, 0, 0, 0, 0]
        padding[(2 - axis) * 2] = radius
        padding[(2 - axis) * 2 + 1] = radius
        data = torch.nn.functional.conv3d(
            torch.nn.functional.pad(data, padding, mode="replicate"),
            kernel.view(*shape),
        )
    expected = data[0, 0]
    actual = torch.from_numpy(
        blur(
            source.numpy(),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            sigma,
            sigma,
        )
    )
    error = torch.abs(actual - expected)
    assert float(error.max()) <= 2.0e-6
    assert float(error.mean()) <= 2.0e-7


@pytest.mark.parametrize("mask_kind", ("all", "partial", "none", "constant"))
def test_native_value_transform_matches_torch_reference(mask_kind: str) -> None:
    native_module = pytest.importorskip("vc.volume")
    transform = getattr(native_module, "fiber3d_value_transform", None)
    if transform is None:
        pytest.skip("volume-cartographer binding lacks native value transform")
    shape = (17, 19, 23)
    generator = torch.Generator().manual_seed(53)
    source = torch.randn(shape, generator=generator)
    if mask_kind == "all":
        valid = torch.ones(shape, dtype=torch.bool)
    elif mask_kind == "partial":
        valid = torch.rand(shape, generator=generator) > 0.2
    elif mask_kind == "none":
        valid = torch.zeros(shape, dtype=torch.bool)
    else:
        source = torch.full(shape, 3.25, dtype=torch.float32)
        valid = torch.rand(shape, generator=generator) > 0.2
    expected = _normalize_image(source, valid, "zscore")
    center = expected[valid].mean() if bool(valid.any()) else expected.mean()
    expected = (expected - center) * 1.7 + center + 0.13
    valid_values = expected[valid] if bool(valid.any()) else expected.reshape(-1)
    lo = valid_values.min()
    hi = valid_values.max()
    scaled = ((expected - lo) / (hi - lo).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    expected = torch.pow(scaled, 0.71) * (hi - lo) + lo
    actual = torch.from_numpy(
        transform(source.numpy(), valid.numpy(), "zscore", 1.7, 0.13, 0.71)
    )
    error = torch.abs(actual - expected)
    scale = max(1.0, float(torch.max(torch.abs(expected))))
    assert float(error.max()) <= 1.0e-6 * scale
    assert float(error.mean()) <= 1.0e-7 * scale
    assert actual.shape == source.shape
    assert actual.dtype == torch.float32


def test_native_value_transform_validates_contract() -> None:
    native_module = pytest.importorskip("vc.volume")
    transform = getattr(native_module, "fiber3d_value_transform", None)
    if transform is None:
        pytest.skip("volume-cartographer binding lacks native value transform")
    source = np.zeros((3, 4, 5), dtype=np.float32)
    valid = np.ones(source.shape, dtype=np.bool_)
    with pytest.raises(ValueError, match="shape must match"):
        transform(source, valid[:, :, :-1], "zscore", 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="unsupported"):
        transform(source, valid, "other", 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="must be positive"):
        transform(source, valid, "zscore", 0.0, 0.0, 1.0)


@pytest.mark.parametrize("kind", ("ramp", "impulse", "high_contrast"))
def test_native_anisotropic_blur_structured_inputs(kind: str) -> None:
    native_module = pytest.importorskip("vc.volume")
    blur = getattr(native_module, "anisotropic_blur_3d", None)
    if blur is None:
        pytest.skip("volume-cartographer binding lacks native anisotropic blur")
    shape = (17, 19, 23)
    if kind == "ramp":
        z, y, x = torch.meshgrid(
            *(torch.linspace(-2.0, 2.0, length) for length in shape),
            indexing="ij",
        )
        source = (z + 0.3 * y - 0.7 * x).contiguous()
    elif kind == "impulse":
        source = torch.zeros(shape, dtype=torch.float32)
        source[8, 9, 11] = 10.0
    else:
        source = 100.0 * torch.randn(
            shape, generator=torch.Generator().manual_seed(47)
        )
    axis = np.asarray([-0.52, 0.17, 0.84], dtype=np.float32)
    expected = _anisotropic_blur_3d(
        source,
        axis_zyx=axis,
        sigma_along=1.5,
        sigma_across=0.25,
    )
    actual = torch.from_numpy(blur(source.numpy(), axis, 1.5, 0.25))
    scale = max(1.0, float(torch.max(torch.abs(source))))
    error = torch.abs(actual - expected)
    assert float(error.max()) <= 1.0e-4 * scale
    assert float(error.mean()) <= 3.0e-6 * scale


def test_native_anisotropic_blur_validates_input_and_zero_sigma() -> None:
    native_module = pytest.importorskip("vc.volume")
    blur = getattr(native_module, "anisotropic_blur_3d", None)
    if blur is None:
        pytest.skip("volume-cartographer binding lacks native anisotropic blur")
    source = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    assert np.array_equal(blur(source, axis, 0.0, 0.0), source)
    with pytest.raises(ValueError, match="positive finite norm"):
        blur(source, np.zeros((3,), dtype=np.float32), 1.0, 0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        blur(source, axis, -1.0, 0.0)
    reversed_source = source[:, :, ::-1]
    assert np.array_equal(blur(reversed_source, axis, 0.0, 0.0), reversed_source)


def test_value_augmentation_backend_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    import vc.volume

    if not hasattr(vc.volume, "fiber3d_value_transform"):
        pytest.skip("volume-cartographer binding lacks native value transform")
    common = {
        "patch_shape_zyx": [17, 15, 13],
        "augment_blur_sigma": 0.5,
        "augment_anisotropic_blur_probability": 1.0,
        "augment_anisotropic_blur_sigma_along": 1.5,
        "augment_anisotropic_blur_sigma_across": 0.25,
    }
    native_loader = _loader_with_mapping(
        {**common, "value_augmentation_backend": "native"}
    )
    raw = native_loader.load_raw_sample(2, profile=True)
    native_sample = native_loader.augment_raw_sample(raw, profile=True)
    native_unprofiled = native_loader.augment_raw_sample(raw, profile=False)
    assert native_sample.profile_timings_ms is not None
    assert native_sample.profile_timings_ms["value_native_transform_used"] == 1.0
    assert torch.equal(native_sample.volume, native_unprofiled.volume)

    monkeypatch.delattr(vc.volume, "fiber3d_value_transform")
    auto_loader = _loader_with_mapping(
        {**common, "value_augmentation_backend": "auto"}
    )
    auto_sample = auto_loader.augment_raw_sample(raw, profile=True)
    assert auto_sample.profile_timings_ms is not None
    assert auto_sample.profile_timings_ms["value_native_transform_used"] == 0.0
    explicit_loader = _loader_with_mapping(
        {**common, "value_augmentation_backend": "native"}
    )
    with pytest.raises(RuntimeError, match="requires a rebuilt volume-cartographer"):
        explicit_loader.augment_raw_sample(raw, profile=True)

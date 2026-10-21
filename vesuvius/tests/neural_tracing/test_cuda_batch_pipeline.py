"""Async preparation extraction: CPU lifecycle plus opt-in manager-owned CUDA."""
from dataclasses import fields, replace
import os
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from tests.neural_tracing.test_fiber_trace_3d import _loader, _TARGET_MODE_CP_ONLY
from tests.neural_tracing.test_sample_pipeline import _real_paris4_loader
from vesuvius.neural_tracing.fiber_trace_3d import cuda_batch_pipeline as module
from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    FiberTrace3DBatch, collate_fiber_trace_3d_samples,
)
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import (
    DistributedSampleIndex, FiberTrace3DTrainingBatchPipeline,
    FiberTrace3DSampleQueue, SampleQueueSource,
)
from vesuvius.neural_tracing.fiber_trace_3d.queued_batches import QueuedBatchView
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets


def _wait(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("asynchronous preparation stalled")


def _same_tensors(actual, expected):
    for field in fields(expected):
        if field.name == "profile_timings_ms":
            continue
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(b, torch.Tensor):
            torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
        else:
            assert a == b, field.name


@pytest.mark.parametrize("device,depth", [("cpu", 2), ("cuda", 0), ("cuda", -1)])
def test_constructor_rejects_invalid_device_or_depth_without_cuda(device, depth):
    with pytest.raises(ValueError):
        module.FiberTrace3DCudaBatchPipeline(iter(()), config=None,
            device=torch.device(device), output_stride=1, depth=depth)


# Regression extracted from the later cooperative integration suite.
@pytest.mark.parametrize("device,index", [("cuda", 3), ("cuda:1", 1)])
def test_cuda_preparation_resolves_device_before_background_thread(monkeypatch, device, index):
    stream_devices = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *, device: stream_devices.append(device))
    monkeypatch.setattr(module.threading.Thread, "start", lambda self: None)
    pipeline = module.FiberTrace3DCudaBatchPipeline(
        iter(()), config=None, device=torch.device(device), output_stride=4)
    assert pipeline.device == torch.device("cuda", index)
    assert stream_devices == [pipeline.device]


def test_nonblocking_cpu_transfer_preserves_tensors_and_cpu_index_metadata():
    loader = _loader(augment_enabled=False)
    source = materialize_targets(loader.load_batch(0), loader.config)
    actual = source.to("cpu", non_blocking=True)
    _same_tensors(actual, source)
    for name in ("target_modes", "target_segment_offsets", "target_segment_counts",
                 "target_segment_bbox_lo_zyx", "target_segment_bbox_hi_zyx"):
        assert getattr(actual, name) is getattr(source, name)


def test_default_target_stride_is_exactly_stride_one():
    loader = _loader(augment_enabled=False)
    raw = loader.load_batch(0)
    _same_tensors(materialize_targets(raw, loader.config),
                  materialize_targets(raw, loader.config, output_stride=1))


class _Source:
    """Small deterministic batch source; CPU-only coordination harness."""
    def __init__(self, batch, count=8, failure=None):
        self.batch, self.count, self.failure = batch, count, failure
        self.admitted = self.released = 0
        self.closed = False

    def __next__(self):
        if self.failure is not None:
            raise self.failure
        if self.admitted >= self.count:
            raise StopIteration
        index = self.admitted
        self.admitted += 1
        return replace(self.batch, stream_indices=torch.full_like(self.batch.stream_indices, index))

    def release_current(self):
        self.released += 1

    def close(self):
        self.closed = True


@pytest.fixture
def fake_cuda(monkeypatch):
    """No driver use: exercise real thread/queue lifecycle with CPU tensors."""
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self, stream):
            pass
        def synchronize(self):
            pass
        def elapsed_time(self, other):
            return 0.

    from contextlib import nullcontext
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(FiberTrace3DBatch, "pin_memory", lambda batch: replace(batch,
        volume=batch.volume.clone(), valid_mask=batch.valid_mask.clone()))
    monkeypatch.setattr(FiberTrace3DBatch, "to", lambda batch, device, **kwargs: batch)


def test_async_pipeline_bounds_preparation_pauses_and_preserves_order(fake_cuda):
    loader = _loader(augment_enabled=False)
    source = _Source(loader.load_batch(0))
    pipeline = module.FiberTrace3DCudaBatchPipeline(source, config=loader.config,
        device=torch.device("cuda:0"), output_stride=1, depth=2)
    try:
        _wait(lambda: source.released == 2)
        assert source.admitted == 2
        pipeline.pause()
        admitted = source.admitted
        time.sleep(.03)
        assert source.admitted == admitted
        pipeline.resume()
        for index in range(source.count):
            batch = next(pipeline)
            assert batch.stream_indices.tolist() == [index] * loader.config.batch_size
            assert batch.presence_target is not None
        with pytest.raises(StopIteration):
            next(pipeline)
    finally:
        pipeline.close()
        pipeline.close()
    assert source.closed and not pipeline._thread.is_alive()
    assert source.released == source.count
    with pytest.raises(RuntimeError, match="closed"):
        next(pipeline)


@pytest.mark.parametrize("stage", ["source", "pin", "targets"])
def test_async_pipeline_propagates_failure_and_closes(fake_cuda, monkeypatch, stage):
    loader = _loader(augment_enabled=False)
    failure = ValueError("injected preparation failure")
    source = _Source(loader.load_batch(0), failure=failure if stage == "source" else None)
    def fail(*args, **kwargs):
        raise failure
    if stage == "pin":
        monkeypatch.setattr(FiberTrace3DBatch, "pin_memory", fail)
    if stage == "targets":
        monkeypatch.setattr(module, "materialize_targets", fail)
    pipeline = module.FiberTrace3DCudaBatchPipeline(source, config=loader.config,
        device=torch.device("cuda:0"), output_stride=1)
    try:
        with pytest.raises(RuntimeError, match="preparation failed") as error:
            next(pipeline)
        assert error.value.__cause__ is failure
    finally:
        pipeline.close()
    assert source.closed and not pipeline._thread.is_alive()
    assert source.released == (0 if stage == "source" else 1)


def test_close_unblocks_a_waiting_source(fake_cuda):
    loader = _loader(augment_enabled=False)
    entered, closed = threading.Event(), threading.Event()
    class BlockingSource(_Source):
        def __next__(self):
            entered.set()
            if not closed.wait(10):
                raise AssertionError("source shutdown did not unblock preparation")
            raise StopIteration
        def close(self):
            super().close()
            closed.set()
    source = BlockingSource(loader.load_batch(0))
    pipeline = module.FiberTrace3DCudaBatchPipeline(source, config=loader.config,
        device=torch.device("cuda:0"), output_stride=1)
    assert entered.wait(5)
    pipeline.close()
    assert not pipeline._thread.is_alive()


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_cuda_real_paris4_async_queue_matches_serial_targets(tmp_path, stride):
    if os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1":
        pytest.skip("run in an externally manager-reserved CUDA process")
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), "a manager GPU lease is required"
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    loader = _real_paris4_loader(tmp_path)
    mapping = DistributedSampleIndex(batch_size=2, world_size=2, rank=1, start_step=3)
    cpu_source = FiberTrace3DTrainingBatchPipeline(config_path=loader.config,
        index_map=mapping, batch_size=2, patch_shape_zyx=(32, 32, 32),
        batch_count=4, workers=2, load_affinity_workers=1, slot_count=4,
        initial_in_flight=2, sample_index_limit=None, sample_mode="random", profile=True)
    pipeline = None
    try:
        pipeline = module.FiberTrace3DCudaBatchPipeline(cpu_source, config=loader.config,
            device=device, output_stride=stride, depth=2)
        # Synchronize the independent reference stream in advance: no test-only
        # synchronization is inserted between queue submission and consumption.
        expected = [materialize_targets(loader.load_batch(mapping(i * 2),
            sample_mode="random").to(device), loader.config, output_stride=stride)
            for i in range(4)]
        torch.cuda.synchronize(device)
        pipeline.pause()
        pipeline.resume()
        for reference in expected:
            actual = next(pipeline)
            _same_tensors(actual, reference)
            assert actual.volume.is_cuda
            assert actual.profile_timings_ms["cuda_pipeline_pinned_bytes"] > 0
        with pytest.raises(StopIteration):
            next(pipeline)
    finally:
        pipeline.close() if pipeline is not None else cpu_source.close()
    assert not pipeline._thread.is_alive()
    assert cpu_source.pipeline.closed
    print(f"Paris4 async CUDA stride={stride}: exact augmented inputs, masks, metadata and targets")


def test_cuda_real_shared_train_test_train_keeps_cpu_workers(tmp_path):
    if os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1":
        pytest.skip("run in an externally manager-reserved CUDA process")
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), "a manager GPU lease is required"
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    loader = _real_paris4_loader(tmp_path)
    test_loader = type(loader)(replace(loader.config, augment_enabled=False))
    owner = FiberTrace3DSampleQueue(sources={
        "train": SampleQueueSource(loader.config, sample_mode="flat"),
        "test": SampleQueueSource(test_loader.config, sample_mode="flat")},
        workers=2, slot_count=3, initial_in_flight=2)
    pids = [process.pid for process in owner.pipeline.processes]
    try:
        for phase in ("train", "test", "train"):
            view = QueuedBatchView(owner, phase, ((0, 1), (2,)))
            config = loader.config if phase == "train" else test_loader.config
            serial_loader = loader if phase == "train" else test_loader
            pipeline = module.FiberTrace3DCudaBatchPipeline(view, config=config,
                device=device, output_stride=1, depth=2)
            try:
                for indices in ((0, 1), (2,)):
                    cpu = collate_fiber_trace_3d_samples([
                        serial_loader.load_sample(index, sample_mode="flat") for index in indices])
                    reference = materialize_targets(cpu.to(device), config)
                    _same_tensors(next(pipeline), reference)
                with pytest.raises(StopIteration):
                    next(pipeline)
            finally:
                pipeline.close()
            assert not owner.pipeline.closed
            assert [process.pid for process in owner.pipeline.processes] == pids
    finally:
        owner.close()
    print("Paris4 asynchronous train/test/train: exact batches, partial ring wraps, persistent CPU PIDs")


# Target-stride regressions extracted unchanged from3f0aeecaa.
def test_materialize_targets_uses_native_coarse_lattice_without_losing_cp() -> None:
    loader = _loader(augment_enabled=False)
    raw = loader.load_batch(0)
    coarse = materialize_targets(raw, loader.config, output_stride=4)

    assert coarse.presence_target is not None
    assert coarse.presence_mask is not None
    assert coarse.direction_indices_bzyx is not None
    assert coarse.presence_target.shape[-3:] == (4, 4, 4)
    assert coarse.presence_mask.shape[-3:] == (4, 4, 4)
    assert int(coarse.direction_indices_bzyx.shape[0]) > 0
    assert torch.all(coarse.direction_indices_bzyx[:, 1:] >= 0)
    assert torch.all(coarse.direction_indices_bzyx[:, 1:] < 4)
    assert float(coarse.presence_target.sum()) > 0.0


def test_native_coarse_targets_use_ceil_shape_and_conservative_validity() -> None:
    loader = _loader(augment_enabled=False)
    raw = loader.load_batch(0)
    batch_size = int(raw.volume.shape[0])
    volume = torch.zeros((batch_size, 1, 9, 10, 11), dtype=raw.volume.dtype)
    valid = torch.ones_like(volume, dtype=torch.bool)
    valid[0, 0, 1, 1, 1] = False
    cp = torch.tensor(
        [[1.2, 1.2, 1.2], [8.2, 9.2, 10.2]],
        dtype=torch.float32,
    )[:batch_size]
    raw = replace(
        raw,
        volume=volume,
        valid_mask=valid,
        cp_local_zyx=cp,
        target_modes=torch.full_like(raw.target_modes, _TARGET_MODE_CP_ONLY),
    )
    config = replace(loader.config, presence_negative_edge_margin_voxels=1)
    coarse = materialize_targets(raw, config, output_stride=4)

    assert coarse.presence_target.shape[-3:] == (3, 3, 3)
    assert not bool(coarse.presence_mask[0, 0, 0, 0, 0])
    assert bool(coarse.presence_mask[0, 0, 1, 1, 1])
    assert not bool(coarse.presence_mask[0, 0, 2, 2, 2])
    assert tuple(coarse.direction_indices_bzyx.shape[1:]) == (4,)


def test_native_coarse_cp_fallback_keeps_fractional_sub_stride_radius() -> None:
    loader = _loader(augment_enabled=False)
    raw = loader.load_batch(0)
    cp = torch.tensor(
        [[2.2, 3.3, 4.4], [14.7, 15.2, 15.8]],
        dtype=torch.float32,
    )
    raw = replace(
        raw,
        cp_local_zyx=cp,
        target_modes=torch.full_like(raw.target_modes, _TARGET_MODE_CP_ONLY),
    )
    config = replace(loader.config, presence_radius_voxels=0.1)
    coarse = materialize_targets(raw, config, output_stride=4)

    indices = {tuple(int(v) for v in row) for row in coarse.direction_indices_bzyx}
    assert (0, 0, 0, 1) in indices
    assert (1, 3, 3, 3) in indices
    assert float(coarse.presence_target.sum()) >= 2.0

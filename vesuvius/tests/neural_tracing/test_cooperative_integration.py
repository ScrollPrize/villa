"""CPU-only integration: no CUDA contexts or driver inventory queries."""

from dataclasses import dataclass, replace
import json
import os
import queue
import time
from types import SimpleNamespace

import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.cooperative_resources import MemoryCalibration
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_worker import FiberWorkerFactory


@dataclass(frozen=True)
class CpuFactory(FiberWorkerFactory):
    def __call__(self, rank, grant, state):
        from tests.neural_tracing.test_fiber_trace_3d import _loader_with_mapping
        from vesuvius.neural_tracing.fiber_trace_3d import train
        assert not torch.cuda.is_initialized()
        loader = _loader_with_mapping({"patch_shape_zyx": [8, 8, 8], "batch_size": 2,
                                      "augment_enabled": False})
        loader_type = type(loader)

        def make_loader(config):
            object.__setattr__(config, "_array_records", loader.config._array_records)
            return loader_type(config)

        train.load_config = lambda _: loader.config
        train.config_from_mapping = lambda mapping, **kwargs: loader.config
        train.FiberTrace3DLoader = make_loader
        train.build_fiber_trace_3d_model = lambda _: torch.nn.Conv3d(1, 7, 1)
        return super().__call__(rank, grant, state)


class CpuBroker:
    effective_batch_size = 5
    end_index = 15

    def __init__(self):
        from tests.neural_tracing.test_fiber_trace_3d import _loader_with_mapping
        self.loader = _loader_with_mapping({"patch_shape_zyx": [8, 8, 8], "batch_size": 2,
                                           "augment_enabled": False})
        self.frontier = 0
        self.commits = []
        self.pipeline = SimpleNamespace(end_index=15, shared_workers=1, closed=False, processes=[])

    def poll(self):
        pass

    def begin_test(self, step):
        assert step * self.effective_batch_size == self.frontier

    def end_test(self, step):
        assert step * self.effective_batch_size == self.frontier

    def select_test_source(self, step, source):
        assert step * self.effective_batch_size == self.frontier
        assert source == "historical"

    def test_batch(self, step, indices):
        assert step * self.effective_batch_size == self.frontier
        return self.batch(indices)

    def release_test(self, step, indices):
        assert step * self.effective_batch_size == self.frontier

    def batch(self, indices):
        from vesuvius.neural_tracing.fiber_trace_3d.loader import collate_fiber_trace_3d_samples
        return collate_fiber_trace_3d_samples([self.loader.load_sample(index) for index in indices])

    def commit(self, step):
        assert step * 5 == self.frontier + 5
        self.commits.append(step)
        self.frontier = step * 5

    def set_worker_limit(self, value):
        self.limit = value

    def workers_within_limit(self):
        return True

    def close(self):
        self.pipeline.closed = True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("resume_during_test", [False, True])
def test_cpu_cohort_queue_training_and_restore_on_different_rank_count(monkeypatch, tmp_path, device, resume_during_test):
    from vesuvius.neural_tracing.fiber_trace_3d import cooperative_cohort
    from vesuvius.neural_tracing.fiber_trace_3d.cooperative_logging import WriterHub
    if device == "cuda":
        # Explicit opt-in: never commandeer GPUs during the ordinary CPU suite.
        devices = os.environ.get("FIBER_COOPERATIVE_TEST_GPUS", "").split(",")
        if len(devices) != 2 or not all(devices):
            pytest.skip("set FIBER_COOPERATIVE_TEST_GPUS to two explicitly available device IDs")
        unit = 1024**3
        first_grant = {gpu: 10 * unit for gpu in devices}
        second_grant = {devices[0]: 9 * unit}
        # At most 6 GiB in the allocator, plus 4 GiB context/NCCL reserve.
        calibration = MemoryCalibration(4 * unit, unit, 0, 0, 4 * unit, 2,
                                        ((1, 5 * unit), (2, 6 * unit)))
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        first_grant = {"GPU-fake-a": 30, "GPU-fake-b": 30}
        second_grant = {"GPU-fake-a": 20}
        calibration = MemoryCalibration(10, 10, 0, 0, 0, 2, ((1, 20), (2, 30)))
    monkeypatch.setattr(cooperative_cohort, "FiberWorkerFactory", CpuFactory)
    config = {"training": {"device": device, "bounded_gradient_accumulation": True,
        "ddp_broadcast_buffers": False, "sync_batchnorm": False,
        "optimizer_global_batch_size": 5, "max_steps": 3, "initialization_seed": 17,
        "run_path": str(tmp_path), "run_datestr": "test", "tensorboard_enabled": True,
        "scalar_log_interval": 1,
        "ema_decays": [0.99, 0.999],
        "test_interval": 100, "checkpoint_interval": 100, "sample_vis_interval": 1,
        "sample_pipeline_workers": 2}}
    if resume_during_test:
        # Four original test groups: first two are completed on two ranks,
        # remaining groups resume on one rank through the real test RPC.
        config["test_datasets"] = [{"name": "in-memory-fixture"}]
        config["historical_test_datasets"] = [{"name": "in-memory-fixture"}]
        config["training"]["test_control_points"] = 8
        config["training"]["test_trace2cp_enabled"] = False
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    broker = CpuBroker()
    state = None
    events = []

    class Writer:
        def add_text(self, tag, *args, **kwargs):
            events.append(("text", tag))

        def add_scalar(self, tag, value, step):
            events.append((tag, step))

        def add_image(self, tag, image, step, **kwargs):
            events.append(("image", tag, step))

        def flush(self):
            pass

        def close(self):
            events.append(("closed", None))

    hub = WriterHub(lambda path, enabled: Writer())

    def wait_until(cohort, predicate):
        deadline = time.monotonic() + 40
        while not predicate():
            hub.poll()
            cohort.poll()
            assert time.monotonic() < deadline, "CPU cohort failed to progress"
            time.sleep(0.002)

    for devices, target in ((first_grant, 0 if resume_during_test else 1), (second_grant, 3)):
        cohort = cooperative_cohort.FiberCohort(config_path=path,
            grant={"cpu_cores": 3, "gpus": devices}, state=state, broker=broker,
            calibration=calibration, writer_factory=hub.proxy_factory)
        old_state = state
        try:
            cohort.authorize()
            wait_until(cohort, lambda: cohort.ready)
            while broker.frontier // 5 < target:
                cohort.start_unit()
                wait_until(cohort, lambda: cohort.idle)
            cohort.drain()
            wait_until(cohort, lambda: cohort.closed)
            state = cohort.state
            payload = state.load()
            assert payload["progress"]["step"] == target
            assert payload["progress"]["samples_seen"] == target * 5
            assert payload["progress"]["optimizer_global_batch_size"] == 5
            assert all(tensor.device.type == "cpu" for tensor in payload["model"].values())
            assert all(track["updates"] == target for track in payload["ema"]["tracks"].values())
            if resume_during_test and target == 0:
                assert len(payload["progress"]["pending_test"]["rows"]) == 2
        finally:
            if not cohort.closed:
                cohort.abort()
            if old_state is not None:
                old_state.close()
    actual = state.load()
    state.close()
    reference_broker = CpuBroker()
    reference = cooperative_cohort.FiberCohort(config_path=path,
        grant={"cpu_cores": 2, "gpus": {next(iter(second_grant)): next(iter(first_grant.values()))}}, state=None,
        broker=reference_broker, calibration=calibration,
        training_overrides={"tensorboard_enabled": False})
    try:
        reference.authorize()
        wait_until(reference, lambda: reference.ready)
        while not reference.done:
            reference.start_unit()
            wait_until(reference, lambda: reference.idle)
        reference.drain()
        wait_until(reference, lambda: reference.closed)
        expected = reference.state.load()
        for name, tensor in expected["model"].items():
            torch.testing.assert_close(actual["model"][name], tensor, rtol=1e-5, atol=1e-6)
        for key, values in expected["optimizer"]["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(actual["optimizer"]["state"][key][name], value, rtol=1e-5, atol=1e-6)
        for name, track in expected["ema"]["tracks"].items():
            assert actual["ema"]["tracks"][name]["updates"] == track["updates"] == 3
            for key, value in track["model"].items():
                torch.testing.assert_close(actual["ema"]["tracks"][name]["model"][key], value, rtol=1e-5, atol=1e-6)
    finally:
        if not reference.closed:
            reference.abort()
        else:
            reference.state.close()
    hub.close()
    assert broker.commits == [1, 2, 3]
    assert [event[1] for event in events if event[0] == "train/loss_total"] == [5, 10, 15]
    assert events.count(("text", "config/json")) == 1
    assert events.count(("closed", None)) == 1
    if resume_during_test:
        assert any(event[0] == "image" for event in events)
        for prefix in ("test_historical", "test_ema_0p99", "test_historical_ema_0p999"):
            assert sum(event[0] == prefix + "/loss_total" for event in events) == 1
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("device,index", [("cuda", 3), ("cuda:1", 1)])
def test_cuda_preparation_resolves_device_before_background_thread(monkeypatch, device, index):
    from vesuvius.neural_tracing.fiber_trace_3d import sample_pipeline
    stream_devices = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *, device: stream_devices.append(device))
    monkeypatch.setattr(sample_pipeline.threading.Thread, "start", lambda self: None)
    pipeline = sample_pipeline.FiberTrace3DCudaBatchPipeline(
        iter(()), config=None, device=torch.device(device), output_stride=4)
    assert pipeline.device == torch.device("cuda", index)
    assert stream_devices == [pipeline.device]


def test_queue_publication_failure_is_not_silently_counted_as_delivery():
    from vesuvius.neural_tracing.fiber_trace_3d.cooperative_cohort import FiberCohort
    cohort = FiberCohort.__new__(FiberCohort)
    cohort._publication_errors = queue.SimpleQueue()
    cohort._publication_failed(TypeError("cannot pickle batch"), None)
    with pytest.raises(RuntimeError, match="publication failed: TypeError: cannot pickle batch"):
        cohort._check_publication()


def test_cohort_constructor_failure_reaps_children(monkeypatch):
    from villa_gpu_manager import cohort as process_cohort, processes
    from vesuvius.neural_tracing.fiber_trace_3d.cooperative_cohort import FiberCohort
    events = []

    class GatedCohort:
        processes = [{"pid": 123}]

        def __init__(self, *args, **kwargs):
            events.append("spawned")

        def abort(self):
            events.append("joined")

    def fail_monitor(*args, **kwargs):
        raise RuntimeError("monitor setup failed")

    monkeypatch.setattr(process_cohort, "ProcessCohort", GatedCohort)
    monkeypatch.setattr(processes, "NvidiaMonitor", fail_monitor)
    broker = SimpleNamespace(frontier=0, effective_batch_size=128)
    calibration = MemoryCalibration(10, 10, 0, 0, 0, 1, ((1, 20),))
    with pytest.raises(RuntimeError, match="monitor setup failed"):
        FiberCohort(config_path="unused", grant={"cpu_cores": 1, "gpus": {"GPU-fake": 30}},
                    state=None, broker=broker, calibration=calibration,
                    calibration_observations=[{"gpu_name": "fake", "torch_version": "test", "cuda_version": "test"}])
    assert events == ["spawned", "joined"]


def test_driver_memory_checks_only_owned_processes_and_fails_closed():
    from vesuvius.neural_tracing.fiber_trace_3d.cooperative_cohort import FiberCohort
    cohort = FiberCohort.__new__(FiberCohort)
    cohort.grant = {"gpus": {"GPU-fake": 100}}
    cohort._driver_peaks = {"GPU-fake": 0}
    cohort._gpu_pids = {123}
    cohort._authorized_at = time.monotonic()
    monitor = SimpleNamespace(poll=lambda pids: None, last_snapshot_at=time.monotonic(),
        last_snapshot={"GPU-fake": {"processes": {123: 100, 456: 900}}})
    cohort._memory_monitor = monitor
    cohort._check_driver_memory()
    monitor.last_snapshot["GPU-fake"]["processes"][123] = 101
    with pytest.raises(RuntimeError, match="exceeded VRAM grant"):
        cohort._check_driver_memory()
    monitor.last_snapshot = {}
    with pytest.raises(RuntimeError, match="disappeared"):
        cohort._check_driver_memory()
    monitor.last_snapshot_at = time.monotonic() - 21
    with pytest.raises(RuntimeError, match="cannot verify"):
        cohort._check_driver_memory()

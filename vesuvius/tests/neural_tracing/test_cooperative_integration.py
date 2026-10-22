"""Real Paris4 training with persistent CPU loading and replaceable cohorts."""

from dataclasses import dataclass, replace
from copy import deepcopy
import json
import os
import pickle
import queue
import time
from types import SimpleNamespace

import pytest
import torch

from tests.neural_tracing.test_resumable_training import _bounded_config, _finish, _assert_state_close
from tests.neural_tracing.test_training_queue_integration import _Writer
from vesuvius.neural_tracing.fiber_trace_3d import train, cooperative_cohort
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_resources import (
    EffectiveBatchSampleIndex, MemoryCalibration, partition_grant,
)
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_setup import (
    load_calibration, prepare_settings, derive_request, task_priority,
)
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_training import launch_settings
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_worker import FiberWorkerFactory
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_logging import WriterHub
from vesuvius.neural_tracing.fiber_trace_3d.loader import config_from_mapping
from vesuvius.neural_tracing.fiber_trace_3d.sample_broker import CanonicalSampleBroker
from vesuvius.neural_tracing.fiber_trace_3d.memory_calibration import (
    calibration_from_observations, calibration_report_path, StartupMemoryCalibration,
)
from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
from vesuvius.neural_tracing.fiber_trace_3d.training_state import config_fingerprint


@dataclass(frozen=True)
class ReferenceFactory(FiberWorkerFactory):
    def __call__(self, rank, grant, state):
        # Same explicit numerical-comparison backend as the preceding extraction
        # tests; production defaults are not changed by this test adapter.
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.allow_tf32 = False
        from vesuvius.neural_tracing.fiber_trace_3d import train as worker_train
        def forbidden(*a, **kw):
            raise AssertionError("replacement consumer rebuilt persistent datasets")
        worker_train.FiberTrace3DLoader = forbidden
        return super().__call__(rank, grant, state)


def _cohort_equivalence(tmp_path, monkeypatch, phase, *, device="cpu", stride=1):
    raw, path = _bounded_config(tmp_path, device=device,
                                depth=2 if device.startswith("cuda") else 0)
    raw["model"]["output_stride"] = stride
    raw["training"].update(tensorboard_enabled=True, sample_vis_interval=0,
        test_sample_vis_interval=0)
    path.write_text(json.dumps(raw))
    expected, _, _ = _finish(path, physical=(5,))
    broker = CanonicalSampleBroker.create(config_path=path,
        loader_config=config_from_mapping(raw, config_dir=path.parent),
        start_step=0, max_steps=4, effective_batch_size=5, workers=2,
        slot_count=5, initial_in_flight=2)
    pids = tuple(p.pid for p in broker.pipeline.processes)
    metadata = pickle.loads(pickle.dumps(broker.dataset_metadata))
    assert metadata["train"].sample_count == broker.loaders["train"].sample_count
    assert not hasattr(metadata["train"], "records")
    writer = _Writer()
    hub = WriterHub(lambda *a, **k: writer)
    monkeypatch.setattr(cooperative_cohort, "FiberWorkerFactory", ReferenceFactory)
    if device.startswith("cuda"):
        devices = os.environ["FIBER_COOPERATIVE_TEST_GPUS"].split(",")
        assert len(devices) == 2
        unit = 1024**3
        calibration = MemoryCalibration(4 * unit, unit, 0, 0, 4 * unit, 2,
                                        ((1, 5 * unit), (2, 6 * unit)))
        first = {gpu: 10 * unit for gpu in devices}
        second = {devices[0]: 9 * unit}
    else:
        calibration = MemoryCalibration(10, 10, 0, 0, 0, 2, ((1, 20), (2, 30)))
        first = {"GPU-test-a": 30, "GPU-test-b": 30}
        second = {"GPU-test-a": 20}
    state = cohort = None

    def wait(predicate):
        deadline = time.monotonic() + 60
        while not predicate():
            hub.poll()
            cohort.poll()
            assert time.monotonic() < deadline, "real-data cohort failed to progress"
            time.sleep(.002)

    try:
        for stage, gpus in enumerate((first, second)):
            previous = state
            cohort = cooperative_cohort.FiberCohort(config_path=path,
                grant={"cpu_cores": 4, "gpus": gpus}, state=state, broker=broker,
                calibration=calibration, writer_factory=hub.proxy_factory)
            cohort.authorize()
            wait(lambda: cohort.ready)
            # Ready does not mean a test or optimizer step was already run.
            assert broker.frontier == (0 if stage == 0 else checkpoint_step * 5)
            if stage == 0 and phase == "before_test":
                cohort.drain()
            else:
                while not cohort.done:
                    cohort.start_unit()
                    if stage == 0 and phase == "during_update" and broker.frontier == 5:
                        cohort.drain()
                        break
                    wait(lambda: cohort.idle)
                    step = broker.frontier // 5
                    in_test = broker.test_step == step
                    if stage == 0 and ((phase == "update" and step == 1)
                            or (phase == "test" and step == 2 and in_test)):
                        cohort.drain()
                        break
                else:
                    assert cohort.done
                if cohort.done:
                    cohort.drain()
            wait(lambda: cohort.closed)
            state = cohort.state
            payload = state.load()
            checkpoint_step = payload["progress"]["step"]
            assert payload["progress"]["sample_frontier"] == broker.frontier
            assert tuple(p.pid for p in broker.pipeline.processes) == pids
            assert not any(t.is_cuda for t in payload["model"].values())
            if previous is not None:
                previous.close()
            cohort = None
        actual = state.load()
        _assert_state_close(actual["model"], expected["model"], rtol=8e-5, atol=8e-6)
        _assert_state_close(actual["optimizer"], expected["optimizer"], rtol=8e-5, atol=8e-6)
        assert actual["progress"]["sample_frontier"] == broker.frontier == 20
        assert broker.test_step is None
        assert not broker.queue._owned and not broker.queue._ready
    finally:
        if cohort is not None:
            cohort.abort()
        if state is not None:
            state.close()
        hub.close()
        broker.close()
    assert writer.closed
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize("phase", ["before_test", "test", "update", "during_update"])
def test_real_paris4_cohort_retains_cpu_datasets_and_resizes(tmp_path, monkeypatch, phase):
    with torch.backends.mkldnn.flags(enabled=False):
        _cohort_equivalence(tmp_path, monkeypatch, phase)


@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires manager reservation")
@pytest.mark.parametrize("phase,stride", [("before_test", 1), ("test", 2), ("during_update", 1)])
def test_cuda_real_paris4_cohort_resize_and_early_output(tmp_path, monkeypatch, phase, stride):
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS")
    with torch.backends.cudnn.flags(enabled=False, allow_tf32=False):
        _cohort_equivalence(tmp_path, monkeypatch, phase, device="cuda:0", stride=stride)


def test_queue_publication_failure_surfaces():
    cohort = cooperative_cohort.FiberCohort.__new__(cooperative_cohort.FiberCohort)
    cohort._publication_errors = queue.SimpleQueue()
    cohort._publication_failed(TypeError("cannot pickle batch"), None)
    with pytest.raises(RuntimeError, match="publication failed: TypeError"):
        cohort._check_publication()


def test_runtime_cohort_does_not_create_vram_monitor(monkeypatch, tmp_path):
    from villa_gpu_manager import cohort as process_cohort, processes
    events = []
    class Gated:
        processes = []
        def __init__(self, *a, **kw):
            events.append("spawn")
        def abort(self):
            events.append("join")
    monkeypatch.setattr(process_cohort, "ProcessCohort", Gated)
    monkeypatch.setattr(processes, "NvidiaMonitor", lambda *a, **k: pytest.fail("normal-training telemetry"))
    (tmp_path / "unused").write_text("{}")
    broker = SimpleNamespace(frontier=0, effective_batch_size=5, dataset_metadata={})
    cal = MemoryCalibration(10, 10, 0, 0, 0, 1, ((1, 20),))
    cohort = cooperative_cohort.FiberCohort(config_path=tmp_path / "unused",
        grant={"cpu_cores": 2, "gpus": {"GPU-test": 30}}, state=None,
        broker=broker, calibration=cal,
        calibration_observations=[{"gpu_name": "test", "torch_version": "test", "cuda_version": "test"}])
    cohort.gpu.abort()
    cohort._close_queues(abort=True)
    cohort._temp.cleanup()
    assert cohort._memory_monitor is None
    assert events == ["spawn", "join"]


def test_calibration_constructor_failure_reaps_gated_workers(monkeypatch, tmp_path):
    from villa_gpu_manager import cohort as process_cohort, processes
    events = []
    class Gated:
        processes = []
        def __init__(self, *a, **kw):
            events.append("spawn")
        def abort(self):
            events.append("join")
    monkeypatch.setattr(process_cohort, "ProcessCohort", Gated)
    def fail(*a, **kw):
        raise RuntimeError("monitor failed")
    monkeypatch.setattr(processes, "NvidiaMonitor", fail)
    (tmp_path / "unused").write_text("{}")
    broker = SimpleNamespace(frontier=0, effective_batch_size=5, dataset_metadata={})
    with pytest.raises(RuntimeError, match="monitor failed"):
        cooperative_cohort.FiberCohort(config_path=tmp_path / "unused",
            grant={"cpu_cores": 2, "gpus": {"GPU-test": 30}}, state=None,
            broker=broker, calibration=None, startup_microbatch_size=1)
    assert events == ["spawn", "join"]


def test_consumer_metadata_allows_only_allocation_local_changes(tmp_path):
    from tests.neural_tracing.test_sample_pipeline import _real_paris4_loader
    loader = _real_paris4_loader(tmp_path)
    metadata = pickle.loads(pickle.dumps(loader.dataset_metadata))
    config = replace(loader.config, batch_size=1, volume_io_threads=1,
                     volume_decode_threads=1, volume_sample_threads=1)
    assert metadata.for_config(config).sample_count == loader.sample_count
    with pytest.raises(ValueError, match="differs"):
        metadata.for_config(replace(config, seed=config.seed + 1))
    assert metadata.for_config(config).config.datasets is not metadata.config.datasets


def test_default_cuda_launch_uses_manager_without_fallback(monkeypatch, tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d import cooperative_training
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"training": {"device": "cuda"}}))
    monkeypatch.setattr(train, "training_units", lambda *a, **k: pytest.fail("unmanaged fallback"))
    def managed(*a, **kw):
        assert kw["socket"] is None
        raise ValueError("manager unavailable")
    monkeypatch.setattr(cooperative_training, "run_cooperative_training", managed)
    with pytest.raises(ValueError, match="manager unavailable"):
        train.run_training(config)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit, match="manager supervisor launches GPU workers"):
        train.run_training(config)


def test_normal_worker_never_reads_allocator_usage(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d.cooperative_worker import FiberTrainingWorker
    worker = FiberTrainingWorker.__new__(FiberTrainingWorker)
    worker.cuda = True
    worker.factory = SimpleNamespace(measure_startup_memory=False)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: pytest.fail("normal-training telemetry"))
    worker._check_memory()


def test_real_training_manager_pause_resize_disconnect_and_finish(tmp_path, monkeypatch):
    """Actual Unix protocol and real CPU training, without driver/GPU access."""
    import tempfile
    import threading
    from villa_gpu_manager.server import Server
    from villa_gpu_manager.client import Client
    from villa_gpu_manager.supervisor import Supervisor

    class DisconnectableClient(Client):
        offline = False
        def call(self, *a, **kw):
            if self.offline:
                raise ConnectionError("injected communication outage")
            return super().call(*a, **kw)

    raw, path = _bounded_config(tmp_path)
    raw["training"].update(tensorboard_enabled=False, sample_vis_interval=0,
                           test_sample_vis_interval=0)
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(cooperative_cohort, "FiberWorkerFactory", ReferenceFactory)
    broker = CanonicalSampleBroker.create(config_path=path,
        loader_config=config_from_mapping(raw, config_dir=path.parent), start_step=0,
        max_steps=4, effective_batch_size=5, workers=2, slot_count=5, initial_in_flight=2)
    pids = tuple(p.pid for p in broker.pipeline.processes)
    grant = {"cpu_cores": 4, "gpus": {"GPU-test-a": 30, "GPU-test-b": 30}}
    request = dict(max_cpu_cores=4, max_gpus=2, min_vram_bytes_per_gpu=20,
        max_vram_bytes_per_gpu=30, allowed_gpu_uuids=[], priority=0, exclusive=False)
    calibration = MemoryCalibration(10, 10, 0, 0, 0, 2, ((1, 20), (2, 30)))
    generations = []
    now = [0.0]
    stop = threading.Event()
    errors = []
    with tempfile.TemporaryDirectory(prefix="fiber-manager-", dir=tmp_path.parent) as directory:
        with Server(os.path.join(directory, "m", "s"), grant,
                    engine_options={"protected_processes": lambda: []}) as server:
            def serve():
                try:
                    while not stop.is_set():
                        server.step(.002)
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=serve)
            thread.start()
            client = DisconnectableClient(server.path)
            def factory(allocation, state):
                generations.append(allocation)
                return cooperative_cohort.FiberCohort(config_path=path, grant=allocation,
                    state=state, broker=broker, calibration=calibration)
            supervisor = Supervisor(client, label="real-paris4-test", request=request,
                cohort_factory=factory, clock=lambda: now[0], poll_interval=.01,
                reconnect_interval=.01, disconnect_grace_seconds=60,
                dispose_state=lambda state: state.close())
            def until(predicate):
                deadline = time.monotonic() + 60
                while not predicate():
                    assert not errors
                    now[0] += .01
                    supervisor.tick()
                    if supervisor.error is not None:
                        raise supervisor.error
                    assert time.monotonic() < deadline, "managed training did not progress"
                    time.sleep(.002)
            try:
                until(lambda: broker.frontier >= 5)
                client.pause(True)
                until(lambda: supervisor.cohort is None and supervisor.status == "paused")
                assert client.task["held"] == {"cpu_cores": 0, "gpus": {}}
                assert tuple(p.pid for p in broker.pipeline.processes) == pids
                client.set_request({**request, "max_gpus": 1, "max_vram_bytes_per_gpu": 20})
                client.pause(False)
                until(lambda: supervisor.status == "running")
                assert len(supervisor.grant["gpus"]) == 1
                client.offline = True
                supervisor.next_poll = 0
                until(lambda: not supervisor.connected)
                assert supervisor.cohort is not None  # grace retains authorized work
                now[0] += 61
                until(lambda: supervisor.cohort is None)
                held_step = broker.frontier
                assert supervisor.state is not None
                client.offline = False
                supervisor.next_poll = 0
                until(lambda: supervisor.finished)
                assert broker.frontier == 20 >= held_step
                assert client.task["terminal"]
                assert client.task["held"] == {"cpu_cores": 0, "gpus": {}}
                assert len(generations) >= 3
                assert tuple(p.pid for p in broker.pipeline.processes) == pids
            finally:
                if supervisor.cohort is not None:
                    supervisor.cohort.abort()
                if supervisor.state is not None:
                    supervisor.state.close()
                client.close()
                stop.set()
                thread.join(5)
                broker.close()
            assert not thread.is_alive() and not errors


@dataclass(frozen=True)
class LocalFailureFactory(ReferenceFactory):
    def __call__(self, rank, grant, state):
        worker = super().__call__(rank, grant, state)
        if rank == 1:
            from vesuvius.neural_tracing.fiber_trace_3d import train as worker_train
            def fail(*a, **kw):
                raise RuntimeError("originating rank-local forward failure")
            worker_train._forward_loss_statistics = fail
            worker_train._distributed_cleanup = lambda _: time.sleep(60)
        return worker


def test_real_local_failure_reported_before_blocking_cleanup(tmp_path, monkeypatch):
    raw, path = _bounded_config(tmp_path)
    raw["training"].update(tensorboard_enabled=False, sample_vis_interval=0,
                           test_sample_vis_interval=0)
    path.write_text(json.dumps(raw))
    broker = CanonicalSampleBroker.create(config_path=path,
        loader_config=config_from_mapping(raw, config_dir=path.parent), start_step=0,
        max_steps=4, effective_batch_size=5, workers=2, slot_count=5, initial_in_flight=2)
    monkeypatch.setattr(cooperative_cohort, "FiberWorkerFactory", LocalFailureFactory)
    cohort = cooperative_cohort.FiberCohort(config_path=path,
        grant={"cpu_cores": 4, "gpus": {"GPU-test-a": 30, "GPU-test-b": 30}},
        state=None, broker=broker,
        calibration=MemoryCalibration(10, 10, 0, 0, 0, 2, ((1, 20), (2, 30))))
    try:
        cohort.authorize()
        deadline = time.monotonic() + 30
        with pytest.raises(RuntimeError, match="originating rank-local forward failure"):
            while time.monotonic() < deadline:
                cohort.poll()
                if cohort.ready and cohort.idle:
                    cohort.start_unit()
                time.sleep(.002)
            pytest.fail("rank-local failure hidden behind collective cleanup")
    finally:
        cohort.abort()
        broker.close()
    assert cohort.gpu.closed


def configuration():
    raw = {"training": {"device": "cuda", "optimizer_global_batch_size": 128,
        "max_steps": 100, "sample_pipeline_workers": 32, "sample_pipeline_slots": 256,
        "sync_batchnorm": False, "ddp_broadcast_buffers": False}}
    fingerprint = config_fingerprint(raw)
    rows = [{"microbatch_size": size, "initialization_reserved_bytes": 20,
             "peak_reserved_bytes": 20 + 10 * size, "peak_allocated_bytes": 10 + 10 * size,
             "driver_peak_bytes": 30 + 10 * size, "gpu_name": "fake",
             "torch_version": "test", "cuda_version": "test", "world_size": 8,
             "config_fingerprint": fingerprint} for size in (1, 4)]
    raw["training"]["cooperative"] = {
        "socket": "/tmp/example.sock",
        "request": {"max_cpu_cores": 32, "max_gpus": 8, "min_vram_bytes_per_gpu": 50,
                    "max_vram_bytes_per_gpu": 200, "allowed_gpu_uuids": [], "priority": 0, "exclusive": False},
        "memory_calibration": calibration_from_observations(fingerprint, rows, reserve_bytes=10),
    }
    return raw


@pytest.mark.parametrize("override,expected", [(None, 107), ("2", 2)])
def test_probe_is_gated_by_exclusive_lease(monkeypatch, tmp_path, override, expected):
    from villa_gpu_manager import bootstrap, cohort, exclusive, processes
    from vesuvius.neural_tracing.fiber_trace_3d import memory_calibration as module, train
    raw = configuration()
    raw["training"]["cooperative"]["priority"] = 7
    monkeypatch.setattr(train, "_load_raw_config", lambda _: raw)
    grant = {"cpu_cores": 2, "gpus": {"GPU-a": 100}}
    monkeypatch.setattr(bootstrap, "ensure_manager", lambda _: ("fake", grant))
    events = []
    class Lease:
        def __init__(self, path, allocation, **kwargs):
            assert kwargs["priority"] == expected
            assert allocation == grant
        def __enter__(self):
            events.append("grant")
            return self
        def authorize(self):
            events.append("authorize")
        def check(self):
            events.append("check")
        def __exit__(self, *args):
            assert self.cohort.closed
            events.append("release")
    class FakeCohort:
        closed = False
        ready = True
        idle = True
        done = True
        processes = [{"pid": 123}]
        def __init__(self, factory, *args, **kwargs):
            assert events == ["grant"]
            assert factory.rendezvous.startswith("file://")
            events.append("spawn")
            self.state = SimpleNamespace(load=lambda: [{"gpu_uuid": "GPU-a"}], close=lambda: None)
        def poll(self):
            assert "authorize" in events
        def drain(self):
            self.closed = True
        def abort(self):
            self.closed = True
    class Monitor:
        def __init__(self, *args, **kwargs):
            self.last_snapshot = {"GPU-a": {"processes": {}, "total": 100, "used": 0}}
            self.last_snapshot_at = module.time.monotonic()
        def poll(self, pids):
            self.last_snapshot["GPU-a"]["processes"] = {p: 50 for p in pids}
        def close(self):
            pass
    monkeypatch.setattr(exclusive, "ExclusiveTask", Lease)
    monkeypatch.setattr(cohort, "ProcessCohort", FakeCohort)
    monkeypatch.setattr(processes, "NvidiaMonitor", Monitor)
    monkeypatch.setattr(module, "calibration_from_observations", lambda _, rows, **kw: rows)
    output = tmp_path / "report.json"
    args = [str(tmp_path / "config.json"), "--gpu-uuids", "GPU-a", "--microbatch-sizes", "1",
            "--cpu-workers", "2", "--vram-limit-bytes", "100", "--reserve-bytes", "10",
            "--output", str(output)]
    if override is not None:
        args += ["--priority", override]
    assert module.main(args) == 0
    assert events == ["grant", "spawn", "authorize", "check", "release"]
    assert json.loads(output.read_text())[0]["driver_peak_bytes"] == 50


def test_calibration_bounds_observations_and_does_not_extrapolate():
    settings = launch_settings(configuration())
    for size, peak in settings.calibration.measured_peaks:
        assert settings.calibration.estimated_bytes(size) >= peak + 10
    assert settings.calibration.microbatch_for(10**9) == 4
    assert settings.calibration.microbatch_for(49) == 0
    assert settings.batch_size == 128


def test_socket_default_and_override_precedence(monkeypatch, tmp_path):
    from villa_gpu_manager import paths
    from villa_gpu_manager.paths import default_socket_path
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(paths, "_listening", lambda _: False)
    monkeypatch.setattr(paths, "_writable_location", lambda _: False)
    raw = configuration()
    assert launch_settings(raw, socket="/tmp/override.sock").socket == "/tmp/override.sock"
    assert launch_settings(raw).socket == "/tmp/example.sock"
    del raw["training"]["cooperative"]["socket"]
    assert launch_settings(raw).socket == default_socket_path()
    assert default_socket_path() == str(tmp_path / ".local/run/villa-gpu-manager/manager.sock")
    with pytest.raises(ValueError, match="socket path"):
        launch_settings(raw, socket="")


def test_operational_controls_do_not_change_calibration_identity():
    raw = configuration()
    original = config_fingerprint(raw)
    raw["training"]["cooperative"]["request"]["priority"] = 10
    assert config_fingerprint(raw) == original
    assert launch_settings(raw).request["priority"] == 10
    raw["training"]["optimizer_global_batch_size"] = 64
    with pytest.raises(ValueError, match="different training config"):
        launch_settings(raw)


def test_cannot_supply_unrelated_peak_records():
    raw = configuration()
    raw["training"]["cooperative"]["memory_calibration"]["observations"][0]["driver_peak_bytes"] += 1
    with pytest.raises(ValueError, match="recorded observations"):
        launch_settings(raw)


def test_queue_slots_must_hold_current_effective_batch():
    raw = configuration()
    raw["training"]["sample_pipeline_slots"] = 64
    # Re-identify synthetic observations to test the slot constraint itself.
    identity = config_fingerprint(raw)
    calibration = raw["training"]["cooperative"]["memory_calibration"]
    calibration["config_fingerprint"] = identity
    for row in calibration["observations"]:
        row["config_fingerprint"] = identity
    with pytest.raises(ValueError, match="one effective batch"):
        launch_settings(raw)


def test_per_device_calibration_supports_more_replicas():
    raw = configuration()
    raw["training"]["cooperative"]["memory_calibration"]["observations"][1]["world_size"] = 1
    assert launch_settings(raw).request["max_gpus"] == raw["training"]["cooperative"]["request"]["max_gpus"]


def test_production_configuration_rejects_cpu():
    raw = configuration()
    raw["training"]["device"] = "cpu"
    identity = config_fingerprint(raw)
    calibration = raw["training"]["cooperative"]["memory_calibration"]
    calibration["config_fingerprint"] = identity
    for row in calibration["observations"]:
        row["config_fingerprint"] = identity
    with pytest.raises(ValueError, match="requires CUDA"):
        launch_settings(raw)


# cooperative resources


def calibration():
    return MemoryCalibration(fixed_bytes=100, bytes_per_sample=10,
                             preparation_bytes_per_sample=5, preparation_depth=2,
                             reserve_bytes=20, max_microbatch=16,
                             measured_peaks=((1, 120), (4, 180), (16, 420)))


def test_memory_grants_and_no_extrapolation():
    c = calibration()
    assert c.microbatch_for(139) == 0
    assert c.microbatch_for(140) == 1
    assert c.microbatch_for(199) == 1
    assert c.microbatch_for(200) == 4
    assert c.microbatch_for(10000) == 16
    with pytest.raises(ValueError, match="underestimates"):
        MemoryCalibration(10, 1, 0, 0, 0, 1, ((1, 100),))


def test_unequal_vram_exact_batch():
    grant = {"cpu_cores": 16, "gpus": {"GPU-c": 440, "GPU-a": 140, "GPU-b": 200}}
    partitions = partition_grant(grant, calibration(), step=7, effective_batch_size=128)
    assert [p.microbatch_size for p in partitions] == [1, 4, 16]
    assert [i for p in partitions for batch in p.batches() for i in batch] == list(range(896, 1024))
    assert partition_grant({"gpus": {}}, calibration(), step=0, effective_batch_size=128) == ()


def test_insufficient_grant_fails_not_silently_shrinks_batch():
    with pytest.raises(ValueError, match="single-sample"):
        partition_grant({"gpus": {"GPU-a": 30}}, calibration(), step=0, effective_batch_size=128)


@pytest.mark.parametrize("sizes,batch", [((1, 4, 16), 128), ((16,) * 8, 128), ((1, 1, 1), 2), ((3, 5), 17)])
def test_canonical_index_partial_and_zero_work_ranks(sizes, batch):
    maps = [EffectiveBatchSampleIndex(batch, sizes, rank, start_step=4) for rank in range(len(sizes))]
    for step in range(3):
        samples = []
        for mapping in maps:
            assert sum(mapping.batch_sizes) == mapping.local_samples
            samples.extend(mapping(step * mapping.local_samples + i) for i in range(mapping.local_samples))
        assert samples == list(range((step + 4) * batch, (step + 5) * batch))


# startup calibration


def observation(size=1, updates=0, tested=False):
    return {"microbatch_size": size, "initialization_reserved_bytes": 10,
            "peak_reserved_bytes": 30, "peak_allocated_bytes": 20,
            "driver_peak_bytes": 35, "gpu_name": "fake", "torch_version": "test",
            "cuda_version": "test", "config_fingerprint": "config", "world_size": 1,
            "gpu_uuid": "GPU-test", "optimizer_steps": updates,
            "initial_test_completed": tested}


def test_only_real_completed_updates_and_full_test_allow_size_transition():
    session = StartupMemoryCalibration("config", target_microbatch=4, reserve_bytes=5)
    session.observe([observation()], {"phase": "test"})
    session.observe([observation(updates=2, tested=True)], {"phase": "update"})
    assert not session.ready
    with pytest.raises(RuntimeError, match="incomplete"):
        session.complete_size({"gpus": {"GPU-test": 1000}})
    session.observe([observation(updates=3, tested=True)], {"phase": "update"})
    assert session.complete_size({"gpus": {"GPU-test": 1000}}) == 2
    assert session.report()["estimate"]["max_microbatch"] == 1
    assert session.report()["observations"][0]["evaluation_microbatch_size"] == 1
    session.observe([observation(size=2, updates=3)], {"phase": "update"})
    assert session.complete_size({"gpus": {"GPU-test": 1000}}) == 4
    assert session.report()["estimate"]["max_microbatch"] == 2


def test_no_extrapolated_size_is_published_when_capacity_prevents_ascent():
    session = StartupMemoryCalibration("config", target_microbatch=16, reserve_bytes=5)
    session.observe([observation()], {"phase": "test"})
    session.observe([observation(updates=3, tested=True)], {"phase": "update"})
    assert session.complete_size({"gpus": {"GPU-test": 100}}) is None
    report = session.report()
    assert report["estimate"]["max_microbatch"] == 1
    assert report["estimate"]["measured_peaks"] == ((1, 35),)
    assert report["observations"][0]["peak_reserved_bytes"] == 30
    assert report["observations"][0]["driver_peak_bytes"] == 35
    assert report["observations"][0]["evaluation_overhead_bytes"] == 35
    assert report["estimate"]["fixed_bytes"] == 45


def test_incomplete_test_or_missing_driver_measurement_cannot_publish():
    session = StartupMemoryCalibration("config", target_microbatch=1, reserve_bytes=5)
    session.observe([observation(updates=3)], {"phase": "update"})
    assert not session.ready
    with pytest.raises(RuntimeError, match="unmeasured"):
        session.report()
    row = observation(updates=3, tested=True)
    row["driver_peak_bytes"] = 0
    session.observe([row], {"phase": "test"})
    session.observe([row], {"phase": "update"})
    with pytest.raises(RuntimeError, match="driver"):
        session.complete_size({"gpus": {"GPU-test": 1000}})


@pytest.mark.parametrize("run_done", [False, True])
def test_real_run_startup_retains_broker_writer_and_handoff(monkeypatch, tmp_path, run_done):
    from copy import deepcopy
    from types import SimpleNamespace
    import torch
    from villa_gpu_manager import client, supervisor
    from vesuvius.neural_tracing.fiber_trace_3d import (
        cooperative_training as module, cooperative_setup, cooperative_cohort,
        cooperative_logging, sample_broker, loader, train,
    )
    raw = {"batch_size": 1, "training": {"device": "cuda", "max_steps": 100,
        "optimizer_global_batch_size": 8, "sample_pipeline_workers": 2,
        "sample_pipeline_slots": 8, "run_path": str(tmp_path)}}
    grant = {"cpu_cores": 2, "gpus": {"GPU-test": 20 * 1024**3}}
    request = {"max_cpu_cores": 2, "max_gpus": 1, "min_vram_bytes_per_gpu": 20 * 1024**3,
               "max_vram_bytes_per_gpu": 20 * 1024**3, "allowed_gpu_uuids": ["GPU-test"],
               "priority": 100, "exclusive": True}
    events = []
    def prepared(path, value, **kwargs):
        result = deepcopy(value)
        settings = result["training"].setdefault("cooperative", {})
        settings["socket"] = "test"
        settings["request"] = {**request, "exclusive": "memory_calibration" not in settings}
        if "memory_calibration" not in settings:
            settings["startup_calibration"] = True
        return result
    monkeypatch.setattr(cooperative_setup, "prepare_settings", prepared)
    monkeypatch.setattr(train, "_load_raw_config", lambda _: deepcopy(raw))
    monkeypatch.setattr(loader, "load_config", lambda _: object())
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    class Client:
        task = None
        task_id = "fake"
        def __init__(self, path):
            pass
        def close(self):
            pass
    monkeypatch.setattr(client, "Client", Client)
    class Hub:
        proxy_factory = object()
        def __init__(self, factory):
            events.append("writer")
        def poll(self):
            pass
        def close(self):
            events.append("writer_closed")
    monkeypatch.setattr(cooperative_logging, "WriterHub", Hub)
    broker = SimpleNamespace(close=lambda: events.append("broker_closed"))
    def make_broker(**kwargs):
        events.append("broker")
        return broker
    monkeypatch.setattr(sample_broker.CanonicalSampleBroker, "create", make_broker)
    handoff = SimpleNamespace(close=lambda: events.append("state_closed"))
    class Cohort:
        def __init__(self, **kwargs):
            self.options = kwargs
            assert kwargs["broker"] is broker
            assert kwargs["training_overrides"]["evaluation_microbatch_size"] == 1
            if kwargs["startup_microbatch_size"] is None:
                assert kwargs["state"] is handoff
    monkeypatch.setattr(cooperative_cohort, "FiberCohort", Cohort)
    class Supervisor:
        error = None
        generation = 1
        status = "running"
        def __init__(self, client, **kwargs):
            self.options = kwargs
            self.finished = False
            self.state = kwargs.get("state")
            self.cohort = None
            self.grant = grant
            events.append(("register", kwargs.get("pausable", True), kwargs.get("adaptable", True)))
        def stop(self, reason):
            events.append("joined_before_finish")
            self.cohort = None
            self.state = handoff
            self.finished = True
        def tick(self):
            self.cohort = self.options["cohort_factory"](grant, self.state)
            callback = self.cohort.options["on_startup_observation"]
            if callback:
                row = observation(updates=3, tested=True)
                row.update(config_fingerprint=module.config_fingerprint(raw), initial_test_peak_bytes=35)
                callback([row], {"step": 3, "phase": "update", "done": run_done})
            else:
                self.cohort = None
                self.finished = True
        def fail(self, error):
            raise error
    monkeypatch.setattr(supervisor, "Supervisor", Supervisor)
    module.run_cooperative_training(tmp_path / "config.json")
    assert events.count("broker") == events.count("writer") == 1
    assert events.count("state_closed") == 1
    if run_done:
        assert ("register", True, True) not in events
    else:
        assert events.index("joined_before_finish") < events.index(("register", True, True))
    assert ("register", False, False) in events
    assert len(list((tmp_path / ".memory-calibrations").glob("*.json"))) == 1


# cooperative setup


@pytest.mark.parametrize("settings,override,expected", [
    ({}, None, 0), ({"priority": 7}, None, 7),
    ({"request": {"priority": 3}}, None, 3),
    ({"priority": 7, "request": {"priority": 3}}, None, 7),
    ({"priority": 7}, 0, 0),
])
def test_priority_precedence(settings, override, expected):
    assert task_priority(settings, override) == expected


@pytest.mark.parametrize("value", [True, 1.5, "7"])
def test_priority_rejects_non_integer(value):
    with pytest.raises(ValueError, match="integer"):
        task_priority({"priority": value})


def fixture_config(tmp_path):
    raw = configuration()
    raw["training"]["run_path"] = str(tmp_path / "runs")
    observation = raw["training"]["cooperative"]["memory_calibration"]
    identity = config_fingerprint(raw)
    observation["config_fingerprint"] = identity
    for row in observation["observations"]:
        row.update(config_fingerprint=identity, torch_version=str(torch.__version__), cuda_version=torch.version.cuda or "cpu-test")
    return raw


def test_zero_manager_config_loads_report_and_derives_request(monkeypatch, tmp_path):
    from villa_gpu_manager import bootstrap
    monkeypatch.setattr(torch.version, "cuda", "cpu-test")
    raw = fixture_config(tmp_path)
    observation = raw["training"].pop("cooperative")["memory_calibration"]
    config = tmp_path / "config.json"
    target = calibration_report_path(config, raw)
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(observation))
    original = deepcopy(raw)
    inventory = {"cpu_cores": 16, "gpus": {"GPU-a": 200, "GPU-b": 200, "GPU-c": 200}}
    monkeypatch.setattr(bootstrap, "ensure_manager", lambda path: (path, inventory))
    monkeypatch.setattr(bootstrap, "compatible_gpu_uuids", lambda names: ["GPU-a", "GPU-b"])
    prepared = prepare_settings(config, raw, socket="/tmp/test.sock")
    settings = launch_settings(prepared)
    assert settings.request["max_gpus"] == 2
    assert settings.request["allowed_gpu_uuids"] == ["GPU-a", "GPU-b"]
    assert settings.workers == 16
    assert settings.batch_size == 128
    assert raw == original
    assert config_fingerprint(prepared) == config_fingerprint(raw)


def test_missing_calibration_requests_exclusive_real_startup(monkeypatch, tmp_path):
    from villa_gpu_manager import bootstrap
    inventory = {"cpu_cores": 224, "gpus": {"GPU-a": 80 * 1024**3, "GPU-b": 80 * 1024**3}}
    monkeypatch.setattr(bootstrap, "ensure_manager", lambda path: (path, inventory))
    monkeypatch.setattr(bootstrap, "visible_gpu_uuids", lambda: ["GPU-a", "GPU-b"])
    raw = fixture_config(tmp_path)
    raw["training"].pop("cooperative")
    raw["training"]["sample_pipeline_workers"] = 128
    raw["test_datasets"] = ["test-fixture"]
    raw["training"]["test_interval"] = 10
    prepared = prepare_settings(tmp_path / "config.json", raw)
    settings = launch_settings(prepared)
    assert settings.calibration is None
    assert settings.request["exclusive"]
    assert settings.request["priority"] == 100
    assert settings.request["max_gpus"] == 2
    assert settings.workers == 128
    assert "memory_calibration" not in prepared["training"]["cooperative"]


@pytest.mark.parametrize('scope', ['global', 'visible_gpus'])
def test_missing_calibration_accepts_explicit_visible_managed_subset(monkeypatch, tmp_path, scope):
    from villa_gpu_manager import bootstrap
    inventory = {
        "cpu_cores": 224,
        "gpus": {"GPU-a": 80 * 1024**3, "GPU-b": 80 * 1024**3},
    }
    monkeypatch.setattr(bootstrap, "ensure_manager", lambda path: (path, inventory))
    monkeypatch.setattr(bootstrap, "visible_gpu_uuids", lambda: ["GPU-a"])
    raw = fixture_config(tmp_path)
    raw["training"].pop("cooperative")
    raw['training']['cooperative'] = {'startup_calibration_scope': scope}
    raw["test_datasets"] = ["test-fixture"]
    raw["training"]["test_interval"] = 10
    prepared = prepare_settings(tmp_path / "config.json", raw)
    request = prepared["training"]["cooperative"]["request"]
    assert request["exclusive"] == (scope == 'global')
    assert request['min_vram_bytes_per_gpu'] == request['max_vram_bytes_per_gpu'] == inventory['gpus']['GPU-a']
    assert request["max_gpus"] == 1
    assert request["allowed_gpu_uuids"] == ["GPU-a"]
    from villa_gpu_manager.supervisor import Supervisor
    settings = launch_settings(prepared)
    supervisor = Supervisor(None, label='test', request=request, cohort_factory=None,
                            pausable=False, adaptable=False,
                            disconnect_grace_seconds=(None if scope == 'global' else 60))
    assert supervisor.disconnect_grace_seconds == (None if scope == 'global' else 60)


def test_single_gpu_calibration_does_not_limit_replica_count(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.version, "cuda", "cpu-test")
    raw = fixture_config(tmp_path)
    for row in raw["training"]["cooperative"]["memory_calibration"]["observations"]:
        row["world_size"] = 1
    obs, cal = load_calibration(tmp_path / "config.json", raw)
    inventory = {"cpu_cores": 128, "gpus": {f"GPU-{i}": 200 for i in range(8)}}
    request = derive_request(raw["training"], inventory, list(inventory["gpus"]), obs, cal)
    assert request["max_gpus"] == 8
    with pytest.raises(ValueError, match="no visible"):
        derive_request(raw["training"], inventory, [], obs, cal)


@pytest.mark.parametrize("requested,expected", [(5, 5), (3, 3), (8, 5)])
def test_single_gpu_profile_preserves_requested_count_and_cap(monkeypatch, tmp_path, requested, expected):
    from villa_gpu_manager import bootstrap
    monkeypatch.setattr(torch.version, "cuda", "cpu-test")
    raw = fixture_config(tmp_path)
    settings = raw["training"]["cooperative"]
    for row in settings["memory_calibration"]["observations"]:
        row["world_size"] = 1
    settings["request"].update(max_gpus=requested, max_vram_bytes_per_gpu=180,
                               allowed_gpu_uuids=[])
    visible = [f"GPU-{i}" for i in range(5)]
    monkeypatch.setattr(bootstrap, "ensure_manager", lambda path: (path,
        {"cpu_cores": 128, "gpus": {f"GPU-{i}": 200 for i in range(8)}}))
    monkeypatch.setattr(bootstrap, "compatible_gpu_uuids", lambda _: visible)
    prepared = prepare_settings(tmp_path / "config.json", raw)
    launch = launch_settings(prepared)
    assert launch.request["max_gpus"] == expected
    assert launch.request["allowed_gpu_uuids"] == visible
    assert launch.request["max_vram_bytes_per_gpu"] == 180


def test_explicit_request_cannot_bypass_visibility(monkeypatch, tmp_path):
    from villa_gpu_manager import bootstrap
    monkeypatch.setattr(torch.version, "cuda", "cpu-test")
    raw = fixture_config(tmp_path)
    monkeypatch.setattr(bootstrap, "ensure_manager", lambda path: (path,
        {"cpu_cores": 128, "gpus": {"GPU-a": 200, "GPU-b": 200}}))
    monkeypatch.setattr(bootstrap, "compatible_gpu_uuids", lambda _: ["GPU-b"])
    prepared = prepare_settings(tmp_path / "config.json", raw)
    assert prepared["training"]["cooperative"]["request"]["allowed_gpu_uuids"] == ["GPU-b"]
    assert prepared["training"]["cooperative"]["request"]["max_gpus"] == 1
    monkeypatch.setattr(bootstrap, "compatible_gpu_uuids", lambda _: [])
    with pytest.raises(ValueError, match="no visible"):
        prepare_settings(tmp_path / "config.json", raw)

@pytest.mark.parametrize("stride", [1, 2, 4])
def test_real_paris4_coarse_output_supervision_and_visualization(tmp_path, stride):
    from tests.neural_tracing.test_resumable_training import _bounded_config, _finish
    raw, path = _bounded_config(tmp_path)
    raw["model"].update(features_per_stage=[2, 4, 8], output_stride=stride,
                        normalization="group")
    raw["training"].update(max_steps=1, sample_vis_interval=1, test_sample_vis_interval=2)
    path.write_text(json.dumps(raw))
    state, writer, boundaries = _finish(path, physical=(2,))
    assert state["progress"]["step"] == 1
    assert boundaries[-1] == (1, "update")
    assert torch.isfinite(torch.as_tensor(writer.scalars[("train/loss_total", 5)]))
    model = build_fiber_trace_3d_model(raw)
    # Native output avoids constructing unused high-resolution decoder stages.
    assert len(model.net.decoder.stages) == 2 - {1: 0, 2: 1, 4: 2}[stride]
    output = model(torch.zeros(1, 1, 32, 32, 32))
    assert output.shape == (1, 7, 32 // stride, 32 // stride, 32 // stride)


def test_model_default_output_stride_preserves_snapshot_layout():
    raw = {"model": {"features_per_stage": [2, 4], "normalization": "none"}}
    torch.manual_seed(17)
    old = build_fiber_trace_3d_model(raw)
    torch.manual_seed(17)
    explicit = build_fiber_trace_3d_model({"model": {**raw["model"], "output_stride": 1}})
    assert old.state_dict().keys() == explicit.state_dict().keys()
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    assert old.net.decoder.final_transpconv is not None


def test_output_lattice_default_preserves_anisotropic_strides():
    from vesuvius.neural_tracing.nets.output_lattice import output_stage_index
    assert output_stage_index(((1, 1, 1), (1, 2, 2)), 1) == 0
    with pytest.raises(ValueError, match="isotropic"):
        output_stage_index(((1, 1, 1), (1, 2, 2)), 2)
    with pytest.raises(ValueError, match="not represented"):
        output_stage_index(((1, 1, 1), (2, 2, 2)), 3)


def test_coarse_output_inference_adapter_metadata():
    from vesuvius.neural_tracing.fiber_trace_3d.inference_adapter import FiberTrace3DPredictAdapter
    adapter = FiberTrace3DPredictAdapter({"model": {"features_per_stage": [2, 4],
                                                     "output_stride": 2}})
    assert adapter.native_output_stride == 2


def test_training_manager_dependency_matches_locked_monorepo_source():
    from pathlib import Path
    import tomllib
    project = Path(__file__).resolve().parents[2]
    config = tomllib.loads((project / "pyproject.toml").read_text())
    lock = tomllib.loads((project / "uv.lock").read_text())
    package = next(row for row in lock["package"] if row["name"] == "vesuvius")
    for extra in ("models", "all"):
        assert "villa-gpu-manager" in config["project"]["optional-dependencies"][extra]
        assert {"name": "villa-gpu-manager"} in package["optional-dependencies"][extra]
    manager = next(row for row in lock["package"] if row["name"] == "villa-gpu-manager")
    assert manager["source"]["editable"] == config["tool"]["uv"]["sources"]["villa-gpu-manager"]["path"]
    assert manager["dependencies"] == [{"name": "psutil"}]


def _run_default_managed_training(path, socket, device):
    # Production supervisor owns no CUDA context. Run it in a fresh process,
    # not the pytest parent which has already run reference GPU forwards.
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    from vesuvius.neural_tracing.fiber_trace_3d import cooperative_training
    cooperative_training.run_cooperative_training(path, socket=socket)


@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires manager reservation")
def test_cuda_real_default_startup_calibrates_then_trains_same_run(tmp_path):
    import multiprocessing as mp
    import tempfile
    import threading
    from villa_gpu_manager.server import Server
    from vesuvius.neural_tracing.fiber_trace_3d.memory_calibration import calibration_report_path
    raw, path = _bounded_config(tmp_path, device="cuda:0", depth=2)
    raw["batch_size"] = 1
    raw["training"].update(tensorboard_enabled=False, sample_vis_interval=0,
        test_sample_vis_interval=0, max_steps=4, sample_pipeline_workers=2,
        sample_pipeline_slots=5, sample_pipeline_initial_in_flight=2,
        sync_batchnorm=False, ddp_broadcast_buffers=False)
    path.write_text(json.dumps(raw))
    devices = os.environ["FIBER_COOPERATIVE_TEST_GPUS"].split(",")
    inventory = {"cpu_cores": 4, "gpus": {gpu: 20 * 1024**3 for gpu in devices}}
    stopped = threading.Event()
    errors = []
    with tempfile.TemporaryDirectory(prefix="fiber-startup-", dir=tmp_path.parent) as directory:
        with Server(os.path.join(directory, "m", "s"), inventory,
                    engine_options={"protected_processes": lambda: []}) as server:
            def serve():
                try:
                    while not stopped.is_set():
                        server.step(.002)
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=serve)
            thread.start()
            process = mp.get_context("spawn").Process(target=_run_default_managed_training,
                args=(str(path), str(server.path), ",".join(devices)))
            process.start()
            try:
                deadline = time.monotonic() + 90
                while process.is_alive():
                    assert not errors
                    assert time.monotonic() < deadline, "automatic real-update calibration stalled"
                    process.join(.05)
                assert process.exitcode == 0
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(20)
                if process.is_alive():
                    process.kill()
                    process.join(5)
                stopped.set()
                thread.join(5)
            assert not errors and not thread.is_alive()
            tasks = sorted(server.engine.tasks.values(), key=lambda task: task["order"])
            assert len(tasks) == 2
            assert tasks[0]["request"]["exclusive"]
            assert not tasks[1]["request"]["exclusive"]
            assert all(task["terminal"] and not task["held"]["gpus"] for task in tasks)
    report = json.loads(calibration_report_path(path, train._load_raw_config(path)).read_text())
    assert report["estimate"]["max_microbatch"] == 1
    assert report["observations"][0]["initial_test_completed"]
    snapshots = list(path.parent.glob("runs/*/snapshots/current.pt"))
    assert len(snapshots) == 1
    payload = torch.load(snapshots[0], map_location="cpu", weights_only=False)
    assert payload["step"] == 4 and payload["samples_seen"] == 20

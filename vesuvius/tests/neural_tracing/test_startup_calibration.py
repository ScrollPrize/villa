import pytest

from vesuvius.neural_tracing.fiber_trace_3d.startup_calibration import StartupMemoryCalibration


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

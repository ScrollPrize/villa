"""CPU tests for the opt-in reference validation/measurement harness."""

import pytest
import torch


def test_benchmark_requires_successful_applied_resize_evidence(tmp_path):
    import copy
    import json
    from vesuvius.neural_tracing.fiber_trace_3d.cooperative_reference_lifecycle import validate_benchmark_transitions
    full = {"GPU-a": 100, "GPU-b": 100}
    partial = {"GPU-a": 40, "GPU-b": 40}
    shrink = {"GPU-a": 40}
    transitions = [{"frontier": step * 128, "grant": {"cpu_cores": 8, "gpus": gpus}}
                   for step, gpus in ((0, full), (1, shrink), (4, full), (20, partial),
                                      (40, full), (60, shrink), (80, full))]
    units = [{"step": step + 1, "gpus": count, "update_completed": True}
             for step, count in ((20, 2), (40, 2), (60, 1), (80, 2))]
    saved = {"dynamic": {"transitions": transitions, "units": units}}
    path = tmp_path / "lifecycle.json"
    path.write_text(json.dumps(saved))
    assert len(validate_benchmark_transitions(path)) == 4
    for index in range(4):
        broken = copy.deepcopy(saved)
        del broken["dynamic"]["units"][index]
        with pytest.raises(AssertionError, match="successfully applied"):
            validate_benchmark_transitions(broken)
    broken = copy.deepcopy(saved)
    broken["dynamic"]["transitions"][3]["grant"]["gpus"] = full
    with pytest.raises(AssertionError, match="partial"):
        validate_benchmark_transitions(broken)
    # A failed factory immediately replaced at the same frontier owns no units.
    broken = copy.deepcopy(saved)
    broken["dynamic"]["transitions"].insert(4, {"frontier": 20 * 128,
        "grant": {"cpu_cores": 8, "gpus": full}})
    with pytest.raises(AssertionError, match="partial"):
        validate_benchmark_transitions(broken)


def test_reference_diagnostics_accept_reused_payload_without_local_files(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.cooperative_reference_lifecycle import describe_state_differences
    state = {"model": {"weight": torch.ones(2)},
             "optimizer": {"state": {0: {"step": torch.tensor(1.), "exp_avg": torch.ones(2)}}},
             "progress": {"step": 1, "sample_frontier": 128, "best_metric": 2., "pending_test": None}}
    result = describe_state_differences(tmp_path, payloads={"fixed": state, "dynamic": state})
    assert result["model"]["strict_mismatches"] == 0
    assert result["optimizer"]["relative_l2"] == 0


def test_reference_benchmark_statistics():
    from vesuvius.neural_tracing.fiber_trace_3d.cooperative_reference_benchmark import timing_summary
    result = timing_summary([1, 2, 3, 4, 5])
    assert result["n"] == 5 and result["mean"] == 3 and result["p50"] == 3
    assert result["p95"] == pytest.approx(4.8)
    with pytest.raises(ValueError):
        timing_summary([])


def test_profile_sums_repeated_microbatch_stages(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import training_profile
    ticks = iter([0., 1., 2., 3., 5.])
    monkeypatch.setattr(training_profile.time, "perf_counter", lambda: next(ticks))
    profile = training_profile.TrainingProfile({"profile_enabled": True, "profile_trace_steps": 0},
                                              tmp_path, 0, torch.device("cpu"))
    try:
        profile.begin(1)
        with profile.stage("component_backward"):
            pass
        with profile.stage("component_backward"):
            pass
        assert profile.row["component_backward_wall_ms"] == 3000
    finally:
        profile.close()


def test_unmanaged_reference_driver_drains_before_finishing():
    from vesuvius.neural_tracing.fiber_trace_3d.cooperative_reference_lifecycle import _DirectCohortRunner
    class Cohort:
        ready = idle = True
        done = closed = False
        state = "state"
        def authorize(self):
            self.authorized = True
        def start_unit(self):
            self.done = True
        def poll(self):
            pass
        def drain(self):
            self.closed = True
    cohort = Cohort()
    runner = _DirectCohortRunner(lambda grant, state: cohort, {})
    runner.tick()
    assert cohort.authorized and not runner.finished
    runner.tick()
    assert runner.finished and runner.state == "state" and runner.cohort is None

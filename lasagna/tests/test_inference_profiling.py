import threading

import pytest

from lasagna import inference_profiling as profiling


def test_stage_clocks_and_exception_are_recorded(monkeypatch):
    wall, cpu = iter([10., 12.]), iter([3., 3.25])
    usages = iter([{"ru_nvcsw": 2}, {"ru_nvcsw": 5}])
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(wall))
    monkeypatch.setattr(profiling.time, "thread_time", lambda: next(cpu))
    monkeypatch.setattr(profiling, "_thread_usage", lambda: next(usages))
    rows = []
    with pytest.raises(ValueError):
        with profiling.stage_timing(rows, "sqlite") as details:
            details["test_detail"] = 7
            raise ValueError("fixture")
    assert len(rows) == 1
    assert rows[0]["wall_seconds"] == 2
    assert rows[0]["thread_cpu_seconds"] == .25
    assert rows[0]["off_cpu_seconds_estimate"] == 1.75
    assert rows[0]["ru_nvcsw"] == 3
    assert rows[0]["test_detail"] == 7


def test_disabled_stages_do_not_read_clocks(monkeypatch):
    def forbidden():
        raise AssertionError("disabled timer called")
    monkeypatch.setattr(profiling.time, "perf_counter", forbidden)
    monkeypatch.setattr(profiling.time, "thread_time", forbidden)
    monkeypatch.setattr(profiling, "_thread_usage", forbidden)
    with profiling.stage_timing(None, "disabled") as details:
        assert details is None


def test_unavailable_thread_counters_are_absent(monkeypatch):
    monkeypatch.setattr(profiling, "resource", None)
    rows = []
    with profiling.stage_timing(rows, "no-platform-counters"):
        pass
    assert not any(k.startswith("ru_") for k in rows[0])


def test_unsupported_thread_usage_call_is_absent(monkeypatch):
    from types import SimpleNamespace
    def unsupported(_):
        raise OSError("not supported")
    monkeypatch.setattr(profiling, "resource", SimpleNamespace(RUSAGE_THREAD=1, getrusage=unsupported))
    assert profiling._thread_usage() == {}


def test_parallel_stage_records_keep_caller_thread_ids():
    from concurrent.futures import ThreadPoolExecutor
    barrier = threading.Barrier(8)
    rows = []
    def work(_):
        with profiling.stage_timing(rows, "parallel"):
            barrier.wait(timeout=5)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(work, range(8)))
    assert len(rows) == len({r["thread_id"] for r in rows}) == 8
    assert all(r["wall_seconds"] >= 0 and r["thread_cpu_seconds"] >= 0 for r in rows)

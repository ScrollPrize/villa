from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_distribution_reports_samples_and_percentiles():
    from vesuvius.neural_tracing.fiber_trace_3d.live_inference_profile import distribution
    assert distribution([]) == {"count": 0}
    result = distribution([1, 2, 3, 4])
    assert result["count"] == 4
    assert result["sum"] == 10
    assert result["mean"] == result["p50"] == 2.5
    assert result["p95"] == pytest.approx(3.85)


def test_responsiveness_probe_records_and_cleans_up_on_error():
    import threading
    import time
    from vesuvius.neural_tracing.fiber_trace_3d.live_inference_profile import responsiveness_probe
    report = responsiveness_probe(lambda: time.sleep(.005), iterations=3, interval=.001)
    assert report["iterations"] == 3 and report["heartbeat_gap_seconds"]["count"] > 0
    assert report["stages"]["contained"]["operation"]["wall_seconds"]["count"] == 3
    assert report["stages"]["contained"]["result_release"]["wall_seconds"]["count"] == 3
    def fail():
        raise RuntimeError("probe fixture")
    with pytest.raises(RuntimeError, match="probe fixture"):
        responsiveness_probe(fail)
    assert not any(t.name == "inference-profile-heartbeat" for t in threading.enumerate())
    with pytest.raises(ValueError, match="positive"):
        responsiveness_probe(fail, iterations=0)


def test_native_reader_settings_and_classic_concurrency():
    import json
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import native_reader_options
    config_path = Path(__file__).resolve().parents[2] / "src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_collection.json"
    config = json.loads(config_path.read_text())
    settings = config["evaluation"]["inference"]
    assert settings["input_workers"] == 128
    assert settings["input_slots"] == 256 and settings["result_slots"] == 128
    assert settings["accumulator_layout"] == "chunk-major"
    assert config["evaluation"]["workers"] == 0  # native automatic CPU budget, not a case limit
    assert native_reader_options(settings) == dict(cache_budget_bytes=4 << 30,
        io_threads=64, decode_threads=8, sample_threads=1)
    assert native_reader_options({})["io_threads"] == 16  # legacy callers unchanged


def test_stage_summary_excludes_crossing_window_from_totals():
    from vesuvius.neural_tracing.fiber_trace_3d.live_inference_profile import stage_report
    rows = [{"stage": "submit", "start_seconds": start, "end_seconds": end,
             "thread_id": 1, "wall_seconds": end - start, "thread_cpu_seconds": .1}
            for start, end in [(1, 2), (9, 11), (11, 12), (19, 21), (21, 22)]]
    report = stage_report(rows, 10, 20)
    assert report["contained"]["submit"]["wall_seconds"]["sum"] == 1
    assert report["contained"]["submit"]["wall_seconds"]["count"] == 1
    assert report["crossing_counts"] == {"submit": 2}
    assert len(report["records"]) == 3
    assert report["records"][0]["start_seconds"] == -1


def test_process_counters_parse_linux_fields_and_missing(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.live_inference_profile import process_counters
    import os
    assert process_counters(77, tmp_path) == {}
    proc = tmp_path / "77"
    proc.mkdir()
    fields = ["0"] * 50
    fields[0] = "D"
    fields[7], fields[9], fields[11], fields[12], fields[39] = "11", "3", "20", "40", "6"
    (proc / "stat").write_text("77 (name with ) spaces) " + " ".join(fields))
    (proc / "io").write_text("rchar: 1000\nwchar: 2000\nread_bytes: 4096\nwrite_bytes: 8192\n")
    (proc / "schedstat").write_text("1000000000 2000000000 33")
    result = process_counters(77, tmp_path)
    assert result["minor_faults"] == 11 and result["major_faults"] == 3
    assert result["user_seconds"] == 20 / os.sysconf("SC_CLK_TCK")
    assert result["block_io_delay_seconds"] == 6 / os.sysconf("SC_CLK_TCK")
    assert result["main_thread_runnable_wait_seconds"] == 2
    assert result["read_bytes"] == 4096 and result["rchar"] == 1000

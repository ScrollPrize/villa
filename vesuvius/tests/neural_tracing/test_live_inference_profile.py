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

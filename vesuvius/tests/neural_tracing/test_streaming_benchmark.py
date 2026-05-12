"""Smoke test for the streaming benchmark — runs it in synthetic mode and
asserts that the cached path beats the naive path on a tiny CPU model."""

from __future__ import annotations

import json
from pathlib import Path

from vesuvius.models.benchmarks.benchmark_trace_fiber import main as benchmark_main


def test_benchmark_synthetic_writes_report_and_shows_speedup(tmp_path: Path) -> None:
    out_path = tmp_path / "report.json"
    rc = benchmark_main(["--max-steps", "12", "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["config"] == "synthetic"
    assert "cached" in payload and "naive" in payload
    cached_sps = float(payload["cached"]["steps_per_second"])
    naive_sps = float(payload["naive"]["steps_per_second"])
    assert cached_sps > 0
    assert naive_sps > 0
    # On a 2-decoder-block CPU model with 12 steps + 2 reanchors, the cached
    # path's amortised cost beats naive even at this tiny scale. The ratio is
    # modest here; on the real config (16 blocks, dim 768, 500+ steps) the
    # gap is ~10x because per-step cost goes from O(T^2) -> O(T).
    assert payload["speedup_cached_vs_naive"] >= 1.0
    assert isinstance(payload["chunk_cache_stats"], dict)
    assert payload["chunk_cache_stats"]["misses"] >= 0

"""Exercise the native-only benchmark without optional comparison dependencies."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_native_benchmark_all_cases_without_torch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    for name in ("torch", "vesuvius", "scipy"):
        monkeypatch.setitem(sys.modules, name, None)
    bench = importlib.import_module("bench_augment")
    array = tmp_path / "ct.zarr"
    array.mkdir()
    # Small uncompressed reader fixture; performance measurements use real CT.
    (array / ".zarray").write_text(json.dumps(dict(
        zarr_format=2, shape=[16] * 3, chunks=[16] * 3, dtype="|u1",
        compressor=None, filters=None, fill_value=0, order="C")))
    data = np.arange(16 ** 3).astype(np.uint8).reshape((16,) * 3)
    (array / "0.0.0").write_bytes(data.tobytes())
    destination = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [
        "bench_augment.py", "--native-only", "--array", str(array),
        "--start", "0", "0", "0", "--additional-start", "1", "1", "1",
        "--shape", "12", "12", "12", "--reps", "2", "--warmup", "0",
        "--profile", "all", "--output", str(destination)])
    bench.main()
    result = json.loads(destination.read_text())
    assert result["mode"] == "native-only"
    assert len(result["cases"]) == len(bench.cases((12,) * 3)) == 26
    names = {case["name"] for case in result["cases"]}
    assert not names & {
        "CopyOnly", "CropPad", "Morphology", "ReplaceLabel", "Regions", "OneHot",
        "MaskImage", "DeepSupervision", "MedialSurface", "Pseudo2DRoundTrip", "IntensityPipeline",
        "RemoveComponents", "ClipOutliers",
    }
    assert {"RandomBinaryOperator", "Flip", "Rotate90", "Transpose"} <= names
    assert len(result["crops"]) == 2
    times = [case["latency"]["p50_ms"] for case in result["cases"]]
    assert times == sorted(times, reverse=True)
    for case in result["cases"]:
        assert len(case["latency"]["samples_ms"]) == 4
        assert case["latency"]["min_ms"] > 0
        assert "python" not in case
        assert all("native_steps" in crop and crop["output"] for crop in case["crops"])
    assert (array / "0.0.0").read_bytes() == data.tobytes()


def test_native_benchmark_rejects_unknown_case(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    bench = importlib.import_module("bench_augment")
    monkeypatch.setattr(sys, "argv", ["bench_augment.py", "--native-only", "--only", "Typo"])
    with pytest.raises(SystemExit, match="Unknown cases: Typo"):
        bench.main()


@pytest.mark.parametrize("size", [64, 128])
def test_default_workload_keeps_original_crops_and_adds_dense_regions(monkeypatch, size):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    runner = importlib.import_module("bench_augment_native")
    args = SimpleNamespace(start=None, additional_start=[], shape=(size,) * 3)
    starts = runner.crop_starts(args)
    offset = (128 - size) // 2
    original = [(8960, 4480, 3200), (5120, 3840, 3840), (5120, 2816, 5120)]
    assert starts[:3] == [tuple(v + offset for v in origin) for origin in original]
    assert len(set(starts)) == 6
    args.start, args.additional_start = (1, 2, 3), [(4, 5, 6)]
    assert runner.crop_starts(args) == [(1, 2, 3), (4, 5, 6)]

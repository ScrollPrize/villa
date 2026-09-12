import copy
import importlib.util
import json
from pathlib import Path

import pytest

from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import (
    reference_segments, segment_statistics, correct_summary, complete_report,
)


def case(length=10, failures=(4,), **extra):
    return dict(reference_length_base_voxels=length,
                evaluated_through_base_voxels=length, evaluation_complete=True,
                failures=[dict(directional_reference_arc_base_voxels=x) for x in failures], **extra)


def summary():
    sources = []
    for name, scale, c in [("a", 1000, case()), ("b", 2000, case(5, (), direction="reverse"))]:
        sources.append(dict(source_id=name, version=4, cases=[c],
                            backend_config=dict(coordinate_space=name + "/scan"),
                            summary=dict(base_voxel_size_um=scale, directed_reference_length_base_voxels=c["reference_length_base_voxels"])))
    return dict(completed=True, cases=2, failures=1, directed_reference_length_mm=20, sources=sources)


def test_one_restart_and_terminal():
    values = reference_segments(case())
    assert values == [4, 6]
    stats = segment_statistics(values, 1)
    assert stats["mean_segment_length_mm"] == 5
    assert stats["median_segment_length_mm"] == 5
    assert stats["p05_segment_length_mm"] == pytest.approx(4.1)
    assert stats["p95_segment_length_mm"] == pytest.approx(5.9)


def test_zero_failures_repeats_endpoints_and_reverse():
    assert reference_segments(case(failures=())) == [10]
    assert reference_segments(case(failures=(0, 4, 4, 10), direction="reverse")) == [0, 4, 0, 6, 0]
    assert segment_statistics([10, 20], 2)["mean_segment_length_mm"] == 15
    assert segment_statistics([10], 1)["p05_segment_length_mm"] == 10
    assert segment_statistics([], 0)["reliability_segments"] == 0


@pytest.mark.parametrize("arcs", [(-1,), (11,), (5, 4), (float("nan"),)])
def test_bad_arcs(arcs):
    with pytest.raises(ValueError):
        reference_segments(case(failures=arcs))


def test_incomplete():
    c = case()
    c["evaluation_complete"] = False
    with pytest.raises(ValueError):
        reference_segments(c)


def test_pooled_units_and_idempotence():
    s = summary()
    before = copy.deepcopy(s)
    fixed = correct_summary(s)
    assert s == before
    assert fixed["mean_segment_length_mm"] == pytest.approx(20 / 3)
    assert fixed["reliability_segments"] == 3
    assert fixed["sources"][1]["summary"]["mean_segment_length_mm"] == 10
    assert fixed["sources"][0]["cases"] == s["sources"][0]["cases"]
    assert correct_summary(fixed) == fixed
    assert complete_report(fixed)["global_result"]["median_segment_length_mm"] == 6


def test_saved_migration_preserves_original_and_cases(tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts/correct_saved_replay_metrics.py"
    spec = importlib.util.spec_from_file_location("migrate_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "summary.json"
    raw = json.dumps(summary()).encode()
    output.write_bytes(raw)
    trace = tmp_path / "trace.json"
    trace.write_text("untouched")
    assert module.migrate(output)["status"] == "would-correct"
    assert output.read_bytes() == raw
    assert module.migrate(output, True)["status"] == "corrected"
    assert output.with_name("summary.before-complete-segments.json").read_bytes() == raw
    assert trace.read_text() == "untouched"
    fixed = output.read_bytes()
    assert module.migrate(output, True)["status"] == "already-correct"
    assert output.read_bytes() == fixed

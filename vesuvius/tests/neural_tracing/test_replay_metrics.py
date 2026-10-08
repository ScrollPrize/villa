import copy
import importlib.util
import json
from pathlib import Path

import pytest

from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import (
    reference_segments, segment_statistics, correct_summary, complete_report,
    split_report,
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


def test_split_report_uses_same_complete_partition():
    s=summary()
    for source in s['sources']:
        source['backend_config']['fibers']=[source['source_id']]
        source['cases'][0]['source_index']=0
    grouped=split_report(s,{'a':{'split':'fit'},'b':{'split':'test'}})
    assert grouped['fit']==segment_statistics([4,6],1)
    assert grouped['test']==segment_statistics([10],1)
    from vesuvius.neural_tracing.fiber_trace_3d.embedding_integration_report import validate_coverage
    rows=[dict(name='a',metrics=complete_report(s)),dict(name='b',metrics=complete_report(s))]
    validate_coverage(rows)
    rows[1]['metrics']['global_result']['directions']+=1
    with pytest.raises(ValueError,match='coverage'):validate_coverage(rows)


def test_zero_failures_repeats_endpoints_and_reverse():
    assert reference_segments(case(failures=())) == [10]
    assert reference_segments(case(failures=(0, 4, 4, 10), direction="reverse")) == [0, 10]
    assert reference_segments(case(failures=(0, 4, 4, 10)), episode_distance=0) == [0, 4, 0, 6, 0]
    assert segment_statistics([10, 20], 2)["mean_segment_length_mm"] == 30
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


def test_combined_restart_denominator_and_unchanged_quantiles():
    stats = segment_statistics([10, 20, 30, 40, 50, 60, 100, 115.1609036763345], 6)
    assert stats["failures"] == 2
    assert stats["reliability_segments"] == 8
    assert stats["mean_denominator"] == 3
    assert stats["mean_segment_length_mm"] == pytest.approx(425.1609036763345 / 3)
    assert stats["median_segment_length_mm"] == 45
    assert segment_statistics([10, 20, 30], 3)["mean_segment_length_mm"] == 60
    assert segment_statistics([], 0)["mean_denominator"] == 0


def test_previously_filtered_summary_keeps_counts_with_new_mean():
    old = correct_summary(summary())
    old.pop("mean_definition")
    old.pop("mean_denominator")
    old["mean_segment_length_mm"] = 20 / 3
    new = correct_summary(old)
    assert new["failures"] == old["failures"]
    assert new["mean_segment_length_mm"] == 10
    assert new["median_segment_length_mm"] == old["median_segment_length_mm"]
    assert correct_summary(new) == new


def test_pooled_units_and_idempotence():
    s = summary()
    before = copy.deepcopy(s)
    fixed = correct_summary(s)
    assert s == before
    assert fixed["mean_segment_length_mm"] == pytest.approx(20 / 2)
    assert fixed["mean_denominator"] == 2
    assert fixed["mean_segment_length_percent"] == 50
    assert fixed["reliability_segments"] == 3
    assert fixed["sources"][1]["summary"]["mean_segment_length_mm"] == 10
    assert fixed["sources"][0]["cases"][0]["failures"] == s["sources"][0]["cases"][0]["failures"]
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


def test_episodes_chain_boundary_and_run_reset():
    assert reference_segments(case(300, (0, 70, 140, 220))) == [0, 220, 80]
    assert reference_segments(case(300, (10, 89, 169))) == [10, 159, 131]
    assert reference_segments(case(300, (10, 90))) == [10, 80, 210]
    assert reference_segments(case(300, (10, 90), direction="reverse")) == [10, 80, 210]
    assert reference_segments(case(20, (2,))) == [2, 18]  # first event always counts
    for distance in (-1, float("nan")):
        with pytest.raises(ValueError):
            reference_segments(case(), episode_distance=distance)


def test_filtered_summary_is_idempotent_and_validates_original_counts():
    s = summary()
    c = s["sources"][0]["cases"][0]
    c["failures"] = [{"directional_reference_arc_base_voxels": v} for v in (1, 2, 3)]
    s["failures"] = 3
    fixed = correct_summary(s)
    assert fixed["failures"] == 1
    assert len(fixed["sources"][0]["cases"][0]["failures"]) == 3
    assert correct_summary(fixed) == fixed
    assert complete_report(fixed, episode_distance=0)["global_result"]["failures"] == 3
    s["failures"] = 2
    with pytest.raises(ValueError, match="failure total"):
        complete_report(s)


def test_infrastructure_limited_cases_are_not_regular_restart_directions():
    s = summary()
    # The native benchmark preserves the requested case count while omitting a
    # route-state-limited case from ordinary geometric replay outcomes.
    s['sources'][1]['cases'] = []
    s['sources'][1]['summary']['directed_reference_length_base_voxels'] = 0
    s.update(evaluated_cases=1, route_state_limit_cases=1,
             failures=1, directed_reference_length_mm=10,
             requested_directed_reference_length_mm=20)
    report = complete_report(s)
    assert report['global_result']['directions'] == 1
    assert report['global_result']['failures'] == 1
    fixed = correct_summary(s)
    assert fixed['cases'] == 2
    assert fixed['evaluated_cases'] == 1
    assert fixed['route_state_limit_cases'] == 1
    assert fixed['failures'] == 1

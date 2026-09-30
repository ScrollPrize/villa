import pytest

from vesuvius.neural_tracing.fiber_trace_3d.neighborhood_readout_report import summarize


def inputs():
    log = ('independent_reference_summary\n'
           'reference,gauge,calibrated,best_ties,lower_unbounded_best,upper_unbounded_best\n'
           '0,0,1,1,0,0\n0.5,0,1,2,0,1\n'
           'independent_reference_candidates\n'
           'reference,gauge,rank,winding,raw_winding,hard_sign_errors\n'
           '0,0,1,0,0,0\n'
           '0.5,0,1,1.5,1.5,0\n'
           'independent_reference_end\n')
    cohort = dict(reference_nodes_in_solve=0, reference_cross_factors_in_solve=0,
                  ordinary_source_trace_ids=[7], ordinary_original_piece_ids=[3, 4])
    conditioned = dict(traces=[dict(source_trace=7)],
                       pieces=[dict(original_piece=3), dict(original_piece=4)])
    return log, cohort, conditioned


def test_existing_accuracy_and_shift_definitions_and_ambiguity():
    result = summarize(*inputs())
    assert result['independent_percent'] == 50
    assert result['ladder_shift_percent'] == 50
    assert result['ladder_shifts'] == 1
    assert result['tied_best_rows'] == result['unbounded_best_rows'] == 1


def test_rejects_changed_cohort_or_reference_leakage():
    log, cohort, conditioned = inputs()
    cohort['ordinary_source_trace_ids'] = [8]
    with pytest.raises(ValueError, match='cohort differs'):
        summarize(log, cohort, conditioned)
    cohort['reference_nodes_in_solve'] = 1
    with pytest.raises(ValueError, match='leaked'):
        summarize(log, cohort, conditioned)


def test_multiple_gauges_do_not_get_cherry_picked():
    log, cohort, conditioned = inputs()
    log = log.replace('0.5,0,1,2,0,1', '0,1,1,2,0,1')
    log = log.replace('0.5,0,1,1.5,1.5,0', '0,1,1,1.5,1.5,0')
    result = summarize(log, cohort, conditioned)
    assert result['independent_percent'] is None
    assert 'multiple independent gauges' in result['metric_unavailable']
    assert {r['gauge'] for r in result['per_gauge']} == {0, 1}
    assert len(result['full_coverage_gauges']) == 2


def test_missing_reference_invalidates_shift_score_not_denominator():
    log, cohort, conditioned = inputs()
    log = log.replace('0.5,0,1,1.5,1.5,0\n', '')
    result = summarize(log, cohort, conditioned)
    assert result['independent_percent'] == 50
    assert result['independent_missing'] == 1
    assert result['ladder_shift_percent'] is None


def test_same_piece_count_with_different_ids_is_rejected():
    log, cohort, conditioned = inputs()
    cohort['ordinary_original_piece_ids'] = [3, 5]
    with pytest.raises(ValueError, match='cohort differs'):
        summarize(log, cohort, conditioned)

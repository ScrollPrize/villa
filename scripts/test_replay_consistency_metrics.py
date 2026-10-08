import pytest
from reviewed_fiber_evaluation import ROOT  # establish project import paths
from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import consistency_report


def test_filtered_stops_do_not_become_ordinary_restarts_or_credit_skipped_length():
    case = dict(reference_length_base_voxels=1000, traced_length_base_voxels=700,
                restart_count=2, reverse_checks=4, filtered_stops=[[100, 300, 400]])
    summary = dict(sources=[dict(source_id='scroll', summary=dict(base_voxel_size_um=1000), cases=[case])])
    result = consistency_report(summary)['combined']
    assert result['ordinary_restarts'] == 2
    assert result['filtered_stops'] == 1
    assert result['filtered_skipped_mm'] == 300
    assert result['retained_percent'] == 70
    assert result['other_untraced_mm'] == 0
    assert result['mean_retained_mm_per_termination'] == 175
    case['filtered_stops'].append([200, 250, 500])
    with pytest.raises(ValueError, match='overlapping'):
        consistency_report(summary)

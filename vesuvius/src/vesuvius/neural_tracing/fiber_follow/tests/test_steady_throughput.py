"""Throughput includes loader/logging gaps and uses a continuous warm window."""
import pytest

from vesuvius.neural_tracing.fiber_follow.regression.benchmark_feature_training import steady_summary


def rows():
    return [dict(step=i+1, compiled_graphs=int(i == 3), optimizer_ms=1000.,
                 between_updates_seconds=None if i == 0 else .5,
                 observed_states=12, endpoint_states=2, supervised_states=4)
            for i in range(7)]


def test_recompilation_restarts_contiguous_window_and_wall_time_includes_gaps():
    summary, measured = steady_summary(rows(), 2)
    assert [r['step'] for r in measured] == [5, 6, 7]
    assert summary['steady_wall_seconds'] == 4.5
    assert summary['observations_per_second'] == 8.
    assert summary['observations_per_optimizer_second'] == 12.
    assert summary['endpoints_per_second'] == pytest.approx(4/3)


def test_warmup_and_missing_first_gap_are_not_timed():
    data = rows()
    for row in data:
        row['compiled_graphs'] = 0
    _, measured = steady_summary(data, 0)
    assert measured[0]['step'] == 2
    _, measured = steady_summary(data, 5)
    assert measured[0]['step'] == 6
    with pytest.raises(RuntimeError, match='compilation-free'):
        steady_summary(data, len(data))


def test_synthetic_microbatch_preserves_decisions_and_candidate_supervision():
    import torch
    from slab_fixtures import cfg
    from vesuvius.neural_tracing.fiber_follow.regression.benchmark_slabs import (
        synthetic_decisions, decision_microbatches,
    )
    rows = synthetic_decisions(cfg(), 3)
    grouped = decision_microbatches(rows, 3)[0]
    assert len(grouped['hist']) == 3
    torch.testing.assert_close(grouped['x']['fine'], torch.cat([r['x']['fine'] for r in rows]))
    assert not grouped['candidate_mask'][0].any()
    torch.testing.assert_close(grouped['candidate_mask'][1:], torch.cat([r['candidate_mask'] for r in rows[1:]]))

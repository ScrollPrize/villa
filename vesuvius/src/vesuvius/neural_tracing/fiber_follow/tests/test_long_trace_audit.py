import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_follow.evaluation.long_trace_audit import length_weighted_order, truncate_path


def test_length_weighting_is_reproducible_and_without_replacement():
    lengths = np.array([1., 3., 12.])
    a = length_weighted_order(lengths, 3, np.random.default_rng(19))
    np.testing.assert_array_equal(a, length_weighted_order(lengths,3,np.random.default_rng(19)))
    assert sorted(a)==[0,1,2]
    rng = np.random.default_rng(27)
    counts=np.bincount([length_weighted_order(lengths,1,rng)[0] for _ in range(5000)],minlength=3)
    np.testing.assert_allclose(counts/counts.sum(),lengths/lengths.sum(),atol=.025)
    with pytest.raises(ValueError):length_weighted_order([1,0],1,rng)


def test_horizon_clips_by_travel_not_annotation_progress():
    p=np.array([[0.,0,0],[3.,0,0],[3.,4,0]])
    np.testing.assert_allclose(truncate_path(p,5),[[0,0,0],[3,0,0],[3,2,0]])
    np.testing.assert_array_equal(truncate_path(p,9),p)
    np.testing.assert_array_equal(truncate_path(p[:1],5),p[:1])


def test_paired_ratios_match_seed_identity_and_use_total_lengths():
    from vesuvius.neural_tracing.fiber_follow.evaluation.summarize_long_audit import paired_length_metrics
    def row(fiber, correct, wrong):
        return dict(source='test',cohort='length_weighted',fiber=fiber,t0=0.,sign=1.,
                    correct=correct,offtrack=wrong,
                    local=dict(local_correct_length=correct,local_scored_length=correct+wrong,
                               recovered_coverage_length=correct,recovered_available=correct+wrong))
    base=[row(0,1.,1.),row(1,9.,1.)]
    same=paired_length_metrics(base,list(reversed(base)),repeats=30)['all']
    assert same['strict_precision']['base']==pytest.approx(10/12)
    for value in same.values():
        assert value['delta']==0
        assert value['ci95']==[0.,0.]
    changed=paired_length_metrics(base,[row(1,8.,2.),row(0,0.,2.)],repeats=30)['all']
    assert changed['strict_precision']['delta']==pytest.approx(-2/12)
    assert changed['local_correct_length']['delta']==-2.

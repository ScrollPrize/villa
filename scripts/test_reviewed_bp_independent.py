import pytest

from reviewed_bp_independent import independent_args, readout_metrics


def test_independent_args_preserves_objective_and_does_not_mutate():
    original = ['direction-ablation', 'trace.zarr', '--reference-prune-offenders',
        '--reference-prune-policy', 'oracle-inliers', '--reference-oracle-rounds', '1',
        '--winding-sign-weights', '1,1', '--threads', '128']
    result = independent_args(original)
    assert result == ['direction-ablation', 'trace.zarr', '--winding-sign-weights',
                      '1,1', '--threads', '128', '--reference-independent']
    assert '--reference-prune-offenders' in original
    assert independent_args(result) == result


@pytest.mark.parametrize('args', [['--reference-prune-policy'],
    ['--reference-oracle-rounds', '--threads', '128']])
def test_reject_missing_oracle_value(args):
    with pytest.raises(ValueError, match='missing value'):
        independent_args(args)


def test_incomplete_ladder_keeps_26_denominator():
    summaries = ['reference,calibrated,evaluated,best_ties,lower_unbounded_best,upper_unbounded_best']
    candidates = ['reference,winding,hard_sign_errors,rank']
    for i in range(26):
        present = i not in (8, 24)
        summaries.append(f'{i/2},{int(present)},{int(present)},1,0,0')
        if present:
            candidates.append(f'{i/2},{i/2},0,1')
    log = ('independent_reference_summary\n'+'\n'.join(summaries)+'\n'
           'independent_reference_candidates\n'+'\n'.join(candidates)+'\nindependent_reference_end')
    result = readout_metrics(log)
    assert result['independent_percent'] == pytest.approx(100*24/26)
    assert result['ladder_shift_percent'] is None
    assert result['observed_ladder_shift']['references'] == 24
    assert result['observed_ladder_shift']['ladder_shift_percent'] == 100
    with pytest.raises(ValueError, match='uncalibrated'):
        readout_metrics(log.replace('0.0,1,1,1,0,0', '0.0,0,1,1,0,0'))
    with pytest.raises(ValueError, match='uncalibrated'):
        readout_metrics(log.replace('0.0,1,1,1,0,0', '0.0,0,0,1,0,0'))
    satellite=log.replace('0.0,1,1,1,0,0', '0.0,1,1,1,0,0\n0.0,0,0,0,0,0')
    assert readout_metrics(satellite)==result
    with pytest.raises(ValueError,match='multiple independent gauges'):
        readout_metrics(log.replace('0.0,1,1,1,0,0','0.0,1,1,1,0,0\n0.0,1,1,1,0,0'))

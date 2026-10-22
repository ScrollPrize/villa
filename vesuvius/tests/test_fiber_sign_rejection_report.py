import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_sign_rejection_report import cohort_metrics
from vesuvius.neural_tracing.fiber_trace_3d.fiber_sign_rejection_report import markdown_report


def test_cohort_boundaries_and_retention():
    events = [dict(residual=x) for x in [-.5, .5, 2.]]
    order = np.array([0., -1., -2.])
    row = cohort_metrics(events, order, np.array([True, True, False]))
    assert row == dict(events=2, signs=1, l1=.5, correct_percent=50.)
    empty = cohort_metrics(events, order, np.zeros(3, bool))
    assert empty == dict(events=0, signs=0, l1=None, correct_percent=None)


def test_report_includes_all_runs_signs_retention_and_fit():
    report=markdown_report([])
    assert report.startswith('# Sign weight versus disabling cost\n')
    for label in ('All supplied runs','Disabled fibers',
                  'Internal perp signs','GT perp signs','Perp L1','Width/GT',
                  'All-fiber versus enabled signs','Shared retained GT comparison'):
        assert label in report


def test_arbitrary_weight_values_are_not_filtered_into_historical_grids():
    def row(weight,cost):
        metric=dict(events=3,signs=1,l1=.2,correct_percent=66.)
        return dict(weight=weight,rejection=cost,retention=dict(total=5,fibers=4,length_percent=80.),
            internal=[dict(kind=kind,cohort=cohort,violations=1,signed_events=3)
                for kind in ('all','parallel','perpendicular') for cohort in ('all','both_enabled')],
            cohorts={kind:{cohort:metric for cohort in ('all','enabled','baseline_common','trial_common')}
                for kind in ('parallel','perpendicular')},
            score=dict(coverage_by_type={kind:dict(percent=90) for kind in ('parallel','perpendicular')}),
            objective=1,width_factor=1,offset=0,seconds=2)
    report=markdown_report([row(13.7,.83),row(26.2,8.1)])
    assert '|13.7|0.83|' in report and '|26.2|8.1|' in report

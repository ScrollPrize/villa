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


def test_report_has_both_grids_signs_retention_and_fit():
    report=markdown_report([])
    assert report.startswith('# Sign weight versus disabling cost\n')
    for label in ('Matched-factor grid','Original grid','Disabled fibers',
                  'Internal perp signs','GT perp signs','Perp L1','Width/GT',
                  'All-fiber versus enabled signs','Shared retained GT comparison'):
        assert label in report

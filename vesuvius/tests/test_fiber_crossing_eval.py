"""The crossing eval reports the measured residual, not the solver's arcs."""
import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d import fiber_crossing_eval as evaluation


def events(ids, a, b, target, distance):
    return dict(ids=np.array(ids), a=np.array(a, np.int64), b=np.array(b, np.int64),
                target=np.array(target, float), distance=np.array(distance, float),
                weight=np.ones(len(a)))


def test_metrics_separate_the_rounded_and_unrounded_residual():
    """A third of a turn off is a miss unrounded and a hit once rounded."""
    m = evaluation.metrics(np.array([0., 0., 1.]), np.array([0.3, -0.3, 1.2]))
    assert m['crossings'] == 3
    assert m['rounded_exact_percent'] == 100.0
    assert m['rounded_median'] == 0
    assert m['unrounded_median'] == pytest.approx(-0.2)
    assert m['unrounded_mean'] == pytest.approx(-0.0666667)


def test_a_systematic_shift_shows_up_in_the_rounded_median():
    """Every crossing a full turn out: the rounded median is that turn."""
    m = evaluation.metrics(np.zeros(5), np.ones(5))
    assert m['rounded_median'] == -1
    assert m['rounded_exact_percent'] == 0.0


def test_focus_keeps_only_crossings_that_cross_the_group_boundary():
    """Inside the scaffold and inside the traced volume both say nothing here."""
    ids = ['s0', 's1', 't0', 't1']
    data = events(ids, [0, 0, 2], [1, 2, 3], [0., 1., 0.], [0., 0., 0.])
    winding = {u: 0 for u in ids}
    winding['t0'] = 1
    report = evaluation.evaluate(data, winding, {}, focus=('s0', 's1'), largest_component_only=False)
    # Only s0->t0 crosses the boundary: step 1 - 0 = 1, target 1, residual 0.
    assert report['cuts'][0]['crossings'] == 1
    assert report['cuts'][0]['rounded_exact_percent'] == 100.0


def test_only_the_largest_component_is_scored():
    """Dust components carry their own gauge and would add noise, not signal."""
    ids = ['a', 'b', 'c', 'd']
    data = events(ids, [0, 2], [1, 3], [0., 0.], [0., 0.])
    winding = dict(a=0, b=0, c=0, d=5)
    component = dict(a=1, b=1, c=2, d=2)
    report = evaluation.evaluate(data, winding, component, largest_component_only=True)
    assert report['cuts'][0]['crossings'] == 1
    assert report['cuts'][0]['rounded_exact_percent'] == 100.0


def test_tables_render_the_summary_and_the_expanded_per_winding_block():
    m = evaluation.metrics(np.zeros(2), np.zeros(2))
    text = evaluation.tables(dict(cuts={0: m, 1: m}, per_winding={-3: m}))
    assert '| d<=0 |' in text and '| d<=1 |' in text
    assert '<details><summary>per winding' in text
    assert '| -3 |' in text

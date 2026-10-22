import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_split import Pieces
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import solve, split_fractional_edges, corrected_targets


def data():
    arcs = np.array([100, 400, 700, 1000, 1400, 1700, 2000, 2300.])
    return dict(source_id=np.array([10, 20]), links=np.empty((0, 4)),
                node_ds=np.array([[0, 1500, 3000], [0, -1500, -3000]]),
                start=np.zeros(2), end=np.full(2, 3000.),
                i=np.zeros(8, int), j=np.ones(8, int), ds_a=arcs, ds_b=-arcs,
                parallel=np.zeros(8, bool), event_ids=np.arange(8))


def test_eligible_split_reversed_arcs_cut_ownership_and_protection():
    d = data(); p = Pieces(d)
    np.testing.assert_array_equal(p.parent, [0, 0, 1, 1])
    np.testing.assert_array_equal(p.start, [0, 1200, 0, 1200])
    np.testing.assert_array_equal(p.endpoints('a'), [0]*4+[1]*4)
    np.testing.assert_array_equal(p.endpoints('b'), [2]*4+[3]*4)
    assert p.locate(10, 1200) == 1
    assert p.locate(10, 3000) == 1
    assert p.locate(20, 0) == 2
    with pytest.raises(ValueError): p.locate(10, 4000)
    protected = Pieces(d, protected_sources=[20])
    np.testing.assert_array_equal(protected.parent, [0, 0, 1])
    np.testing.assert_array_equal(protected.endpoints('b'), [2]*8)
    whole = Pieces(d, minimum_length=0)
    np.testing.assert_array_equal(whole.endpoints('a'), d['i'])
    np.testing.assert_array_equal(whole.endpoints('b'), d['j'])


def test_duplicate_observations_do_not_create_split_eligibility():
    d = data()
    d['event_ids'] = np.tile(np.arange(4), 2)
    assert len(Pieces(d).parent) == 2
    d = data(); d['parallel'][4:] = True
    assert len(Pieces(d).parent) == 2


def test_remap_preserves_original_chart_targets_and_can_repair_fiber():
    d = data(); p = Pieces(d, protected_sources=[20])
    measured = np.r_[np.zeros(4), np.ones(4)]
    # Nontrivial seam transport cancels; use the original fiber-start chart.
    target = corrected_targets(measured, np.full(8, 2), np.full(8, 3), np.ones(8))
    g, event = split_fractional_edges(d['i'], d['j'], target+.2, np.ones(8))
    transport = dict(flow_event_index=event)
    sg = p.remap(g, transport)
    np.testing.assert_array_equal(sg['target'], g['target'])
    split = solve(len(p.parent), **sg)
    whole = solve(2, **g)
    assert split['certificate']['primal'] == pytest.approx(1.6)
    assert whole['certificate']['primal'] == pytest.approx(4.0)
    assert split['winding'][1]-split['winding'][0] == -1
    # Reference ownership is endpoint-specific, not last node with source ID.
    assert split['winding'][p.locate(10, 200)] != split['winding'][p.locate(10, 2000)]

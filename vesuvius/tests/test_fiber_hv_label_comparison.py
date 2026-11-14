import json

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.fiber_hv_label_comparison import comparison, run
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_initial_fibers
from .fiber_graph_fixture import uid


def test_disagreement_keeps_rejection_and_arc_denominators_separate():
    p = np.array([[.95,.03,.02], [.05,.9,.05], [.01,.01,.98], [0,0,1], [.6,.3,.1]])
    result = comparison(np.arange(5), [0,0,1,1,-1], [100,300,700,900,1],
                        p, p.argmax(1), [False,False,False,True,False], [.5,.95])
    assert result['retained']['fibers'] == 2
    assert result['retained']['mismatches'] == 1
    assert result['retained']['mismatch_percent'] == 50
    assert result['retained']['mismatch_arc_percent'] == 75
    assert result['normal_stage_broken']['fibers'] == 1
    assert result['hv_stage_broken']['fibers'] == 1
    assert result['original_unknown']['fibers'] == 1
    assert result['by_confidence'][1]['mismatches'] == 0
    assert result['confusion']['H']['V']['fibers'] == 1
    assert result['confusion']['V']['normal_stage_broken']['fibers'] == 1


def test_empty_retained_set_is_not_perfect_agreement():
    result = comparison([1], [0], [100], [[0,0,1]], [2], [True], [.5])
    assert result['retained']['mismatch_percent'] is None
    assert result['retained']['mismatch_arc_percent'] is None


def test_length_histogram_boundaries_and_empty_groups():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_hv_label_comparison import length_histogram
    result = length_histogram([0, 99, 100, 250, 100000, 200000], [0, 0, 1, 1, 2, 2])
    assert result['groups']['all']['counts'][:3] == [2, 1, 1]
    assert result['groups']['all']['counts'][-1] == 2
    assert sum(result['groups']['all']['counts']) == 6
    assert length_histogram([100], [0])['groups']['broken']['mean'] is None


def test_reject_inconsistent_saved_hard_labels():
    with pytest.raises(ValueError, match='Invalid'):
        comparison([1], [0], [100], [[.1,.8,.1]], [0], [False], [.5])


def test_run_matches_ids_before_comparing(tmp_path):
    fibers=tmp_path/'fibers.vcobj'
    write_initial_fibers(fibers,[(dict(id=uid(n),family=family),np.array([[0.,0,0],[length,0,0]]))
        for n,family,length in [(3,'H',100),(9,'V',200)]],volume=uid(999),workers=2,packed=True)
    solution = tmp_path/'solution.npz'
    np.savez(solution, ids=[uid(3),uid(9)], labels=[1,1], probabilities=[[.1,.8,.1],[.05,.9,.05]], inherited_broken=[False,False])
    config = dict(fibers=str(fibers), solution=str(solution), report=str(tmp_path/'report.json'))
    assert run(config)['retained']['mismatches'] == 1
    assert json.loads((tmp_path/'report.json').read_text())['confusion']['H']['V']['fibers'] == 1
    np.savez(solution, ids=[uid(3),uid(10)], labels=[1,1], probabilities=[[.1,.8,.1],[.05,.9,.05]], inherited_broken=[False,False])
    with pytest.raises(ValueError, match='IDs'):
        run(config)

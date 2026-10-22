import json
import sqlite3

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.fiber_hv_label_comparison import comparison, run


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


def test_reject_inconsistent_saved_hard_labels():
    with pytest.raises(ValueError, match='Invalid'):
        comparison([1], [0], [100], [[.1,.8,.1]], [0], [False], [.5])


def test_run_matches_ids_before_comparing(tmp_path):
    afv = tmp_path/'fibers.afv'
    with sqlite3.connect(afv) as db:
        db.execute('CREATE TABLE fibers(id INTEGER,family TEXT,length REAL)')
        db.executemany('INSERT INTO fibers VALUES(?,?,?)', [(3,'H',100), (9,'V',200)])
    solution = tmp_path/'solution.npz'
    np.savez(solution, ids=[3,9], labels=[1,1], probabilities=[[.1,.8,.1],[.05,.9,.05]], inherited_broken=[False,False])
    config = dict(afv=str(afv), solution=str(solution), report=str(tmp_path/'report.json'))
    assert run(config)['retained']['mismatches'] == 1
    assert json.loads((tmp_path/'report.json').read_text())['confusion']['H']['V']['fibers'] == 1
    np.savez(solution, ids=[3,10], labels=[1,1], probabilities=[[.1,.8,.1],[.05,.9,.05]], inherited_broken=[False,False])
    with pytest.raises(ValueError, match='IDs'):
        run(config)

import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_audit import sign_constrained


def U(n):
    return f'00000000-0000-4000-8000-{n:012d}'


def test_sign_tradeoff_and_optimal_face():
    d = dict(i=np.array([0, 0, 0]), j=np.array([1, 1, 1]),
        sign=np.array([-1, -1, 1]), family=np.array([0, 0]),
        parallel=np.ones(3, bool), source_id=np.array([U(10), U(20)]), event_ids=np.arange(3))
    s = dict(winding=np.array([0, -1]), a=d['i'], b=d['j'], target=np.array([-1, -1, 1]),
        weight=np.ones(3), fiber_a=np.zeros(3), fiber_b=np.zeros(3), connector=np.zeros(3))
    hard = sign_constrained(d, s)
    assert hard['objective'] == 3 and hard['violations'] == 0
    tied = sign_constrained(d, s, optimal_cost=2)
    assert tied['objective'] == 2 and tied['total_deficit'] == 1


def test_inconsistent_signs_return_concrete_cycle():
    d = dict(i=np.array([0, 1]), j=np.array([1, 0]), sign=np.ones(2),
        family=np.array([0, 1]), parallel=np.zeros(2, bool),
        source_id=np.array([U(10), U(20)]), event_ids=np.array([50, 51]))
    s = dict(winding=np.array([0, 0]), a=d['i'], b=d['j'], target=np.array([0, 1]),
        weight=np.ones(2), fiber_a=np.zeros(2), fiber_b=np.zeros(2), connector=np.zeros(2))
    result = sign_constrained(d, s)
    assert not result['feasible']
    assert result['cycle_bound_sum'] == -1
    assert {e['event_id'] for e in result['contradictory_cycle']} == {50, 51}

import json
import sqlite3
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_audit import (
    objective_cost, compare_targets, internal_details, gt_details, audit_measurements,
)
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import split_fractional_edges, solve


def test_fractional_and_one_sided_sign_analysis():
    d = dict(i=np.array([0, 0]), j=np.array([1, 1]), sign=np.array([-1, 1]),
             family=np.zeros(2, int), parallel=np.ones(2, bool), source_id=np.array([U(10),U(20)]),
             event_ids=np.arange(2))
    g, _ = split_fractional_edges(d['i'], d['j'], np.array([-1.2, 1.]), np.array([2., 1.]))
    result = solve(2, **g)
    s = dict(g, winding=result['winding'], fiber_a=np.zeros(2), fiber_b=np.zeros(2), connector=np.zeros(2))
    hard = sign_constrained(d, s)
    assert hard['objective'] == pytest.approx(3.4)
    tie = sign_constrained(d, s, result['certificate']['primal'])
    assert tie['objective'] == pytest.approx(result['certificate']['primal'])
    assert tie['total_deficit'] == 1
    # A hinge that permits differences >= 2 must not become |difference-2|.
    d.update(i=np.array([0]), j=np.array([1]), sign=np.array([1]), parallel=np.ones(1, bool))
    s = dict(a=np.array([0]), b=np.array([1]), target=np.array([2]), weight=np.ones(1),
             one_sided=np.ones(1, bool), winding=np.array([0,5]),
             fiber_a=np.zeros(1), fiber_b=np.zeros(1), connector=np.zeros(1))
    assert objective_cost(s, s['winding']) == 0
    assert sign_constrained(d, s, 0)['objective'] == 0


def test_split_sign_cycle_names_actual_pieces():
    d = dict(i=np.array([0,0]), j=np.array([1,1]), sign=np.array([1,-1]),
             family=np.array([0,1]), parallel=np.zeros(2,bool),
             source_id=np.array([U(10),U(20)]), event_ids=np.array([50,51]))
    s = dict(winding=np.zeros(3,int), source_id=np.array([U(10),U(10),U(20)]),
             a=np.array([1,2]), b=np.array([2,1]), target=np.ones(2), weight=np.ones(2),
             event_piece_a=np.array([1,1]), event_piece_b=np.array([2,2]),
             fiber_a=np.zeros(2), fiber_b=np.zeros(2), connector=np.zeros(2))
    r = sign_constrained(d,s)
    assert not r['feasible']
    assert {e['from_node'] for e in r['contradictory_cycle']} == {1,2}
    assert {e['from_source'] for e in r['contradictory_cycle']} == {U(10),U(20)}


def test_comparison_checks_expansion_and_reports_changed_fiber():
    shared = dict(source_id=np.array([U(10),U(20)]), valid=np.array([True]),
                  fiber_a=np.zeros(1), fiber_b=np.zeros(1), connector=np.zeros(1), component=np.zeros(2,int))
    old = dict(shared, a=np.array([0]), b=np.array([1]), target=np.array([2]),
               weight=np.ones(1), one_sided=np.zeros(1,bool), winding=np.array([0,2]))
    g, _ = split_fractional_edges(np.array([0]),np.array([1]),np.array([2.5]),np.ones(1))
    new = dict(shared, **g, winding=np.array([7,10]), corrected_target=np.array([2.5]))
    d = dict(i=np.array([0]), j=np.array([1]), parallel=np.ones(1,bool))
    r = compare_targets(old,new,d,{})
    assert r['changed_node_ids'] == [U(20)]
    assert r['fractional_expansion_checked']
    assert r['objectives']['fractional_baseline'] == .5
    assert r['objectives']['rounded_trial'] == 1
    with pytest.raises(AssertionError):
        compare_targets(old,dict(new,corrected_target=np.array([4.])),d,{})


def test_detail_exports_keep_fractional_residual_and_empty_gt():
    d = dict(i=np.array([0]),j=np.array([1]),source_id=np.array([U(10),U(20)]),
             family=np.array([0,1]),sign=np.array([-1]),parallel=np.zeros(1,bool),
             event_ids=np.array([8]),distance=np.array([1.8]))
    s = dict(winding=np.array([0,0]),component=np.zeros(2,int),
             a=np.array([0]),b=np.array([1]),target=np.array([-1]),weight=np.ones(1),
             one_sided=np.zeros(1,bool),valid=np.ones(1,bool),measured_target=np.array([-1.3]),
             fiber_a=np.zeros(1),fiber_b=np.zeros(1),connector=np.zeros(1))
    result, rows = internal_details(d,s,{})
    assert rows[0]['target_residual'] == 1.3
    assert result['missing_next_winding'] == 1
    assert result['all_fiber_counts'] == {U(10):1,U(20):1}
    assert gt_details([])['events'] == 0


def test_measurement_audit_verifies_sources_and_detects_wrong_cache(tmp_path):
    db = tmp_path/'constraints.afc'
    normal = json.dumps(dict(xyz=[1,0,0],valid=True))
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE constraints (id INTEGER, fiber_a TEXT, fiber_b TEXT, ax REAL, ay REAL, az REAL, bx REAL, by REAL, bz REAL, parallel_score REAL, perpendicular_score REAL, winding_distance REAL, parallel_winding_distance REAL, normal_a TEXT, normal_b TEXT)')
        con.execute('INSERT INTO constraints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (8,U(10),U(20),0,0,0,1,0,0,.9,.1,9.,2.,normal,normal))
    stage = tmp_path/'normals.npz'
    np.savez(stage, ids=np.array([U(10),U(20)]), normal_sign=np.ones(2), normal_convention=np.array('inward'))
    cache = tmp_path/'cache.npz'
    np.savez(cache, unused=np.zeros(1))
    d = dict(event_ids=np.array([8]),source_id=np.array([U(10),U(20)]),i=np.array([0]),j=np.array([1]),
             parallel=np.ones(1,bool),sign=np.array([-1]),distance=np.array([2.]))
    sources=dict(database=str(db),stage1=str(stage))
    r = audit_measurements(dict(tensor_cache=str(cache)), d, sources)
    assert r['rows_verified'] == 1
    assert r['source_afc_recomputation'] == 'verified'
    with pytest.raises(ValueError, match='measurements'):
        audit_measurements(dict(tensor_cache=str(cache)), dict(d,sign=np.array([1])), sources)
    with pytest.raises(ValueError, match='identity'):
        audit_measurements(dict(tensor_cache=str(cache)), dict(d,source_id=np.array([U(11),U(20)])), sources)
    r = audit_measurements(dict(tensor_cache=str(cache)), d, dict(sources,database=str(tmp_path/'missing')))
    assert r['source_afc_recomputation'] == 'unavailable'


def test_audit_config_resolves_optional_sources_and_rejects_flags(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_experiment import load_config
    case = dict(name='trial', dataset=dict(fibers='fibers.vcobj',umbilicus='axis.json',tensor_cache='cache.npz'),
                audit=dict(details=True,sign_analysis=False,rounded_baseline='rounded/solution.npz',
                           measurements=dict(database='missing.afc',stage1='normals.npz')))
    (tmp_path/'normals.npz').touch()
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(output_root='results',cases=[case])))
    settings, dependencies = load_config(config)
    audit = settings['cases'][0]['audit']
    assert audit['rounded_baseline'] == str(tmp_path/'rounded/solution.npz')
    assert str(tmp_path/'normals.npz') in dependencies
    assert str(tmp_path/'missing.afc') not in dependencies
    case['audit']['details'] = 'false'
    config.write_text(json.dumps(dict(output_root='results',cases=[case])))
    with pytest.raises(ValueError,match='boolean'):
        load_config(config)

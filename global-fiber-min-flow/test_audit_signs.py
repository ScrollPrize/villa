import numpy as np
from audit_signs import sign_constrained


def test_sign_tradeoff_and_optimal_face():
    d = dict(i=np.array([0, 0, 0]), j=np.array([1, 1, 1]),
        sign=np.array([-1, -1, 1]), family=np.array([0, 0]),
        parallel=np.ones(3, bool), source_id=np.array([10, 20]), event_ids=np.arange(3))
    s = dict(winding=np.array([0, -1]), a=d['i'], b=d['j'], target=np.array([-1, -1, 1]),
        weight=np.ones(3), fiber_a=np.zeros(3), fiber_b=np.zeros(3), connector=np.zeros(3))
    hard = sign_constrained(d, s)
    assert hard['objective'] == 3 and hard['violations'] == 0
    tied = sign_constrained(d, s, optimal_cost=2)
    assert tied['objective'] == 2 and tied['total_deficit'] == 1


def test_inconsistent_signs_return_concrete_cycle():
    d = dict(i=np.array([0, 1]), j=np.array([1, 0]), sign=np.ones(2),
        family=np.array([0, 1]), parallel=np.zeros(2, bool),
        source_id=np.array([10, 20]), event_ids=np.array([50, 51]))
    s = dict(winding=np.array([0, 0]), a=d['i'], b=d['j'], target=np.array([0, 1]),
        weight=np.ones(2), fiber_a=np.zeros(2), fiber_b=np.zeros(2), connector=np.zeros(2))
    result = sign_constrained(d, s)
    assert not result['feasible']
    assert result['cycle_bound_sum'] == -1
    assert {e['event_id'] for e in result['contradictory_cycle']} == {50, 51}

import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_paired_weight_search import better,refine,rank_pair


def test_numeric_cache_key_ignores_json_number_representation():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_paired_weight_search import numeric_key
    assert numeric_key(dict(x=1,y=.5))==numeric_key(dict(x=1.,y=.5))


def row(sign,l1,passed=True):
    return dict(passed=passed,perp_sign_sum=sign,perp_l1_sum=l1)


def test_lexicographic_sign_first_and_guard():
    assert better(row(1,100),row(2,0))
    assert not better(row(3,0),row(2,100))
    assert better(row(2,.5),row(2,1))
    assert not better(row(0,0,False),row(9,9))
    assert better(row(99,99),row(0,0,False))
    assert not better(row(2,1-1e-8),row(2,1))


def test_rank_uses_both_problems():
    def report(sign,l1):
        return dict(signs=dict(perpendicular=dict(violations=sign)),
                    score=dict(gt_perpendicular=dict(l1=l1)),final=dict(loss=1))
    assert rank_pair(dict(free=report(3,.2),joint=report(4,.3)),dict(passed=True))==row(7,.5)
    with pytest.raises(ValueError):rank_pair(dict(free=report(3,float('nan')),joint=report(4,.3)),dict(passed=True))


def test_recenter_goes_beyond_initial_grid_and_restarts():
    seen=[];accepted=[]
    def evaluate(v):
        seen.append(v['x'])
        return dict(parameters=v,ranking=row(0,abs(v['x']-8)))
    result=refine(dict(x=1),evaluate,['x'],[[.5,2]],dict(x=[.125,32]),
                  lambda *args:accepted.append(args))
    assert result['parameters']['x']==8
    assert len(accepted)==3
    assert 16 in seen


def test_search_can_leave_zero_and_return_to_zero():
    def evaluate(v):return dict(parameters=v,ranking=row(0,abs(v['x']-.1)))
    result=refine(dict(x=0),evaluate,['x'],[[.5,2]],dict(x=[0,1]),lambda *args:None,
                  absolute_values=dict(x=[0,.1,1]))
    assert result['parameters']['x']==.1


def test_optimistic_pruning_uses_sign_count_before_l1():
    from vesuvius.neural_tracing.fiber_trace_3d.paired_search_schedule import can_improve
    def free(sign,l1):return dict(signs=dict(perpendicular=dict(violations=sign)),score=dict(gt_perpendicular=dict(l1=l1)))
    assert not can_improve(free(16,0),row(15,1))
    assert can_improve(free(14,100),row(15,1))
    assert can_improve(free(15,.9),row(15,1))
    assert not can_improve(free(15,1),row(15,1))
    assert not can_improve(free(15,1-1e-8),row(15,1))


def test_cyclic_search_does_not_restart_earlier_coordinates():
    from vesuvius.neural_tracing.fiber_trace_3d.paired_search_schedule import cyclic_refine
    visited=[]
    def evaluate(v):
        return dict(parameters=dict(v),ranking=row(0,abs(v['a']-1)+abs(v['b']-4)+abs(v['c']-2)))
    def batch(vs,best):return [evaluate(v) for v in vs]
    result=cyclic_refine(dict(a=1,b=1,c=1),evaluate,batch,['a','b','c'],[[.5,2]],
        {k:[.5,8] for k in 'abc'},lambda *args:None,
        progress=lambda sweep,name,*args:visited.append((sweep,name)))
    assert result['parameters']==dict(a=1,b=4,c=2)
    assert [name for sweep,name in visited if sweep==1]==['a','b','b','b','c','c']
    assert [name for sweep,name in visited if sweep==2]==['a','b','c']

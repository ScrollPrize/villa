import time
from vesuvius.neural_tracing.fiber_trace_3d.paired_async_schedule import candidate_queue,asynchronous_refine
from vesuvius.neural_tracing.fiber_trace_3d.fiber_paired_weight_search import numeric_key, DEFAULT_FACTOR_GRIDS


def test_default_search_includes_fine_and_ultrafine_steps():
    q = candidate_queue(dict(a=100.), ['a'], DEFAULT_FACTOR_GRIDS, dict(a=[1, 1000]))
    assert [v['a'] for _, v in q] == [50., 200., 75., 150., 90., 110., 97., 103., 99., 101.]


def test_candidates_interleave_parameters_not_batches():
    q=candidate_queue(dict(a=1,b=1,c=1),['a','b','c'],[[.5,2]],{n:[.1,10] for n in 'abc'})
    assert [name for name,_ in q]==['a','b','c','a','b','c']


def test_acceptance_does_not_wait_for_slow_other_parameter():
    def error(v):return abs(v['a']-2)+abs(v['b']-2)
    def result(v):return dict(key=numeric_key(v),parameters=dict(v),ranking=dict(passed=True,perp_sign_sum=0,perp_l1_sum=error(v)))
    initial=result(dict(a=1,b=1));cache={initial['key']:initial};events=[];calls=[];accepted=[]
    def work(v,stage,device):
        calls.append((dict(v),stage,device))
        time.sleep(.12 if stage=='free' and v==dict(a=2,b=1) else .001)
        return dict(signs=dict(perpendicular=dict(violations=0)),score=dict(gt_perpendicular=dict(l1=error(v)/2)))
    best=asynchronous_refine(initial,['a','b'],[[2]],{n:[1,2] for n in 'ab'},['0','1'],['guard'],
        numeric_key,cache,{},work,lambda key,v,parts:result(v),lambda r:None,
        lambda name,v,*args:accepted.append((name,dict(v))),events.append)
    assert best['parameters']==dict(a=2,b=2)
    assert accepted[0]==('b',dict(a=1,b=2))
    assert any(e['revision']>0 and any(j['stale'] for j in e['running']) for e in events)
    assert not any(stage!='free' and v==dict(a=2,b=1) for v,stage,_ in calls)


def test_impossible_screen_never_schedules_followups():
    initial=dict(key='initial',parameters=dict(a=1),ranking=dict(passed=True,perp_sign_sum=1,perp_l1_sum=1))
    calls=[];cache={}
    def work(v,stage,device):
        calls.append(stage)
        return dict(signs=dict(perpendicular=dict(violations=2)),score=dict(gt_perpendicular=dict(l1=0)))
    best=asynchronous_refine(initial,['a'],[[2]],dict(a=[1,2]),['0'],['guard'],numeric_key,cache,{},
        work,lambda *args:None,lambda r:None,lambda *args:None,lambda s:None)
    assert best is initial
    assert calls==['free']
    assert next(iter(cache.values()))['screened_out']

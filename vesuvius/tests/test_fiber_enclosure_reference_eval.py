import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_enclosure_reference_eval import stats,cross_metrics,integer_offset,cross_rows
from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph
from .fiber_graph_fixture import uid, measured_store


def test_cross_offset_and_family_signs():
    target=np.array([0.,1.,2.]);local=target+7
    shift=np.median(local-target)
    r=cross_metrics(local-shift,target,target,np.ones(3),np.array([1,-1,0]),np.array([False,False,True]))
    assert r['all']['l1']==0 and r['all']['correct_step_percent']==100
    assert r['all']['sign_violations']==0
    r=cross_metrics(np.array([-1.,0.,-2.]),target,target,np.ones(3),np.array([1,-1,0]),np.array([False,False,True]))
    assert r['perpendicular']['sign_violations']==2
    assert r['parallel']['sign_violations']==1


def test_stats_coverage_and_weighting():
    r=stats(np.array([0.,2.,np.nan]),np.array([3.,1.,9.]))
    assert r['count']==2 and r['l1']==.5 and r['within_half_percent']==75
    assert stats([])['l1'] is None


def test_integer_gauge_minimizes_l1_without_fractional_sign_shift():
    v=np.array([3.1,3.2,3.3,4.1])
    assert integer_offset(v)==3
    assert integer_offset(v-9)==-6
    assert integer_offset([]) is None
    assert integer_offset([-.5,.5])==0


def test_cross_graph_uses_uuid_membership_not_dense_remapping(tmp_path):
    graph=FiberGraph(measured_store(tmp_path, [(1,10,30,False,.5),(2,40,20,True,2.)]))
    graph.arc_a=np.array([3.,4.])
    graph.arc_b=np.array([5.,6.])
    rows=cross_rows(graph,np.array([uid(10),uid(20)]),np.array([uid(30),uid(40)]),np.ones(2,bool))
    assert rows[:,:4].tolist()==[[0,0,3,5],[1,1,6,4]]
    assert rows[:,6].tolist()==[.5,2.]
    assert cross_rows(graph,np.array([uid(10),uid(20)]),np.array([uid(30),uid(40)]),np.array([False,True])).shape==(1,8)
